from torch.autograd import Variable
import torch.nn as nn
import torch
import numpy as np
from collections import defaultdict


def cosine_sim(im, s):
    """Cosine similarity between all the image and sentence pairs
    """
    return im.mm(s.t())


def order_sim(im, s):
    """Order embeddings similarity measure $max(0, s-im)$
    """
    YmX = (s.unsqueeze(1).expand(s.size(0), im.size(0), s.size(1))
           - im.unsqueeze(0).expand(s.size(0), im.size(0), s.size(1)))
    score = -YmX.clamp(min=0).pow(2).sum(2).sqrt().t()
    return score


class Triplet(nn.Module):
    """
    Compute contrastive loss
    """

    def __init__(self, margin=0.2, measure=False, max_violation=False, count_grads=False):
        super(Triplet, self).__init__()
        self.margin = margin
        if measure == 'order':
            self.sim = order_sim
        else:
            self.sim = cosine_sim

        self.max_violation = max_violation
        self.count_grads = count_grads

    def forward(self, im, s):
        # compute image-sentence score matrix
        scores = self.sim(im, s)
        diagonal = scores.diag().view(im.size(0), 1)
        d1 = diagonal.expand_as(scores)
        d2 = diagonal.t().expand_as(scores)

        # compare every diagonal score to scores in its column
        # caption retrieval
        cost_s = (self.margin + scores - d1).clamp(min=0)
        # compare every diagonal score to scores in its row
        # image retrieval
        cost_im = (self.margin + scores - d2).clamp(min=0)

        # clear diagonals
        mask = torch.eye(scores.size(0)) > .5
        I = Variable(mask)
        if torch.cuda.is_available():
            I = I.cuda()
        cost_s = cost_s.masked_fill_(I, 0)
        cost_im = cost_im.masked_fill_(I, 0)

        # keep the maximum violating negative for each query
        if self.max_violation:
            cost_s = cost_s.max(1)[0]
            cost_im = cost_im.max(0)[0]
        if self.count_grads:
            return self.gradient_counter(cost_s, cost_im)
        else:
            return cost_s.sum() + cost_im.sum()

    def gradient_counter(self, cost_s, cost_im):

        batch_size = cost_im.shape[0]

        i2t_grads = (cost_s > 0).float()
        t2i_grads = (cost_im > 0).float()

        if cost_s.dim() > 1:
            i2t_grads = i2t_grads.sum(dim=1)
            t2i_grads = t2i_grads.sum(dim=1)

        zeros_i2t = (i2t_grads == 0).sum()
        zeros_t2i = (t2i_grads == 0).sum()

        return ('i2t_grads_mean', float(i2t_grads.sum() / (batch_size - zeros_i2t))), ('i2t_grads_sum', float(i2t_grads.sum())), ('zeros_i2t', float(zeros_i2t)),\
               ('t2i_grads_mean', float(t2i_grads.sum() / (batch_size - zeros_t2i))), ('t2i_grad_sum', float(t2i_grads.sum())), ('zeros_t2i', float(zeros_t2i))


class NTXent(nn.Module):

    def __init__(self, tau=0.1):

        super(NTXent, self).__init__()

        self.loss = nn.LogSoftmax(dim=1)
        self.tau = tau
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, images, captions, count_gradients=False, threshold=0.05):

        t2i = cosine_sim(captions, images) / self.tau
        i2t = cosine_sim(images, captions) / self.tau

        if count_gradients:
            return self.gradient_counter(t2i, i2t, threshold)

        image_retrieval_loss = - self.loss(t2i).diag().mean()
        caption_retrieval_loss = - self.loss(i2t).diag().mean()

        loss = 0.5 * caption_retrieval_loss + 0.5 * image_retrieval_loss

        return loss

    def gradient_counter(self, t2i, i2t, threshold=0.05):

        s_i2t = self.softmax(i2t)
        s_t2i = self.softmax(t2i)

        batch_size = t2i.shape[0]

        negative_idx = torch.ones(batch_size).cuda() - torch.eye(batch_size).cuda()

        s_i2t_negatives = s_i2t * (negative_idx)
        s_t2i_negatives = s_t2i * (negative_idx)

        i2t_low_grad = (s_i2t_negatives * (s_i2t_negatives < threshold).int()).sum(dim=1).mean()
        i2t_high_grad = (s_i2t_negatives * (s_i2t_negatives > threshold).int()).sum(dim=1).mean()
        i2t_n_high_grad = (s_i2t_negatives > threshold).sum(dim=1).float().mean()

        t2i_low_grad = (s_t2i_negatives * (s_t2i_negatives < threshold).int()).sum(dim=1).mean()
        t2i_high_grad = (s_t2i_negatives * (s_t2i_negatives > threshold).int()).sum(dim=1).mean()
        t2i_n_high_grad = (s_t2i_negatives > threshold).sum(dim=1).float().mean()

        return ('1-s_i2t.diag.mean', float(1-s_i2t.diag().mean())), ('1-s_t2i.diag.mean', float(1-s_t2i.diag().mean())), \
               ('i2t_low_grad', float(i2t_low_grad)), ('i2t_high_grad', float(i2t_high_grad)), ('i2t_n_high_grad', float(i2t_n_high_grad)), \
               ('t2i_low_grad', float(t2i_low_grad)), ('t2i_high_grad', float(t2i_high_grad)), ('t2i_n_high_grad', float(t2i_n_high_grad))


class SmoothAP(torch.nn.Module):

    def __init__(self, n=5, n_negatives=None, tau=0.01, threshold = 0.01, use_rankings = True):
        super(SmoothAP, self).__init__()

        self.n_positives = n
        self.n_negatives= n_negatives
        self.tau = tau
        self.threshold = threshold
        self.use_rankings = use_rankings

    def sigmoid(self, tensor, temp=1.0):
        """ temperature controlled sigmoid
        takes as input a torch tensor (tensor) and passes it through a sigmoid, controlled by temperature: temp
        """
        exponent = -tensor / temp
        # clamp the input tensor for stability
        exponent = torch.clamp(exponent, min=-50, max=50)
        y = 1.0 / (1.0 + torch.exp(exponent))
        return y

    def forward(self, images, captions, count_gradients=False):
        """
        :param images:
        :param captions:
        :return:
        """
        batch_size = images.shape[0]

        scores = images.mm(captions.t()).unsqueeze(1)
        gt = torch.eye(batch_size).repeat_interleave(self.n_positives, dim=1).unsqueeze(1)

        if torch.cuda.is_available():
            gt = gt.cuda()

        if scores.dim() == 3:

            i2t = self.batch_forward(scores, gt, count_gradients, task='i2t')
            t2i = self.batch_forward(scores.transpose(2, 0), gt.transpose(2, 0), count_gradients, task='t2i')

            if count_gradients:
                return i2t, t2i

            loss = 0.5 * (i2t + t2i)

            return loss

        else:
            raise Exception("Wrong dims")

    def single_forward(self, scores, gt):
        """
        :param scores:
        :param gt:
        :return:
        """
        """
        # scores: predicted relevance scores (1 x m)
        # gt: groundtruth relevance scores (1 x m)
        
        # repeat the number row-wise.
        m = scores.shape[1]
        s1 = scores.repeat(m, 1)  # s1: m x m
        # repeat the number column-wise.
        s2 = s1.t()  # s2: m x m
        # compute difference matrix
        D = s1 - s2
        # approximating heaviside
        D_ = self.sigmoid(D, temp=self.tau) * (1 - torch.eye(m))
        # ranking of each instance
        R = 1 + torch.sum(D_, 1)
        # compute positive ranking
        R_pos = (1 + torch.sum(D_ * gt, 1)) * gt
        # compute AP
        AP = (1 / torch.sum(gt)) * torch.sum(R_pos / R)
        loss = 1 - AP
        return loss
        """
        raise NotImplementedError

    def batch_forward(self, scores, gt, count_gradients=False, task=None):
        """
        :param scores: (N, 1, M)
        :param gt:
        :return:
        """

        m = scores.shape[2]

        s1 = scores.repeat(1, m, 1)

        s2 = s1.transpose(2, 1)

        D = s1 - s2

        eye = torch.eye(m)

        if torch.cuda.is_available():
            eye = eye.cuda()

        D_ = self.sigmoid(D, temp=self.tau) * (1 - eye)

        if count_gradients:
            return self.gradient_counter(D_, task)

        R = 1 + torch.sum(D_, 2)

        R_pos = (1 + torch.sum(D_ * gt, 2)) * gt.squeeze(1)

        AP = (1 / torch.sum(gt, dim=2)) * torch.sum(R_pos / R, dim=1).unsqueeze(1)

        loss = 1 - AP

        loss.mean()

        return loss.mean()

    def gradient_counter(self, D_, task):

        batch_size = D_.shape[0]
        grad = D_ * (1 - D_)

        _sum = 0
        zeros = 0
        for i in range(batch_size):
            positive_grads = grad[i, i * 5:(i * 5) + 5]
            if self.use_rankings:
                inverse_ranking_squared = ((D_[i,i * 5 :(i * 5) + 5].sum(dim=1) + 1) ** -2).unsqueeze(1)
                positive_grads *= inverse_ranking_squared

            if task == 'i2t':
                div = 5
            else:
                div = 1
            _sum += (positive_grads > self.threshold).sum() / div
            zeros += ((positive_grads > self.threshold).sum() == 0).int()

        batch_div = batch_size - zeros

        if batch_div > 0:
            _sum = _sum/(batch_size - zeros)
        else:
            _sum = 0
        return ('sum_' + task, float(_sum)), ('zeros_' + task, float(zeros))
