import pickle
import os
import time
import shutil

import torch

import data
from vocab import Vocabulary  # NOQA
from model import VSE
from evaluation import i2t, t2i, AverageMeter, LogCollector, encode_data

import logging
# import tensorboard_logger as tb_logger

import argparse
import wandb
from losses import order_sim



def main():
    # Hyper Parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='/Users/mauritsbleeker/Documents/PhD/Github/Projects/MMC_reproduce/data',
                        help='path to datasets')
    parser.add_argument('--data_name', default='f30k',
                        help='{coco,f8k,f30k,10crop}_precomp|coco|f8k|f30k')
    parser.add_argument('--vocab_path', default='/Users/mauritsbleeker/Documents/PhD/Github/Projects/MMC_reproduce/data',
                        help='Path to saved vocabulary pickle files.')
    parser.add_argument('--margin', default=0.2, type=float,
                        help='Rank loss margin.')
    parser.add_argument('--num_epochs', default=30, type=int,
                        help='Number of training epochs.')
    parser.add_argument('--batch_size', default=128, type=int,
                        help='Size of a training mini-batch.')
    parser.add_argument('--word_dim', default=300, type=int,
                        help='Dimensionality of the word embedding.')
    parser.add_argument('--embed_size', default=1024, type=int,
                        help='Dimensionality of the joint embedding.')
    parser.add_argument('--grad_clip', default=2., type=float,
                        help='Gradient clipping threshold.')
    parser.add_argument('--crop_size', default=224, type=int,
                        help='Size of an image crop as the CNN input.')
    parser.add_argument('--num_layers', default=1, type=int,
                        help='Number of GRU layers.')
    parser.add_argument('--learning_rate', default=.0002, type=float,
                        help='Initial learning rate.')
    parser.add_argument('--lr_update', default=15, type=int,
                        help='Number of epochs to update the learning rate.')
    parser.add_argument('--workers', default=0, type=int,
                        help='Number of data loader workers.')
    parser.add_argument('--log_step', default=10, type=int,
                        help='Number of steps to print and record the log.')
    parser.add_argument('--val_step', default=500, type=int,
                        help='Number of steps to run validation.')
    parser.add_argument('--logger_name', default='runs/runX',
                        help='Path to save the model and Tensorboard log.')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--max_violation', action='store_true',
                        help='Use max instead of sum in the rank loss.')
    parser.add_argument('--img_dim', default=4096, type=int,
                        help='Dimensionality of the image embedding.')
    parser.add_argument('--finetune', action='store_true',
                        help='Fine-tune the image encoder.')
    parser.add_argument('--cnn_type', default='resnet50',
                        help="""The CNN used for image encoder
                        (e.g. vgg19, resnet152)""")
    parser.add_argument('--use_restval', action='store_true',
                        help='Use the restval data for training on MSCOCO.')
    parser.add_argument('--measure', default='cosine',
                        help='Similarity measure used (cosine|order)')
    parser.add_argument('--use_abs', action='store_true',
                        help='Take the absolute value of embedding vectors.')
    parser.add_argument('--no_imgnorm', action='store_true',
                        help='Do not normalize the image embeddings.')
    parser.add_argument('--reset_train', action='store_true',
                        help='Ensure the training is always done in '
                        'train mode (Not recommended).')
    parser.add_argument('--experiment_name', default='reproduce_vsepp',
                        help='name of the experiment')
    parser.add_argument('--criterion',
                        default='triplet',
                        help='loss function')
    parser.add_argument('--tau', default=0.1, type=float,
                        help='Tau value NTXent or SmoothAP.')
    parser.add_argument('--ranking_based', action='store_true',
                        help='Fine tune as a ranking.')
    parser.add_argument('--n_sp', default=5, type=int,
                        help='Number of positive samples')

    opt = parser.parse_args()
    print(opt)

    logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)

    if torch.cuda.is_available():
        dir = '/ivi/ilps/personal/mbleeke1/mmc_reproduce/vsepp'
    else:
        dir = '.'

    wandb.init(project='vsepp_paper_experimentsgit s', entity='mbleeker', tags=["vsepp", "paper_experiments_vsepp"], name=opt.experiment_name, dir=dir)

    config = wandb.config

    config.name = opt.experiment_name
    config.batch_size = opt.batch_size
    config.learning_rate = opt.learning_rate
    config.lr_update = opt.lr_update
    config.cnn_type = opt.cnn_type
    config.finetune = opt.finetune
    config.embed_size = opt.embed_size
    config.img_dim = opt.img_dim
    config.data_name = opt.data_name
    config.max_violation = opt.max_violation
    config.n_sp = opt.n_sp
    config.ranking_based = opt.ranking_based
    config.criterion = opt.criterion
    config.tau = opt.tau


    # Load Vocabulary Wrapper
    vocab = pickle.load(open(os.path.join(
        opt.vocab_path, '%s_vocab.pkl' % opt.data_name), 'rb'))
    opt.vocab_size = len(vocab)

    # Load data loaders
    train_loader, val_loader = data.get_loaders(
        opt.data_name, vocab, opt.crop_size, opt.batch_size, opt.workers, opt)

    # Construct the model
    model = VSE(opt)

    # optionally resume from a checkpoint

    wandb.watch([model.txt_enc, model.img_enc], log='all', log_freq=200, idx=0)


    if opt.resume:
        if os.path.isfile(opt.resume):
            print("=> loading checkpoint '{}'".format(opt.resume))
            checkpoint = torch.load(opt.resume)
            start_epoch = checkpoint['epoch']
            best_rsum = checkpoint['best_rsum']
            model.load_state_dict(checkpoint['model'])
            # Eiters is used to show logs as the continuation of another
            # training
            model.Eiters = checkpoint['Eiters']
            print("=> loaded checkpoint '{}' (epoch {}, best_rsum {})"
                  .format(opt.resume, start_epoch, best_rsum))
            validate(opt, val_loader, model)
        else:
            print("=> no checkpoint found at '{}'".format(opt.resume))

    # Train the Model
    best_rsum = 0
    for epoch in range(opt.num_epochs):
        adjust_learning_rate(opt, model.optimizer, epoch)

        # train for one epoch
        train(opt, train_loader, model, epoch, val_loader)

        # evaluate on validation set
        rsum = validate(opt, val_loader, model)

        # remember best R@ sum and save checkpoint
        is_best = rsum > best_rsum
        best_rsum = max(rsum, best_rsum)
        save_checkpoint({
            'epoch': epoch + 1,
            'model': model.state_dict(),
            'best_rsum': best_rsum,
            'opt': opt,
            'Eiters': model.Eiters,
        }, is_best, prefix=opt.logger_name + '/')


def train(opt, train_loader, model, epoch, val_loader):
    # average meters to record the training statistics
    batch_time = AverageMeter()
    data_time = AverageMeter()
    train_logger = LogCollector()

    # switch to train mode
    model.train_start()

    end = time.time()
    for i, train_data in enumerate(train_loader):
        # Always reset to train mode, this is not the default behavior
        model.train_start()

        # measure data loading time
        data_time.update(time.time() - end)

        # make sure train logger is used
        model.logger = train_logger

        # Update the model
        model.train_emb(*train_data)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # Print log info
        if model.Eiters % opt.log_step == 0:
            logging.info(
                'Epoch: [{0}][{1}/{2}]\t'
                '{e_log}\t'
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                .format(
                    epoch, i, len(train_loader), batch_time=batch_time,
                    data_time=data_time, e_log=str(model.logger)))

            # Record logs in tensorboard
            wandb.log({'epoch': epoch, 'step': i, }, step=model.Eiters)
            wandb.log({'Eit': model.Eiters, 'lr': model.optimizer.param_groups[0]['lr']}, step=model.Eiters)
            wandb.log({'Le': model.loss_value.data}, step=model.Eiters)

        # validate at every val_step
        if model.Eiters % opt.val_step == 0:
            validate(opt, val_loader, model)


def validate(opt, val_loader, model):
    # compute the encoding for all the validation images and captions
    img_embs, cap_embs = encode_data(
        model, val_loader, opt.log_step, logging.info)

    # caption retrieval
    (r1, r5, r10, medr, meanr, map5, map10) = i2t(img_embs, cap_embs, measure=opt.measure)
    logging.info("Image to text: %.1f, %.1f, %.1f, %.1f, %.1f" %
                 (r1, r5, r10, medr, meanr))
    # image retrieval
    (r1i, r5i, r10i, medri, meanri) = t2i(
        img_embs, cap_embs, measure=opt.measure)
    logging.info("Text to image: %.1f, %.1f, %.1f, %.1f, %.1f" %
                 (r1i, r5i, r10i, medri, meanri))
    # sum of recalls to be used for early stopping
    currscore = r1 + r5 + r10 + r1i + r5i + r10i

    # record metrics in tensorboard
    # tb_logger.log_value('r1', r1, step=model.Eiters)
    # tb_logger.log_value('r5', r5, step=model.Eiters)
    # tb_logger.log_value('r10', r10, step=model.Eiters)
    # tb_logger.log_value('medr', medr, step=model.Eiters)
    # tb_logger.log_value('meanr', meanr, step=model.Eiters)
    # tb_logger.log_value('r1i', r1i, step=model.Eiters)
    # tb_logger.log_value('r5i', r5i, step=model.Eiters)
    # tb_logger.log_value('r10i', r10i, step=model.Eiters)
    # tb_logger.log_value('medri', medri, step=model.Eiters)
    # tb_logger.log_value('meanri', meanri, step=model.Eiters)
    # tb_logger.log_value('rsum', currscore, step=model.Eiters)

    wandb.log({'r1': r1, 'r5': r5, 'r10': r10, 'medr': medr, 'meanr': meanr,
               'r1i': r1i, 'r5i': r5i, 'r10i': r10i, 'medri': medri, 'meanr': meanr,
               'rsum': currscore}, step=model.Eiters
    )

    return currscore


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar', prefix=''):
    if not os.path.exists(prefix):
        os.makedirs(prefix)

    torch.save(state, prefix + filename)
    if is_best:
        shutil.copyfile(prefix + filename, prefix + 'model_best.pth.tar')


def adjust_learning_rate(opt, optimizer, epoch):
    """Sets the learning rate to the initial LR
       decayed by 10 every 30 epochs"""
    lr = opt.learning_rate * (0.1 ** (epoch // opt.lr_update))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
    main()
