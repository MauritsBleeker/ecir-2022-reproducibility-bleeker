import numpy as np
import torch
from shared.evaluation.recall import RecallMetric
import os
from collections import defaultdict


def eval_rank_mean_std(path, f_eval_rank, split='test', n_experiments=5, model_file='model_best.pth.tar'):
    recall = RecallMetric()
    for i in range(0, n_experiments):
        r_i2t, r_t2i, rsum, ar_i2t, ar_t2i = f_eval_rank(os.path.join(path, str(i), model_file), split=split)
        recall.add_recals(r_i2t=r_i2t, r_t2i=r_t2i, rsum=rsum, ar_i2t=ar_i2t, ar_t2i=ar_t2i, map5=r_i2t[5],
                          map10=r_i2t[6])

    print('Done')
    recall.print_results()


def average_precision(inds, index, k=10):

    idx = np.where(np.logical_and(inds[:k] >= index * 5, inds[:k] < 5 * index + 5))[0] + 1
    if len(idx) >= 1:
        ap = np.mean(np.array(range(1, len(idx) + 1)) / idx)

        return ap
    else:
        return 0


class GradStats(object):

    def __init__(self):

        self.data = defaultdict(list)

    def add_stats(self, data):

        for key, value in data:
            self.data[key].append(value)

    def print_stats(self):
        for key, value in self.data.items():
            # print(key, np.mean(value), np.std(value))
            print("%s: %.2f $\pm$  %.2f"%(key, np.mean(value), np.std(value)))

    def reset_stats(self):

        self.data = defaultdict(list)
