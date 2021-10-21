import numpy as np
import scipy
from collections import defaultdict


class RecallMetric(object):

	def __init__(self):

		self.store = {
			"i2t": defaultdict(list),
			"t2i": defaultdict(list),
			"rsum": list(),
			"map5": list(),
			"map10": list(),
		}

	def add_recals(self, r_i2t, r_t2i, rsum, ar_i2t, ar_t2i, map5=None, map10=None):

		self.store['i2t']['r@1'].append(r_i2t[0])
		self.store['i2t']['r@5'].append(r_i2t[1])
		self.store['i2t']['r@10'].append(r_i2t[2])
		self.store['i2t']['ar'].append(ar_i2t)

		self.store['rsum'].append(rsum)

		if map5 and map10:
			self.store['map5'].append(map5)
			self.store['map10'].append(map10)

		self.store['t2i']['r@1'].append(r_t2i[0])
		self.store['t2i']['r@5'].append(r_t2i[1])
		self.store['t2i']['r@10'].append(r_t2i[2])
		self.store['t2i']['ar'].append(ar_t2i)

	def print_results(self):
		self.mean_confidence_interval('i2t', 'r@1')
		self.mean_confidence_interval('i2t', 'r@5')
		self.mean_confidence_interval('i2t', 'r@10')
		self.mean_confidence_interval('i2t', 'ar')

		self.mean_confidence_interval('t2i', 'r@1')
		self.mean_confidence_interval('t2i', 'r@5')
		self.mean_confidence_interval('t2i', 'r@10')
		self.mean_confidence_interval('t2i', 'ar')

		self.mean_confidence_interval('rsum')

		if len(self.store['map5']) >= 1 and len(self.store['map10']) >= 1:

			self.mean_confidence_interval('map5')
			self.mean_confidence_interval('map10')

	def mean_confidence_interval(self, task, metric=None, confidence=0.95):

		if metric:
			data = self.store[task][metric]
		else:
			data = self.store[task]

		a = 1.0 * np.array(data)

		mean = np.mean(a)
		std = np.std(a)


		print("Task: %s, Metric: %s, \n Mean: %.2f , Std: %.2f" % (task, metric, mean , std))
		print("$ %.2f \pm %.2f $" %(mean, std))
		print("______________")

