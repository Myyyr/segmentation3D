import torch
import numpy as np

class DiceScore():
	def __init__(self, classes, epsilon = 1e-7):
		"""
		classes : list : list of classe's names
		"""
		self.n_classes = len(classes)
		self.classes = classes

		self.n_count = 0
		self.accum = {}
		self.all_dices = {}
		for k in self.classes : 
			self.accum[k] = {'inter_sum':0, 'pred_sum':0, 'out_sum':0, 'all':0}
			self.all_dices[k] = 0

		self.epsilon = epsilon
		self.mean_dice = 0

	

	def __call__(self, output, pred):
		bs = output.shape[0]
		for c in range(self.n_classes):
			cout_sum, cpred_sum, cint_sum = self.dice_values(self.get_mask(output, c), self.get_mask(pred, c))
			self.accum[self.classes[c]]['inter_sum'] += cint_sum
			self.accum[self.classes[c]]['pred_sum'] += cpred_sum
			self.accum[self.classes[c]]['out_sum'] += cout_sum

			self.all_dices[self.classes[c]] = self.dice(self.accum[self.classes[c]]['inter_sum'],
														self.accum[self.classes[c]]['pred_sum'],
														self.accum[self.classes[c]]['out_sum'])
			self.mean_dice = np.mean(list(self.all_dices.values()))

		self.n_count += bs

	def get_dice_scores(self):
		return self.all_dices()

	def get_mean_dice_score(self, exeptions = []):
		a = []
		for k in self.classes:
			if k not in exeptions:
				a.append(self.all_dices[k])
		return np.mean(a)

	def dice(self, xy, x, y):
		return (2*xy + self.epsilon)/(x + y + self.epsilon)

	def dice_values(self, x,y):
		x_sum = x.sum().item()
		y_sum = y.sum().item()
		int_sum = (x*y).sum().item()
		return x_sum, y_sum, int_sum

	def get_mask(self, labels, i):
	    return (labels == i)*1

