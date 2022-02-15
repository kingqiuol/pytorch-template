# coding:utf-8
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau


class WarmUpLR(_LRScheduler):
	"""warmup_training learning rate scheduler
	Args:
		optimizer: optimzier(e.g. SGD)
		total_iters: totoal_iters of warmup phase
	"""

	def __init__(self, optimizer, total_iters, last_epoch=-1):
		self.total_iters = total_iters
		super().__init__(optimizer, last_epoch)

	def get_lr(self):
		"""we will use the first m batches, and set the learning
		rate to base_lr * m / total_iters
		"""
		return [base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]


class PolynomialLRDecay(_LRScheduler):
	"""Polynomial learning rate decay until step reach to max_decay_step

	Args:
		optimizer (Optimizer): Wrapped optimizer.
		max_decay_steps: after this step, we stop decreasing learning rate
		end_learning_rate: scheduler stoping learning rate decay, value of learning rate must be this value
		power: The power of the polynomial.
	"""

	def __init__(self, optimizer, max_decay_steps, end_learning_rate=0.0001, power=1.0):
		if max_decay_steps <= 1.:
			raise ValueError('max_decay_steps should be greater than 1.')
		self.max_decay_steps = max_decay_steps
		self.end_learning_rate = end_learning_rate
		self.power = power
		self.last_step = 0
		super().__init__(optimizer)

	def get_lr(self):
		if self.last_step > self.max_decay_steps:
			return [self.end_learning_rate for _ in self.base_lrs]

		return [(base_lr - self.end_learning_rate) *
				((1 - self.last_step / self.max_decay_steps) ** (self.power)) +
				self.end_learning_rate for base_lr in self.base_lrs]

	def step(self, step=None):
		if step is None:
			step = self.last_step + 1
		self.last_step = step if step != 0 else 1
		if self.last_step <= self.max_decay_steps:
			decay_lrs = [(base_lr - self.end_learning_rate) *
						 ((1 - self.last_step / self.max_decay_steps) ** (self.power)) +
						 self.end_learning_rate for base_lr in self.base_lrs]
			for param_group, lr in zip(self.optimizer.param_groups, decay_lrs):
				param_group['lr'] = lr


class GradualWarmupScheduler(_LRScheduler):
	""" Gradually warm-up(increasing) learning rate in optimizer.
	Proposed in 'Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour'.

	Args:
		optimizer (Optimizer): Wrapped optimizer.
		multiplier: target learning rate = base lr * multiplier if multiplier > 1.0. if multiplier = 1.0, lr starts from 0 and ends up with the base_lr.
		total_epoch: target learning rate is reached at total_epoch, gradually
		after_scheduler: after target_epoch, use this scheduler(eg. ReduceLROnPlateau)
	"""

	def __init__(self, optimizer, multiplier, total_epoch, after_scheduler=None):
		self.multiplier = multiplier
		if self.multiplier < 1.:
			raise ValueError('multiplier should be greater thant or equal to 1.')
		self.total_epoch = total_epoch
		self.after_scheduler = after_scheduler
		self.finished = False
		super(GradualWarmupScheduler, self).__init__(optimizer)

	def get_lr(self):
		if self.last_epoch > self.total_epoch:
			if self.after_scheduler:
				if not self.finished:
					self.after_scheduler.base_lrs = [base_lr * self.multiplier for base_lr in self.base_lrs]
					self.finished = True
				return self.after_scheduler.get_last_lr()
			return [base_lr * self.multiplier for base_lr in self.base_lrs]

		if self.multiplier == 1.0:
			return [base_lr * (float(self.last_epoch) / self.total_epoch) for base_lr in self.base_lrs]
		else:
			return [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in
					self.base_lrs]

	def step_ReduceLROnPlateau(self, metrics, epoch=None):
		if epoch is None:
			epoch = self.last_epoch + 1
		self.last_epoch = epoch if epoch != 0 else 1  # ReduceLROnPlateau is called at the end of epoch, whereas others are called at beginning
		if self.last_epoch <= self.total_epoch:
			warmup_lr = [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in
						 self.base_lrs]
			for param_group, lr in zip(self.optimizer.param_groups, warmup_lr):
				param_group['lr'] = lr
		else:
			if epoch is None:
				self.after_scheduler.step(metrics, None)
			else:
				self.after_scheduler.step(metrics, epoch - self.total_epoch)

	def step(self, epoch=None, metrics=None):
		if type(self.after_scheduler) != ReduceLROnPlateau:
			if self.finished and self.after_scheduler:
				if epoch is None:
					self.after_scheduler.step(None)
				else:
					self.after_scheduler.step(epoch - self.total_epoch)
				self._last_lr = self.after_scheduler.get_last_lr()
			else:
				return super(GradualWarmupScheduler, self).step(epoch)
		else:
			self.step_ReduceLROnPlateau(metrics, epoch)


class MyStepLR:
	'''
	自定义分步学习率衰减策略
	'''
	def __init__(self, optimizer, learning_rate: float, total_epochs: int):
		self.optimizer = optimizer
		self.total_epochs = total_epochs
		self.base = learning_rate

	def __call__(self, epoch):
		if epoch < self.total_epochs * 3 / 10:
			lr = self.base
		elif epoch < self.total_epochs * 6 / 10:
			lr = self.base * 0.2
		elif epoch < self.total_epochs * 8 / 10:
			lr = self.base * 0.2 ** 2
		else:
			lr = self.base * 0.2 ** 3

		for param_group in self.optimizer.param_groups:
			param_group["lr"] = lr

	def lr(self) -> float:
		return self.optimizer.param_groups[0]["lr"]


if __name__ == '__main__':
	import torch
	from torch.optim.lr_scheduler import StepLR, ExponentialLR
	from torch.optim import lr_scheduler
	from torch.optim.sgd import SGD
	import matplotlib.pyplot as plt

	model = [torch.nn.Parameter(torch.randn(2, 2, requires_grad=True))]
	optim = SGD(model, 0.1)

	# warmup
	# scheduler_warmup is chained with schduler_steplr
	# scheduler_steplr = StepLR(optim, step_size=10, gamma=0.1) # 等间隔调整学习率
	# scheduler_steplr = lr_scheduler.MultiStepLR(optim, milestones=[2,4,5,8,10,12,14,16,18,20], gamma=0.1) # 多步间隔调整学习率
	# scheduler_steplr=torch.optim.lr_scheduler.ExponentialLR(optim, gamma=0.99, last_epoch=-1) # 指数衰减调整学习率
	# scheduler_steplr = torch.optim.lr_scheduler.CosineAnnealingLR(optim, 200, eta_min=0, last_epoch=-1)  # 余弦退火调整学习率
	# scheduler = GradualWarmupScheduler(optim, multiplier=1, total_epoch=5, after_scheduler=scheduler_steplr)
	# 多项式衰减
	# scheduler = PolynomialLRDecay(optim, 200, end_learning_rate=1e-6, power=0.5)

	scheduler_steplr=MyStepLR(optim,0.1,200)

	plt.figure()
	x = list(range(200))
	y = []
	for epoch in range(200):
		# scheduler.step()
		# lr = scheduler.get_lr()
		# print(epoch, optim.param_groups[0]['lr'])
		# y.append(scheduler.get_lr()[0])

		scheduler_steplr(epoch)
		y.append(scheduler_steplr.lr())

	plt.xlabel("epoch")
	plt.ylabel("learning rate")
	plt.plot(x, y)
	plt.show()
