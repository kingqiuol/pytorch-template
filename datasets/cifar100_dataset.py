""" train and test dataset

author baiyu
"""
import os
import sys
import pickle

from skimage import io
import matplotlib.pyplot as plt
import numpy as np
import torch

import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from utils.augment.augment import get_transform


class CIFAR100Train(Dataset):
	"""cifar100 test dataset, derived from
	torch.utils.data.DataSet
	"""

	def __init__(self, path, transform=None):
		# if transform is given, we transoform data using
		with open(os.path.join(path, 'train'), 'rb') as cifar100:
			self.data = pickle.load(cifar100, encoding='bytes')
		self.transform = transform

	def __len__(self):
		return len(self.data['fine_labels'.encode()])

	def __getitem__(self, index):
		label = self.data['fine_labels'.encode()][index]
		r = self.data['data'.encode()][index, :1024].reshape(32, 32)
		g = self.data['data'.encode()][index, 1024:2048].reshape(32, 32)
		b = self.data['data'.encode()][index, 2048:].reshape(32, 32)
		image = np.dstack((r, g, b))

		if self.transform:
			image = self.transform(image)
		return label, image


class CIFAR100Test(Dataset):
	"""cifar100 test dataset, derived from
	torch.utils.data.DataSet
	"""

	def __init__(self, path, transform=None):
		with open(os.path.join(path, 'test'), 'rb') as cifar100:
			self.data = pickle.load(cifar100, encoding='bytes')
		self.transform = transform

	def __len__(self):
		return len(self.data['data'.encode()])

	def __getitem__(self, index):
		label = self.data['fine_labels'.encode()][index]
		r = self.data['data'.encode()][index, :1024].reshape(32, 32)
		g = self.data['data'.encode()][index, 1024:2048].reshape(32, 32)
		b = self.data['data'.encode()][index, 2048:].reshape(32, 32)
		image = np.dstack((r, g, b))

		if self.transform:
			image = self.transform(image)
		return label, image


def get_training_dataloader(cfg,batch_size=16, num_workers=2, shuffle=True):
	""" return training dataloader
	Args:
		mean: mean of cifar100 training dataset
		std: std of cifar100 training dataset
		path: path to cifar100 training python dataset
		batch_size: dataloader batchsize
		num_workers: dataloader num_works
		shuffle: whether to shuffle
	Returns: train_data_loader:torch dataloader object
	"""


	transform_train=get_transform(cfg)

	# transform_train = transforms.Compose([
	# 	# transforms.ToPILImage(),
	# 	transforms.RandomCrop(32, padding=4),
	# 	transforms.RandomHorizontalFlip(),
	# 	transforms.RandomRotation(15),
	# 	transforms.ToTensor(),
	# 	transforms.Normalize(cfg.TRAIN_MEAN, cfg.TRAIN_STD),
	#
	# ])
	# cifar100_training = CIFAR100Train(cfg.PATH, transform=transform_train)
	cifar100_training = torchvision.datasets.CIFAR100(root='./data', train=True, download=True,
													  transform=transform_train)
	cifar100_training_loader = DataLoader(
		cifar100_training, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)

	return cifar100_training_loader

def get_training_kfold_dataloader(cfg,k,n,batch_size=16, num_workers=2, shuffle=True):
	'''
	return k fold datasets
	:param cfg:
	:param k:
	:param n:
	:param batch_size:
	:param num_workers:
	:param shuffle:
	:return:
	'''
	transform_train=get_transform(cfg)

	# transform_train = transforms.Compose([
	# 	# transforms.ToPILImage(),
	# 	transforms.RandomCrop(32, padding=4),
	# 	transforms.RandomHorizontalFlip(),
	# 	transforms.RandomRotation(15),
	# 	transforms.ToTensor(),
	# 	transforms.Normalize(cfg.TRAIN_MEAN, cfg.TRAIN_STD),
	#
	# ])
	# cifar100_training = CIFAR100Train(cfg.PATH, transform=transform_train)
	cifar100_training = torchvision.datasets.CIFAR100(root='./data', train=True, download=True,
													  transform=transform_train)

	length = len(cifar100_training)
	shuffle_dataset = True
	random_seed = 42  # fixed random seed
	indices = list(range(length))

	if shuffle_dataset:
		np.random.seed(random_seed)
		np.random.shuffle(indices)  # shuffle

	val_indices = indices[int(length / k) * n:int(length / k) * (n + 1)]
	train_indices = list(set(indices).difference(set(val_indices)))
	train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)  # build Sampler
	valid_sampler = torch.utils.data.SubsetRandomSampler(val_indices)

	cifar100_training_loader = DataLoader(cifar100_training,
										  shuffle=shuffle,
										  sampler=train_sampler,
										  num_workers=num_workers,
										  batch_size=batch_size)

	cifar100_val_loader = DataLoader(cifar100_training,
										  shuffle=shuffle,
										  sampler=valid_sampler,
										  num_workers=num_workers,
										  batch_size=batch_size)

	return cifar100_training_loader,cifar100_val_loader



def get_test_dataloader(cfg, batch_size=16, num_workers=2, shuffle=True):
	""" return training dataloader
	Args:
		mean: mean of cifar100 test dataset
		std: std of cifar100 test dataset
		path: path to cifar100 test python dataset
		batch_size: dataloader batchsize
		num_workers: dataloader num_works
		shuffle: whether to shuffle
	Returns: cifar100_test_loader:torch dataloader object
	"""
	transform_test=get_transform(cfg,is_train=False)
	# transform_test = transforms.Compose([
	# 	transforms.ToTensor(),
	# 	transforms.Normalize(cfg.TRAIN_MEAN, cfg.TRAIN_STD)
	# ])

	# cifar100_test = CIFAR100Test(cfg.PATH, transform=transform_test)
	cifar100_test = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
	cifar100_test_loader = DataLoader(
		cifar100_test, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)

	return cifar100_test_loader

