# -*- encoding: utf-8 -*-

import os
import sys
import argparse
import time
import yaml
from easydict import EasyDict as edict
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from conf import settings

from datasets.cifar100_dataset import get_test_dataloader, get_training_dataloader,get_training_kfold_dataloader
from models.get_networks import get_network
from utils.optim import Ranger,SAM
from utils.scheduler import WarmUpLR, GradualWarmupScheduler
from utils.utils import most_recent_folder, most_recent_weights, last_epoch, best_acc_weights
from utils.initialize import initialize
from utils.log import Log
from utils.bypass_bn import disable_running_stats,enable_running_stats
from utils.losses import LabelSmoothingCrossEntropy,FocalLoss


def train(epoch):
	start = time.time()
	net.train()
	for batch_index, (images, labels) in enumerate(cifar100_training_loader):

		if cfg.MODEL.USE_GPU:
			labels = labels.cuda()
			images = images.cuda()

		if cfg.SOLVER.OPTIMIZER == "SAM":
			enable_running_stats(net)
		optimizer.zero_grad()
		outputs = net(images)
		loss = loss_function(outputs, labels)
		loss.backward()
		if cfg.SOLVER.OPTIMIZER == "SAM":
			optimizer.first_step(zero_grad=True)
		else:
			optimizer.step()

		if cfg.SOLVER.OPTIMIZER == "SAM":
			disable_running_stats(net)
			loss_function(net(images), labels).mean().backward()
			optimizer.second_step(zero_grad=True)


		n_iter = (epoch - 1) * len(cifar100_training_loader) + batch_index + 1

		last_layer = list(net.children())[-1]
		for name, para in last_layer.named_parameters():
			if 'weight' in name:
				writer.add_scalar('LastLayerGradients/grad_norm2_weights', para.grad.norm(), n_iter)
			if 'bias' in name:
				writer.add_scalar('LastLayerGradients/grad_norm2_bias', para.grad.norm(), n_iter)

		print('Training Epoch: {epoch} [{trained_samples}/{total_samples}]\tLoss: {:0.4f}\tLR: {:0.6f}'.format(
			loss.item(),
			optimizer.param_groups[0]['lr'],
			epoch=epoch,
			trained_samples=batch_index * cfg.SOLVER.BATCH_SIZE + len(images),
			total_samples=len(cifar100_training_loader.dataset)
		))

		# with torch.no_grad():
		# 	correct=torch.argmax(outputs.data,1)==labels
		# 	log(net,loss.cpu(),correct.cpu(),optimizer.param_groups[0]['lr'])

		# update training loss for each iteration
		writer.add_scalar('Train/loss', loss.item(), n_iter)

		# if epoch <= args.warm:
		#     warmup_scheduler.step()
		scheduler.step(epoch=epoch)

	for name, param in net.named_parameters():
		layer, attr = os.path.splitext(name)
		attr = attr[1:]
		writer.add_histogram("{}/{}".format(layer, attr), param, epoch)

	finish = time.time()

	print('epoch {} training time consumed: {:.2f}s'.format(epoch, finish - start))


@torch.no_grad()
def eval_training(epoch=0, tb=True):
	start = time.time()
	net.eval()

	test_loss = 0.0  # cost function error
	correct = 0.0

	for (images, labels) in cifar100_test_loader:

		if cfg.MODEL.USE_GPU:
			images = images.cuda()
			labels = labels.cuda()

		outputs = net(images)
		loss = loss_function(outputs, labels)

		test_loss += loss.item()
		_, preds = outputs.max(1)
		correct += preds.eq(labels).sum()

	finish = time.time()
	if cfg.MODEL.USE_GPU:
		print('GPU INFO.....')
		print(torch.cuda.memory_summary(), end='')
	print('Evaluating Network.....')
	print('Test set: Epoch: {}, Average loss: {:.4f}, Accuracy: {:.4f}, Time consumed:{:.2f}s'.format(
		epoch,
		test_loss / len(cifar100_test_loader.dataset),
		correct.float() / len(cifar100_test_loader.dataset),
		finish - start
	))

	# correct = torch.argmax(outputs, 1) == labels
	# log(net,loss.cpu(),correct.cpu())

	# add informations to tensorboard
	if tb:
		writer.add_scalar('Test/Average loss', test_loss / len(cifar100_test_loader.dataset), epoch)
		writer.add_scalar('Test/Accuracy', correct.float() / len(cifar100_test_loader.dataset), epoch)

	return correct.float() / len(cifar100_test_loader.dataset)


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("-config",type=str,default="./conf/config.ymal",help="training model configuration information")
	args = parser.parse_args()

	# 设置随机化种子
	initialize(seed=42)

	# 加载配置文件
	yaml_path = args.config
	with open(yaml_path, "r", encoding='utf-8') as f:
		cfg = yaml.load(f, Loader=yaml.FullLoader)
		cfg = edict(cfg)

	# 打印日志设置
	# log=Log(log_each=10)

	# networks
	net = get_network(cfg)

	if cfg.SOLVER.LOSS=="CE":
		loss_function = nn.CrossEntropyLoss()
	elif cfg.SOLVER.LOSS=="smooth_CE":
		loss_function= LabelSmoothingCrossEntropy()
	elif cfg.SOLVER.LOSS=="FOCAL":
		loss_function=FocalLoss(num_classes=100)

	if cfg.SOLVER.OPTIMIZER == "SGD":
		optimizer = optim.SGD(net.parameters(), lr=cfg.SOLVER.LR, momentum=cfg.SOLVER.MOMENTUM, weight_decay=cfg.SOLVER.WEIGHT_DECAY)
	elif cfg.SOLVER.OPTIMIZER == "Adam":
		optimizer = optim.Adam(net.parameters(), lr=cfg.SOLVER.LR, weight_decay=cfg.SOLVER.WEIGHT_DECAY, amsgrad=True)
	elif cfg.SOLVER.OPTIMIZER == "Ranger":
		optimizer = Ranger(params=filter(lambda p: p.requires_grad, net.parameters()), lr=cfg.SOLVER.LR)
	elif cfg.SOLVER.OPTIMIZER == "SAM":
		base_optimizer=torch.optim.SGD
		optimizer=SAM(net.parameters(),base_optimizer=base_optimizer,rho=cfg.SOLVER.RHO,adaptive=cfg.SOLVER.ADAPTIVE,
					  lr=cfg.SOLVER.LR,momentum=cfg.SOLVER.MOMENTUM, weight_decay=cfg.SOLVER.WEIGHT_DECAY)

	# train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=settings.MILESTONES, gamma=0.2) #learning rate decay
	# iter_per_epoch = len(cifar100_training_loader)

	if cfg.SOLVER.USE_WARMUP:
		scheduler_steplr = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg.SOLVER.EPOCH, eta_min=1e-5,
																	  last_epoch=-1)  # 余弦退火调整学习率
		scheduler = GradualWarmupScheduler(optimizer, 1, cfg.SOLVER.WARMUP_EPOCH, scheduler_steplr)
	else:
		scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, settings.EPOCH, eta_min=1e-5, last_epoch=-1)

	TIME_NOW = datetime.now().strftime(cfg.OUTPUT.DATE_FORMAT)
	if cfg.SOLVER.RESUME:
		recent_folder = most_recent_folder(os.path.join(cfg.OUTPUT.CHECKPOINT_PATH, cfg.MODEL.NAME), fmt=cfg.OUTPUT.DATE_FORMAT)
		if not recent_folder:
			raise Exception('no recent folder were found')
		checkpoint_path = os.path.join(cfg.OUTPUT.CHECKPOINT_PATH, cfg.MODEL.NAME, recent_folder)
	else:
		checkpoint_path = os.path.join(cfg.OUTPUT.CHECKPOINT_PATH,cfg.MODEL.NAME, TIME_NOW)

	# use tensorboard
	if not os.path.exists(cfg.OUTPUT.LOG_DIR):
		os.mkdir(cfg.OUTPUT.LOG_DIR)

	# since tensorboard can't overwrite old values
	# so the only way is to create a new tensorboard log
	writer = SummaryWriter(log_dir=os.path.join(
		cfg.OUTPUT.LOG_DIR, cfg.MODEL.NAME, TIME_NOW))
	input_tensor = torch.Tensor(1, 3, 32, 32)
	if cfg.MODEL.USE_GPU:
		input_tensor = input_tensor.cuda()
	writer.add_graph(net, input_tensor)

	# create checkpoint folder to save model
	if not os.path.exists(checkpoint_path):
		os.makedirs(checkpoint_path)
	checkpoint_path = os.path.join(checkpoint_path, '{net}-{epoch}-{type}-{k_fold}.pth')

	best_acc = 0.0
	if cfg.SOLVER.RESUME:
		best_weights = best_acc_weights(os.path.join(cfg.OUTPUT.CHECKPOINT_PATH, cfg.MODEL.NAME, recent_folder))
		if best_weights:
			weights_path = os.path.join(settings.CHECKPOINT_PATH, args.MODEL.NAME, recent_folder, best_weights)
			print('found best acc weights file:{}'.format(weights_path))
			print('load best training file to test acc...')
			net.load_state_dict(torch.load(weights_path))
			best_acc = eval_training(tb=False)
			print('best acc is {:0.2f}'.format(best_acc))

		recent_weights_file = most_recent_weights(os.path.join(settings.CHECKPOINT_PATH, cfg.MODEL.NAME, recent_folder))
		if not recent_weights_file:
			raise Exception('no recent weights file were found')
		weights_path = os.path.join(settings.CHECKPOINT_PATH, cfg.MODEL.NAME, recent_folder, recent_weights_file)
		print('loading weights file {} to resume training.....'.format(weights_path))
		net.load_state_dict(torch.load(weights_path))

		resume_epoch = last_epoch(os.path.join(settings.CHECKPOINT_PATH, cfg.MODEL.NAME, recent_folder))

	if not cfg.DATALOADER.K_FOLD:
		# data preprocessing
		cifar100_training_loader = get_training_dataloader(
			cfg,
			num_workers=cfg.DATALOADER.NUM_WORKERS,
			batch_size=cfg.SOLVER.BATCH_SIZE,
			shuffle=True
		)

		cifar100_test_loader = get_test_dataloader(
			cfg,
			num_workers=cfg.DATALOADER.NUM_WORKERS,
			batch_size=cfg.SOLVER.BATCH_SIZE,
			shuffle=True
		)


		for epoch in range(1, settings.EPOCH + 1):
			# if epoch > args.warm:
			#     train_scheduler.step(epoch)

			if cfg.SOLVER.RESUME:
				if epoch <= resume_epoch:
					continue

			# log.train(len_dataset=len(cifar100_training_loader))
			train(epoch)
			# log.eval(len_dataset=len(cifar100_test_loader))
			acc = eval_training(epoch)

			# start to save best performance model after learning rate decay to 0.01
			if epoch > settings.MILESTONES[1] and best_acc < acc:
				weights_path = checkpoint_path.format(net=cfg.MODEL.NAME, epoch=epoch, type='best',k_fold=0)
				print('saving weights file to {}'.format(weights_path))
				torch.save(net.state_dict(), weights_path)
				best_acc = acc
				continue

			if not epoch % settings.SAVE_EPOCH:
				weights_path = checkpoint_path.format(net=cfg.MODEL.NAME, epoch=epoch, type='regular',k_fold=0)
				print('saving weights file to {}'.format(weights_path))
				torch.save(net.state_dict(), weights_path)
	else:
		for n in range(cfg.DATALOADER.K):
			print('Training kfold set {}'.format(n))


			cifar100_training_loader, cifar100_test_loader=get_training_kfold_dataloader(cfg=cfg,
																						k=cfg.DATALOADER.K,n=n,
																						num_workers=cfg.DATALOADER.NUM_WORKERS,
																						batch_size=cfg.SOLVER.BATCH_SIZE,
																						shuffle=False)
			for epoch in range(1, settings.EPOCH + 1):
				# if epoch > args.warm:
				#     train_scheduler.step(epoch)

				if cfg.SOLVER.RESUME:
					if epoch <= resume_epoch:
						continue

				# log.train(len_dataset=len(cifar100_training_loader))
				train(epoch)
				# log.eval(len_dataset=len(cifar100_test_loader))
				acc = eval_training(epoch)

				# start to save best performance model after learning rate decay to 0.01
				if epoch > settings.MILESTONES[1] and best_acc < acc:
					weights_path = checkpoint_path.format(net=cfg.MODEL.NAME, epoch=epoch, type='best',k_fold=n)
					print('saving weights file to {}'.format(weights_path))
					torch.save(net.state_dict(), weights_path)
					best_acc = acc
					continue

				if not epoch % settings.SAVE_EPOCH:
					weights_path = checkpoint_path.format(net=cfg.MODEL.NAME, epoch=epoch, type='regular',k_fold=n)
					print('saving weights file to {}'.format(weights_path))
					torch.save(net.state_dict(), weights_path)

	# log.flush()
	writer.close()