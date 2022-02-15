#!/usr/bin/python
# encoding: utf-8
import cv2
import abc
import random
import math
import numpy as np
import torchvision
import torchvision.transforms as transforms
from PIL import Image, ImageEnhance, ImageFilter, ImageOps, ImageDraw

from utils.augment.random_Erasing import RandomErasing
from utils.augment.autoaugment import Augmentation
from utils.augment.common.achive import autoaug_policy


def get_transform(cfg,is_train=True):
	'''
	获取数据增强
	:param is_train:
	:return:
	'''

	if is_train:
		if cfg.INPUT.USE_AUTOAUG:
			image_transform_list=[
				#自动增强
				Augmentation(autoaug_policy()),

				transforms.RandomCrop(32, padding=4),
				transforms.RandomHorizontalFlip(),
				transforms.RandomRotation(15),
				transforms.ToTensor(),
				transforms.Normalize(cfg.INPUT.TRAIN_MEAN, cfg.INPUT.TRAIN_STD),
				# 随机擦除
				RandomErasing(probability=cfg.INPUT.RANDOM_ERASE.RE_PROB,
							  sh=cfg.INPUT.RANDOM_ERASE.RE_MAX_RATIO,
							  mean=cfg.INPUT.TRAIN_MEAN),
			]
		else:
			image_transform_list=[
				transforms.RandomCrop(32, padding=4),
				transforms.RandomHorizontalFlip(),
				transforms.RandomRotation(15),
				transforms.ToTensor(),
				transforms.Normalize(cfg.INPUT.TRAIN_MEAN, cfg.INPUT.TRAIN_STD),
				# 随机擦除
				RandomErasing(probability=cfg.INPUT.RANDOM_ERASE.RE_PROB,
							  sh=cfg.INPUT.RANDOM_ERASE.RE_MAX_RATIO,
							  mean=cfg.INPUT.TRAIN_MEAN),
			]

		transform = transforms.Compose(image_transform_list)
	else:
		transform = transforms.Compose([
			transforms.ToTensor(),
			transforms.Normalize(cfg.INPUT.TRAIN_MEAN, cfg.INPUT.TRAIN_STD)
		])


	return transform

