# test.py
# !/usr/bin/env python3

""" test neuron network performace
print top1 and top5 err on test dataset
of a model

author baiyu
"""
import yaml
from easydict import EasyDict as edict
import argparse
from matplotlib import pyplot as plt
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from sklearn.metrics import roc_curve, auc, f1_score, precision_recall_curve, average_precision_score

from conf import settings
from models.get_networks import get_network
from datasets.cifar100_dataset import get_test_dataloader
from utils.metric import plot_pr_curve, plot_roc_curve, get_confusion_matrix

from utils.tta.base import Compose
from utils.tta.tta_wrappers import ClassificationTTAWrapper
from utils.tta.transforms import (
    HorizontalFlip, VerticalFlip, Rotate90, Scale, Add, Multiply, FiveCrops, Resize
)

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("-config",type=str,default="./conf/config.ymal",help="training model configuration information")
	args = parser.parse_args()

	# 加载配置文件
	yaml_path = args.config
	with open(yaml_path, "r", encoding='utf-8') as f:
		cfg = yaml.load(f, Loader=yaml.FullLoader)
		cfg = edict(cfg)


	net = get_network(cfg)



	cifar100_test_loader = get_test_dataloader(
		cfg,
		num_workers=cfg.DATALOADER.NUM_WORKERS,
		batch_size=cfg.SOLVER.BATCH_SIZE,
		shuffle=True
	)

	net.load_state_dict(torch.load(cfg.OUTPUT.WEIGHTS))
	net.eval()

	if cfg.SOLVER.USE_TTA:
		tta_transforms = Compose(
			[
				FiveCrops(32, 32),
				HorizontalFlip(),
				# VerticalFlip(),
				Rotate90(angles=list(range(15)))
			]
		)
		net = ClassificationTTAWrapper(net, tta_transforms)

	correct_1 = 0.0
	correct_5 = 0.0
	total = 0
	score_list = []  # 存储预测得分
	label_list = []  # 存储标签
	with torch.no_grad():
		for n_iter, (image, label) in enumerate(cifar100_test_loader):
			print("iteration: {}\ttotal {} iterations".format(n_iter + 1, len(cifar100_test_loader)))

			if cfg.MODEL.USE_GPU:
				image = image.cuda()
				label = label.cuda()
				# print('GPU INFO.....')
				# print(torch.cuda.memory_summary(), end='')

			output = net(image)
			_, pred = output.topk(5, 1, largest=True, sorted=True)

			score_list.extend(output.detach().cpu().numpy())
			label_list.extend(label.cpu().numpy())

			label = label.view(label.size(0), -1).expand_as(pred)
			correct = pred.eq(label).float()

			# compute top 5
			correct_5 += correct[:, :5].sum()

			# compute top1
			correct_1 += correct[:, :1].sum()

	if cfg.MODEL.USE_GPU:
		print('GPU INFO.....')
		print(torch.cuda.memory_summary(), end='')

	# 计算topk
	print("Top 1 err: ", 1 - correct_1 / len(cifar100_test_loader.dataset))
	print("Top 5 err: ", 1 - correct_5 / len(cifar100_test_loader.dataset))
	print("Parameter numbers: {}".format(sum(p.numel() for p in net.parameters())))

	score_array = np.array(score_list)
	# 将label转换成onehot形式
	label_tensor = torch.tensor(label_list)
	label_tensor = label_tensor.reshape((label_tensor.shape[0], 1))
	label_onehot = torch.zeros(label_tensor.shape[0], cfg.MODEL.N_CLASS)
	label_onehot.scatter_(dim=1, index=label_tensor, value=1)
	label_onehot = np.array(label_onehot)
	print("score_array:", score_array.shape)  # (batchsize, classnum) softmax
	print("label_onehot:", label_onehot.shape)

	pred = np.argmax(score_array, axis=1)
	label = np.array(label_list)
	print(pred.shape, label.shape)

	# 绘制PR,ROC
	plot_pr_curve(score_array, label_onehot, cfg.MODEL.N_CLASS)
	plot_roc_curve(score_array, label_onehot, cfg.MODEL.N_CLASS)

	# 绘制混淆矩阵
	get_confusion_matrix(pred, label, labels_name=["1" for i in range(100)])
