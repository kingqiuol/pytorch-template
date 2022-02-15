# -*- coding: utf-8 -*-
# @Time : 2022/1/12 13:55
# @Author : jinqiu
# @Site : 
# @File : metric.py
# @Software: PyCharm

import time
import torch

from scipy import interp
from itertools import cycle

import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from thop import profile
from sklearn.metrics import confusion_matrix

from sklearn.metrics import roc_curve, auc, f1_score, precision_recall_curve, average_precision_score
from sklearn.preprocessing import label_binarize


def plot_roc_curve(output,target,num_class):
	'''
	绘制ROC曲线
	:param output:
	:param target:
	:param num_class:
	:return:
	'''
	# 调用sklearn库，计算每个类别对应的fpr和tpr
	fpr_dict = dict()
	tpr_dict = dict()
	roc_auc_dict = dict()
	for i in range(num_class):
		fpr_dict[i], tpr_dict[i], _ = roc_curve(target[:, i], output[:, i])
		roc_auc_dict[i] = auc(fpr_dict[i], tpr_dict[i])
	# micro
	fpr_dict["micro"], tpr_dict["micro"], _ = roc_curve(target.ravel(), output.ravel())
	roc_auc_dict["micro"] = auc(fpr_dict["micro"], tpr_dict["micro"])

	# macro
	# First aggregate all false positive rates
	all_fpr = np.unique(np.concatenate([fpr_dict[i] for i in range(num_class)]))
	# Then interpolate all ROC curves at this points
	mean_tpr = np.zeros_like(all_fpr)
	for i in range(num_class):
		mean_tpr += interp(all_fpr, fpr_dict[i], tpr_dict[i])
	# Finally average it and compute AUC
	mean_tpr /= num_class
	fpr_dict["macro"] = all_fpr
	tpr_dict["macro"] = mean_tpr
	roc_auc_dict["macro"] = auc(fpr_dict["macro"], tpr_dict["macro"])

	# 绘制所有类别平均的roc曲线
	plt.figure()
	lw = 2
	plt.plot(fpr_dict["micro"], tpr_dict["micro"],
			 label='micro-average ROC curve (area = {0:0.2f})'
				   ''.format(roc_auc_dict["micro"]),
			 color='deeppink', linestyle=':', linewidth=4)

	plt.plot(fpr_dict["macro"], tpr_dict["macro"],
			 label='macro-average ROC curve (area = {0:0.2f})'
				   ''.format(roc_auc_dict["macro"]),
			 color='navy', linestyle=':', linewidth=4)

	colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
	for i, color in zip(range(num_class), colors):
		# plt.plot(fpr_dict[i], tpr_dict[i], color=color, lw=lw,
		# 		 label='class {0} (area = {1:0.2f})'
		# 			   ''.format(i, roc_auc_dict[i]))
		plt.plot(fpr_dict[i], tpr_dict[i], color=color, lw=lw,label=None)
	plt.plot([0, 1], [0, 1], 'k--', lw=lw)
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.05])
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('Some extension of Receiver operating characteristic to multi-class')
	plt.legend(loc="lower right")
	plt.savefig('./doc/roc.jpg')
	plt.show()


def plot_pr_curve(output,target,num_class):
	'''
	绘制PR曲线
	:param output:
	:param target:
	:param num_class:
	:return:
	'''
	# 调用sklearn库，计算每个类别对应的precision和recall
	precision_dict = dict()
	recall_dict = dict()
	average_precision_dict = dict()
	for i in range(num_class):
		precision_dict[i], recall_dict[i], _ = precision_recall_curve(target[:, i], output[:, i])
		average_precision_dict[i] = average_precision_score(target[:, i], output[:, i])
		# print(precision_dict[i].shape, recall_dict[i].shape, average_precision_dict[i])

	# micro
	precision_dict["micro"], recall_dict["micro"], _ = precision_recall_curve(target.ravel(),
																			  output.ravel())
	average_precision_dict["micro"] = average_precision_score(target, output, average="micro")
	print('Average precision score, micro-averaged over all classes: {0:0.2f}'.format(average_precision_dict["micro"]))

	# 绘制所有类别平均的pr曲线
	plt.figure()
	plt.step(recall_dict['micro'], precision_dict['micro'], where='post')

	plt.xlabel('Recall')
	plt.ylabel('Precision')
	plt.ylim([0.0, 1.05])
	plt.xlim([0.0, 1.0])
	plt.title(
		'Average precision score, micro-averaged over all classes: AP={0:0.2f}'
			.format(average_precision_dict["micro"]))
	plt.savefig("./doc/pr_curve.jpg")
	plt.show()



def plot_confusion_matrix(cm, labels_name, title):
    '''
    绘制混淆矩阵图
    :param cm:
    :param labels_name:
    :param title:
    :return:
    '''
    plt.figure(figsize=(8, 8))
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]  # 归一化
    plt.imshow(cm, interpolation='nearest')  # 在特定的窗口上显示图像
    plt.title(title)  # 图像标题
    plt.colorbar()
    num_local = np.array(range(len(labels_name)))
    plt.xticks(num_local, labels_name, rotation=90)  # 将标签印在x轴坐标上
    plt.yticks(num_local, labels_name)  # 将标签印在y轴坐标上
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig('./doc/%s.jpg' % title, bbox_inches='tight')


def get_confusion_matrix(output, target,labels_name,title="CIFAR100_cm"):
    '''
    绘制混淆矩阵
    :param output:
    :param target:
    :return:
    '''
    cm = confusion_matrix(output, target, labels=None, sample_weight=None)
    print("confusion_matrix:{}".format(np.diagonal(cm)))
    plot_confusion_matrix(cm, labels_name=labels_name, title=title)



def topk_accuracy(output, target, topk=(1,)):
    """
    计算前K个。N表示样本数，C表示类别数
    :param output: 大小为[N, C]，每行表示该样本计算得到的C个类别概率
    :param target: 大小为[N]，每行表示指定类别
    :param topk: tuple，计算前top-k的accuracy
    :return: list
    """
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, largest=True, sorted=True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


@torch.no_grad()
def compute_accuracy(data_loader, model, device=None, isErr=False, topk=(1, 5)):
    '''
    计算准确率
    :param data_loader:
    :param model:
    :param device:
    :param isErr:
    :param topk:
    :return:
    '''
    if device:
        model = model.to(device)

    epoch_top1_acc = 0.0
    epoch_top5_acc = 0.0
    for inputs, targets in data_loader:
        if device:
            inputs = inputs.to(device)
            targets = targets.to(device)

        # forward
        # track history if only in train
        with torch.no_grad():
            outputs = model(inputs)
            # print(outputs.shape)
            # _, preds = torch.max(outputs, 1)

            # statistics
            res_acc = topk_accuracy(outputs, targets, topk=topk)
            epoch_top1_acc += res_acc[0]
            epoch_top5_acc += res_acc[1]

    if isErr:
        top_1_err = 100 - epoch_top1_acc / len(data_loader)
        top_5_err = 100 - epoch_top5_acc / len(data_loader)
        return top_1_err, top_5_err
    else:
        return epoch_top1_acc / len(data_loader), epoch_top5_acc / len(data_loader)


def compute_gflops_and_model_size(model):
    '''
    计算模型的flops和大小
    :param model:
    :return:
    '''
    input = torch.randn(1, 3, 224, 224)
    macs, params = profile(model, inputs=(input,), verbose=False)

    GFlops = macs * 2.0 / pow(10, 9)
    model_size = params * 4.0 / 1024 / 1024
    return GFlops, model_size


def compute_params(model):
    assert isinstance(model, nn.Module)
    return sum([param.numel() for param in model.parameters()])


@torch.no_grad()
def compute_fps(model, shape, epoch=100, device=None):
    """
    frames per second
    :param shape: 输入数据大小
    """
    total_time = 0.0

    if device:
        model = model.to(device)
    for i in range(epoch):
        data = torch.randn(shape)
        if device:
            data = data.to(device)

        start = time.time()
        outputs = model(data)
        end = time.time()

        total_time += (end - start)

    return total_time / epoch
