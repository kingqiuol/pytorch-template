# 深度学习分类优化实战

近期做了一些与分类相关得实验，主要研究了模型有过过程中的一些优化手段，这里记录下，本文对相关模型和算法进行了实现并运行测试，整体来说，有的优化手段可以增加模型的准确率，有的可能没啥效果，总的记录如下文。本文使用得数据集为CIFAR-100 。

**代码地址：**[传送门](https://github.com/kingqiuol/pytorch-template.git)

## 一、优化策略

### 1、CIFAR-100 数据集简介

首先，我们需要拿到数据和明确我们的任务。这里以cifar-100为例，它是8000万个微小图像数据集的子集，他们由Alex Krizhevsky，Vinod Nair和Geoffrey Hinton收集。CIFAR -100数据集（100 个类别）是 Tiny Images 数据集的子集，由 60000 个 32x32 彩色图像组成。CIFAR-100 中的 100 个类分为 20 个超类。每个类有 600 张图像。每个图像都带有一个“精细”标签（它所属的类）和一个“粗略”标签（它所属的超类）。每个类有 500 个训练图像和 100 个测试图像。

简单来说，我们需要针对CIFAR-100 数据集，设计、搭建、训练机器学习模型，能够尽可能准确地分辨出测试数据地标签。

**参考连接：**

[CIFAR100数据集介绍及使用方法](https://blog.csdn.net/qq_45589658/article/details/109440786)

### 2、模型评估指标

对于分类模型，最主要的是看模型的准确率。当然，光从准确率不能完全评估模型的性能，我还需要从混淆矩阵来看每一类的分类情况，PR曲线分析我们模型的准确率和召回率，ROC曲线评估模型的泛化能力。具体实现可以参考本文代码`utils/metric.py`。

* 混淆矩阵

<img src="doc\CIFAR100_cm.jpg" alt="CIFAR100_cm" style="zoom:67%;" />

通过观察，可以看出模型对每一类都能很好的进行分类。

* PR曲线

<img src="doc\pr_curve.jpg" alt="pr_curve" style="zoom:67%;" />

* ROC曲线

<img src="doc\roc.jpg" alt="roc" style="zoom:67%;" />

### 3、数据！数据！数据！

#### 3.1、数据增强

数据增强是解决过拟合一个比较好的手段，它的本质是在一定程度上扩充训练数据样本，避免模型拟合到训练集中的噪声，所以设计一个好的数据增强方案尤为必要。在CV任务中，常用的数据增强包括RandomCrop(随机扣取)、Padding(补丁)、RandomHorizontalFlip(随机水平翻转)、ColorJilter(颜色抖动)等。还有一些其他高级的数据增强技巧，比如RandomEreasing(随机擦除)、MixUp、CutMix、AutoAugment，以及最新的AugMix和GridMask等。在实际训练中，如何选择，需要以具体实验为主，主要需要参考一些优秀论文，借鉴何使用。在此次任务中我们除了一些常用的增强方法外，也选择了一些加分点的优化手段，然后通过选择实验对比，选择较合适的数据增强方案。具体实现`utils/augment/augment.py`。

主要对比如下：

| method                                                       | acc  |
| ------------------------------------------------------------ | ---- |
| RandomCrop+RandomHorizontalFlip+RandomRotation               | 0.78 |
| RandomCrop+RandomHorizontalFlip+RandomRotation+random_erase  | 0.79 |
| RandomCrop+RandomHorizontalFlip+RandomRotation+random_erase+autoaugment | 0.81 |

#### 3.2、数据分布

本文使用的CIFAR-100数据集的每一个类属于数据比较均衡的，但在实际分类中，大多数是不均衡的长尾数据，这个时候需要减少这种不均衡对预测的影响。当然，除了长尾分布的影响，还有类间相似的影响，比如两个类比较接近，无论形状、大小或颜色等，需要算法进一步区分或尽量减少对分类的影响。常用的解决长尾分布手手段有：重采样（需要在不影响原始分布的情况，如异常检测，这种情况重采样会改变数据原始分布，反而会降低准确率，因为本来就是正/负样本多）、重新设计loss（如Focal loss、OHEM、Class Balanced Loss）、或者转化为异常检测以及One-class分类模型等。

对于多类别问题，同一张图片可能有多个类，此时传统的CE loss的设计就有一定缺陷了。因为在多标签分类中，一个数据点中可以有多个正确的类。因此，多标签分类问题的需要检测图像中存在的每个对象。而CE loss会尽可能拟合one-hot标签，容易造成过拟合，无法保证模型的泛化能力,同时由于无法保证标签百分百正确，可能存在一些错误标签，但模型也会拟合这些错误标签，由于以上原因，提出了标签平滑，为软标签，属于正则化的一种，可以防止过拟合。label smoothing标签平滑实现见`utils/losses.py`。

**参考链接：**

[样本不均衡、长尾分布问题的方法整理（文献+代码）](https://blog.csdn.net/Bit_Coders/article/details/117460999)

[视觉分类任务中处理不平衡问题的loss比较](https://cloud.tencent.com/developer/article/2008018)

[长尾分布分类问题解决方法](https://www.jianshu.com/p/aa60059582af)

### 4、模型选择

模型的选择优先考虑最新最好的模型，可以参考[传送门](https://paperswithcode.com/task/image-classification)，选择合适的模型。这里，我选择的ResNet模型作为baseline backbone。

<img src="doc\1.png" alt="1" style="zoom: 80%;" />

这里我们进行不同的模型比较，实验如下：

| method    | acc  |
| --------- | ---- |
| resnet18  | 0.75 |
| resnet50  | 0.78 |
| resnet101 | 0.79 |

可以看出模型越复杂，能提升我们的模型准确率。所以后续我们也选择了wideresnet这样的大的模型来训练这个对模型的准确率也有很大的提升。

当然，后续还可以选择当前最新的transformer模型，如：VIT、Swin、CaiT等，作为我们的训练模型。

**参考链接：**

[一文窥探近期大火的Transformer以及在图像分类领域的应用_果菌药的博客-程序员ITS401_transformer图像分类](https://its401.com/article/qq_40688292/112540486)

[Transformer小试牛刀（一）：Vision Transformer](https://www.heywhale.com/mw/project/60d980b694c44a0017dc0c5f)

### 5、模型优化

#### 5.1、学习率选择

我们通过枚举不同学习率下的loss值选择最优学习率（具体实现`tool/lr_finder.py`），绘制曲线如下：

<img src="doc\find_lr.jpg" alt="find_lr" style="zoom:67%;" />

通过观察可知，**lr=0.1**时loss最低，此时学习率最优。

#### 5.2、优化器选择

对于深度学习来说，优化器比较多，如：SGD、Adagrad、Adadelta、RMSprop、Adam等。当然，也有最新的优化器，如：Ranger、SAM等（具体实现`utils/optim.py`）。

这里我们对不同的优化器比较，实验如下：

| method | acc    |
| ------ | ------ |
| SGD    | 0.79   |
| adam   | 0.79   |
| ranger | 0.65   |
| SAM    | 0.8311 |

通过观察可知，选择**SAM**优化器最优。

**参考链接：**

[深度学习——优化器算法Optimizer详解（BGD、SGD、MBGD、Momentum、NAG、Adagrad、Adadelta、RMSprop、Adam）](https://www.cnblogs.com/guoyaohua/p/8542554.html)

[再也不用担心过拟合的问题了](https://jishuin.proginn.com/p/763bfbd5461b)

#### 5.3、学习率更新策略选择

这里我们选择**warmup**预热更新策略，具体实现`utils/scheduler.py`

#### 5.4、loss选择

在前面的数据分析中，我们讨论了数据分布的问题，由于我们的数据是多分类问题，所以我们需要在交叉熵损失函数的基础上加入标签平滑，这样能够更好的训练，防止过拟合。

这里我们对不同的损失函数比较，实验如下：

| method    | acc    |
| --------- | ------ |
| CE        | 0.8311 |
| smooth_CE | 0.833  |

### 6、整体思路

我们初步训练resnet50作为基础模型，实验测试过程如下：

|     network     |                      method                       |  acc   |
| :-------------: | :-----------------------------------------------: | :----: |
|    resnet18     |                   SGD+warmup+CE                   |  0.75  |
|    resnet50     |                   SGD+warmup+CE                   |  0.78  |
|    resnet101    |                   SGD+warmup+CE                   |  0.79  |
|    resnet50     |            SGD+warmup+random_erase+CE             |  0.79  |
|    resnet50     |      SGD+warmup+random_erase+autoaugment+CE       | 0.815  |
|    resnet50     |      adam+warmup+random_erase+autoaugment+CE      |  0.79  |
|    resnet50     |     ranger+warmup+random_erase+autoaugment+CE     |  0.65  |
|    resnet50     |      SAM+warmup+random_erase+autoaugment+CE       | 0.8311 |
|    resnet50     |   SAM+warmup+random_erase+autoaugment+smooth_CE   | 0.833  |
| wideresnet40_10 |   SAM+warmup+random_erase+autoaugment+smooth_CE   | 0.840  |
| wideresnet40_10 | SAM+warmup+random_erase+autoaugment+smooth_CE+TTA | 0.8437 |

通过实验，我们最终选择wideresnet40_10作为特征提取模型，实验过程中将Accuracy由78%提升到84.37%。

## 二、pytorch实战

1. **安装要求**
   * python3.6
   * pytorch1.6.0+cu101
   * tensorboard 2.2.2(optional)

2. **运行tensorboard**

```bash
$ cd pytorch-cifar100
$ mkdir runs
$ tensorboard --logdir='runs' --port=6006 --host='localhost'
```

3. **训练模型**

```bash
$ python train.py -gpu
```

4. **测试模型**

```bash
$ python test.py 
```

**模型参考链接：**

- vgg [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556v6)
- googlenet [Going Deeper with Convolutions](https://arxiv.org/abs/1409.4842v1)
- inceptionv3 [Rethinking the Inception Architecture for Computer Vision](https://arxiv.org/abs/1512.00567v3)
- inceptionv4, inception_resnet_v2 [Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning](https://arxiv.org/abs/1602.07261)
- xception [Xception: Deep Learning with Depthwise Separable Convolutions](https://arxiv.org/abs/1610.02357)
- resnet [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385v1)
- resnext [Aggregated Residual Transformations for Deep Neural Networks](https://arxiv.org/abs/1611.05431v2)
- resnet in resnet [Resnet in Resnet: Generalizing Residual Architectures](https://arxiv.org/abs/1603.08029v1)
- densenet [Densely Connected Convolutional Networks](https://arxiv.org/abs/1608.06993v5)
- shufflenet [ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile Devices](https://arxiv.org/abs/1707.01083v2)
- shufflenetv2 [ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design](https://arxiv.org/abs/1807.11164v1)
- mobilenet [MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications](https://arxiv.org/abs/1704.04861)
- mobilenetv2 [MobileNetV2: Inverted Residuals and Linear Bottlenecks](https://arxiv.org/abs/1801.04381)
- residual attention network [Residual Attention Network for Image Classification](https://arxiv.org/abs/1704.06904)
- senet [Squeeze-and-Excitation Networks](https://arxiv.org/abs/1709.01507)
- squeezenet [SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and <0.5MB model size](https://arxiv.org/abs/1602.07360v4)
- nasnet [Learning Transferable Architectures for Scalable Image Recognition](https://arxiv.org/abs/1707.07012v4)
- wide residual network[Wide Residual Networks](https://arxiv.org/abs/1605.07146)
- stochastic depth networks[Deep Networks with Stochastic Depth](https://arxiv.org/abs/1603.09382)

