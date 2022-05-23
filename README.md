# 深度学习分类优化实战

近期做了一些与分类相关得实验，主要研究了模型有过过程中的一些优化手段，这里记录下，本文对相关模型和算法进行了实现并运行测试，整体来说，有的优化手段可以增加模型的准确率，有的可能没啥效果，总的记录下文。本文使用得数据集为cifar100 

## 一、优化策略



## 二、pytorch实战

1. **安装要求**
   * python3.6
   * pytorch1.6.0+cu101
   * tensorboard 2.2.2(optional)

2. **运行tensorboard**

```bash
$ cd pytorch-cifar100
$ pip install tensorboard
$ mkdir runs
Run tensorboard
$ tensorboard --logdir='runs' --port=6006 --host='localhost'
```

3. **训练模型**

```bash
# use gpu to train vgg16
$ python train.py -net vgg16 -gpu
```

4. **测试模型**

```bash
$ python test.py -net vgg16 -weights path_to_vgg16_weights_file
```

## 三、测试结果

不同优化策略的比较：

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

## 四、相关网络

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

