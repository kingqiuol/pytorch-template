import sys


def get_network(args):
	""" return given network
	"""
	if args.MODEL.NAME == 'vgg16':
		from models.vgg import vgg16_bn
		net = vgg16_bn()
	elif args.MODEL.NAME == 'vgg13':
		from models.vgg import vgg13_bn
		net = vgg13_bn()
	elif args.MODEL.NAME == 'vgg11':
		from models.vgg import vgg11_bn
		net = vgg11_bn()
	elif args.MODEL.NAME == 'vgg19':
		from models.vgg import vgg19_bn
		net = vgg19_bn()
	elif args.MODEL.NAME == 'densenet121':
		from models.densenet import densenet121
		net = densenet121()
	elif args.MODEL.NAME == 'densenet161':
		from models.densenet import densenet161
		net = densenet161()
	elif args.MODEL.NAME == 'densenet169':
		from models.densenet import densenet169
		net = densenet169()
	elif args.MODEL.NAME == 'densenet201':
		from models.densenet import densenet201
		net = densenet201()
	elif args.MODEL.NAME == 'googlenet':
		from models.googlenet import googlenet
		net = googlenet()
	elif args.MODEL.NAME == 'inceptionv3':
		from models.inceptionv3 import inceptionv3
		net = inceptionv3()
	elif args.MODEL.NAME == 'inceptionv4':
		from models.inceptionv4 import inceptionv4
		net = inceptionv4()
	elif args.MODEL.NAME == 'inceptionresnetv2':
		from models.inceptionv4 import inception_resnet_v2
		net = inception_resnet_v2()
	elif args.MODEL.NAME == 'xception':
		from models.xception import xception
		net = xception()
	elif args.MODEL.NAME == 'resnet18':
		from models.resnet import resnet18
		net = resnet18()
	elif args.MODEL.NAME == 'resnet34':
		from models.resnet import resnet34
		net = resnet34()
	elif args.MODEL.NAME == 'resnet50':
		from models.resnet import resnet50
		net = resnet50()
	elif args.MODEL.NAME == 'resnet101':
		from models.resnet import resnet101
		net = resnet101()
	elif args.MODEL.NAME == 'resnet152':
		from models.resnet import resnet152
		net = resnet152()
	elif args.MODEL.NAME == 'preactresnet18':
		from models.preactresnet import preactresnet18
		net = preactresnet18()
	elif args.MODEL.NAME == 'preactresnet34':
		from models.preactresnet import preactresnet34
		net = preactresnet34()
	elif args.MODEL.NAME == 'preactresnet50':
		from models.preactresnet import preactresnet50
		net = preactresnet50()
	elif args.MODEL.NAME == 'preactresnet101':
		from models.preactresnet import preactresnet101
		net = preactresnet101()
	elif args.MODEL.NAME == 'preactresnet152':
		from models.preactresnet import preactresnet152
		net = preactresnet152()
	elif args.MODEL.NAME == 'resnext50':
		from models.resnext import resnext50
		net = resnext50()
	elif args.MODEL.NAME == 'resnext101':
		from models.resnext import resnext101
		net = resnext101()
	elif args.MODEL.NAME == 'resnext152':
		from models.resnext import resnext152
		net = resnext152()
	elif args.MODEL.NAME == 'shufflenet':
		from models.shufflenet import shufflenet
		net = shufflenet()
	elif args.MODEL.NAME == 'shufflenetv2':
		from models.shufflenetv2 import shufflenetv2
		net = shufflenetv2()
	elif args.MODEL.NAME == 'squeezenet':
		from models.squeezenet import squeezenet
		net = squeezenet()
	elif args.MODEL.NAME == 'mobilenet':
		from models.mobilenet import mobilenet
		net = mobilenet()
	elif args.MODEL.NAME == 'mobilenetv2':
		from models.mobilenetv2 import mobilenetv2
		net = mobilenetv2()
	elif args.MODEL.NAME == 'nasnet':
		from models.nasnet import nasnet
		net = nasnet()
	elif args.MODEL.NAME == 'attention56':
		from models.attention import attention56
		net = attention56()
	elif args.MODEL.NAME == 'attention92':
		from models.attention import attention92
		net = attention92()
	elif args.MODEL.NAME == 'seresnet18':
		from models.senet import seresnet18
		net = seresnet18()
	elif args.MODEL.NAME == 'seresnet34':
		from models.senet import seresnet34
		net = seresnet34()
	elif args.MODEL.NAME == 'seresnet50':
		from models.senet import seresnet50
		net = seresnet50()
	elif args.MODEL.NAME == 'seresnet101':
		from models.senet import seresnet101
		net = seresnet101()
	elif args.MODEL.NAME == 'seresnet152':
		from models.senet import seresnet152
		net = seresnet152()
	elif args.MODEL.NAME == 'wideresnet':
		from models.wideresidual import wideresnet
		net = wideresnet()
	elif args.MODEL.NAME == 'stochasticdepth18':
		from models.stochasticdepth import stochastic_depth_resnet18
		net = stochastic_depth_resnet18()
	elif args.MODEL.NAME == 'stochasticdepth34':
		from models.stochasticdepth import stochastic_depth_resnet34
		net = stochastic_depth_resnet34()
	elif args.MODEL.NAME == 'stochasticdepth50':
		from models.stochasticdepth import stochastic_depth_resnet50
		net = stochastic_depth_resnet50()
	elif args.MODEL.NAME == 'stochasticdepth101':
		from models.stochasticdepth import stochastic_depth_resnet101
		net = stochastic_depth_resnet101()

	elif args.MODEL.NAME == 'vit':
		from models.vit import vit
		net =vit()
	else:
		print('the network name you have entered is not supported yet')
		sys.exit()

	if args.MODEL.USE_GPU:  # use_gpu
		net = net.cuda()

	return net
