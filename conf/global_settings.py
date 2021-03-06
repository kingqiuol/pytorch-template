# -*- encoding: utf-8 -*-


import os
from datetime import datetime

#dataset path (python version)
PATH = r'D:\PycharmProjects\pytorch-template\data\cifar-100-python'

#mean and std of  dataset
TRAIN_MEAN = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
TRAIN_STD = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)

#directory to save weights file
CHECKPOINT_PATH = 'checkpoint'

#total training epoches
EPOCH = 200
MILESTONES = [60, 120, 160]

#initial learning rate
#INIT_LR = 0.1

DATE_FORMAT = '%A_%d_%B_%Y_%Hh_%Mm_%Ss'
#time of we run the script
TIME_NOW = datetime.now().strftime(DATE_FORMAT)

#tensorboard log dir
LOG_DIR = 'runs'

#save weights file per SAVE_EPOCH epoch
SAVE_EPOCH = 10


# 使用自动增强
USE_AUTOAUG=True
# 使用随机擦除
USE_RANDOM_ERASE=True
RE_PROB=0.5
RE_MAX_RATIO=0.4








