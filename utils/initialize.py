# -*- coding: utf-8 -*-
# @Time : 2022/1/25 14:24
# @Author : jinqiu
# @Site : 
# @File : initialize.py
# @Software: PyCharm

import random
import torch
import numpy as np


def initialize(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
