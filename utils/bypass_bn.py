# -*- coding: utf-8 -*-
# @Time : 2022/1/25 14:57
# @Author : jinqiu
# @Site : 
# @File : bypass_bn.py
# @Software: PyCharm

import torch
import torch.nn as nn

def disable_running_stats(model):
    def _disable(module):
        if isinstance(module, nn.BatchNorm2d):
            module.backup_momentum = module.momentum
            module.momentum = 0

    model.apply(_disable)

def enable_running_stats(model):
    def _enable(module):
        if isinstance(module, nn.BatchNorm2d) and hasattr(module, "backup_momentum"):
            module.momentum = module.backup_momentum

    model.apply(_enable)
