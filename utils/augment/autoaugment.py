# -*- coding: utf-8 -*-
# @Time : 2022/1/25 13:52
# @Author : jinqiu
# @Site : 
# @File : autoaugment.py
# @Software: PyCharm

import numpy as np
import random
import math

from utils.augment.common.augmentations import apply_augment


class Augmentation(object):
    def __init__(self, policies):
        self.policies = policies

    def __call__(self, img):
        for _ in range(1):
            policy = random.choice(self.policies)
            for name, pr, level in policy:
                if random.random() > pr:
                    continue
                img = apply_augment(img, name, level)
        return img