# -*- coding: utf-8 -*-
# @Time : 2022/2/14 17:12
# @Author : jinqiu
# @Site : 
# @File : tta_wrappers.py
# @Software: PyCharm

import torch
import torch.nn as nn
from typing import Optional, Mapping, Union

from utils.tta.base import Merger, Compose


class ClassificationTTAWrapper(nn.Module):
	'''
	Wrap PyTorch nn.Module (classification model) with test time augmentation transforms
	'''
	def __init__(self,model:nn.Module,
				 transforms:Compose,
				 merge_mode:str="mean",
				 output_label_key: Optional[str] = None,
				 ):
		super().__init__()

		self.model=model
		self.transforms=transforms
		self.merge_mode=merge_mode
		self.output_key=output_label_key

	def forward(self,image: torch.Tensor, *args
				)-> Union[torch.Tensor, Mapping[str, torch.Tensor]]:
		merger=Merger(type=self.merge_mode,n=len(self.transforms))

		for transformer in self.transforms:
			aug_img=transformer.augment_image(image)
			aug_out=self.model(aug_img,*args)
			if self.output_key is not None:
				aug_out=aug_out[self.output_key]
			deaugmented_output = transformer.deaugment_label(aug_out)
			merger.append(deaugmented_output)

		result = merger.result
		if self.output_key is not None:
			result = {self.output_key: result}

		return result