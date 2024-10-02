# Copyright 2023 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
# Modified from RT-DETR (https://github.com/lyuwenyu/RT-DETR/tree/main/rtdetr_pytorch)
# ============================================================================

import torch 
import torch.nn as nn 
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

from src.core import register


__all__ = ['AdamW', 'SGD', 'Adam', 'MultiStepLR', 'CosineAnnealingLR', 'OneCycleLR', 'LambdaLR']



SGD = register(optim.SGD)
Adam = register(optim.Adam)
AdamW = register(optim.AdamW)


MultiStepLR = register(lr_scheduler.MultiStepLR)
CosineAnnealingLR = register(lr_scheduler.CosineAnnealingLR)
OneCycleLR = register(lr_scheduler.OneCycleLR)
LambdaLR = register(lr_scheduler.LambdaLR)
