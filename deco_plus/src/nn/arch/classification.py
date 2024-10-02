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

from src.core import register


__all__ = ['Classification', 'ClassHead']


@register
class Classification(nn.Module):
    __inject__ = ['backbone', 'head']

    def __init__(self, backbone: nn.Module, head: nn.Module=None):
        super().__init__()
        
        self.backbone = backbone
        self.head = head

    def forward(self, x):
        x = self.backbone(x)

        if self.head is not None:
            x = self.head(x)

        return x 


@register
class ClassHead(nn.Module):
    def __init__(self, hidden_dim, num_classes):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Linear(hidden_dim, num_classes)  

    def forward(self, x):
        x = x[0] if isinstance(x, (list, tuple)) else x 
        x = self.pool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.proj(x)
        return x 
