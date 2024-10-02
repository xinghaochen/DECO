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
import torch.nn.functional as F 
from collections import OrderedDict
from .common import get_activation, ConvNormLayer, FrozenBatchNorm2d
from src.core import register

__all__ = ['ResNet']

ResNet_cfg = {
    18: [2, 2, 2, 2],
    50: [3, 4, 6, 3],
}

ckpt_path = {
    18: 'pretrained/resnet18_pretrained.pth',
    50: 'pretrained/resnet50_pretrained.pth',
}

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, ch_in, ch_out, stride, shortcut, act='relu'):
        super().__init__()

        self.shortcut = shortcut

        if not shortcut:
            if stride == 2:
                self.short = nn.Sequential(OrderedDict([
                    ('pool', nn.AvgPool2d(2, 2, 0, ceil_mode=True)),
                    ('conv', ConvNormLayer(ch_in, ch_out, 1, 1))
                ]))
            else:
                self.short = ConvNormLayer(ch_in, ch_out, 1, stride)

        self.branch1 = ConvNormLayer(ch_in, ch_out, 3, stride, act=act)
        self.branch2 = ConvNormLayer(ch_out, ch_out, 3, 1, act=None)
        self.act = nn.Identity() if act is None else get_activation(act) 


    def forward(self, x):
        out = self.branch1(x)
        out = self.branch2(out)
        if self.shortcut:
            short = x
        else:
            short = self.short(x)
        
        out = out + short
        out = self.act(out)

        return out


class BottleNeck(nn.Module):
    expansion = 4

    def __init__(self, ch_in, ch_out, stride, shortcut, act='relu'):
        super().__init__()

        stride1, stride2 = 1, stride

        width = ch_out 

        self.branch1 = ConvNormLayer(ch_in, width, 1, stride1, act=act)
        self.branch2 = ConvNormLayer(width, width, 3, stride2, act=act)
        self.branch3 = ConvNormLayer(width, ch_out * self.expansion, 1, 1)

        self.shortcut = shortcut
        if not shortcut:
            if stride == 2:
                self.short = nn.Sequential(OrderedDict([
                    ('pool', nn.AvgPool2d(2, 2, 0, ceil_mode=True)),
                    ('conv', ConvNormLayer(ch_in, ch_out * self.expansion, 1, 1))
                ]))
            else:
                self.short = ConvNormLayer(ch_in, ch_out * self.expansion, 1, stride)

        self.act = nn.Identity() if act is None else get_activation(act) 

    def forward(self, x):
        out = self.branch1(x)
        out = self.branch2(out)
        out = self.branch3(out)

        if self.shortcut:
            short = x
        else:
            short = self.short(x)

        out = out + short
        out = self.act(out)

        return out


class Blocks(nn.Module):
    def __init__(self, block, ch_in, ch_out, count, stage_num, act='relu'):
        super().__init__()

        self.blocks = nn.ModuleList()
        for i in range(count):
            self.blocks.append(
                block(
                    ch_in, 
                    ch_out,
                    stride=2 if i == 0 and stage_num != 2 else 1, 
                    shortcut=False if i == 0 else True,
                    act=act)
            )

            if i == 0:
                ch_in = ch_out * block.expansion

    def forward(self, x):
        out = x
        for block in self.blocks:
            out = block(out)
        return out


@register
class ResNet(nn.Module):
    def __init__(
        self, 
        depth, 
        num_stages=4, 
        return_idx=[0, 1, 2, 3], 
        act='relu',
        freeze_at=-1, 
        freeze_norm=True, 
        pretrained=False):
        super().__init__()

        block_nums = ResNet_cfg[depth]
        ch_in = 64
        conv_def = [
            [3, ch_in // 2, 3, 2, "conv1_1"],
            [ch_in // 2, ch_in // 2, 3, 1, "conv1_2"],
            [ch_in // 2, ch_in, 3, 1, "conv1_3"],
        ]

        self.conv1 = nn.Sequential(OrderedDict([
            (_name, ConvNormLayer(c_in, c_out, k, s, act=act)) for c_in, c_out, k, s, _name in conv_def
        ]))

        ch_out_list = [64, 128, 256, 512]
        block = BottleNeck if depth >= 50 else BasicBlock

        _out_channels = [block.expansion * v for v in ch_out_list]
        _out_strides = [4, 8, 16, 32]

        self.layers = nn.ModuleList()
        for i in range(num_stages):
            stage_num = i + 2
            self.layers.append(
                Blocks(block, ch_in, ch_out_list[i], block_nums[i], stage_num, act=act)
            )
            ch_in = _out_channels[i]

        self.return_idx = return_idx
        self.out_channels = [_out_channels[_i] for _i in return_idx]
        self.out_strides = [_out_strides[_i] for _i in return_idx]

        if freeze_at >= 0:
            self._freeze_parameters(self.conv1)
            for i in range(min(freeze_at, num_stages)):
                self._freeze_parameters(self.layers[i])

        if freeze_norm:
            self._freeze_norm(self)

        if pretrained:
            state = torch.load(ckpt_path[depth], map_location='cpu')
            self.load_state_dict(state)
            print(f'Load ResNet{depth} state_dict')
            
    def _freeze_parameters(self, m: nn.Module):
        for p in m.parameters():
            p.requires_grad = False

    def _freeze_norm(self, m: nn.Module):
        if isinstance(m, nn.BatchNorm2d):
            m = FrozenBatchNorm2d(m.num_features)
        else:
            for name, child in m.named_children():
                _child = self._freeze_norm(child)
                if _child is not child:
                    setattr(m, name, _child)
        return m

    def forward(self, x):
        conv1 = self.conv1(x)
        x = F.max_pool2d(conv1, kernel_size=3, stride=2, padding=1)
        outs = []
        for idx, stage in enumerate(self.layers):
            x = stage(x)
            if idx in self.return_idx:
                outs.append(x)
        return outs


