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

"""
DECO ConvNet classes.
"""

import copy
from typing import Optional
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from timm.models.layers import DropPath

from .encoder_module import *

class DECO_ConvNet(nn.Module):
    '''DECO ConvNet class, including encoder and decoder'''

    def __init__(self, num_queries=100, d_model=512, enc_dims=[120,240,480], enc_depth=[2,6,2], 
                 num_decoder_layers=6, normalize_before=False, return_intermediate_dec=False, qH=10):
        super().__init__()

        # object query shape
        self.qH = qH
        self.qW = int(np.float(num_queries)/np.float(self.qH))
        print('query shape {}x{}'.format(self.qH, self.qW))  

        # encoder
        self.encoder = DecoEncoder(enc_dims=enc_dims, enc_depth=enc_depth)     

        # decoder
        decoder_layer = DecoDecoderLayer(d_model, normalize_before, self.qH, self.qW)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = DecoDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                          return_intermediate=return_intermediate_dec, qH=self.qH, qW=self.qW)

        # other initialization
        self.tgt = nn.Embedding(num_queries, d_model)
        self._reset_parameters()
        self.d_model = d_model
        
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, query_embed):
        bs, c, h, w = src.shape

        tgt=self.tgt.weight.unsqueeze(1).repeat(1, bs, 1)
        memory = self.encoder(src)
       
        query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)
        query_embed = query_embed.permute(1, 2, 0).view(bs, self.d_model,self.qH,self.qW)

        hs = self.decoder(tgt, memory, bs=bs, d_model=self.d_model, query_pos=query_embed)
        
        return hs.transpose(1, 2), memory


class DecoEncoder(nn.Module):
    '''Define Deco Encoder'''
    def __init__(self, enc_dims=[120,240,480], enc_depth=[2,6,2]): 
        super().__init__()
        self._encoder = ConvNeXt(depths=enc_depth, dims=enc_dims) 

    def forward(self, src):    
        output = self._encoder(src)
        return output


class DecoDecoder(nn.Module):
    '''Define Deco Decoder'''
    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False, qH=10, qW=10):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate
        self.qH = qH
        self.qW = qW

    def forward(self, tgt, memory, bs, d_model, query_pos: Optional[Tensor] = None):
        output = tgt
        intermediate = []

        for layer in self.layers:
            output=output.permute(1, 2, 0).view(bs, d_model,self.qH,self.qW)
            output = layer(output, memory, query_pos=query_pos)
            output=output.flatten(2).permute(2, 0, 1)
            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)
        
        return output.unsqueeze(0)

class DecoDecoderLayer(nn.Module):
    '''Define a layer for Deco Decoder'''
    def __init__(self,d_model, normalize_before=False, qH=10, qW=10,
                 drop_path=0.,layer_scale_init_value=1e-6):
        super().__init__()
        self.normalize_before = normalize_before
        self.qH = qH
        self.qW = qW

        # The SIM module   
        self.dwconv1 = nn.Conv2d(d_model, d_model, kernel_size=9, padding=4, groups=d_model) 
        self.norm1 = LayerNorm(d_model, eps=1e-6)
        self.pwconv1_1 = nn.Linear(d_model, 4 * d_model) 
        self.act1 = nn.GELU()
        self.pwconv1_2 = nn.Linear(4 * d_model, d_model)
        self.gamma1 = nn.Parameter(layer_scale_init_value * torch.ones((d_model)), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        # The CIM module
        self.dwconv2 = nn.Conv2d(d_model, d_model, kernel_size=9, padding=4, groups=d_model) 
        self.norm2 = LayerNorm(d_model, eps=1e-6)
        self.pwconv2_1 = nn.Linear(d_model, 4 * d_model) 
        self.act2 = nn.GELU()
        self.pwconv2_2 = nn.Linear(4 * d_model, d_model)
        self.gamma2 = nn.Parameter(layer_scale_init_value * torch.ones((d_model)), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()


    def forward(self, tgt, memory, query_pos: Optional[Tensor] = None):
        # SIM
        b, d, h, w = memory.shape
        tgt2 = tgt + query_pos
        tgt2 = self.dwconv1(tgt2)
        tgt2 = tgt2.permute(0, 2, 3, 1) # (b,d,10,10)->(b,10,10,d)
        tgt2 = self.norm1(tgt2)
        tgt2 = self.pwconv1_1(tgt2)
        tgt2 = self.act1(tgt2)
        tgt2 = self.pwconv1_2(tgt2)
        if self.gamma1 is not None:
            tgt2 = self.gamma1 * tgt2
        tgt2 = tgt2.permute(0,3,1,2) # (b,10,10,d)->(b,d,10,10)
        tgt = tgt + self.drop_path1(tgt2)

        # CIM
        tgt = F.interpolate(tgt, size=[h,w]) 
        tgt2 = tgt + memory 
        tgt2 = self.dwconv2(tgt2)
        tgt2 = tgt2+tgt 
        tgt2 = tgt2.permute(0, 2, 3, 1) # (b,d,h,w)->(b,h,w,d)
        tgt2=self.norm2(tgt2)
        
        # FFN
        tgt = tgt2
        tgt2 = self.pwconv2_1(tgt2)
        tgt2 = self.act2(tgt2)
        tgt2 = self.pwconv2_2(tgt2)
        if self.gamma2 is not None:
            tgt2 = self.gamma2 * tgt2
        tgt2 = tgt2.permute(0,3,1,2) # (b,h,w,d)->(b,d,h,w)
        tgt = tgt.permute(0,3,1,2) # (b,h,w,d)->(b,d,h,w)
        tgt = tgt + self.drop_path1(tgt2)

        # pooling
        m = nn.AdaptiveMaxPool2d((self.qH, self.qW))
        tgt = m(tgt)

        return tgt


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def build_deco_convnet(args):
    return DECO_ConvNet(num_queries=args.num_queries,
                        d_model=args.hidden_dim,
                        num_decoder_layers=args.dec_layers,
                        normalize_before=args.pre_norm,
                        return_intermediate_dec=True,
                        qH=args.qH,
                        )

class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x
