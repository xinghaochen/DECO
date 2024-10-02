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
# Partially modified from RT-DETR (https://github.com/lyuwenyu/RT-DETR/tree/main/rtdetr_pytorch)
# ============================================================================


import math 
import copy 
from collections import OrderedDict

import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import torch.nn.init as init 

from .utils import get_activation, inverse_sigmoid
from .utils import bias_init_with_prob

from timm.models.layers import DropPath

from src.core import register

import matplotlib.pyplot as plt


__all__ = ['DecoPlusDecoderModule']


class InceptionDWConv2d(nn.Module):
    """ Inception depthweise convolution
    """
    def __init__(self, in_channels, square_kernel_size=3, band_kernel_size_h=13, band_kernel_size_w=15, branch_ratio=0.25):
        super().__init__()
        
        gc = int(in_channels * branch_ratio) # channel numbers of a convolution branch
        self.dwconv_hw = nn.Conv2d(gc, gc, square_kernel_size, padding=square_kernel_size//2, groups=gc)
        self.dwconv_w = nn.Conv2d(gc, gc, kernel_size=(1, band_kernel_size_w), padding=(0, band_kernel_size_w//2), groups=gc)
        self.dwconv_h = nn.Conv2d(gc, gc, kernel_size=(band_kernel_size_h, 1), padding=(band_kernel_size_h//2, 0), groups=gc)
        self.split_indexes = (in_channels - 3 * gc, gc, gc, gc)
        
    def forward(self, x):
        x_id, x_hw, x_w, x_h = torch.split(x, self.split_indexes, dim=1)
        return torch.cat((x_id, self.dwconv_hw(x_hw), self.dwconv_w(x_w), self.dwconv_h(x_h)), dim=1,)


class ConvNormLayer(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size, stride, padding=None, bias=False, act=None):
        super().__init__()
        self.conv = nn.Conv2d(
            ch_in, 
            ch_out, 
            kernel_size, 
            stride, 
            padding=(kernel_size-1)//2 if padding is None else padding, 
            bias=bias)
        self.norm = nn.BatchNorm2d(ch_out)
        self.act = nn.Identity() if act is None else get_activation(act) 

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, act='relu'):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))
        self.act = nn.Identity() if act is None else get_activation(act)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = self.act(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


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


class DecoPlusDecoderLayer(nn.Module):
    '''Define a layer for DECO+ Decoder'''
    def __init__(self,d_model, dropout=0.,
                 layer_scale_init_value=1e-6, normalize_before=False, qH=15, qW=20, inceptH=13, inceptW=15, branch_ratio=0.25): 

        super().__init__()
        self.normalize_before = normalize_before
        self.qH = qH
        self.qW = qW
        self.inceptH = inceptH
        self.inceptW = inceptW

        # The SIM module   
        self.dwconv1 = InceptionDWConv2d(d_model, square_kernel_size=3, band_kernel_size_h=self.inceptH, band_kernel_size_w=self.inceptW, branch_ratio=branch_ratio)
        self.norm1 = LayerNorm(d_model, eps=1e-6)
        self.pwconv1_1 = nn.Linear(d_model, 4 * d_model) 
        self.act1 = nn.GELU()
        self.pwconv1_2 = nn.Linear(4 * d_model, d_model)
        self.gamma1 = nn.Parameter(layer_scale_init_value * torch.ones((d_model)), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path1 = DropPath(dropout) if dropout > 0. else nn.Identity()
        
        # The CIM module
        self.dwconv2 = nn.Conv2d(d_model, d_model, kernel_size=9, padding=4, groups=d_model) 
        self.norm2 = LayerNorm(d_model, eps=1e-6)
        self.pwconv2_1 = nn.Linear(d_model, 4 * d_model) 
        self.act2 = nn.GELU()
        self.pwconv2_2 = nn.Linear(4 * d_model, d_model)
        self.gamma2 = nn.Parameter(layer_scale_init_value * torch.ones((d_model)), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path2 = DropPath(dropout) if dropout > 0. else nn.Identity()

    def forward_ffn(self, tgt):
        return self.linear2(self.dropout3(self.activation(self.linear1(tgt))))

    def forward(self, tgt, memory,
                query_pos_embed=None):
        
        tgt = tgt.permute(0,2,1)
        bs, n_embed, _ = tgt.shape
        tgt = tgt.reshape(bs, n_embed, self.qH, self.qW)
        query_pos_embed = query_pos_embed.permute(0,2,1)
        query_pos_embed = query_pos_embed.reshape(bs, n_embed, self.qH, self.qW)
        
        # SIM
        _, _, h, w = memory.shape
        tgt2 = tgt + query_pos_embed
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

        # reshape back
        tgt = tgt.reshape(bs, n_embed, self.qH*self.qW)
        tgt = tgt.permute(0,2,1)
        return tgt


class DecoPlusDecoder(nn.Module):
    def __init__(self, hidden_dim, decoder_layer, num_layers, eval_idx=-1):
        super(DecoPlusDecoder, self).__init__()
        self.layers = nn.ModuleList([copy.deepcopy(decoder_layer) for _ in range(num_layers)])
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.eval_idx = eval_idx if eval_idx >= 0 else num_layers + eval_idx

    def forward(self,
                tgt,
                ref_points_unact,
                memory,
                bbox_head,
                score_head,
                query_pos_head):
        
        output = tgt
        dec_out_bboxes = []
        dec_out_logits = []
        ref_points_detach = F.sigmoid(ref_points_unact)

        for i, layer in enumerate(self.layers):
            query_pos_embed = query_pos_head(ref_points_detach)

            output = layer(output, memory, query_pos_embed=query_pos_embed)

            inter_ref_bbox = F.sigmoid(bbox_head[i](output) + inverse_sigmoid(ref_points_detach))

            if self.training:
                dec_out_logits.append(score_head[i](output))
                if i == 0:
                    dec_out_bboxes.append(inter_ref_bbox)
                else:
                    dec_out_bboxes.append(F.sigmoid(bbox_head[i](output) + inverse_sigmoid(ref_points)))

            elif i == self.eval_idx:
                dec_out_logits.append(score_head[i](output))
                dec_out_bboxes.append(inter_ref_bbox)
                break

            ref_points = inter_ref_bbox
            ref_points_detach = inter_ref_bbox.detach(
            ) if self.training else inter_ref_bbox

        return torch.stack(dec_out_bboxes), torch.stack(dec_out_logits)


@register
class DecoPlusDecoderModule(nn.Module):
    __share__ = ['num_classes']
    def __init__(self,
                 num_classes=80,
                 hidden_dim=256,
                 num_queries=300,
                 qH=15,
                 qW=20,
                 inceptH = 13,
                 inceptW = 15,
                 feat_channels=[512, 1024, 2048],
                 feat_strides=[8, 16, 32],
                 num_levels=3,
                 memory_level=1,
                 num_decoder_layers=6,
                 dropout=0.,
                 downsample_act = 'silu',
                 label_noise_ratio=0.5,
                 box_noise_scale=1.0,
                 eval_spatial_size=None,
                 eval_idx=-1,
                 eps=1e-2, 
                 aux_loss=True):

        super(DecoPlusDecoderModule, self).__init__()
        assert len(feat_channels) <= num_levels
        assert len(feat_strides) == len(feat_channels)
        for _ in range(num_levels - len(feat_strides)):
            feat_strides.append(feat_strides[-1] * 2)

        self.hidden_dim = hidden_dim
        self.feat_strides = feat_strides
        self.num_levels = num_levels
        self.num_classes = num_classes
        self.num_queries = num_queries
        self.qH = qH
        self.qW = self.num_queries//self.qH
        assert self.qW == qW
        self.inceptH = inceptH
        self.inceptW = inceptW
        self.memory_level = memory_level

        self.eps = eps
        self.num_decoder_layers = num_decoder_layers
        self.eval_spatial_size = eval_spatial_size
        self.aux_loss = aux_loss

        self.label_noise_ratio = label_noise_ratio
        self.box_noise_scale = box_noise_scale

        # backbone feature projection
        self._build_input_proj_layer(feat_channels)
        self.output_proj =  nn.Sequential(OrderedDict([
                    ('conv', nn.Conv2d(3*self.hidden_dim, self.hidden_dim, 1, bias=False)), 
                    ('norm', nn.BatchNorm2d(self.hidden_dim,))])
                )

        # DECOplus decoder module
        decoder_layer = DecoPlusDecoderLayer(hidden_dim, dropout, qH=self.qH, qW=self.qW, inceptH=self.inceptH, inceptW=self.inceptW)
        self.decoder = DecoPlusDecoder(hidden_dim, decoder_layer, num_decoder_layers, eval_idx)

        # decoder embedding
        self.query_pos_head = MLP(4, 2 * hidden_dim, hidden_dim, num_layers=2)

        # encoder head
        self.enc_output = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim,)
        )
        self.enc_score_head = nn.Linear(hidden_dim, num_classes)
        self.enc_bbox_head = MLP(hidden_dim, hidden_dim, 4, num_layers=3)

        # decoder head
        self.dec_score_head = nn.ModuleList([
            nn.Linear(hidden_dim, num_classes)
            for _ in range(num_decoder_layers)
        ])
        self.dec_bbox_head = nn.ModuleList([
            MLP(hidden_dim, hidden_dim, 4, num_layers=3)
            for _ in range(num_decoder_layers)
        ])

        # downsample convs
        if self.memory_level>0:
            self.downsample_convs = nn.ModuleList()
            for i in range(self.memory_level):
                self.downsample_convs.append(ConvNormLayer(hidden_dim, hidden_dim, 2*abs(self.memory_level-i), 2*abs(self.memory_level-i), act=downsample_act,))

        # init encoder output anchors and valid_mask
        if self.eval_spatial_size:
            self.anchors, self.valid_mask = self._generate_anchors()

        self._reset_parameters()

    def _reset_parameters(self):
        bias = bias_init_with_prob(0.01)

        init.constant_(self.enc_score_head.bias, bias)
        init.constant_(self.enc_bbox_head.layers[-1].weight, 0)
        init.constant_(self.enc_bbox_head.layers[-1].bias, 0)

        for cls_, reg_ in zip(self.dec_score_head, self.dec_bbox_head):
            init.constant_(cls_.bias, bias)
            init.constant_(reg_.layers[-1].weight, 0)
            init.constant_(reg_.layers[-1].bias, 0)
        
        init.xavier_uniform_(self.enc_output[0].weight)
        init.xavier_uniform_(self.query_pos_head.layers[0].weight)
        init.xavier_uniform_(self.query_pos_head.layers[1].weight)


    def _build_input_proj_layer(self, feat_channels):
        self.input_proj = nn.ModuleList()
        for in_channels in feat_channels:
            self.input_proj.append(
                nn.Sequential(OrderedDict([
                    ('conv', nn.Conv2d(in_channels, self.hidden_dim, 1, bias=False)), 
                    ('norm', nn.BatchNorm2d(self.hidden_dim,))])
                )
            )

        in_channels = feat_channels[-1]
        for _ in range(self.num_levels - len(feat_channels)):
            self.input_proj.append(
                nn.Sequential(OrderedDict([
                    ('conv', nn.Conv2d(in_channels, self.hidden_dim, 3, 2, padding=1, bias=False)),
                    ('norm', nn.BatchNorm2d(self.hidden_dim))])
                )
            )
            in_channels = self.hidden_dim


    def _get_encoder_input_noFlatten(self, feats):
        # get projection features
        proj_feats = [self.input_proj[i](feat) for i, feat in enumerate(feats)]
        if self.num_levels > len(proj_feats):
            len_srcs = len(proj_feats)
            for i in range(len_srcs, self.num_levels):
                if i == len_srcs:
                    proj_feats.append(self.input_proj[i](feats[-1]))
                else:
                    proj_feats.append(self.input_proj[i](proj_feats[-1]))

        # get encoder inputs
        feat_flatten = []
        spatial_shapes = []
        for i, feat in enumerate(proj_feats):
            _, _, h, w = feat.shape
            feat_flatten.append(feat.flatten(2).permute(0, 2, 1))
            spatial_shapes.append([h, w])

        # merge the three scale features into the self.memory_level scale
        _, _, h, w = proj_feats[self.memory_level].shape
        feat_fullscale = []
        for i, feat in enumerate(proj_feats):
            if i>self.memory_level:
                feat_fullscale.append(F.interpolate(feat, size=[h,w]))
            elif i<self.memory_level:
                feat_fullscale.append(self.downsample_convs[i](feat))
            else:
                feat_fullscale.append(feat)
        feat_fullscale = torch.concat(feat_fullscale, 1)
        feat_fullscale = self.output_proj(feat_fullscale)

        return (feat_fullscale, feat_flatten, spatial_shapes)

    def _generate_anchors(self,
                          spatial_shapes=None,
                          grid_size=0.05,
                          dtype=torch.float32,
                          device='cpu',
                          level=0):
        if spatial_shapes is None:
            spatial_shapes = [[int(self.eval_spatial_size[0] / s), int(self.eval_spatial_size[1] / s)]
                for s in self.feat_strides
            ]
        anchors = []
        for lvl, (h, w) in enumerate(spatial_shapes):
            grid_y, grid_x = torch.meshgrid(\
                torch.arange(end=h, dtype=dtype), \
                torch.arange(end=w, dtype=dtype), indexing='ij')
            grid_xy = torch.stack([grid_x, grid_y], -1)
            valid_WH = torch.tensor([w, h]).to(dtype)
            grid_xy = (grid_xy.unsqueeze(0) + 0.5) / valid_WH
            if len(spatial_shapes)==1: 
                wh = torch.ones_like(grid_xy) * grid_size * (2.0 ** level)
            else:
                wh = torch.ones_like(grid_xy) * grid_size * (2.0 ** lvl)
            anchors.append(torch.concat([grid_xy, wh], -1).reshape(-1, h * w, 4))

        anchors = torch.concat(anchors, 1).to(device)
        valid_mask = ((anchors > self.eps) * (anchors < 1 - self.eps)).all(-1, keepdim=True)
        anchors = torch.log(anchors / (1 - anchors))
        anchors = torch.where(valid_mask, anchors, torch.inf)

        return anchors, valid_mask


    def _get_decoder_input(self,
                           memory,
                           spatial_shapes):
        
        # prepare input for decoder
        if self.training or self.eval_spatial_size is None:
            anchors, valid_mask = self._generate_anchors(spatial_shapes, device=memory.device)
        else:
            anchors, valid_mask = self.anchors.to(memory.device), self.valid_mask.to(memory.device)

        memory = valid_mask.to(memory.dtype) * memory  # TODO fix type error for onnx export 

        output_memory = self.enc_output(memory)

        enc_outputs_class = self.enc_score_head(output_memory)
        enc_outputs_coord_unact = self.enc_bbox_head(output_memory) + anchors

        _, topk_ind = torch.topk(enc_outputs_class.max(-1).values, self.num_queries, dim=1)

        reference_points_unact = enc_outputs_coord_unact.gather(dim=1, \
            index=topk_ind.unsqueeze(-1).repeat(1, 1, enc_outputs_coord_unact.shape[-1]))

        enc_topk_bboxes = F.sigmoid(reference_points_unact)
        
        enc_topk_logits = enc_outputs_class.gather(dim=1, \
            index=topk_ind.unsqueeze(-1).repeat(1, 1, enc_outputs_class.shape[-1]))

        # extract region features
        target = output_memory.gather(dim=1, \
            index=topk_ind.unsqueeze(-1).repeat(1, 1, output_memory.shape[-1]))
        target = target.detach() 

        return target, reference_points_unact.detach(), enc_topk_bboxes, enc_topk_logits 


    def forward(self, feats):

        # input projection and embedding
        (memory_noFlat, memory_list, spatial_shapes) = self._get_encoder_input_noFlatten(feats)
        memory = torch.concat(memory_list, 1)

        target, init_ref_points_unact, enc_topk_bboxes, enc_topk_logits = self._get_decoder_input(memory, spatial_shapes)

        # decoder
        out_bboxes, out_logits = self.decoder(
            target,
            init_ref_points_unact,
            memory_noFlat,
            self.dec_bbox_head,
            self.dec_score_head,
            self.query_pos_head)

        out = {'pred_logits': out_logits[-1], 'pred_boxes': out_bboxes[-1]}

        if self.training and self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(out_logits[:-1], out_bboxes[:-1])
            out['aux_outputs'].extend(self._set_aux_loss([enc_topk_logits], [enc_topk_bboxes]))

        return out


    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_boxes': b}
                for a, b in zip(outputs_class, outputs_coord)]
