from __future__ import division

import torch
import torch.nn as nn
import numpy as np
from mmcv.cnn import normal_init

from mmdet.core import distance2bbox, force_fp32, multi_apply, multiclass_nms,AnchorGenerator,anchor_target,delta2bbox
from ..builder import build_loss
from ..registry import HEADS
from ..utils import ConvModule, Scale, bias_init_with_prob


INF = 1e8

@HEADS.register_module

class TWO_BRANCESHead(nn.Module):
    
    def __init__(self,
                 num_classes,
                 in_channels,
                 feat_channels = 256,
                 stacked_convs = 4,
                 strides = (4,8,16,32,64),
                 regress_ranges=((-1, 64), (64, 128), (128, 256), (256, 512),
                                 (512, INF)),
                 loss_cls = dict(
                     type = 'FocalLoss'
                     use_sigmoid = True,
                     gamma = 2.0,
                     alpha = 0.25,
                     loss_weight = 1.0),
                 loss_bbox = dict(type='IoULoss',loss_weight=1.0),
                 loss_centerness = dict(
                     type = 'CrossEntropyLoss',
                     use_sigmoid = True,
                     loss_weight = 1.0),
                 conv_cfg = None,
                 norm_cfg = dict('type
                     

