import torch
import torch.nn as nn
from mmcv.cnn import normal_init
from mmdet.core import distance2bbox, force_fp32, multi_apply, multiclass_nms
from ..builder import build_loss
from ..registry import HEADS
from ..utils import ConvModule, Scale, bias_init_with_prob
import pdb
INF = 1e8

@HEADS.register_module
class PAR_SAF_HEAD(nn.Module):
    def __init__(self,
                 num_classes,
                 in_channels,
                 feat_channels=256,
                 stacked_convs=4,
                 strides=(4, 8, 16, 32, 64),
                 regress_ranges=((-1, 64), (64, 128), (128, 256), (256, 512),
                                 (512, INF)),
                 loss_cls_large=dict(
                     type='FocalLoss',
                     use_sigmoid=True,
                     gamma=2.0,
                     alpha=0.25,
                     loss_weight=1.0),
                 loss_cls_medium=dict(
                     type='FocalLoss',
                     use_sigmoid=True,
                     gamma=2.0,
                     alpha=0.25,
                     loss_weight=1.0),
                 loss_cls_small=dict(
                     type='FocalLoss',
                     use_sigmoid=True,
                     gamma=2.0,
                     alpha=0.25,
                     loss_weight=1.0),
                 loss_bbox_large=dict(type='IoULoss', loss_weight=1.0),
                 loss_bbox_medium=dict(type='IoULoss', loss_weight=1.0),
                 loss_bbox_small=dict(type='IoULoss', loss_weight=1.0),
                 num_parallel = 3,
                 sample_threshold = [0.30,0.35,0.40],
                 square_sample = False,
                 conv_cfg=None,
                 norm_cfg=dict(type='GN', num_groups=32, requires_grad=True)):
        super(PAR_SAF_HEAD, self).__init__()
        self.num_classes = num_classes
        self.cls_out_channels = num_classes - 1
        self.in_channels = in_channels
        self.feat_channels = feat_channels
        self.stacked_convs = stacked_convs
        self.strides = strides
        self.regress_ranges = regress_ranges
        self.loss_cls_large = build_loss(loss_cls_large)
        self.loss_cls_medium = build_loss(loss_cls_medium)
        self.loss_cls_small = build_loss(loss_cls_small)
        self.loss_bbox_large = build_loss(loss_bbox_large)
        self.loss_bbox_medium = build_loss(loss_bbox_medium)
        self.loss_bbox_small = build_loss(loss_bbox_small)        
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.sample_threshold = sample_threshold
        self.square_sample = square_sample
        self.fp16_enabled = False
        self.num_parallel = num_parallel
        self.share_weights = True
        self._init_layers()

    def _init_layers(self):
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            #four consecutive conv layers
            chn = self.in_channels if i == 0 else self.feat_channels
            self.cls_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    bias=self.norm_cfg is None))
            self.reg_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    bias=self.norm_cfg is None))
        self.fcos_cls_large = nn.Conv2d(
            self.feat_channels, self.cls_out_channels, 3, padding=1)
        self.fcos_reg_large = nn.Conv2d(self.feat_channels, 4, 3, padding=1)
        self.fcos_cls_medium = nn.Conv2d(
            self.feat_channels, self.cls_out_channels, 3, padding=1)
        self.fcos_reg_medium = nn.Conv2d(self.feat_channels, 4, 3, padding=1)
        self.fcos_cls_small = nn.Conv2d(
            self.feat_channels, self.cls_out_channels, 3, padding=1)
        self.fcos_reg_small = nn.Conv2d(self.feat_channels, 4, 3, padding=1)
        
        self.scales = nn.ModuleList([Scale(1.0) for _ in self.strides])

    def init_weights(self):
        for m in self.cls_convs:
            normal_init(m.conv, std=0.01)
        for m in self.reg_convs:
            normal_init(m.conv, std=0.01)
        bias_cls = bias_init_with_prob(0.01)
        normal_init(self.fcos_cls_large, std=0.01, bias=bias_cls)
        normal_init(self.fcos_reg_large, std=0.01)
        normal_init(self.fcos_cls_medium, std=0.01, bias=bias_cls)
        normal_init(self.fcos_reg_medium, std=0.01)
        normal_init(self.fcos_cls_small, std=0.01, bias=bias_cls)
        normal_init(self.fcos_reg_small, std=0.01)
#

    def forward(self, feats):
        return multi_apply(self.forward_single, feats, self.scales)

    def forward_single(self, x, scale):
        cls_feat = x
        reg_feat = x
        
        cls_score = []
        bbox_pred = []
        
        for cls_layer in self.cls_convs:
            cls_feat = cls_layer(cls_feat)
        cls_score_large = self.fcos_cls_large(cls_feat)
        #cls_score.append(cls_score_large)
        cls_score_medium = self.fcos_cls_medium(cls_feat)
        #cls_score.append(cls_score_medium)
        cls_score_small = self.fcos_cls_small(cls_feat)
       # cls_score.append(cls_score_small)

        for reg_layer in self.reg_convs:
            reg_feat = reg_layer(reg_feat)
        
        # float to avoid overflow when enabling FP16
        bbox_pred_large = scale(self.fcos_reg_large(reg_feat)).float().exp()
        #bbox_pred.append(bbox_pred_large)
        bbox_pred_medium = scale(self.fcos_reg_medium(reg_feat)).float().exp()
        #bbox_pred.append(bbox_pred_medium)
        bbox_pred_small = scale(self.fcos_reg_small(reg_feat)).float().exp()
       # bbox_pred.append(bbox_pred_small)

        return cls_score_large,cls_score_medium, cls_score_small, bbox_pred_large, bbox_pred_medium,bbox_pred_small#,cls_score_small, bbox_pred_large, bbox_pred_medium,bbox_pred_small

    @force_fp32(apply_to=('cls_scores', 'bbox_preds'))
    def loss(self,
             cls_score_large,
             cls_score_medium,
             cls_score_small,
             bbox_pred_large,
             bbox_pred_medium,
             bbox_pred_small,
             gt_bboxes,
             gt_labels,
             img_metas,
             cfg,
             gt_bboxes_ignore=None):
        assert len(cls_score_large) == len(bbox_pred_large) 
        assert len(cls_score_medium) == len(bbox_pred_medium) 
        assert len(cls_score_small) == len(bbox_pred_small) 

        #　【-2:】 could get [height,width]
        #cls_scores & bbox_preds are outputs
        #pdb.set_trace()
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_score_large]
        all_level_points = self.get_points(featmap_sizes, bbox_pred_large[0].dtype,
                                           bbox_pred_large[0].device)
        #with points, [points+bbox] pred could apply for bbox_pred loss
        labels, bbox_targets = self.fcos_target(all_level_points, gt_bboxes,
                                                gt_labels)
        
        num_imgs = cls_score_large[0].size(0)
        # flatten cls_scores, bbox_preds and centerness
        
        flatten_cls_scores_large = [
            cls_score_large_single.permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels)
            for cls_score_large_single in cls_score_large
        ]
        
        flatten_cls_scores_medium = [
            cls_score_medium_single.permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels)
            for cls_score_medium_single in cls_score_medium
        ]
        
        flatten_cls_scores_small = [
            cls_score_small_single.permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels)
            for cls_score_small_single in cls_score_small
        ]
        
        flatten_bbox_preds_large = [
            bbox_pred_large_single.permute(0, 2, 3, 1).reshape(-1, 4)
            for bbox_pred_large_single in bbox_pred_large
        ]
        flatten_bbox_preds_medium = [
            bbox_pred_medium_single.permute(0, 2, 3, 1).reshape(-1, 4)
            for bbox_pred_medium_single in bbox_pred_medium
        ]
        flatten_bbox_preds_small = [
            bbox_pred_small_single.permute(0, 2, 3, 1).reshape(-1, 4)
            for bbox_pred_small_single in bbox_pred_small
        ]
      
        flatten_cls_scores_large = torch.cat(flatten_cls_scores_large)
        flatten_cls_scores_medium = torch.cat(flatten_cls_scores_medium)
        flatten_cls_scores_small = torch.cat(flatten_cls_scores_small)

        flatten_bbox_preds_large = torch.cat(flatten_bbox_preds_large)
        flatten_bbox_preds_medium = torch.cat(flatten_bbox_preds_medium)
        flatten_bbox_preds_small = torch.cat(flatten_bbox_preds_small)

        flatten_labels_large = torch.cat(labels)
        flatten_labels_medium = torch.cat(labels)
        flatten_labels_small = torch.cat(labels)
        flatten_bbox_targets_large = torch.cat(bbox_targets)
        flatten_bbox_targets_medium = torch.cat(bbox_targets)
        flatten_bbox_targets_small = torch.cat(bbox_targets)
        
        # repeat points to align with bbox_preds
        flatten_points_large = torch.cat(
            [points.repeat(num_imgs, 1) for points in all_level_points])
        flatten_points_medium = torch.cat(
            [points.repeat(num_imgs, 1) for points in all_level_points])
        flatten_points_small = torch.cat(
            [points.repeat(num_imgs, 1) for points in all_level_points])
        
        #!!!!positive samples assign with sampler!!!!
        pos_inds_tem_large = flatten_labels_large.nonzero().reshape(-1)
        pos_bbox_targets_tem_large = flatten_bbox_targets_large[pos_inds_tem_large]
        left_right_large = pos_bbox_targets_tem_large[:, [0, 2]]
        top_bottom_large = pos_bbox_targets_tem_large[:, [1, 3]]
        centerness_targets_large = (
            left_right_large.min(dim=-1)[0] / left_right_large.max(dim=-1)[0]) * (
                top_bottom_large.min(dim=-1)[0] / top_bottom_large.max(dim=-1)[0])
       # if self.square_sample:
        centerness_targets_large = torch.sqrt(centerness_targets_large)
                
        ##large part 0.30
        pos_inds_large = pos_inds_tem_large[centerness_targets_large>self.sample_threshold[0]]
        pos_inds_ig_large =pos_inds_tem_large[centerness_targets_large<=self.sample_threshold[0]]
        flatten_labels_large[pos_inds_ig_large] = 0
        
        num_pos_large = len(pos_inds_large)
        loss_cls_large = self.loss_cls_large(
            flatten_cls_scores_large, flatten_labels_large,
            avg_factor=num_pos_large + num_imgs)  # avoid num_pos is 0

        pos_bbox_preds_large = flatten_bbox_preds_large[pos_inds_large]
        
        if num_pos_large > 0:
            pos_bbox_targets_large = flatten_bbox_targets_large[pos_inds_large]
            pos_points_large = flatten_points_large[pos_inds_large]
            pos_decoded_bbox_preds_large = distance2bbox(pos_points_large, pos_bbox_preds_large)
            pos_decoded_target_preds_large = distance2bbox(pos_points_large,
                                                     pos_bbox_targets_large)

            loss_bbox_large = self.loss_bbox_large(
                pos_decoded_bbox_preds_large,
                pos_decoded_target_preds_large)
        else:
            loss_bbox_large = pos_bbox_preds_large.sum()
       
         #medium part 0.35 
       #!!!!positive samples assign with sampler!!!!
        pos_inds_tem_medium = flatten_labels_medium.nonzero().reshape(-1)
        pos_bbox_targets_tem_medium = flatten_bbox_targets_medium[pos_inds_tem_medium]
        left_right_medium = pos_bbox_targets_tem_medium[:, [0, 2]]
        top_bottom_medium = pos_bbox_targets_tem_medium[:, [1, 3]]
        centerness_targets_medium = (
            left_right_medium.min(dim=-1)[0] / left_right_medium.max(dim=-1)[0]) * (
                top_bottom_medium.min(dim=-1)[0] / top_bottom_medium.max(dim=-1)[0])
       # if self.square_sample:
        centerness_targets_medium = torch.sqrt(centerness_targets_medium)
                
        pos_inds_medium = pos_inds_tem_medium[centerness_targets_medium>self.sample_threshold[1]]
        pos_inds_ig_medium =pos_inds_tem_medium[centerness_targets_medium<=self.sample_threshold[1]]
        flatten_labels_medium[pos_inds_ig_medium] = 0
         
        num_pos_medium = len(pos_inds_medium)
        loss_cls_medium = self.loss_cls_medium(
            flatten_cls_scores_medium, flatten_labels_medium,
            avg_factor=num_pos_medium + num_imgs)  # avoid num_pos is 0

        pos_bbox_preds_medium = flatten_bbox_preds_medium[pos_inds_medium]
        
        if num_pos_medium > 0:
            pos_bbox_targets_medium = flatten_bbox_targets_medium[pos_inds_medium]
            pos_points_medium = flatten_points_medium[pos_inds_medium]
            pos_decoded_bbox_preds_medium = distance2bbox(pos_points_medium, pos_bbox_preds_medium)
            pos_decoded_target_preds_medium = distance2bbox(pos_points_medium,
                                                     pos_bbox_targets_medium)

            loss_bbox_medium= self.loss_bbox_medium(
                pos_decoded_bbox_preds_medium,
                pos_decoded_target_preds_medium)
        else:
            loss_bbox_medium = pos_bbox_preds_medium.sum()
        
         #medium part 0.40 
       #!!!!positive samples assign with sampler!!!!
        pos_inds_tem_small = flatten_labels_small.nonzero().reshape(-1)
        pos_bbox_targets_tem_small = flatten_bbox_targets_small[pos_inds_tem_small]
        left_right_small = pos_bbox_targets_tem_small[:, [0, 2]]
        top_bottom_small = pos_bbox_targets_tem_small[:, [1, 3]]
        centerness_targets_small = (
            left_right_small.min(dim=-1)[0] / left_right_small.max(dim=-1)[0]) * (
                top_bottom_small.min(dim=-1)[0] / top_bottom_small.max(dim=-1)[0])
       # if self.square_sample:
        centerness_targets_small = torch.sqrt(centerness_targets_small)
                
        pos_inds_small = pos_inds_tem_small[centerness_targets_small>self.sample_threshold[2]]
        pos_inds_ig_small =pos_inds_tem_small[centerness_targets_small<=self.sample_threshold[2]]
        flatten_labels_small[pos_inds_ig_small] = 0
        
        num_pos_small = len(pos_inds_small)
        loss_cls_small = self.loss_cls_small(
            flatten_cls_scores_small, flatten_labels_small,
            avg_factor=num_pos_small + num_imgs)  # avoid num_pos is 0

        pos_bbox_preds_small = flatten_bbox_preds_small[pos_inds_small]
        
        if num_pos_small > 0:
            pos_bbox_targets_small = flatten_bbox_targets_small[pos_inds_small]
            pos_points_small = flatten_points_small[pos_inds_small]
            pos_decoded_bbox_preds_small = distance2bbox(pos_points_small, pos_bbox_preds_small)
            pos_decoded_target_preds_small = distance2bbox(pos_points_small,
                                                     pos_bbox_targets_small)

            loss_bbox_small= self.loss_bbox_small(
                pos_decoded_bbox_preds_small,
                pos_decoded_target_preds_small)
        else:
            loss_bbox_small = pos_bbox_preds_small.sum()
        
        return dict(
            loss_cls_large = loss_cls_large,
            loss_bbox_large = loss_bbox_large,
            loss_cls_medium = loss_cls_medium,
            loss_bbox_medium = loss_bbox_medium,
            loss_cls_small = loss_cls_small,
            loss_bbox_small = loss_bbox_small,
        )

    @force_fp32(apply_to=('cls_scores', 'bbox_preds'))
    def get_bboxes(self,
                   cls_scores_large,
                   cls_scores_medium,
                   cls_scores_small,
                   bbox_preds_large,
                   bbox_preds_medium,
                   bbox_preds_small,
                   img_metas,
                   cfg,
                   rescale=None):
        assert len(cls_scores_large) == len(bbox_preds_large)
        assert len(cls_scores_medium) == len(bbox_preds_medium)
        assert len(cls_scores_small) == len(bbox_preds_small)

        num_levels = len(cls_scores_large)

        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores_large]
        mlvl_points = self.get_points(featmap_sizes, bbox_preds_large[0].dtype,
                                      bbox_preds_large[0].device)
        result_list = []
        for img_id in range(len(img_metas)):
            cls_score_list_large = [
                cls_scores_large[i][img_id].detach() for i in range(num_levels)
            ]
            bbox_pred_list_large = [
                bbox_preds_large[i][img_id].detach() for i in range(num_levels)
            ]
            cls_score_list_medium = [
                cls_scores_medium[i][img_id].detach() for i in range(num_levels)
            ]
            bbox_pred_list_medium = [
                bbox_preds_medium[i][img_id].detach() for i in range(num_levels)
            ]
            cls_score_list_small = [
                cls_scores_small[i][img_id].detach() for i in range(num_levels)
            ]
            bbox_pred_list_small = [
                bbox_preds_small[i][img_id].detach() for i in range(num_levels)
            ]
            img_shape = img_metas[img_id]['img_shape']
            scale_factor = img_metas[img_id]['scale_factor']
            det_bboxes = self.get_bboxes_single(cls_score_list_large, cls_score_list_medium,
                                                cls_score_list_small, bbox_pred_list_large,
                                                bbox_pred_list_medium, bbox_pred_list_small,
                                                mlvl_points, img_shape,
                                                scale_factor, cfg, rescale)
            result_list.append(det_bboxes)
        return result_list

    def get_bboxes_single(self,
                          cls_scores_large,
                          cls_scores_medium,
                          cls_scores_small,
                          bbox_preds_large,
                          bbox_preds_medium,
                          bbox_preds_small,
                          mlvl_points,
                          img_shape,
                          scale_factor,
                          cfg,
                          rescale=False):
        assert len(cls_scores_large) == len(bbox_preds_large) == len(mlvl_points)
        assert len(cls_scores_medium) == len(bbox_preds_medium) == len(mlvl_points)
        assert len(cls_scores_small) == len(bbox_preds_small) == len(mlvl_points)

        mlvl_bboxes = []
        mlvl_scores = []
        
        for cls_score_large, bbox_pred_large, points in zip(
            cls_scores_large, bbox_preds_large, mlvl_points):
            assert cls_score_large.size()[-2:] == bbox_pred_large.size()[-2:]
            scores = cls_score_large.permute(1, 2, 0).reshape(
                -1, self.cls_out_channels).sigmoid()

            bbox_pred = bbox_pred_large.permute(1, 2, 0).reshape(-1, 4)
            nms_pre = cfg.get('nms_pre', -1)
            if nms_pre > 0 and scores.shape[0] > nms_pre:
                max_scores, _ = (scores).max(dim=1)
                _, topk_inds = max_scores.topk(nms_pre)
                points = points[topk_inds, :]
                bbox_pred = bbox_pred[topk_inds, :]
                scores = scores[topk_inds, :]
            bboxes = distance2bbox(points, bbox_pred, max_shape=img_shape)
            #pdb.set_trace()
            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)
        
        for cls_score_medium, bbox_pred_medium, points in zip(
            cls_scores_medium, bbox_preds_medium, mlvl_points):
            assert cls_score_medium.size()[-2:] == bbox_pred_medium.size()[-2:]
            scores = cls_score_medium.permute(1, 2, 0).reshape(
                -1, self.cls_out_channels).sigmoid()
          #  centerness = centerness.permute(1, 2, 0).reshape(-1).sigmoid()

            bbox_pred = bbox_pred_medium.permute(1, 2, 0).reshape(-1, 4)
            nms_pre = cfg.get('nms_pre', -1)
            if nms_pre > 0 and scores.shape[0] > nms_pre:
                max_scores, _ = (scores).max(dim=1)
                _, topk_inds = max_scores.topk(nms_pre)
                points = points[topk_inds, :]
                bbox_pred = bbox_pred[topk_inds, :]
                scores = scores[topk_inds, :]
            bboxes = distance2bbox(points, bbox_pred, max_shape=img_shape)
            #pdb.set_trace()
            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)
        
        for cls_score_small, bbox_pred_small, points in zip(
            cls_scores_small, bbox_preds_small, mlvl_points):
            assert cls_score_small.size()[-2:] == bbox_pred_small.size()[-2:]
            scores = cls_score_small.permute(1, 2, 0).reshape(
                -1, self.cls_out_channels).sigmoid()

            bbox_pred = bbox_pred_small.permute(1, 2, 0).reshape(-1, 4)
            nms_pre = cfg.get('nms_pre', -1)
            if nms_pre > 0 and scores.shape[0] > nms_pre:
                max_scores, _ = (scores).max(dim=1)
                _, topk_inds = max_scores.topk(nms_pre)
                points = points[topk_inds, :]
                bbox_pred = bbox_pred[topk_inds, :]
                scores = scores[topk_inds, :]
            bboxes = distance2bbox(points, bbox_pred, max_shape=img_shape)
            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)
        #pdb.set_trace()
        
        #pdb.set_trace()
        mlvl_bboxes = torch.cat(mlvl_bboxes)
        if rescale:
            mlvl_bboxes /= mlvl_bboxes.new_tensor(scale_factor)
        mlvl_scores = torch.cat(mlvl_scores)
        padding = mlvl_scores.new_zeros(mlvl_scores.shape[0], 1)
        mlvl_scores = torch.cat([padding, mlvl_scores], dim=1)
        det_bboxes, det_labels = multiclass_nms(
            mlvl_bboxes,
            mlvl_scores,
            cfg.score_thr,
            cfg.nms,
            cfg.max_per_img)
        return det_bboxes, det_labels

    def get_points(self, featmap_sizes, dtype, device):
        """Get points according to feature map sizes.

        Args:
            featmap_sizes (list[tuple]): Multi-level feature map sizes.
            dtype (torch.dtype): Type of points.
            device (torch.device): Device of points.

        Returns:
            tuple: points of each image.
        """
        mlvl_points = []
        for i in range(len(featmap_sizes)):
            mlvl_points.append(
                self.get_points_single(featmap_sizes[i], self.strides[i],
                                       dtype, device))
        return mlvl_points

    def get_points_single(self, featmap_size, stride, dtype, device):
        h, w = featmap_size
        x_range = torch.arange(
            0, w * stride, stride, dtype=dtype, device=device)
        y_range = torch.arange(
            0, h * stride, stride, dtype=dtype, device=device)
        y, x = torch.meshgrid(y_range, x_range)
        points = torch.stack(
            (x.reshape(-1), y.reshape(-1)), dim=-1) + stride // 2
        return points

    def fcos_target(self, points, gt_bboxes_list, gt_labels_list):
        assert len(points) == len(self.regress_ranges)
        #num_levels mean the levels of all feature maps
        num_levels = len(points)
        # expand regress ranges to align with points
        expanded_regress_ranges = [
            points[i].new_tensor(self.regress_ranges[i])[None].expand_as(
                points[i]) for i in range(num_levels)
        ]
        
        # concat all levels points and regress ranges
        concat_regress_ranges = torch.cat(expanded_regress_ranges, dim=0)
        concat_points = torch.cat(points, dim=0)
        
        # get labels and bbox_targets of each image
        
        # use multi_apply() could put every set of parameters in *args to the first function (with the same **kwargs)
        labels_list, bbox_targets_list = multi_apply(
            self.fcos_target_single,
            gt_bboxes_list,
            gt_labels_list,
            points=concat_points,
            regress_ranges=concat_regress_ranges)

        # split to per img, per level
        num_points = [center.size(0) for center in points]
        labels_list = [labels.split(num_points, 0) for labels in labels_list]
        bbox_targets_list = [
            bbox_targets.split(num_points, 0)
            for bbox_targets in bbox_targets_list
        ]

        # concat per level image
        concat_lvl_labels = []
        concat_lvl_bbox_targets = []
        for i in range(num_levels):
            concat_lvl_labels.append(
                torch.cat([labels[i] for labels in labels_list]))
            concat_lvl_bbox_targets.append(
                torch.cat(
                    [bbox_targets[i] for bbox_targets in bbox_targets_list]))
        return concat_lvl_labels, concat_lvl_bbox_targets

    def fcos_target_single(self, gt_bboxes, gt_labels, points, regress_ranges):
        #num of all points
        num_points = points.size(0)
        #num of all gts
        num_gts = gt_labels.size(0)
        if num_gts == 0:
            return gt_labels.new_zeros(num_points), \
                   gt_bboxes.new_zeros((num_points, 4))
        
        # areas of all gt bboxes
        areas = (gt_bboxes[:, 2] - gt_bboxes[:, 0] + 1) * (
            gt_bboxes[:, 3] - gt_bboxes[:, 1] + 1)
        
       
        # TODO: figure out why these two are different
        # areas = areas[None].expand(num_points, num_gts)
        areas = areas[None].repeat(num_points, 1)
        
        regress_ranges = regress_ranges[:, None, :].expand(
            num_points, num_gts, 2)
        gt_bboxes = gt_bboxes[None].expand(num_points, num_gts, 4)
        xs, ys = points[:, 0], points[:, 1]
        xs = xs[:, None].expand(num_points, num_gts)
        ys = ys[:, None].expand(num_points, num_gts)

        left = xs - gt_bboxes[..., 0]
        right = gt_bboxes[..., 2] - xs
        top = ys - gt_bboxes[..., 1]
        bottom = gt_bboxes[..., 3] - ys
        
        # from the points(xs,ys) to generate the gt_targets(l,r,t,b)
        bbox_targets = torch.stack((left, top, right, bottom), -1)

        # condition1: inside a gt bbox
        inside_gt_bbox_mask = bbox_targets.min(-1)[0] > 0

        # condition2: limit the regression range for each location
        max_regress_distance = bbox_targets.max(-1)[0]
        inside_regress_range = (
            max_regress_distance >= regress_ranges[..., 0]) & (
                max_regress_distance <= regress_ranges[..., 1])
        
        # if there are still more than one objects for a location,
        # we choose the one with minimal area
        # use INF to delete those points outsite a gt bbox or exceed the regression range
        # in the next areas.min operation
        areas[inside_gt_bbox_mask == 0] = INF
        areas[inside_regress_range == 0] = INF
        min_area, min_area_inds = areas.min(dim=1)
        
        labels = gt_labels[min_area_inds]
        labels[min_area == INF] = 0
        bbox_targets = bbox_targets[range(num_points), min_area_inds]

        return labels, bbox_targets

    def centerness_target(self, pos_bbox_targets):
        # only calculate pos centerness targets, otherwise there may be nan
        left_right = pos_bbox_targets[:, [0, 2]]
        top_bottom = pos_bbox_targets[:, [1, 3]]
        centerness_targets = (
            left_right.min(dim=-1)[0] / left_right.max(dim=-1)[0]) * (
                top_bottom.min(dim=-1)[0] / top_bottom.max(dim=-1)[0])
        return torch.sqrt(centerness_targets)
