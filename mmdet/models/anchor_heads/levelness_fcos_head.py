import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import normal_init
from mmdet.core import distance2bbox, force_fp32, multi_apply, multiclass_nms
from ..builder import build_loss
from ..registry import HEADS
from ..utils import ConvModule, Scale, bias_init_with_prob
import pdb

INF = 1e8

@HEADS.register_module
class levelness_FCOSHead(nn.Module):
    
    def __init__(self,
                 num_classes,
                 in_channels,
                 feat_channels=256,
                 stacked_convs=4,
                 strides=(4, 8, 16, 32, 64),
                 regress_ranges=((-1, 64), (64, 128), (128, 256), (256, 512),
                                 (512, INF)),
                 loss_cls=dict(
                     type='FocalLoss',
                     use_sigmoid=True,
                     gamma=2.0,
                     alpha=0.25,
                     loss_weight=1.0),
                 loss_bbox=dict(type='IoULoss', loss_weight=1.0),
                 loss_centerness=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=True,
                     loss_weight=1.0),
                 loss_levelness = dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=True,
                     loss_weight=0.1),
                 centerness_reg = False,
                 level_test = False,
                 ciou = False,
                 ciou_threshold = 0.4,
                 conv_cfg=None,
                 use_cc = False,
                 norm_cfg=dict(type='GN', num_groups=32, requires_grad=True)):
        super(levelness_FCOSHead, self).__init__()

        self.num_classes = num_classes
        self.cls_out_channels = num_classes - 1
        self.in_channels = in_channels
        self.feat_channels = feat_channels
        self.stacked_convs = stacked_convs
        self.strides = strides
        self.regress_ranges = regress_ranges
        self.loss_cls = build_loss(loss_cls)
        self.loss_bbox = build_loss(loss_bbox)
        self.loss_centerness = build_loss(loss_centerness)
        self.loss_levelness = build_loss(loss_levelness)
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.centerness_reg = centerness_reg
        self.ciou = ciou
        self.ciou_threshold = ciou_threshold
        self.fp16_enabled = False
        self.level_test = level_test
        self.use_cc =use_cc

        self._init_layers()

    def _init_layers(self): 
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
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
        self.fcos_cls = nn.Conv2d(
            self.feat_channels, self.cls_out_channels, 3, padding=1)

        self.fcos_reg = nn.Conv2d(self.feat_channels, 4, 3, padding=1)
        self.fcos_centerness = nn.Conv2d(self.feat_channels, 1, 3, padding=1)
        if self.use_cc:
            self.fcos_levelness = nn.Conv2d(self.feat_channels*5, 5, 3, padding=1)
        else:
            self.fcos_levelness = nn.Conv2d(self.feat_channels, 5, 3, padding=1)
        self.scales = nn.ModuleList([Scale(1.0) for _ in self.strides])

    def init_weights(self):
        for m in self.cls_convs:
            normal_init(m.conv, std=0.01)
        for m in self.reg_convs:
            normal_init(m.conv, std=0.01)
        bias_cls = bias_init_with_prob(0.01)
        #bias_level = bias_init_with_prob(0.01)
        normal_init(self.fcos_cls, std=0.01, bias=bias_cls)
        normal_init(self.fcos_reg, std=0.01)
        normal_init(self.fcos_centerness, std=0.01)
        normal_init(self.fcos_levelness, std=0.01)

    def forward(self, feats):
        cls_scores = []
        bbox_preds = []
        centernesses = []
        levelnesses = []
        height_4 = feats[0].shape[2]*2
        width_4 = feats[0].shape[3]*2
        up_layer_set = []
        for i in range(len(feats)):
            cls_feat = feats[i]
            reg_feat = feats[i]
            for cls_layer in self.cls_convs:
                cls_feat = cls_layer(cls_feat)
            cls_score = self.fcos_cls(cls_feat)

            for reg_layer in self.reg_convs:
                reg_feat = reg_layer(reg_feat)
            
            #up_layer.append(F.interpolate(reg_feat,size=[height_4,width_4],mode='nearest'))

            if self.use_cc:
                up_layer = F.interpolate(reg_feat,size=[height_4,width_4],mode='nearest')
                up_layer_set.append(up_layer)
            else:
                if i == 0:
                    up_layer = F.interpolate(reg_feat,size=[height_4,width_4],mode='nearest')
                else:
                    up_layer += F.interpolate(reg_feat,size=[height_4,width_4],mode='nearest')


           # pdb.set_trace()
            if self.centerness_reg:
                centerness = self.fcos_centerness(reg_feat)
            else:
                centerness = self.fcos_centerness(cls_feat)

            bbox_pred = self.scales[i](self.fcos_reg(reg_feat)).float().exp()
            cls_scores.append(cls_score)
            bbox_preds.append(bbox_pred)
            centernesses.append(centerness)
        if self.use_cc:
            up_layer = torch.cat(up_layer_set,dim=1)
        
        levelness = self.fcos_levelness(up_layer)
        levelnesses.append(levelness)
        #concat = torch.cat(up_layer,1)
        return tuple([cls_scores,bbox_preds,centernesses,levelnesses])
        #multi_apply(self.forward_single, feats, self.scales)

    def forward_single(self, x, scale):
        cls_feat = x
        reg_feat = x
        for cls_layer in self.cls_convs:
            cls_feat = cls_layer(cls_feat)
        cls_score = self.fcos_cls(cls_feat)

        for reg_layer in self.reg_convs:
            reg_feat = reg_layer(reg_feat)

        if self.centerness_reg:
            centerness = self.fcos_centerness(reg_feat)
        else:
            centerness = self.fcos_centerness(cls_feat)
        #pdb
        # scale the bbox_pred of different level
        # float to avoid overflow when enabling FP16
        bbox_pred = scale(self.fcos_reg(reg_feat)).float().exp()
        return cls_score, bbox_pred, centerness

    @force_fp32(apply_to=('cls_scores', 'bbox_preds', 'centernesses'))
    def loss(self,
             cls_scores,
             bbox_preds,
             centernesses,
             levelnesses,
             gt_bboxes,
             gt_labels,
             img_metas,
             cfg,
             gt_bboxes_ignore=None):
        assert len(cls_scores) == len(bbox_preds) == len(centernesses)
        levelset = []
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        all_level_points = self.get_points(featmap_sizes, bbox_preds[0].dtype,
                                           bbox_preds[0].device)
        #all_level_points returns coordinate(x,y) of points on feature map pyramids in the original image
        labels, bbox_targets = self.fcos_target(all_level_points, gt_bboxes,
                                               gt_labels)
        num_imgs = cls_scores[0].size(0)
        level_lables = labels.copy()
        height = levelnesses[0].shape[2]
        width = levelnesses[0].shape[3]
        batchsize = levelnesses[0].shape[0]
        for i in range(len(level_lables)):
            new = level_lables[i].reshape([batchsize,1,featmap_sizes[i][0],featmap_sizes[i][1]]).float()
            new_level = F.interpolate(new,size=[height,width],mode='nearest')
            #pdb.set_trace()
            level = new_level.clone() 
            level[new_level!=0] = 1
            levelset.append(level)
        level = torch.cat(levelset,dim=1)
        level = level.permute(0,2,3,1).reshape(-1,5)
        flatten_levelness_labels = [level] 
        # flatten cls_scores, bbox_preds and centerness
        flatten_cls_scores = [
            cls_score.permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels)
            for cls_score in cls_scores
        ]
        flatten_bbox_preds = [
            bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4)
            for bbox_pred in bbox_preds
        ]
        flatten_centerness = [
            centerness.permute(0, 2, 3, 1).reshape(-1)
            for centerness in centernesses
        ]
        flatten_levelness = [
            levelness.permute(0, 2, 3, 1).reshape(-1, 5)
            for levelness in levelnesses
        ]
        #pdb.set_trace()
        # flatten_cls_scores[A,80]
        # flatten_bbox_preds[A,4]
        #  flatten_centerness[A,1]
        # A = 5 * points on each level(i)(=Batchsize*H(i)*W(i))
        flatten_cls_scores = torch.cat(flatten_cls_scores)
        flatten_bbox_preds = torch.cat(flatten_bbox_preds)
        flatten_centerness = torch.cat(flatten_centerness)

        # flatten_levelness[B,6]
        # B = Batchsize* H(1/4 ori H) * W (1/4 ori W)
        flatten_levelness = torch.cat(flatten_levelness)
        flatten_levelness_labels = torch.cat(flatten_levelness_labels).long()
        #  flatten_labels[A,1]
        #  A = 5 * points on each level(i)(=Batchsize*H(i)*W(i))
        flatten_labels = torch.cat(labels)
        # flatten_bbox_targets[A,4] 
        flatten_bbox_targets = torch.cat(bbox_targets)

        # repeat points to align with bbox_preds
        flatten_points = torch.cat(
            [points.repeat(num_imgs, 1) for points in all_level_points])
        #pdb.set_trace()
        #original fcos code
        
        if self.ciou:
            pos_inds_tem = flatten_labels.nonzero().reshape(-1)        
            pos_bbox_targets_tem = flatten_bbox_targets[pos_inds_tem]
            #pdb.set_trace()
            left = pos_bbox_targets_tem[:, 0]
            right = pos_bbox_targets_tem[:, 2]
            top = pos_bbox_targets_tem[:, 1]
            bottom = pos_bbox_targets_tem[:, 3]
            inter_left = left.clone()
            inter_right = right.clone()
            inter_top = top.clone()
            inter_bottom = bottom.clone()
            half_w = (left+right)/2
            half_h = (top+bottom)/2
            for i in range(len(left)):
                if half_w[i]<left[i]:
                    inter_left[i] = half_w[i]
                if half_w[i]<right[i]:
                    inter_right[i] = half_w[i]
                if half_h[i]<top[i]:
                    inter_top[i] = half_h[i]
                if half_h[i]<bottom[i]:
                    inter_bottom[i] = half_h[i]
            area_u = (left+right)*(top+bottom)
            area_i = (inter_left+inter_right)*(inter_top+inter_bottom)
            iou_target = area_i/(area_u+area_u-area_i)
        
            pos_inds = pos_inds_tem[iou_target>self.ciou_threshold]
            pos_inds_ignore = pos_inds_tem[iou_target<=self.ciou_threshold]
            flatten_labels[pos_inds_ignore] = 0
        
        else:
            pos_inds = flatten_labels.nonzero().reshape(-1)

        num_pos = len(pos_inds)
        loss_cls = self.loss_cls(
            flatten_cls_scores, flatten_labels,
            avg_factor=num_pos + num_imgs)  # avoid num_pos is 0
        
        #pdb.set_trace()
        num_level = len(flatten_levelness_labels[flatten_levelness_labels!=0])
        loss_levelness = self.loss_levelness(
            flatten_levelness,flatten_levelness_labels,
            avg_factor=num_level + num_imgs)

        pos_bbox_preds = flatten_bbox_preds[pos_inds]
        pos_centerness = flatten_centerness[pos_inds]
        
        if num_pos > 0:
            pos_bbox_targets = flatten_bbox_targets[pos_inds]
            pos_centerness_targets = self.centerness_target(pos_bbox_targets)
            #pos_centerness_targets = pos_centerness.new_ones(pos_levelness.size())  
            pos_points = flatten_points[pos_inds]
            pos_decoded_bbox_preds = distance2bbox(pos_points, pos_bbox_preds)
            pos_decoded_target_preds = distance2bbox(pos_points,
                                                     pos_bbox_targets)
            # centerness weighted iou loss
            loss_bbox = self.loss_bbox(
                pos_decoded_bbox_preds,
                pos_decoded_target_preds,
                weight=pos_centerness_targets,
                avg_factor=pos_centerness_targets.sum())

            #pdb.set_trace()
            loss_centerness = self.loss_centerness(pos_centerness,
                                                   pos_centerness_targets)
        else:
            loss_bbox = pos_bbox_preds.sum()
            loss_centerness = pos_centerness.sum()

        #pdb.set_trace()
        return dict(
            loss_cls=loss_cls,
            loss_bbox=loss_bbox,
            loss_centerness=loss_centerness,
            loss_levelness=loss_levelness)

    @force_fp32(apply_to=('cls_scores', 'bbox_preds', 'centernesses'))
    def get_bboxes(self,
                   cls_scores,
                   bbox_preds,
                   centernesses,
                   levelnesses,
                   img_metas,
                   cfg,
                   rescale=None):
        #pdb.set_trace()
        assert len(cls_scores) == len(bbox_preds)
        num_levels = len(cls_scores)

        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        mlvl_points = self.get_points(featmap_sizes, bbox_preds[0].dtype,
                                      bbox_preds[0].device)
        result_list = []
        ###len(img_metas) is batch size
        for img_id in range(len(img_metas)):
            cls_score_list = [
                cls_scores[i][img_id].detach() for i in range(num_levels)
            ]
            bbox_pred_list = [
                bbox_preds[i][img_id].detach() for i in range(num_levels)
            ]
            centerness_pred_list = [
                centernesses[i][img_id].detach() for i in range(num_levels)
            ]
            levelness_pred_list = [
                levelnesses[0][img_id].detach()
            ]
            img_shape = img_metas[img_id]['img_shape']
            scale_factor = img_metas[img_id]['scale_factor']
            det_bboxes = self.get_bboxes_single(cls_score_list, bbox_pred_list,
                                                centerness_pred_list,
                                                levelness_pred_list,
                                                mlvl_points, img_shape,
                                                scale_factor, cfg, rescale)
            result_list.append(det_bboxes)
        return result_list

    def get_bboxes_single(self,
                          cls_scores,
                          bbox_preds,
                          centernesses,
                          levelnesses,
                          mlvl_points,
                          img_shape,
                          scale_factor,
                          cfg,
                          rescale=False):
        assert len(cls_scores) == len(bbox_preds) == len(mlvl_points)
        mlvl_bboxes = []
        mlvl_scores = []
        mlvl_centerness = []
       #pdb.set_trace()
        levelnesses = levelnesses[0].permute(1,2,0).sigmoid()
        max_inds = levelnesses.argmax(dim=-1).reshape(levelnesses.shape[0],levelnesses.shape[1],1)
        max_new = torch.zeros_like(levelnesses)
        one_hot = max_new.scatter_(-1,max_inds,1)      
        level = 0
        for cls_score, bbox_pred, centerness, points in zip(
                cls_scores, bbox_preds, centernesses, mlvl_points):
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
            h = cls_score.shape[1] 
            w = cls_score.shape[2] 
            scores = cls_score.permute(1, 2, 0).reshape(
                -1, self.cls_out_channels).sigmoid()
            centerness = centerness.permute(1, 2, 0).reshape(-1).sigmoid()
            bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 4)
            bboxes = distance2bbox(points, bbox_pred, max_shape=img_shape)
            nms_pre = cfg.get('nms_pre', -1)
            if self.level_test:
                bboxes = bboxes.permute(1,0).reshape(1,4,h,w)
                scores = scores.permute(1,0).reshape(1,self.cls_out_channels,h,w)
                centerness = centerness.reshape(1,1,h,w)
                bboxes = F.interpolate(bboxes,size=[levelnesses.shape[0],levelnesses.shape[1]],mode='nearest').permute(0,2,3,1).reshape(-1,4)
                scores = F.interpolate(scores,size=[levelnesses.shape[0],levelnesses.shape[1]],mode='nearest').permute(0,2,3,1).reshape(-1,self.cls_out_channels)
                centerness = F.interpolate(centerness,size=[levelnesses.shape[0],levelnesses.shape[1]],mode='nearest').permute(0,2,3,1).reshape(-1)
                #level_index = one_hot[:,:,level].reshape(-1)
                level_index = levelnesses[:,:,level].reshape(-1)
                level_index = level_index*centerness
                if nms_pre > 0 and scores.shape[0] > nms_pre:
                    max_scores, _ = (scores * centerness[:, None]).max(dim=1)
                    _, topk_inds = max_scores.topk(nms_pre)
                    bboxes = bboxes[topk_inds, :]
                    scores = scores[topk_inds, :]
                    #centerness = centerness[to pk_inds]
                    centerness = level_index[topk_inds]
            else:
                if nms_pre > 0 and scores.shape[0] > nms_pre:
                    max_scores, _ = (scores * centerness[:, None]).max(dim=1)
                    _, topk_inds = max_scores.topk(nms_pre)
                    #points = points[topk_inds, :]
                    #bbox_pred = bbox_pred[topk_inds, :]
                    bboxes = bboxes[topk_inds,:]
                    scores = scores[topk_inds, :]
                    centerness = centerness[topk_inds]
            #pdb.set_trace()
            #bboxes = distance2bbox(points, bbox_pred, max_shape=img_shape)
            level += 1
            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)
            mlvl_centerness.append(centerness)

        mlvl_bboxes = torch.cat(mlvl_bboxes)
        if rescale:
            mlvl_bboxes /= mlvl_bboxes.new_tensor(scale_factor)
        mlvl_scores = torch.cat(mlvl_scores)
        padding = mlvl_scores.new_zeros(mlvl_scores.shape[0], 1)
        mlvl_scores = torch.cat([padding, mlvl_scores], dim=1)
        mlvl_centerness = torch.cat(mlvl_centerness)
        det_bboxes, det_labels = multiclass_nms(
            mlvl_bboxes,
            mlvl_scores,
            #cfg.score_thr,
            0.01,
            cfg.nms,
            cfg.max_per_img,
            score_factors=mlvl_centerness)
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
        labels_list, bbox_targets_list = multi_apply(
            self.fcos_target_single,
            gt_bboxes_list,
            gt_labels_list,
            points=concat_points,
            regress_ranges=concat_regress_ranges)
        #pdb.set_trace()
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
        num_points = points.size(0)
        num_gts = gt_labels.size(0)
        if num_gts == 0:
            return gt_labels.new_zeros(num_points), \
                   gt_bboxes.new_zeros((num_points, 4))

        areas = (gt_bboxes[:, 2] - gt_bboxes[:, 0] + 1) * (
            gt_bboxes[:, 3] - gt_bboxes[:, 1] + 1)
        # TODO: figure out why these two are different
        # areas = areas[None].expand(num_points, num_gts)
        # each point is assigned with areas of all gt

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
        bbox_targets = torch.stack((left, top, right, bottom), -1)
        # bbox_target is a tensor with dimension [num_points, num_gts , 4]
        # condition1: inside a gt bbox
        # inside_gt_bbox_mask is a tensor with dimension [num_points, num_gts]
        inside_gt_bbox_mask = bbox_targets.min(-1)[0] > 0

        # condition2: limit the regression range for each location
        max_regress_distance = bbox_targets.max(-1)[0]
        inside_regress_range = (
            max_regress_distance >= regress_ranges[..., 0]) & (
                max_regress_distance <= regress_ranges[..., 1])

        # if there are still more than one objects for a location,
        # we choose the one with minimal area
        areas[inside_gt_bbox_mask == 0] = INF
        areas[inside_regress_range == 0] = INF
        min_area, min_area_inds = areas.min(dim=1)
        #labels is a tensor [num_points]
        labels = gt_labels[min_area_inds]
        labels[min_area == INF] = 0
        bbox_targets = bbox_targets[range(num_points), min_area_inds]
        #bbox_targets now is [num_points,4]
        return labels, bbox_targets

    def centerness_target(self, pos_bbox_targets):
        # only calculate pos centerness targets, otherwise there may be nan
        left_right = pos_bbox_targets[:, [0, 2]]
        top_bottom = pos_bbox_targets[:, [1, 3]]
        centerness_targets = (
            left_right.min(dim=-1)[0] / left_right.max(dim=-1)[0]) * (
                top_bottom.min(dim=-1)[0] / top_bottom.max(dim=-1)[0])
        return torch.sqrt(centerness_targets)
