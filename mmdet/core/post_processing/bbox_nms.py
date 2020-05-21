import torch
import pdb

import numpy as np
from mmdet.ops.nms import nms_wrapper
from IPython import embed

def multiclass_nms(multi_bboxes,
                   multi_scores,
                   score_thr,
                   nms_cfg,
                   max_num=-1,
                   score_factors=None):
    """NMS for multi-class bboxes.

    Args:
        multi_bboxes (Tensor): shape (n, #class*4) or (n, 4)
        multi_scores (Tensor): shape (n, #class)
        score_thr (float): bbox threshold, bboxes with scores lower than it
            will not be considered.
        nms_thr (float): NMS IoU threshold
        max_num (int): if there are more than max_num bboxes after NMS,
            only top max_num will be kept.
        score_factors (Tensor): The factors multiplied to scores before
            applying NMS

    Returns:
        tuple: (bboxes, labels), tensors of shape (k, 5) and (k, 1). Labels
            are 0-based.
    """
    #pdb.set_trace()
    num_classes = multi_scores.shape[1]
    bboxes, labels, bboxes_before = [], [], []
    nms_cfg_ = nms_cfg.copy()
    nms_type = nms_cfg_.pop('type', 'nms')
    nms_op = getattr(nms_wrapper, nms_type)
    for i in range(1, num_classes):
        cls_inds = multi_scores[:, i] > score_thr
        if not cls_inds.any():
            continue
        # get bboxes and scores of this class
        if multi_bboxes.shape[1] == 4:
            _bboxes = multi_bboxes[cls_inds, :]
        else:
            _bboxes = multi_bboxes[cls_inds, i * 4:(i + 1) * 4]
        _scores = multi_scores[cls_inds, i]
        if score_factors is not None:
            _scores *= score_factors[cls_inds]
        cls_dets = torch.cat([_bboxes, _scores[:, None]], dim=1)
        #cls_dets, _ = nms_op(cls_dets, **nms_cfg_)
        #while cls_dets.shape[0]!=0:
        bboxes_before.append(cls_dets)
        cls_dets = nms(cls_dets,nms_cfg['iou_thr'])
        cls_labels = multi_bboxes.new_full((cls_dets.shape[0], ),
                                           i - 1,
                                           dtype=torch.long)
        #stat = calculate_thr(cls_dets)
       # pdb.set_trace()
        bboxes.append(cls_dets)
        labels.append(cls_labels)
    if bboxes:
        bboxes = torch.cat(bboxes)
        labels = torch.cat(labels)
        bboxes_before = torch.cat(bboxes_before)
        stats = calculate_thr(bboxes_before)
        
        if bboxes.shape[0] > max_num:
            _, inds = bboxes[:, -1].sort(descending=True)
            inds = inds[:max_num]
            bboxes = bboxes[inds]
            labels = labels[inds]
    else:
        bboxes = multi_bboxes.new_zeros((0, 5))
        labels = multi_bboxes.new_zeros((0, ), dtype=torch.long)
        stats = np.zeros(5)

    return bboxes, labels#, stats

def calculate_thr(bboxes):
    bboxes = bboxes.cpu().detach().numpy()
    stats = np.zeros(5)
    for i in range(bboxes.shape[0]):
        if bboxes[i,-1] >= 0 and bboxes[i,-1] < 0.2:
            stats[0] += 1
        elif bboxes[i,-1] >= 0.2 and bboxes[i,-1] < 0.4:
            stats[1] += 1
        elif bboxes[i,-1] >= 0.4 and bboxes[i,-1] < 0.6:
            stats[2] += 1
        elif bboxes[i,-1] >= 0.6 and bboxes[i,-1] < 0.8:
            stats[3] += 1
        else:
            stats[4] += 1
    #pdb.set_trace()
    return stats

def nms(dets, thresh):
    #pdb.set_trace()
    dets = dets.cpu().detach().numpy()
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]
    #pdb.set_trace()
    cls_dets = torch.from_numpy(dets[keep]).cuda()
    return cls_dets  


def nms_top2(dets, thresh):
    #pdb.set_trace()
    dets = dets.cpu().detach().numpy()
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 1:
        #pdb.set_trace()
        i = order[0]
        j = order[1]
        keep.append(i)
        keep.append(j)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        #pdb.set_trace()
        order = order[inds + 1]
    #pdb.set_trace()
    cls_dets = torch.from_numpy(dets[keep]).cuda()
    return cls_dets  


def multiclass_nms_with_mask(multi_bboxes,
                   multi_scores,
                   multi_masks,
                   score_thr,
                   nms_cfg,
                   max_num=-1,
                   score_factors=None):
    """NMS for multi-class bboxes.

    Args:
        multi_bboxes (Tensor): shape (n, #class*4) or (n, 4)
        multi_scores (Tensor): shape (n, #class)
        score_thr (float): bbox threshold, bboxes with scores lower than it
            will not be considered.
        nms_thr (float): NMS IoU threshold
        max_num (int): if there are more than max_num bboxes after NMS,
            only top max_num will be kept.
        score_factors (Tensor): The factors multiplied to scores before
            applying NMS

    Returns:
        tuple: (bboxes, labels), tensors of shape (k, 5) and (k, 1). Labels
            are 0-based.
    """
    num_classes = multi_scores.shape[1]
    bboxes, labels, masks = [], [], []
    nms_cfg_ = nms_cfg.copy()
    nms_type = nms_cfg_.pop('type', 'nms')
    nms_op = getattr(nms_wrapper, nms_type)
    for i in range(1, num_classes):
        cls_inds = multi_scores[:, i] > score_thr
        if not cls_inds.any():
            continue
        # get bboxes and scores of this class
        if multi_bboxes.shape[1] == 4:
            _bboxes = multi_bboxes[cls_inds, :]
            _masks  = multi_masks[cls_inds, :]
        else:
            _bboxes = multi_bboxes[cls_inds, i * 4:(i + 1) * 4]
        _scores = multi_scores[cls_inds, i]
        if score_factors is not None:
            _scores *= score_factors[cls_inds]
        cls_dets = torch.cat([_bboxes, _scores[:, None]], dim=1)
        cls_dets, index = nms_op(cls_dets, **nms_cfg_)
        cls_masks = _masks[index]
        cls_labels = multi_bboxes.new_full((cls_dets.shape[0], ),
                                           i - 1,
                                           dtype=torch.long)
        bboxes.append(cls_dets)
        labels.append(cls_labels)
        masks.append(cls_masks)

    if bboxes:
        bboxes = torch.cat(bboxes)
        labels = torch.cat(labels)
        masks = torch.cat(masks)
        if bboxes.shape[0] > max_num:
            _, inds = bboxes[:, -1].sort(descending=True)
            inds = inds[:max_num]
            bboxes = bboxes[inds]
            labels = labels[inds]
            masks = masks[inds]
    else:
        bboxes = multi_bboxes.new_zeros((0, 5))
        labels = multi_bboxes.new_zeros((0, ), dtype=torch.long)
        masks = multi_bboxes.new_zeros((0, 2, 36))

    return bboxes, labels, masks
