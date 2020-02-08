from ..registry import DETECTORS
from .single_stage import SingleStageDetector
import pdb
from mmdet.core import bbox2result

@DETECTORS.register_module
class PAR_SAF(SingleStageDetector):

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(PAR_SAF, self).__init__(backbone, neck, bbox_head, train_cfg,
                                   test_cfg, pretrained)
    
    def simple_test(self, img, img_meta, rescale=False):
        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        bbox_inputs = outs + (img_meta, self.test_cfg, rescale)
        bbox_list = self.bbox_head.get_bboxes(*bbox_inputs)
        bbox_results = [
            #bbox2result from torch.tensor to list[numpy array]
            bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
            for det_bboxes, det_labels in bbox_list
        ]
        return bbox_results[0]