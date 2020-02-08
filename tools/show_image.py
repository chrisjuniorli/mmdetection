from mmdet.apis import init_detector, inference_detector, show_result_pyplot
import mmcv

config_file = '../configs/faster_rcnn_r50_fpn_1x.py'
# download the checkpoint from model zoo and put it in `checkpoints/`
checkpoint_file = '../checkpoints/faster_rcnn_r50_fpn_1x_20181010-3d1b3351.pth'
