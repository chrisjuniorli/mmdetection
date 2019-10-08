#model settings
model = dict(
    type = 'TWO_BRANCHES',
    pretrained = 'open-mmlab://resnet101_caffe',
    backbone = dict(
        type = 'ResNet',
        depth = 101,
        num_stages=4,
        out_indices=(0,1,2,3),
        frozen_stages=1,
        norm_cfg = dict(type='BN',requires_grad = False)
        style = 'caffe'),
    neck = dict(
        type = 'FPN',
        in_channels = [256,512,1024,2048]，
        out_channels = 256,
        start_level = 1,
        add_extra_convs =True,
        extra_convs_on_inputs=False,
        num_outs=5,
        relu_before_extra_convs=True),
    bbox_head = dict(
        type = '
)

train_cfg = dict()
test_cfg = dict()

#dataset settings
dataset_type = 'VOCDataset'
dataset_root = 'data/VOCdevkit'

img_norm_cfg = dict(
    mean=[102.9801, 115.9465, 122.7717], std=[1.0, 1.0, 1.0], to_rgb=False)
train_pipeline = []
test_pipeline = []
data = dict ()

#optimizer 
optimizer = dict()
optimizer_condig = dict()
#learning policy
lr_config = dict()
checkpoint_config = dict()

#yapf:enable
log_config = dict()

#overall params
total_epochs = 8
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './work_dirs/two_branches_1000_600_r101_voc'
load_from = None
resume_from = None
workflow = [('train',1)]