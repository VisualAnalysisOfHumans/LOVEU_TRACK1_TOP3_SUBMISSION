# model settings
model = dict(
    type='Recognizer3D',
    backbone=dict(
        type='ResNet3dCSN',
        pretrained2d=False,
        pretrained=  # noqa: E251
        '/home/notebook/data/personal/mmaction2_task1/data/ircsn_from_scratch_r152_ig65m_20200807-771c4135.pth',  # noqa: E501
        depth=152,
        with_pool2=False,
        bottleneck_mode='ir',
        norm_eval=True,
        bn_frozen=True,
        zero_init_residual=False),
    cls_head=dict(
        type='I3DHead',
        num_classes=2,
        in_channels=2048,
        spatial_type='avg',
        dropout_ratio=0.5,
        init_std=0.01))
# model training and testing settings
train_cfg = None
test_cfg = dict(average_clips='prob')
# dataset settings
dataset_type = 'VideoDataset'
data_root = '' #'data/kinetics400/rawframes_train'
data_root_val = '' #'data/kinetics400/rawframes_val'
ann_file_train = '/home/notebook/data/personal/loveu/data/loveu_wide_2cls_train_annotation_valid.txt'
ann_file_val =   '/home/notebook/data/personal/loveu/data/loveu_wide_2cls_val_annotation_valid.txt'
ann_file_test =  '/home/notebook/data/personal/loveu/data/loveu_wide_2cls_val_annotation_valid.txt'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_bgr=False)
train_pipeline = [
    dict(type='DecordInit'),
    dict(type='SampleFrames', clip_len=16, frame_interval=2, num_clips=1),
    #dict(type='FrameSelector'),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(342, 256), keep_ratio=False),
    dict(type='RandomResizedCrop', area_range=(0.7, 1.0), aspect_ratio_range=(1.0, 4/3)),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='Flip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label'])
]
val_pipeline = [
    dict(type='DecordInit'),
    dict(
        type='SampleFrames',
        clip_len=16,
        frame_interval=2,
        num_clips=1,
        test_mode=True),
    #dict(type='FrameSelector'),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(342, 256), keep_ratio=False),
    dict(type='CenterCrop', crop_size=224),
    dict(type='Flip', flip_ratio=0),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
test_pipeline = [
    dict(type='DecordInit'),
    dict(
        type='SampleFrames',
        clip_len=16,
        frame_interval=2,
        num_clips=1,
        test_mode=True),
    #dict(type='FrameSelector'),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(342, 256), keep_ratio=False),
    dict(type='ThreeCrop', crop_size=256),
    dict(type='Flip', flip_ratio=0),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
data = dict(
    videos_per_gpu=6, #3
    workers_per_gpu=4, #4
    train=dict(
        type=dataset_type,
        ann_file=ann_file_train,
        data_prefix=data_root,
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        data_prefix=data_root_val,
        pipeline=val_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        data_prefix=data_root_val,
        pipeline=test_pipeline))
# optimizer
optimizer = dict(
    type='SGD', lr=0.00025, momentum=0.9,
    weight_decay=0.0001)  # this lr is used for 8 gpus
optimizer_config = dict(grad_clip=dict(max_norm=40, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    step=[32, 48],
    warmup='linear',
    warmup_ratio=0.1,
    warmup_by_epoch=True,
    warmup_iters=16)
total_epochs = 61
checkpoint_config = dict(interval=2)
evaluation = dict(
    interval=62, metrics=['top_k_accuracy', 'mean_class_accuracy']) #'top_k_accuracy', 'mean_class_accuracy'])
log_config = dict(
    interval=20,
    hooks=[dict(type='TextLoggerHook'),
           dict(type='TensorboardLoggerHook')])
# runtime settings
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './work_dirs/ircsn_ig65m_pretrained_bnfrozen_r152_32x2x1_58e_kinetics400_rgb'  # noqa: E501
load_from = None #'/home/notebook/data/personal/mmaction2_sn_test/work_dirs/csn_sn15/epoch_20.pth'
resume_from = '/home/notebook/data/personal/loveu/mmaction2/work_dirs/loveu_2cls_1500ms_csn/epoch_20.pth'
workflow = [('train', 1)]
find_unused_parameters = True
