# model settings
model = dict(
    type='GEBDRecognizer3D',
    backbone=dict(
        type='ResNet3dCSN_Keeptem',
        pretrained2d=False,
        pretrained=  # noqa: E251
        '/home/notebook/data/personal/mmaction2/data/models/pretrained_models/ircsn_from_scratch_r152_ig65m_20200807-771c4135.pth',
        depth=152,
        with_pool2=False,
        bottleneck_mode='ir',
        norm_eval=True,
        bn_frozen=True,
        zero_init_residual=False,
        non_local=False),
    neck = dict(type='BMN_Neck',
            temporal_dim=32,
            boundary_ratio=0.5,
            num_samples=16,
            num_samples_per_bin=3,
            feat_dim= 2048*2, #2048*2, #400
            soft_nms_alpha=0.4,
            soft_nms_low_threshold=0.5,
            soft_nms_high_threshold=0.9,
            post_process_top_k=100,
            feature_extraction_interval=1, # total_frames // feature_extraction_interval * feature_extraction_interval;default=16, no effect if frames=32
            hidden_dim_1d=256, # 256
            hidden_dim_2d=128, # 128
            hidden_dim_3d=512), # 512
    cls_head=dict(
        type='I3DHead',
        num_classes=4,
        in_channels=2048,
        spatial_type='avg',
        dropout_ratio=0.5,
        init_std=0.01))
# model training and testing settings
train_cfg = None
test_cfg = dict(average_clips='prob')
# dataset settings
dataset_type = 'GEBDataset'
data_root = '/home/notebook/data/group/chenen.cc/code/loveu/data/loveu_wide_train_2s_30fps' #'/home/notebook/data/group/zhangchen/action_dataset/kinetics-gebd/trainset' 
data_root_val = '/home/notebook/data/group/chenen.cc/code/loveu/data/loveu_wide_val_2s_30fps' #'/home/notebook/data/group/zhangchen/action_dataset/kinetics-gebd/valset'
ann_file_train = '/home/notebook/data/group/chenen.cc/code/loveu/data/loveu_wide_train_2s_30fps_annotation_valid.txt'
ann_file_val = '/home/notebook/data/group/chenen.cc/code/loveu/data/loveu_wide_val_2s_30fps_annotation_valid.txt'
ann_file_test =  '/home/notebook/data/group/chenen.cc/code/loveu/data/loveu_wide_val_2s_30fps_annotation_valid.txt'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_bgr=False)

train_pipeline = [
    dict(type='DecordInit'),
    dict(type='SampleFrames', clip_len=60, frame_interval=1, num_clips=1),
    dict(type='GEBD_Loading', window_size=32, len_clip_second = 2, sample_rate = 30),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(342, 256), keep_ratio=False),
    dict(type='RandomResizedCrop', area_range=(0.7, 1.0), aspect_ratio_range=(1.0, 4/3)),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='Flip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    #dict(type='Collect', keys=['imgs', 'label', 'gt_box'], meta_keys=[]),
    dict(type='Collect', keys=['imgs', 'label', 'gt_box'], meta_name='video_meta', meta_keys=['video_name']),
    dict(type='ToTensor', keys=['imgs', 'label', 'gt_box'])
]

val_pipeline = [
    dict(type='DecordInit'),
    dict(
        type='SampleFrames',
        clip_len=60,
        frame_interval=1,
        num_clips=1,
        test_mode=True),
    dict(type='GEBD_Loading', window_size=32, len_clip_second = 2, sample_rate = 30),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(342, 256), keep_ratio=False),
    dict(type='CenterCrop', crop_size=224),
    dict(type='Flip', flip_ratio=0),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    #dict(type='Collect', keys=['imgs', 'label', 'gt_box'], meta_keys=[]),
    dict(
        type='Collect',
        keys=['imgs', 'label', 'gt_box'],
        meta_name='video_meta',
        meta_keys=[
            'video_name', 'duration_second', 'duration_frame', 'label', 'gt_box',
        ]),
    dict(type='ToTensor', keys=['imgs', 'label', 'gt_box'])
]

test_pipeline = [
    dict(type='DecordInit'),
    dict(
        type='SampleFrames',
        clip_len=32,
        frame_interval=1,
        num_clips=1,
        test_mode=True),
    #dict(type='GEBD_Loading', window_size=32, len_clip_second = 2, sample_rate = 30),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(342, 256), keep_ratio=False),
    #dict(type='CenterCrop', crop_size=224),
    dict(type='ThreeCrop', crop_size=256),
    dict(type='Flip', flip_ratio=0),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    #dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(
        type='Collect',
        keys=['imgs'],
        #meta_name='video_meta',
        meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]


data = dict(
    videos_per_gpu=4,#4, #3
    workers_per_gpu=4, #4
    test_bmn_dataloader=dict(videos_per_gpu=1),
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
#optimizer = dict(
#    type='SGD', lr=0.00025, momentum=0.9,
#    weight_decay=0.0001)  # this lr is used for 8 gpus, bs=6

optimizer = dict(
    type='SGD', lr=0.000125, momentum=0.9,
    weight_decay=0.0001)

#optimizer = dict(
#    type='SGD', lr=0.0001, momentum=0.9,
#    weight_decay=0.0001)
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
checkpoint_config = dict(interval=1)
evaluation = dict(
    interval=62, metrics=['top_k_accuracy', 'mean_class_accuracy']) #'top_k_accuracy', 'mean_class_accuracy'])
log_config = dict(
    interval=20,
    hooks=[dict(type='TextLoggerHook'),
           dict(type='TensorboardLoggerHook')])
# runtime settings
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './work_dirs/csn_bmn_2s'  # noqa: E501
load_from = None #'/home/notebook/data/personal/mmaction2_sn_test/work_dirs/csn_sn15/epoch_20.pth'
resume_from = None #'/home/notebook/data/personal/loveu/mmaction2/work_dirs/loveu_4cls_1500ms_csn/epoch_20.pth'
workflow = [('train', 1)]
find_unused_parameters = True
