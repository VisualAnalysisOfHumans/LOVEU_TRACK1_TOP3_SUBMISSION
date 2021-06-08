model = dict(
    type='GEBDRecognizer3D',
    backbone=dict(
        type='ResNet3dCSN_Keeptem',
        pretrained2d=False,
        pretrained=
        '/home/notebook/data/personal/mmaction2/data/models/pretrained_models/ircsn_from_scratch_r152_ig65m_20200807-771c4135.pth',
        depth=152,
        with_pool2=False,
        bottleneck_mode='ir',
        norm_eval=True,
        bn_frozen=True,
        zero_init_residual=False,
        non_local=False),
    neck=dict(
        type='BMN_Neck',
        temporal_dim=32,
        boundary_ratio=0.5,
        num_samples=16,
        num_samples_per_bin=3,
        feat_dim=4096,
        soft_nms_alpha=0.4,
        soft_nms_low_threshold=0.5,
        soft_nms_high_threshold=0.9,
        post_process_top_k=100,
        feature_extraction_interval=1,
        hidden_dim_1d=256,
        hidden_dim_2d=128,
        hidden_dim_3d=512),
    cls_head=dict(
        type='I3DHead',
        num_classes=4,
        in_channels=2048,
        spatial_type='avg',
        dropout_ratio=0.5,
        init_std=0.01))
train_cfg = None
test_cfg = dict(average_clips='prob')
dataset_type = 'GEBDataset'
data_root = '/home/notebook/data/personal/dataset/loveu/loveu_wide_trainvals_2s_30fps'
data_root_val = '/home/notebook/data/personal/dataset/loveu/loveu_wide_val_2s_30fps'
ann_file_train = '/home/notebook/data/personal/dataset/loveu/loveu_wide_trainvals_2s_30fps_annotation_valid.txt'
ann_file_val = '/home/notebook/data/personal/dataset/loveu/loveu_wide_val_2s_30fps_annotation_valid.txt'
ann_file_test = '/home/notebook/data/personal/dataset/loveu/loveu_wide_val_2s_30fps_annotation_valid.txt'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_bgr=False)
train_pipeline = [
    dict(type='DecordInit'),
    dict(type='SampleFrames', clip_len=60, frame_interval=1, num_clips=1),
    dict(
        type='GEBD_Loading',
        window_size=32,
        len_clip_second=2,
        sample_rate=30,
        temporal_flip=True,
        thres=0.5),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(342, 256), keep_ratio=False),
    dict(
        type='RandomResizedCrop',
        area_range=(0.7, 1.0),
        aspect_ratio_range=(1.0, 1.3333333333333333)),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='Flip', flip_ratio=0.5),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_bgr=False),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(
        type='Collect',
        keys=['imgs', 'label', 'gt_box'],
        meta_name='video_meta',
        meta_keys=['video_name']),
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
    dict(
        type='GEBD_Loading', window_size=32, len_clip_second=2,
        sample_rate=30),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(342, 256), keep_ratio=False),
    dict(type='CenterCrop', crop_size=224),
    dict(type='Flip', flip_ratio=0),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_bgr=False),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(
        type='Collect',
        keys=['imgs', 'label', 'gt_box'],
        meta_name='video_meta',
        meta_keys=[
            'video_name', 'duration_second', 'duration_frame', 'label',
            'gt_box'
        ]),
    dict(type='ToTensor', keys=['imgs', 'label', 'gt_box'])
]
test_pipeline = [
    dict(type='DecordInit'),
    dict(
        type='SampleFrames',
        clip_len=60,
        frame_interval=1,
        num_clips=1,
        test_mode=True),
    dict(
        type='GEBD_Loading', window_size=32, len_clip_second=2,
        sample_rate=30),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(342, 256), keep_ratio=False),
    dict(type='CenterCrop', crop_size=224),
    dict(type='Flip', flip_ratio=0),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_bgr=False),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(
        type='Collect',
        keys=['imgs'],
        meta_name='video_meta',
        meta_keys=['video_name', 'duration_second', 'duration_frame']),
    dict(type='ToTensor', keys=['imgs'])
]
data = dict(
    videos_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        type='GEBDataset',
        ann_file=
        '/home/notebook/data/personal/dataset/loveu/loveu_wide_trainvals_2s_30fps_annotation_valid.txt',
        data_prefix=
        '/home/notebook/data/personal/dataset/loveu/loveu_wide_trainvals_2s_30fps',
        pipeline=[
            dict(type='DecordInit'),
            dict(
                type='SampleFrames',
                clip_len=60,
                frame_interval=1,
                num_clips=1),
            dict(
                type='GEBD_Loading',
                window_size=32,
                len_clip_second=2,
                sample_rate=30,
                temporal_flip=True,
                thres=0.5),
            dict(type='DecordDecode'),
            dict(type='Resize', scale=(342, 256), keep_ratio=False),
            dict(
                type='RandomResizedCrop',
                area_range=(0.7, 1.0),
                aspect_ratio_range=(1.0, 1.3333333333333333)),
            dict(type='Resize', scale=(224, 224), keep_ratio=False),
            dict(type='Flip', flip_ratio=0.5),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_bgr=False),
            dict(type='FormatShape', input_format='NCTHW'),
            dict(
                type='Collect',
                keys=['imgs', 'label', 'gt_box'],
                meta_name='video_meta',
                meta_keys=['video_name']),
            dict(type='ToTensor', keys=['imgs', 'label', 'gt_box'])
        ]),
    val=dict(
        type='GEBDataset',
        ann_file=
        '/home/notebook/data/personal/dataset/loveu/loveu_wide_val_2s_30fps_annotation_valid.txt',
        data_prefix=
        '/home/notebook/data/personal/dataset/loveu/loveu_wide_val_2s_30fps',
        pipeline=[
            dict(type='DecordInit'),
            dict(
                type='SampleFrames',
                clip_len=60,
                frame_interval=1,
                num_clips=1,
                test_mode=True),
            dict(
                type='GEBD_Loading',
                window_size=32,
                len_clip_second=2,
                sample_rate=30),
            dict(type='DecordDecode'),
            dict(type='Resize', scale=(342, 256), keep_ratio=False),
            dict(type='CenterCrop', crop_size=224),
            dict(type='Flip', flip_ratio=0),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_bgr=False),
            dict(type='FormatShape', input_format='NCTHW'),
            dict(
                type='Collect',
                keys=['imgs', 'label', 'gt_box'],
                meta_name='video_meta',
                meta_keys=[
                    'video_name', 'duration_second', 'duration_frame', 'label',
                    'gt_box'
                ]),
            dict(type='ToTensor', keys=['imgs', 'label', 'gt_box'])
        ]),
    test=dict(
        type='GEBDataset',
        ann_file=
        '/home/notebook/data/personal/dataset/loveu/loveu_wide_val_2s_30fps_annotation_valid.txt',
        data_prefix=
        '/home/notebook/data/personal/dataset/loveu/loveu_wide_val_2s_30fps',
        pipeline=[
            dict(type='DecordInit'),
            dict(
                type='SampleFrames',
                clip_len=60,
                frame_interval=1,
                num_clips=1,
                test_mode=True),
            dict(
                type='GEBD_Loading',
                window_size=32,
                len_clip_second=2,
                sample_rate=30),
            dict(type='DecordDecode'),
            dict(type='Resize', scale=(342, 256), keep_ratio=False),
            dict(type='CenterCrop', crop_size=224),
            dict(type='Flip', flip_ratio=0),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_bgr=False),
            dict(type='FormatShape', input_format='NCTHW'),
            dict(
                type='Collect',
                keys=['imgs'],
                meta_name='video_meta',
                meta_keys=['video_name', 'duration_second', 'duration_frame']),
            dict(type='ToTensor', keys=['imgs'])
        ]))
optimizer = dict(type='SGD', lr=6e-05, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=40, norm_type=2))
lr_config = dict(
    policy='step',
    step=[5, 8],
    warmup='linear',
    warmup_ratio=0.1,
    warmup_by_epoch=True,
    warmup_iters=16)
total_epochs = 10
checkpoint_config = dict(interval=1)
evaluation = dict(
    interval=62, metrics=['top_k_accuracy', 'mean_class_accuracy'])
log_config = dict(
    interval=20,
    hooks=[dict(type='TextLoggerHook'),
           dict(type='TensorboardLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = '/home/notebook/data/personal/mmaction2/data/output/csn_bmn_4cls_2s_32f_bw_16f_bmn_atten_wo_extra_featrues_finetune_bmn_cls_e12'
load_from = '/home/notebook/data/personal/mmaction2/data/output/csn_bmn_4cls_2s_32f_bw_16f_bmn_atten_wo_extra_featrues/epoch_12.pth'
resume_from = None
workflow = [('train', 1)]
find_unused_parameters = True
gpu_ids = range(0, 1)
omnisource = False
module_hooks = []
