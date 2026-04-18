_base_ = [
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_160k.py'
]

crop_size = (512, 512)
data_root = '/root/autodl-tmp/segment/data/tongji'

metainfo = dict(
    classes=(
        'background',
        'crack',
        'lining_falling_off',
        'segment_damage',
        'leakageB',
        'leakageG',
        'leakageW'
    ),
    palette=[
        [0, 0, 0],
        [128, 0, 0],
        [0, 128, 0],
        [128, 128, 0],
        [0, 0, 128],
        [128, 0, 128],
        [0, 128, 128]
    ]
)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', scale=(1024, 512), keep_ratio=True),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='ResizeToMultiple', size_divisor=32),
    dict(type='PackSegInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(1024, 512), keep_ratio=True),
    dict(type='ResizeToMultiple', size_divisor=32),
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs')
]

train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type='BaseSegDataset',
        data_root=data_root,
        data_prefix=dict(
            img_path='images/train',
            seg_map_path='ann/train'
        ),
        pipeline=train_pipeline,
        metainfo=metainfo
    )
)

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='BaseSegDataset',
        data_root=data_root,
        data_prefix=dict(
            img_path='images/val',
            seg_map_path='ann/val'
        ),
        pipeline=test_pipeline,
        metainfo=metainfo
    )
)

test_dataloader = val_dataloader

val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])
test_evaluator = val_evaluator

norm_cfg = dict(type='BN', requires_grad=True)

data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255,
    size=crop_size
)

model = dict(
    type='EncoderDecoder',
    data_preprocessor=data_preprocessor,
    pretrained='open-mmlab://resnet50_v1c',
    backbone=dict(
        type='ResNetV1c',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        dilations=(1, 1, 2, 4),
        strides=(1, 2, 1, 1),
        norm_cfg=norm_cfg,
        norm_eval=False,
        style='pytorch',
        contract_dilation=True
    ),
    decode_head=dict(
        type='DepthwiseSeparableASPPHead',
        in_channels=2048,
        in_index=3,
        channels=512,
        dilations=(1, 12, 24, 36),
        c1_in_channels=256,
        c1_channels=48,
        dropout_ratio=0.1,
        num_classes=7,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=1.0,
            class_weight=[1.0, 5.0, 5.0, 5.0, 2.0, 5.0, 5.0]
        )
    ),
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=1024,
        in_index=2,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=7,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=0.4,
            class_weight=[1.0, 5.0, 5.0, 5.0, 2.0, 5.0, 5.0]
        )
    ),
    train_cfg=dict(),
    test_cfg=dict(mode='whole')
)

train_cfg = dict(type='IterBasedTrainLoop', max_iters=8000, val_interval=1000)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW',
        lr=6e-5,
        betas=(0.9, 0.999),
        weight_decay=0.01
    )
)

param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1e-6,
        by_epoch=False,
        begin=0,
        end=500
    ),
    dict(
        type='PolyLR',
        eta_min=0.0,
        power=0.9,
        begin=500,
        end=8000,
        by_epoch=False
    )
]

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=20, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=1000, max_keep_ckpts=3),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook')
)

work_dir = './work_dirs/deeplabv3plus_tongji'