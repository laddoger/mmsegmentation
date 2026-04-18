
_base_ = [
    '../_base_/models/segformer_mit-b0.py',
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
    dict(type='PackSegInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(1024, 512), keep_ratio=True),
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

# model = dict(
#     data_preprocessor=dict(
#         _delete_=True,
#         type='SegDataPreProcessor',
#         mean=[123.675, 116.28, 103.53],
#         std=[58.395, 57.12, 57.375],
#         bgr_to_rgb=True,
#         pad_val=0,
#         seg_pad_val=255,
#         size=crop_size
#     ),
#     decode_head=dict(num_classes=7,
#                      loss_decode=dict(
#                          type='CrossEntropyLoss',
#                          use_sigmoid=False,
#                          loss_weight=1.0,
#                          class_weight=[1.0, 5.0, 5.0, 5.0, 2.0, 5.0, 5.0]
#                      )
#     )
# )
model = dict(
    backbone=dict(
        init_cfg=dict(
            type='Pretrained',
            checkpoint='/root/autodl-tmp/segment/mmsegmentation/pretrained/mit-b0/mit_b0_mmseg.pth')),
    data_preprocessor=dict(
        _delete_=True,
        type='SegDataPreProcessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_val=0,
        seg_pad_val=255,
        size=crop_size),
    decode_head=dict(
        num_classes=7,
        loss_decode=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=1.0,
            class_weight=[1.0, 5.0, 5.0, 5.0, 2.0, 5.0, 5.0]
        )
    )
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
        weight_decay=0.01,
    )
)

param_scheduler = [
    dict(
        type='PolyLR',
        eta_min=0.0,
        power=1.0,
        begin=0,
        end=8000,
        by_epoch=False)
]

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=1, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=1000, max_keep_ckpts=3),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook')
)

work_dir = './work_dirs/segformer_tongji'