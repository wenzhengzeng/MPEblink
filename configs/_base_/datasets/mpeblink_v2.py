import os

dataset_type = 'MPEblinkV2Dataset'
data_root = os.getenv(
    'MPEBLINK2_ROOT',
    '/data/data4/zengwenzheng/data/dataset_building/mpeblink2_1/')
clip_length = 11
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='LoadAnnotations',
        with_bbox=True,
        with_blink=True,
        with_eye_bbox=True,
        with_id=True),
    dict(type='Resize', img_scale=[(640, 360)], keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(
        type='Collect',
        keys=[
            'img', 'gt_bboxes', 'gt_labels', 'gt_blinks', 'gt_eye_bboxes',
            'gt_ids'
        ]),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', img_scale=(640, 360), keep_ratio=False),
    dict(type='RandomFlip', flip_ratio=0.0),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img']),
]

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=12,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/train.json',
        clip_length=clip_length,
        with_eye_bbox=True,
        img_prefix=data_root + 'train_rawframes/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + '11111val.json',
        clip_length=clip_length,
        with_eye_bbox=True,
        img_prefix=data_root + '21val/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + '11111test.json',
        clip_length=clip_length,
        with_eye_bbox=True,
        img_prefix=data_root + '12test/',
        pipeline=test_pipeline))

evaluation = dict(metric=['segm'])
