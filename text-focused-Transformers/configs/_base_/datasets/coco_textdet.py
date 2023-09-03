# dataset settings
dataset_type = 'MLTSDataset_Det'
data_root = '/home/yuhaiyang/dataset/COCO-Text'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (1024, 1024)
# crop_size = (400, 500)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations_mlt', reduce_zero_label=False),
    dict(type='Resize', img_scale=(2048, 2048), ratio_range=(0.5, 2.0)),
    # dict(type='Resize', img_scale=(1000, 800), ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=0),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg', 'gt_det']),
    # dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(2048, 2048),
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='./images_train/',
        ann_dir='./label_train/',
        det_dir='./det_mask_train_png/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='./images_val/',
        ann_dir='./label_val/',
        det_dir='./det_mask_val_png/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='./images_val/',
        ann_dir='./label_val/',
        det_dir='./det_mask_val_png/',
        pipeline=test_pipeline))
