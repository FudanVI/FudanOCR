_base_ = [
    '../_base_/models/segformer_mit-b0.py',
    '../_base_/datasets/BTS.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_320k.py'
]

# checkpoint = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segformer/mit_b0_20220624-7e0fe6dd.pth'  # noqa

# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
    pretrained=None,
    backbone=dict(
        type='CascadeMixVisionTransformer_V1',
        in_channels=3,
        embed_dims=32,
        num_stages=4,
        num_layers=[2, 2, 2, 2],
        num_heads=[1, 2, 5, 8],
        patch_sizes=[7, 3, 3, 3],
        sr_ratios=[8, 4, 2, 1],
        out_indices=(0, 1, 2, 3),
        mlp_ratio=4,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
        init_cfg=None),
    decode_head=dict(
        type='SegformerHead',
        in_channels=[32, 64, 160, 256],
        in_index=[0, 1, 2, 3],
        channels=256,
        dropout_ratio=0.1,
        num_classes=2,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    # model training and testing settings
    train_cfg=dict(det_loss_ratio=0.1),
    test_cfg=dict(mode='slide', crop_size=(1024, 1024), stride=(768, 768)))

# optimizer
optimizer = dict(
    _delete_=True,
    type='AdamW',
    lr=0.00006,
    betas=(0.9, 0.999),
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys={
            'pos_block': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.),
            'head': dict(lr_mult=10.)
        }))

lr_config = dict(
    _delete_=True,
    policy='poly',
    warmup='linear',
    warmup_iters=1500,
    warmup_ratio=1e-6,
    power=1.0,
    min_lr=0.0,
    by_epoch=False)

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=1,
    train=dict(
        img_dir='./BTS/all_image/',
        ann_dir='./BTS/all_semantic_label_png/',
        det_dir='./BTS/all_det_label_png/'),
        # img_dir='./BTS_TextSeg/image/',
        # ann_dir='./BTS_TextSeg/seg/',
        # det_dir='./BTS_TextSeg/det/'),
    val=dict(
        img_dir='./TextSeg_Release/textseg_test/',
        ann_dir='./TextSeg_Release/semantic_test/',
        det_dir='./TextSeg_Release/det_res_test_png/'),
    test=dict(
        img_dir='./TextSeg_Release/textseg_test/',
        ann_dir='./TextSeg_Release/semantic_test/',
        det_dir='./TextSeg_Release/det_res_test_png/'))