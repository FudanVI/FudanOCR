_base_ = ['./segformer_mit-b0_8x1_160k_textseg.py']

# checkpoint = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segformer/mit_b5_20220624-658746d9.pth'  # noqa
# checkpoint = '/home/yuhaiyang/mmsegmentation/segformer_mit-b5_8x1_1024x1024_160k_cityscapes_20211206_072934-87a052ec.pth'  # noqa
model = dict(
    backbone=dict(
        init_cfg=None,
        # init_cfg=dict(type='Pretrained', checkpoint=checkpoint),
        embed_dims=64,
        num_layers=[3, 6, 40, 3]),
    decode_head=dict(in_channels=[64, 128, 320, 512]))

data = dict(samples_per_gpu=1, workers_per_gpu=1)