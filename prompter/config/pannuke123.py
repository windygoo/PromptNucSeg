prompter = dict(
    backbone=dict(
        model_name='convnext_small',
        pretrained=True,
        num_classes=0,
        global_pool=''
    ),
    neck=dict(
        in_channels=[96, 192, 384, 768],
        out_channels=256,
        num_outs=3,
        add_extra_convs='on_input',
    ),
    dropout=0.1,
    space=8,
    hidden_dim=256
)

data = dict(
    name='pannuke123',
    num_classes=5,
    batch_size_per_gpu=16,
    num_workers=8,
    train=dict(transform=[
        dict(type='RandomGridShuffle', grid=(4, 4), p=0.5),
        dict(type='ColorJitter', brightness=0.25, contrast=0.25, saturation=0.1, hue=0.05, p=0.2),
        dict(type='RandomRotate90', p=0.5),
        dict(type='Downscale', scale_max=0.5, scale_min=0.5, p=0.15),
        dict(type='Blur', blur_limit=10, p=0.2),
        dict(type='GaussNoise', var_limit=50, p=0.25),
        dict(type='ZoomBlur', p=0.1, max_factor=1.05),
        dict(type='HorizontalFlip', p=0.5),
        dict(type='VerticalFlip', p=0.5),
        dict(type='ShiftScaleRotate', shift_limit=0.3, scale_limit=0.1, rotate_limit=0, border_mode=0, value=0, p=0.5),
        dict(type='PadIfNeeded', min_height=None, min_width=None, pad_height_divisor=prompter["space"],
             pad_width_divisor=prompter["space"], position="top_left", p=1),
        dict(type='Normalize'),
    ]),
    val=dict(transform=[
        dict(type='PadIfNeeded', min_height=None, min_width=None, pad_height_divisor=prompter["space"],
             pad_width_divisor=prompter["space"], position="top_left", p=1),
        dict(type='Normalize'),
    ]),
    test=dict(transform=[
        dict(type='PadIfNeeded', min_height=None, min_width=None, pad_height_divisor=prompter["space"],
             pad_width_divisor=prompter["space"], position="top_left", p=1),
        dict(type='Normalize'),
    ]),
)

optimizer = dict(
    type='Adam',
    lr=1e-4,
    weight_decay=1e-4
)

scheduler = dict(
    type='MultiStepLR',
    milestones=[100],
    gamma=0.1
)

criterion = dict(
    matcher=dict(type='HungarianMatcher', dis_type='l2', set_cost_point=0.1, set_cost_class=1),
    eos_coef=0.4,
    reg_loss_coef=5e-3,
    cls_loss_coef=1.0,
    mask_loss_coef=1.0
)

test = dict(nms_thr=12, match_dis=12, filtering=False)
