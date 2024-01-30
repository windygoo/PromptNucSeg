segmentor = dict(
    type='PromptNucSeg-L',
    img_size=256,
    patch_size=16,
    multimask=False
)

input_shape = segmentor['img_size']
data = dict(
    name='pannuke321',
    num_classes=5,
    num_mask_per_img=20,
    batch_size_per_gpu=16,
    num_workers=0,
    num_neg_prompt=0,
    train=dict(transform=[
        dict(type='RandomCrop', height=256, width=256, p=1),
        dict(type='RandomRotate90', p=0.5),
        dict(type='HorizontalFlip', p=0.5),
        dict(type='VerticalFlip', p=0.5),
        dict(type='Downscale', scale_max=0.5, scale_min=0.5, p=0.15),
        dict(type='Blur', blur_limit=10, p=0.2),
        dict(type='GaussNoise', var_limit=50, p=0.25),
        dict(type='ColorJitter', brightness=0.25, contrast=0.25, saturation=0.1, hue=0.05, p=0.2),
        dict(type='Superpixels', p=0.1, p_replace=0.1, n_segments=200, max_size=int(input_shape / 2)),
        dict(type='ZoomBlur', p=0.1, max_factor=1.05),
        dict(type='RandomSizedCrop', min_max_height=(int(input_shape / 2), input_shape),
             height=input_shape,
             width=input_shape,
             p=0.1),
        dict(type='ElasticTransform', p=0.2, sigma=25, alpha=0.5, alpha_affine=15),
        dict(type='Normalize'),
    ]),
    val=dict(transform=[
        dict(type='Normalize'),
    ]),
    test=dict(transform=[
        dict(type='Normalize'),
    ]),
    post=dict(iou_threshold=0.5)
)

optimizer = dict(
    type='Adam',
    lr=1e-4,
    weight_decay=0.
)

scheduler = dict(
    type='MultiStepLR',
    milestones=[100],
    gamma=0.1
)

criterion = dict(
    loss_focal=20,
    loss_dice=1,
    loss_iou=1
)
