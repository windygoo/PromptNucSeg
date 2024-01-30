# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn.functional as F

from functools import partial
from .modeling import ImageEncoderViT, MaskDecoder, PromptEncoder, Sam, TwoWayTransformer


def build_sam_vit_h(cfg):
    return _build_sam(
        cfg,
        encoder_embed_dim=1280,
        encoder_depth=32,
        encoder_num_heads=16,
        encoder_global_attn_indexes=[7, 15, 23, 31],
    )


build_sam = build_sam_vit_h


def build_sam_vit_l(cfg):
    return _build_sam(
        cfg,
        encoder_embed_dim=1024,
        encoder_depth=24,
        encoder_num_heads=16,
        encoder_global_attn_indexes=[5, 11, 17, 23],
    )


def build_sam_vit_b(cfg):
    return _build_sam(
        cfg,
        encoder_embed_dim=768,
        encoder_depth=12,
        encoder_num_heads=12,
        encoder_global_attn_indexes=[2, 5, 8, 11],
    )


sam_model_registry = {
    "default": build_sam_vit_h,
    "vit_h": build_sam_vit_h,
    "vit_l": build_sam_vit_l,
    "vit_b": build_sam_vit_b,
}


def _build_sam(
        cfg,
        encoder_embed_dim,
        encoder_depth,
        encoder_num_heads,
        encoder_global_attn_indexes,
):
    prompt_embed_dim = 256
    image_size = cfg.segmentor.img_size
    vit_patch_size = cfg.segmentor.patch_size

    image_embedding_size = image_size // vit_patch_size
    sam = Sam(
        image_encoder=ImageEncoderViT(
            depth=encoder_depth,
            embed_dim=encoder_embed_dim,
            img_size=image_size,
            mlp_ratio=4,
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            num_heads=encoder_num_heads,
            patch_size=vit_patch_size,
            qkv_bias=True,
            use_rel_pos=True,
            global_attn_indexes=encoder_global_attn_indexes,
            window_size=14,
            out_chans=prompt_embed_dim,
        ),
        prompt_encoder=PromptEncoder(
            embed_dim=prompt_embed_dim,
            image_embedding_size=(image_embedding_size, image_embedding_size),
            input_image_size=(image_size, image_size),
            mask_in_chans=16,
        ),
        mask_decoder=MaskDecoder(
            num_classes=cfg.data.num_classes,
            num_multimask_outputs=3,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=prompt_embed_dim,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=prompt_embed_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
        ),
        num_classes=cfg.data.num_classes,
        multimask=cfg.segmentor.multimask
    )
    sam.eval()

    if cfg.segmentor.type.endswith("B"):
        ckpt = 'pretrained/sam_vit_b_01ec64.pth'
    elif cfg.segmentor.type.endswith("L"):
        ckpt = 'pretrained/sam_vit_l_0b3195.pth'
    elif cfg.segmentor.type.endswith("H"):
        ckpt = 'pretrained/sam_vit_h_4b8939.pth'
    else:
        raise NotImplementedError(f"Unknown model type: {cfg.segmentor.type}")

    with open(ckpt, "rb") as f:
        pretrained_state_dict = torch.load(f, map_location='cpu')

        model_state_dict = sam.state_dict()
        updated_state_dict = {k: v for k, v in pretrained_state_dict.items() if k in model_state_dict}
        model_state_dict.update(updated_state_dict)

        sam.load_state_dict(model_state_dict)

    return sam
