# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import numpy as np
from torch import nn
from torch.nn import functional as F

from typing import Any, Dict, List, Tuple

from .image_encoder import ImageEncoderViT
from .mask_decoder import MaskDecoder
from .prompt_encoder import PromptEncoder
from .common import LayerNorm2d

from utils import point_nms


class Sam(nn.Module):
    mask_threshold: float = 0.0
    image_format: str = "RGB"

    def __init__(
            self,
            image_encoder: ImageEncoderViT,
            prompt_encoder: PromptEncoder,
            mask_decoder: MaskDecoder,
            num_classes: int,
            multimask: bool,
    ) -> None:
        """
        SAM predicts object masks from an image and input prompts.

        Arguments:
          image_encoder (ImageEncoderViT): The backbone used to encode the
            image into image embeddings that allow for efficient mask prediction.
          prompt_encoder (PromptEncoder): Encodes various types of input prompts.
          mask_decoder (MaskDecoder): Predicts masks from the image embeddings
            and encoded prompts.
        """
        super().__init__()
        self.image_encoder = image_encoder
        self.prompt_encoder = prompt_encoder
        self.mask_decoder = mask_decoder
        self.multimask = multimask

    @staticmethod
    def get_anchor_points(images, space):
        bs, _, h, w = images.shape

        anchors = np.stack(
            np.meshgrid(
                np.arange(np.ceil(w / space)) + 0.5,
                np.arange(np.ceil(h / space)) + 0.5
            ), axis=-1) * space

        anchors = torch.from_numpy(anchors).float().to(images.device)
        return anchors.repeat(bs, 1, 1, 1).unsqueeze(-2)

    @property
    def device(self) -> Any:
        return self.pixel_mean.device

    def forward(
            self,
            images,
            prompt_points=None,
            prompt_labels=None,
            cell_nums=None,
            only_det=False
    ):
        # image_embeddings, outputs = self.image_encoder(images)
        image_embeddings = self.image_encoder(images)

        # if only_det:
        #     return outputs

        # if cell_nums.sum() == 0:
        #     return outputs

        outputs = {}
        if prompt_points is not None:
            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points=(prompt_points, prompt_labels),
                boxes=None,
                masks=None,
            )

            low_res_masks, iou_predictions = self.mask_decoder(
            # low_res_masks, iou_predictions, cls_predictions = self.mask_decoder(
                image_embeddings=image_embeddings,
                image_pe=self.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                cell_nums=cell_nums,
                multimask_output=self.multimask
            )

            values, indices = torch.max(iou_predictions, dim=1)
            iou_predictions = values

            if self.multimask:
                low_res_masks = low_res_masks[torch.arange(len(iou_predictions)), indices].unsqueeze(1)

            masks = F.interpolate(
                low_res_masks,
                images.shape[-2:],
                mode="bilinear",
                align_corners=False)[:, 0]

            outputs.update(
                pred_masks=masks,
                pred_ious=iou_predictions,
                # pred_logs=cls_predictions
            )

        return outputs

    def postprocess_masks(
            self,
            masks: torch.Tensor,
            # input_size: Tuple[int, ...],
            # original_size: Tuple[int, ...],
    ) -> torch.Tensor:
        """
        Remove padding and upscale masks to the original image size.

        Arguments:
          masks (torch.Tensor): Batched masks from the mask_decoder,
            in BxCxHxW format.
          input_size (tuple(int, int)): The size of the image input to the
            model, in (H, W) format. Used to remove padding.
          original_size (tuple(int, int)): The original size of the image
            before resizing for input to the model, in (H, W) format.

        Returns:
          (torch.Tensor): Batched masks in BxCxHxW format, where (H, W)
            is given by original_size.
        """
        masks = F.interpolate(
            masks,
            (self.image_encoder.img_size, self.image_encoder.img_size),
            mode="bilinear",
            align_corners=False,
        )
        # masks = masks[..., : input_size[0], : input_size[1]]
        # masks = F.interpolate(masks, original_size, mode="bilinear", align_corners=False)
        return masks

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        # Normalize colors
        x = (x - self.pixel_mean) / self.pixel_std

        # Pad
        h, w = x.shape[-2:]
        padh = self.image_encoder.img_size - h
        padw = self.image_encoder.img_size - w
        x = F.pad(x, (0, padw, 0, padh))  # 填充右边和下边
        return x


# Define block
class ResidualBlock(nn.Module):
    def __init__(self, channel_num):
        super(ResidualBlock, self).__init__()

        # TODO: 3x3 convolution -> relu
        # the input and output channel number is channel_num
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(channel_num, channel_num, 3, padding=1),
            LayerNorm2d(channel_num),
            nn.GELU(),
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(channel_num, channel_num, 3, padding=1),
            LayerNorm2d(channel_num),
        )
        self.gelu = nn.GELU()

    def forward(self, x):
        # TODO: forward
        residual = x
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = x + residual
        out = self.gelu(x)
        return out
