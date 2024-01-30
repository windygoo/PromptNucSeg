# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import timm
import copy
import torch
import numpy as np
import torch.nn.functional as F

from torch import nn
from models.fpn import FPN


class Backbone(nn.Module):
    def __init__(
            self,
            cfg
    ):
        super(Backbone, self).__init__()

        backbone = timm.create_model(
            **cfg.prompter.backbone
        )

        self.backbone = backbone

        self.neck = FPN(
            **cfg.prompter.neck
        )

        new_dict = copy.copy(cfg.prompter.neck)
        new_dict['num_outs'] = 1
        self.neck1 = FPN(
            **new_dict
        )

    def forward(self, images):
        x = self.backbone(images)
        return list(self.neck(x)), self.neck1(x)[0]


class AnchorPoints(nn.Module):
    def __init__(self, space=16):
        super(AnchorPoints, self).__init__()
        self.space = space

    def forward(self, images):
        bs, _, h, w = images.shape
        anchors = np.stack(
            np.meshgrid(
                np.arange(np.ceil(w / self.space)),
                np.arange(np.ceil(h / self.space))),
            -1) * self.space

        origin_coord = np.array([w % self.space or self.space, h % self.space or self.space]) / 2
        anchors += origin_coord

        anchors = torch.from_numpy(anchors).float().to(images.device)
        return anchors.repeat(bs, 1, 1, 1)


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, drop=0.1):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList()

        for n, k in zip([input_dim] + h, h):
            self.layers.append(nn.Linear(n, k))
            self.layers.append(nn.ReLU(inplace=True))
            self.layers.append(nn.Dropout(drop))
        self.layers.append(nn.Linear(hidden_dim, output_dim))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
        return x


class DPAP2PNet(nn.Module):
    """ This is the Proposal-aware P2PNet module that performs cell recognition """

    def __init__(
            self,
            backbone,
            num_levels,
            num_classes,
            dropout=0.1,
            space: int = 16,
            hidden_dim: int = 256,
            with_mask=False
    ):
        """
            Initializes the model.
        """
        super().__init__()
        self.backbone = backbone
        self.get_aps = AnchorPoints(space)
        self.num_levels = num_levels
        self.hidden_dim = hidden_dim
        self.with_mask = with_mask
        self.strides = [2 ** (i + 2) for i in range(self.num_levels)]

        self.deform_layer = MLP(hidden_dim, hidden_dim, 2, 2, drop=dropout)

        self.reg_head = MLP(hidden_dim, hidden_dim, 2, 2, drop=dropout)
        self.cls_head = MLP(hidden_dim, hidden_dim, 2, num_classes + 1, drop=dropout)

        self.conv = nn.Conv2d(hidden_dim * num_levels, hidden_dim, kernel_size=3, padding=1)

        self.mask_head = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.SyncBatchNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, 1, kernel_size=1, padding=1)
        )

    def forward(self,
                images):
        # extract features
        (feats, feats1), proposals = self.backbone(images), self.get_aps(images)

        feat_sizes = [torch.tensor(feat.shape[:1:-1], dtype=torch.float, device=proposals.device) for feat in feats]

        # DPP
        grid = (2.0 * proposals / self.strides[0] / feat_sizes[0] - 1.0)
        roi_features = F.grid_sample(feats[0], grid, mode='bilinear', align_corners=True)
        deltas2deform = self.deform_layer(roi_features.permute(0, 2, 3, 1))
        deformed_proposals = proposals + deltas2deform

        # MSD
        roi_features = []
        for i in range(self.num_levels):
            grid = (2.0 * deformed_proposals / self.strides[i] / feat_sizes[i] - 1.0)
            roi_features.append(F.grid_sample(feats[i], grid, mode='bilinear', align_corners=True))
        roi_features = torch.cat(roi_features, 1)

        roi_features = self.conv(roi_features).permute(0, 2, 3, 1)
        deltas2refine = self.reg_head(roi_features)
        pred_coords = deformed_proposals + deltas2refine

        pred_logits = self.cls_head(roi_features)

        output = {
            'pred_coords': pred_coords.flatten(1, 2),
            'pred_logits': pred_logits.flatten(1, 2),
            'pred_masks': F.interpolate(
                self.mask_head(feats1), size=images.shape[2:], mode='bilinear', align_corners=True)
        }

        return output


def build_model(cfg):
    backbone = Backbone(cfg)

    model = DPAP2PNet(
        backbone,
        num_levels=cfg.prompter.neck.num_outs,
        num_classes=cfg.data.num_classes,
        dropout=cfg.prompter.dropout,
        space=cfg.prompter.space,
        hidden_dim=cfg.prompter.hidden_dim
    )

    return model
