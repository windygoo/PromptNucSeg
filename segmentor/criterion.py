import torch
import torch.nn as nn
from pytorch_toolbelt.losses import BinaryFocalLoss, DiceLoss


class MaskIoULoss(nn.Module):

    def __init__(self, ):
        super(MaskIoULoss, self).__init__()

    def forward(self, pred_mask, ground_truth_mask, pred_iou):
        """
        pred_mask: [B, 1, H, W]
        ground_truth_mask: [B, 1, H, W]
        pred_iou: [B]
        """

        p = torch.sigmoid(pred_mask[:, 0])
        intersection = torch.sum(p * ground_truth_mask, dim=(1, 2))
        union = torch.sum(p, dim=(1, 2)) + torch.sum(ground_truth_mask, dim=(1, 2)) - intersection
        iou = (intersection + 1e-7) / (union + 1e-7)

        iou_loss = torch.mean((iou - pred_iou) ** 2)
        return iou_loss


class Criterion(nn.Module):
    def __init__(self, loss_weight):
        super().__init__()
        self.loss_weight = loss_weight

        self.focal_loss = BinaryFocalLoss()
        self.dice_loss = DiceLoss('binary')
        self.iou_loss = MaskIoULoss()

    def forward(
            self,
            outputs,
            true,
    ):
        pred = outputs['pred_masks'].unsqueeze(1)
        pred_iou = outputs['pred_ious']

        loss_dict = {
            'loss_focal': self.dice_loss(pred, true),
            'loss_dice': self.focal_loss(pred, true.unsqueeze(1)),
            'loss_iou': self.iou_loss(pred, true.float(), pred_iou)
        }

        for k in loss_dict:
            loss_dict[k] *= self.loss_weight[k]

        return loss_dict


def build_criterion(cfg):
    loss_weight = {
        'loss_focal': cfg.criterion.loss_focal,
        'loss_dice': cfg.criterion.loss_dice,
        'loss_iou': cfg.criterion.loss_iou,
    }

    criterion = Criterion(
        loss_weight=loss_weight
    )

    return criterion
