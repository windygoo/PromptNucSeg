import torch
import torch.nn.functional as F

from torch import nn
from matcher import build_matcher
from pytorch_toolbelt.losses import BinaryFocalLoss
from utils import is_dist_avail_and_initialized, get_world_size


class Criterion(nn.Module):
    def __init__(self, num_classes, matcher, class_weight, loss_weight, reg_loss_type='l2'):
        super().__init__()
        self.matcher = matcher
        self.num_classes = num_classes
        self.loss_weight = loss_weight
        self.class_weight = class_weight
        self.reg_loss_type = reg_loss_type

        self.focal_loss = BinaryFocalLoss()

    def loss_reg(self, outputs, targets, indices, num_points):
        """ Regression loss """
        idx = self._get_src_permutation_idx(indices)
        src_points = outputs['pred_coords'][idx]

        target_points = torch.cat([gt_points[J] for gt_points, (_, J) in zip(targets['gt_points'], indices)], dim=0)

        if self.reg_loss_type == 'l2':
            loss_pnt = F.mse_loss(src_points, target_points, reduction='none')
        else:
            loss_pnt = F.l1_loss(src_points, target_points, reduction='none')

        loss_dict = {'loss_reg': loss_pnt.sum() / (num_points + 1e-7)}
        return loss_dict

    def loss_cls(self, outputs, targets, indices, num_points):
        """Classification loss """
        idx = self._get_src_permutation_idx(indices)
        src_logits = outputs['pred_logits']

        target_classes = torch.full(src_logits.shape[:2], self.num_classes, dtype=torch.long, device=src_logits.device)
        target_classes_o = torch.cat([cls[J] for cls, (_, J) in zip(targets['gt_labels'], indices)])
        target_classes[idx] = target_classes_o

        loss_cls = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.class_weight)
        loss_dict = {'loss_cls': loss_cls}

        return loss_dict

    def loss_mask(self, outputs, targets, indices, num_points):
        pred_masks = outputs['pred_masks']
        gt_masks = targets['gt_masks']

        loss_mask = self.focal_loss(pred_masks.squeeze(1), gt_masks)
        # loss_mask = self.focal_loss(pred_masks, gt_masks)
        loss_dict = {'loss_mask': loss_mask}

        # regularization
        # prior = torch.ones(args.num_class)/args.num_class
        # prior = prior.cuda()
        # pred_mean = torch.softmax(logits, dim=1).mean(0)
        # penalty = torch.sum(prior*torch.log(prior/pred_mean))

        return loss_dict

    @staticmethod
    def _get_src_permutation_idx(indices):
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def forward(self, outputs, targets, epoch):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
             args: configuration
        """
        num_points = sum(targets['gt_nums'])
        num_points = torch.as_tensor([num_points], dtype=torch.float, device=outputs['pred_logits'].device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_points)
        num_points = torch.clamp(num_points / get_world_size(), min=1).item()

        losses = {}
        loss_map = {
            'loss_reg': self.loss_reg,
            'loss_cls': self.loss_cls,
            'loss_mask': self.loss_mask
        }

        indices = self.matcher(outputs, targets)
        for loss_func in loss_map.values():
            losses.update(loss_func(outputs, targets, indices, num_points))

        weight_dict = self.loss_weight
        for k in losses:
            assert k in weight_dict
            losses[k] *= weight_dict[k](epoch)

        return losses


def build_criterion(cfg, device):
    class_weight = torch.ones(cfg.data.num_classes + 1, dtype=torch.float).to(device)
    class_weight[-1] = cfg.criterion.eos_coef

    loss_weight = {
        'loss_cls': lambda epoch: cfg.criterion.cls_loss_coef,
        'loss_reg': lambda epoch: cfg.criterion.reg_loss_coef,
        'loss_mask': lambda epoch: cfg.criterion.mask_loss_coef
    }

    matcher = build_matcher(cfg)
    criterion = Criterion(
        cfg.data.num_classes,
        matcher,
        class_weight=class_weight,
        loss_weight=loss_weight
    )

    return criterion
