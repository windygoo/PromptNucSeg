"""
Mostly copy-paste from DETR (https://github.com/facebookresearch/detr).
"""
import torch

from torch import nn
from scipy.optimize import linear_sum_assignment


class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cost_point, cost_class):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the foreground object
            cost_point: This is the relative weight of the L1 error of the points coordinates in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_point = cost_point

    @torch.no_grad()
    def forward(self, outputs, targets):
        """one-to-one label assignment"""
        bs, num_queries = outputs["pred_coords"].shape[:2]

        # Compute the regression cost.
        out_coords = outputs["pred_coords"].flatten(0, 1)
        cost_point = torch.cdist(out_coords.double(), torch.cat(targets['gt_points']).double(), p=2)

        # Compute the classification cost.
        out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)
        cost_class = - out_prob[:, torch.cat(targets['gt_labels'])]

        # Final cost matrix.
        C = self.cost_point * cost_point + self.cost_class * cost_class
        C = C.view(bs, num_queries, -1).detach().cpu()

        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(targets['gt_nums'], -1))]
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]


def build_matcher(cfg):
    return HungarianMatcher(
        cost_point=cfg.criterion.matcher.set_cost_point,
        cost_class=cfg.criterion.matcher.set_cost_class
    )
