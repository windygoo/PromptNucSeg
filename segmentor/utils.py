import torch
import random

import datetime
import errno
import os
import time
import numpy as np
import scipy.spatial as S
import torch.distributed as dist
import torchvision.transforms as T

from collections import defaultdict, deque

from torchvision.transforms.functional import hflip, vflip
from scipy.spatial.distance import directed_hausdorff as hausdorff


def train_collate_fn(batch):
    images, masks, prompt_points, prompt_labels, all_points, all_points_types, cell_nums = [[] for _ in range(7)]
    for x in batch:
        images.append(x[0])
        masks.append(x[1])
        prompt_points.append(x[2])
        prompt_labels.append(x[3])
        all_points_types.append(x[4])
        all_points.append(x[5])
        cell_nums.append(len(x[2]))

    return (torch.stack(images), torch.cat(masks), torch.cat(prompt_points), torch.cat(prompt_labels),
            all_points, all_points_types, torch.as_tensor(cell_nums))


def test_collate_fn(batch):
    (images, inst_maps, type_maps, prompt_points, prompt_labels, prompt_cell_types, cell_nums, ori_sizes, file_inds) = [
        [] for _ in range(9)]

    for x in batch:
        images.append(x[0])
        inst_maps.append(x[1])
        type_maps.append(x[2])
        prompt_points.append(x[3])
        prompt_labels.append(x[4])
        prompt_cell_types.append(x[5])
        cell_nums.append(len(x[3]))
        ori_sizes.append(x[6])
        file_inds.append(x[7])

    return (torch.stack(images), torch.stack(inst_maps), torch.stack(type_maps), torch.cat(prompt_points),
            torch.cat(prompt_labels), torch.cat(prompt_cell_types), torch.as_tensor(cell_nums),
            torch.as_tensor(ori_sizes), torch.as_tensor(file_inds))


def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        args.rank = 0
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}'.format(args.rank, args.dist_url), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def set_seed(args):
    seed = args.seed + get_rank()
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def pre_processing(img):
    trans = T.Compose([
        T.ToTensor(),
        T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    return trans(img).unsqueeze(0)


def point_nms(points, scores, classes, index=None, nms_thr=-1):
    _reserved = np.ones(len(points), dtype=bool)
    dis_matrix = S.distance_matrix(points, points)
    np.fill_diagonal(dis_matrix, np.inf)

    for idx in np.argsort(-scores):
        if _reserved[idx]:
            _reserved[dis_matrix[idx] <= nms_thr] = False

    points = points[_reserved]
    scores = scores[_reserved]
    classes = classes[_reserved]

    if index is not None:
        index = index[_reserved]
        return points, scores, classes, index
    else:
        return points, scores, classes


def reduce_dict(input_dict, average=True):
    """
    Args:
        input_dict (dict): all the values will be reduced
        average (bool): whether to do average or sum
    Reduce the values in the dictionary from all processes so that all processes
    have the averaged results. Returns a dict with the same fields as
    input_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return input_dict
    with torch.inference_mode():
        names = []
        values = []
        # sort the keys so that they are consistent across processes
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])
        values = torch.stack(values, dim=0)
        dist.all_reduce(values)
        if average:
            values /= world_size
        reduced_dict = {k: v for k, v in zip(names, values)}
    return reduced_dict


def all_gather(data):
    """
    Run all_gather on arbitrary picklable data (not necessarily tensors)
    Args:
        data: any picklable object
    Returns:
        list[data]: list of data gathered from each rank
    """
    world_size = get_world_size()
    if world_size == 1:
        return [data]
    data_list = [None] * world_size
    dist.all_gather_object(data_list, data)
    return data_list


@torch.no_grad()
def predict(
        model,
        image,
        ori_shape,
        nms_thr=-1,
        with_mask=False
):
    assert isinstance(image, (torch.Tensor, np.ndarray, list)), f'Invalid input type, expect torch.Tensor or ' \
                                                                f'numpy.ndarray or list, got {type(image)}'

    if isinstance(image, np.ndarray):
        image = pre_processing(image)
    elif isinstance(image, list):
        image = torch.cat([pre_processing(img) for img in image], dim=1)

    image = image.to(next(model.parameters()).device)

    ori_h, ori_w = ori_shape
    outputs = model(image)
    points = outputs['pred_coords'][0].cpu().numpy()
    scores = outputs['pred_logits'][0].softmax(-1).cpu().numpy()

    classes = scores.argmax(axis=-1)

    np.clip(points[:, 0], a_min=0, a_max=ori_w - 1, out=points[:, 0])
    np.clip(points[:, 1], a_min=0, a_max=ori_h - 1, out=points[:, 1])
    valid_flag = classes < (scores.shape[-1] - 1)

    points = points[valid_flag]
    scores = scores[valid_flag].max(1)
    classes = classes[valid_flag]

    index = np.where(valid_flag)[0]

    if with_mask:
        mask = outputs['pred_masks'][0, 0].cpu().numpy()
        mask = mask > 0

        valid_flag = mask[points.astype(int)[:, 1], points.astype(int)[:, 0]]

        points = points[valid_flag]
        scores = scores[valid_flag]
        classes = classes[valid_flag]
        index = np.where(valid_flag)[0]

    if len(points) and nms_thr > 0:
        points, scores, classes, index = point_nms(points, scores, classes, index, nms_thr=nms_thr)

    return points, scores, classes


def mkdir(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def reduce_across_processes(val):
    if not is_dist_avail_and_initialized():
        # nothing to sync, but we still convert to tensor for consistency with the distributed case.
        return torch.tensor(val)

    t = torch.tensor(val, device="cuda")
    dist.barrier()
    dist.all_reduce(t)
    return t


class SmoothedValue:
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{value:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        t = reduce_across_processes([self.count, self.total])
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median, avg=self.avg, global_avg=self.global_avg, max=self.max, value=self.value
        )


class MetricLogger:
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{attr}'")

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(f"{name}: {str(meter)}")
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ""
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt="{avg:.4f}")
        data_time = SmoothedValue(fmt="{avg:.4f}")
        space_fmt = ":" + str(len(str(len(iterable)))) + "d"
        if torch.cuda.is_available():
            log_msg = self.delimiter.join(
                [
                    header,
                    "[{0" + space_fmt + "}/{1}]",
                    "eta: {eta}",
                    "{meters}",
                    "time: {time}",
                    "data: {data}",
                    "max mem: {memory:.0f}",
                ]
            )
        else:
            log_msg = self.delimiter.join(
                [header, "[{0" + space_fmt + "}/{1}]", "eta: {eta}", "{meters}", "time: {time}", "data: {data}"]
            )
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(
                        log_msg.format(
                            i,
                            len(iterable),
                            eta=eta_string,
                            meters=str(self),
                            time=str(iter_time),
                            data=str(data_time),
                            memory=torch.cuda.max_memory_allocated() / MB,
                        )
                    )
                else:
                    print(
                        log_msg.format(
                            i, len(iterable), eta=eta_string, meters=str(self), time=str(iter_time), data=str(data_time)
                        )
                    )
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print(f"{header} Total time: {total_time_str} ({total_time / len(iterable):.4f} s / it)")


def remap_label(pred, by_size=False):
    """
    Rename all instance id so that the id is contiguous i.e [0, 1, 2, 3]
    not [0, 2, 4, 6]. The ordering of instances (which one comes first)
    is preserved unless by_size=True, then the instances will be reordered
    so that bigger nucler has smaller ID

    Args:
        pred    : the 2d array contain instances where each instances is marked
                  by non-zero integer
        by_size : renaming with larger nuclei has smaller id (on-top)
    """
    pred_id = list(np.unique(pred))
    if 0 in pred_id:
        pred_id.remove(0)
    if len(pred_id) == 0:
        return pred  # no label
    if by_size:
        pred_size = []
        for inst_id in pred_id:
            size = (pred == inst_id).sum()
            pred_size.append(size)
        # sort the id by size in descending order
        pair_list = zip(pred_id, pred_size)
        pair_list = sorted(pair_list, key=lambda x: x[1], reverse=True)
        pred_id, pred_size = zip(*pair_list)

    new_pred = np.zeros(pred.shape, np.int32)
    for idx, inst_id in enumerate(pred_id):
        new_pred[pred == inst_id] = idx + 1
    return new_pred


def crop_with_overlap(
        img,
        split_width,
        split_height,
        overlap
):
    def start_points(
            size,
            split_size,
            overlap
    ):
        points = [0]
        counter = 1
        stride = 256 - overlap
        while True:
            pt = stride * counter
            if pt + split_size >= size:
                if split_size == size:
                    break
                points.append(size - split_size)
                break
            else:
                points.append(pt)
            counter += 1
        return points

    _, img_h, img_w = img.shape

    X_points = start_points(img_w, split_width, overlap)
    Y_points = start_points(img_h, split_height, overlap)

    crop_boxes = []
    for y in Y_points:
        for x in X_points:
            crop_boxes.append([x, y, min(x + split_width, img_w), min(y + split_height, img_h)])
    return np.asarray(crop_boxes)


from segment_anything.utils.amg import (
    MaskData,
    area_from_rle,
    batch_iterator,
    batched_mask_to_box,
    box_xyxy_to_xywh,
    calculate_stability_score,
    is_box_near_crop_edge,
    mask_to_rle_pytorch,
    remove_small_regions,
    rle_to_mask,
    uncrop_boxes_xyxy,
    uncrop_masks,
    uncrop_points,
)
from torchvision.ops.boxes import batched_nms

import torch.nn as nn


@torch.inference_mode()
def inference(
        model: nn.Module,
        image: torch.Tensor,
        crop_box: np.ndarray,
        ori_size: tuple,
        prompt_points: torch.Tensor,
        prompt_labels: torch.Tensor,
        prompt_cell_types: torch.Tensor,
        points_per_batch: int = 256,
        mask_threshold: float = .0,
        pred_iou_thresh: float = 0.88,
        stability_score_thresh: float = 0.95,
        stability_score_offset: float = 1.0,
        box_nms_thresh: float = 1.0,
        min_mask_region_area: int = 0,
        inds=None,
        tta=False
):
    orig_h, orig_w = ori_size

    # Generate masks for this crop in batches
    mask_data = MaskData()
    for (points, labels, cell_types, sub_inds) in batch_iterator(points_per_batch, prompt_points, prompt_labels,
                                                                 prompt_cell_types, inds):
        outputs = model(
            image,
            points,
            labels,
            torch.as_tensor([len(points)]).to(points.device),
        )

        if tta:  # used in FullNet and CDNet

            points1 = points.clone()
            points1[..., 0] = 255 - points1[..., 0]

            outputs1 = model(
                hflip(image),
                points1,
                labels,
                torch.as_tensor([len(points)]).to(points.device),
            )

            points2 = points.clone()
            points2[..., 1] = 255 - points2[..., 1]
            outputs2 = model(
                vflip(image),
                points2,
                labels,
                torch.as_tensor([len(points)]).to(points.device),
            )

            theta = np.radians(90)

            c, s = np.cos(theta), np.sin(theta)
            R = np.array(((c, -s), (s, c)))
            R = torch.from_numpy(R).to(points.device).float()
            center = torch.tensor([[128, 128]], device=points.device, dtype=torch.float)
            points3 = points.clone()
            points3 = torch.matmul((points3 - center), R) + center

            outputs3 = model(
                torch.rot90(image, 1, [2, 3]),
                points3,
                labels,
                torch.as_tensor([len(points)]).to(points.device),
            )
            outputs3["pred_masks"] = torch.rot90(outputs3["pred_masks"], 3, [1, 2])

            theta = np.radians(180)

            c, s = np.cos(theta), np.sin(theta)
            R = np.array(((c, -s), (s, c)))
            R = torch.from_numpy(R).to(points.device).float()
            points4 = points.clone()
            points4 = torch.matmul((points4 - center), R) + center

            outputs4 = model(
                torch.rot90(image, 2, [2, 3]),
                points4,
                labels,
                torch.as_tensor([len(points)]).to(points.device),
            )
            outputs4["pred_masks"] = torch.rot90(outputs4["pred_masks"], 2, [1, 2])

            theta = np.radians(270)

            c, s = np.cos(theta), np.sin(theta)
            R = np.array(((c, -s), (s, c)))
            R = torch.from_numpy(R).to(points.device).float()
            points5 = points.clone()
            points5 = torch.matmul((points5 - center), R) + center

            outputs5 = model(
                torch.rot90(image, 3, [2, 3]),
                points5,
                labels,
                torch.as_tensor([len(points)]).to(points.device),
            )
            outputs5["pred_masks"] = torch.rot90(outputs5["pred_masks"], 1, [1, 2])

            masks = (outputs["pred_masks"] + hflip(outputs1["pred_masks"]) + vflip(outputs2["pred_masks"]) +
                     outputs3["pred_masks"] + outputs4["pred_masks"] + outputs5["pred_masks"])
            masks = masks / 6

            iou_preds = (outputs["pred_ious"] + outputs1["pred_ious"] + outputs2["pred_ious"] + outputs3["pred_ious"] +
                         outputs4["pred_ious"] + outputs5["pred_ious"])
            iou_preds = iou_preds / 6

            # points3 = points.clone()
            # points3 = 255 - points3
            # outputs3 = model(
            #     vflip(hflip(image)),
            #     points3,
            #     labels,
            #     torch.as_tensor([len(points)]).to(points.device),
            # )
            #
            # theta = np.radians(90)
            #
            # c, s = np.cos(theta), np.sin(theta)
            # R = np.array(((c, -s), (s, c)))
            # R = torch.from_numpy(R).to(points.device).double()
            # center = torch.tensor([[128, 128]], device=points.device)
            # points4 = points.clone()
            # points4 = torch.matmul((points4 - center), R) + center
            #
            # outputs4 = model(
            #     torch.rot90(image, 1, [2, 3]),
            #     points4,
            #     labels,
            #     torch.as_tensor([len(points)]).to(points.device),
            # )
            # outputs4["pred_masks"] = torch.rot90(outputs4["pred_masks"], 3, [1, 2])
            #
            # points5 = points4.clone()
            # points5[..., 0] = 255 - points5[..., 0]
            # outputs5 = model(
            #     hflip(torch.rot90(image, 1, [2, 3])),
            #     points5,
            #     labels,
            #     torch.as_tensor([len(points)]).to(points.device),
            # )
            # outputs5["pred_masks"] = torch.rot90(hflip(outputs5["pred_masks"]), 3, [1, 2])
            #
            # points6 = points4.clone()
            # points6[..., 1] = 255 - points6[..., 1]
            # outputs6 = model(
            #     vflip(torch.rot90(image, 1, [2, 3])),
            #     points6,
            #     labels,
            #     torch.as_tensor([len(points)]).to(points.device),
            # )
            # outputs6["pred_masks"] = torch.rot90(vflip(outputs6["pred_masks"]), 3, [1, 2])
            #
            # points7 = points4.clone()
            # points7 = 255 - points7
            # outputs7 = model(
            #     vflip(hflip(torch.rot90(image, 1, [2, 3]))),
            #     points7,
            #     labels,
            #     torch.as_tensor([len(points)]).to(points.device),
            # )
            # outputs7["pred_masks"] = torch.rot90(hflip(vflip(outputs7["pred_masks"])), 3, [1, 2])
            #
            # masks = (outputs["pred_masks"] + hflip(outputs1["pred_masks"]) + vflip(outputs2["pred_masks"]) +
            #          hflip(vflip(outputs3["pred_masks"])) + outputs4["pred_masks"] + outputs5["pred_masks"] +
            #          outputs6["pred_masks"] + outputs7["pred_masks"])
            # masks = masks / 8
            #
            # iou_preds = (outputs["pred_ious"] + outputs1["pred_ious"] + outputs2["pred_ious"] + outputs3["pred_ious"] +
            #              outputs4["pred_ious"] + outputs5["pred_ious"] + outputs6["pred_ious"] + outputs7["pred_ious"])
            # iou_preds = iou_preds / 8
        else:
            masks = outputs["pred_masks"]
            iou_preds = outputs["pred_ious"]

        # Serialize predictions and store in MaskData
        batch_data = MaskData(
            masks=masks,
            iou_preds=iou_preds,
            points=points,
            categories=cell_types,
            inds=sub_inds
        )
        del masks

        # Filter by predicted IoU
        if pred_iou_thresh > 0.0:
            keep_mask = batch_data["iou_preds"] > pred_iou_thresh
            batch_data.filter(keep_mask)

        # Calculate stability score
        batch_data["stability_score"] = calculate_stability_score(
            batch_data["masks"], mask_threshold, stability_score_offset
        )
        if stability_score_thresh > 0.0:
            keep_mask = batch_data["stability_score"] >= stability_score_thresh
            batch_data.filter(keep_mask)

        # Threshold masks and calculate boxes
        batch_data["masks"] = batch_data["masks"] > mask_threshold
        batch_data["boxes"] = batched_mask_to_box(batch_data["masks"])

        # Filter boxes that touch crop boundaries
        keep_mask = ~is_box_near_crop_edge(batch_data["boxes"], crop_box, [0, 0, orig_w, orig_h], atol=7)
        # print(keep_mask.shape, batch_data["masks"].shape, batch_data["boxes"].shape)
        if not torch.all(keep_mask):
            batch_data.filter(keep_mask)

        # Compress to RLE
        batch_data["masks"] = uncrop_masks(batch_data["masks"], crop_box, orig_h, orig_w)
        batch_data["rles"] = mask_to_rle_pytorch(batch_data["masks"])
        del batch_data["masks"]

        mask_data.cat(batch_data)
        del batch_data

    # Remove duplicates within this crop.
    keep_by_nms = batched_nms(
        mask_data["boxes"].float(),
        mask_data["iou_preds"],
        torch.zeros_like(mask_data["boxes"][:, 0]),  # apply cross categories
        iou_threshold=box_nms_thresh
    )
    mask_data.filter(keep_by_nms)

    # Return to the original image frame
    mask_data["boxes"] = uncrop_boxes_xyxy(mask_data["boxes"], crop_box)
    mask_data["points"] = uncrop_points(mask_data["points"], crop_box)
    mask_data["crop_boxes"] = torch.tensor([crop_box for _ in range(len(mask_data["rles"]))])

    # Filter small disconnected regions and holes in masks
    if min_mask_region_area > 0:
        crop_nms_thresh = 0.7
        mask_data = postprocess_small_regions(
            mask_data,
            min_mask_region_area,
            max(box_nms_thresh, crop_nms_thresh),
        )

    mask_data["segmentations"] = [rle_to_mask(rle) for rle in mask_data["rles"]]

    # Write mask records
    curr_anns = []
    for idx in range(len(mask_data["segmentations"])):
        ann = {
            "segmentation": mask_data["segmentations"][idx],
            "area": area_from_rle(mask_data["rles"][idx]),
            "bbox": mask_data["boxes"][idx].tolist(),
            "predicted_iou": mask_data["iou_preds"][idx].item(),
            "point_coords": [mask_data["points"][idx].tolist()],
            "stability_score": mask_data["stability_score"][idx].item(),
            "crop_box": box_xyxy_to_xywh(mask_data["crop_boxes"][idx]).tolist(),
            'categories': mask_data['categories'][idx].tolist(),
            'inds': mask_data['inds'][idx].tolist()
        }
        curr_anns.append(ann)

    return curr_anns


def postprocess_small_regions(
        mask_data: MaskData, min_area: int, nms_thresh: float
) -> MaskData:
    """
    Removes small disconnected regions and holes in masks, then reruns
    box NMS to remove any new duplicates.

    Edits mask_data in place.

    Requires open-cv as a dependency.
    """
    if len(mask_data["rles"]) == 0:
        return mask_data

    # Filter small disconnected regions and holes
    new_masks = []
    scores = []

    for rle in mask_data["rles"]:
        mask = rle_to_mask(rle)

        mask, changed = remove_small_regions(mask, min_area, mode="holes")
        unchanged = not changed
        mask, changed = remove_small_regions(mask, min_area, mode="islands")
        unchanged = unchanged and not changed

        new_masks.append(torch.as_tensor(mask).unsqueeze(0))
        # Give score=0 to changed masks and score=1 to unchanged masks
        # so NMS will prefer ones that didn't need postprocessing
        scores.append(float(unchanged))

    # Recalculate boxes and remove any new duplicates
    masks = torch.cat(new_masks, dim=0)
    boxes = batched_mask_to_box(masks)
    keep_by_nms = batched_nms(
        boxes.float(),
        torch.as_tensor(scores),
        torch.zeros_like(boxes[:, 0]),  # categories
        iou_threshold=nms_thresh,
    )

    # Only recalculate RLEs for masks that have changed
    for i_mask in keep_by_nms:
        if scores[i_mask] == 0.0:
            mask_torch = masks[i_mask].unsqueeze(0)
            mask_data["rles"][i_mask] = mask_to_rle_pytorch(mask_torch)[0]
            mask_data["boxes"][i_mask] = boxes[i_mask]  # update res directly
    mask_data.filter(keep_by_nms)

    return mask_data


def get_tp(
        pred_points,
        pred_scores,
        gd_points,
        thr=12,
        return_index=False
):
    sorted_pred_indices = np.argsort(-pred_scores)
    sorted_pred_points = pred_points[sorted_pred_indices]

    unmatched = np.ones(len(gd_points), dtype=bool)
    matched_index = np.full(len(gd_points), -1)

    dis = S.distance_matrix(sorted_pred_points, gd_points)

    for i in range(len(pred_points)):
        min_index = dis[i, unmatched].argmin()
        if dis[i, unmatched][min_index] <= thr:
            matched_index[np.where(unmatched)[0][min_index]] = i
            unmatched[np.where(unmatched)[0][min_index]] = False

        if not np.any(unmatched):
            break

    if return_index:
        return sum(~unmatched), matched_index
    else:
        return sum(~unmatched)


def binarize(x):
    """
    convert multichannel (multiclass) instance segmetation tensor
    to binary instance segmentation (bg and nuclei),

    :param x: B*B*C (for PanNuke 256*256*5 )
    :return: Instance segmentation
    """
    out = np.zeros([x.shape[0], x.shape[1]])
    count = 1
    for i in range(x.shape[2]):
        x_ch = x[:, :, i]
        unique_vals = np.unique(x_ch)
        unique_vals = unique_vals.tolist()
        unique_vals.remove(0)
        for j in unique_vals:
            x_tmp = x_ch == j
            x_tmp_c = 1 - x_tmp
            out *= x_tmp_c
            out += count * x_tmp
            count += 1
    out = out.astype("int32")
    return out
