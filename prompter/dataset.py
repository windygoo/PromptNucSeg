import json
import os

import scipy.io
import torch
import numpy as np
import albumentations as A

from skimage import io
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler


def read_from_json(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


class DataFolder(Dataset):
    def __init__(
            self,
            cfg,
            mode
    ):
        anno_json = read_from_json(f'datasets/{cfg.data.name}/{mode}.json')
        self.classes = anno_json.pop('classes')
        self.data = anno_json
        self.img_paths = list(anno_json.keys())
        self.keys = ['image', 'keypoints'] + [f'keypoints{i}' for i in range(1, cfg.data.num_classes)] + ['mask']

        self.phase = mode
        self.dataset = cfg.data.name

        additional_targets = {}
        for i in range(1, cfg.data.num_classes):
            additional_targets.update({'keypoints%d' % i: 'keypoints'})

        self.transform = A.Compose(
            [getattr(A, tf_dict.pop('type'))(**tf_dict) for tf_dict in cfg.data.get(mode).transform] + [ToTensorV2()],
            p=1, keypoint_params=A.KeypointParams(format='xy'),
            additional_targets=additional_targets, is_check_shapes=False
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        assert index <= len(self), 'index range error'

        index = index % len(self.data)

        img_path = self.img_paths[index]

        values = ([io.imread(f'../segmentor/{img_path}')[..., :3]] +
                  [np.array(point).reshape(-1, 2) for point in self.data[img_path]])

        if self.dataset == 'kumar':
            mask_path = f'{img_path[:-4].replace("images", "labels")}.npy'
            sub_paths = mask_path.split('/')
            sub_paths[-2] += '_ins'
            mask_path = '/'.join(sub_paths)
            mask = np.load(f'../segmentor/{mask_path}')
        elif self.dataset == 'cpm17':
            mask = scipy.io.loadmat(f'../segmentor/{img_path[:-4].replace("Images", "Labels")}.mat')['inst_map']
        else:
            mask = np.load(f'../segmentor/{img_path.replace("Images", "Masks")[:-4]}.npy', allow_pickle=True)[()][
                        'inst_map']
        mask = mask.astype(float)

        values.append(mask)

        ori_shape = values[0].shape[:2]
        sample = dict(zip(self.keys, values))
        res = self.transform(**sample)
        res = list(res.values())

        img = res[0]
        labels = []
        for i in range(1, len(res) - 1):
            res[i] = torch.tensor(res[i])
            labels.append(torch.full((len(res[i]),), i - 1))
        mask = res[-1]
        return img, torch.cat(res[1:-1]), torch.cat(labels), mask, torch.as_tensor(ori_shape)
