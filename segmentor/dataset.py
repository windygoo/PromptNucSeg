import json

import torch
import random
import scipy.io
import numpy as np
import albumentations as A

from skimage import io
from torch.utils.data import Dataset
from albumentations.pytorch import ToTensorV2


class DataFolder(Dataset):
    def __init__(
            self,
            cfg,
            mode
    ):
        super(DataFolder, self).__init__()

        self.mode = mode
        dataset = cfg.data.name

        self.files = np.load(f'datasets/{dataset}_{mode}_files.npy')

        self.dataset = dataset

        self.transform = A.Compose(
            [getattr(A, tf_dict.pop('type'))(**tf_dict) for tf_dict in cfg.data.get(mode).transform]
            + [ToTensorV2()], p=1)
        self.num_mask_per_img = cfg.data.num_mask_per_img
        self.num_neg_prompt = cfg.data.num_neg_prompt

        if 'pannuke' in self.dataset:
            if mode == 'train':
                fid = self.dataset[-3]
            elif mode == 'val':
                fid = self.dataset[-2]
            else:
                fid = self.dataset[-1]

            self.files = [f'datasets/pannuke/Images/{fid}_{i}.png' for i in range(len(self.files))]
            self.types = np.load(f'datasets/pannuke/Fold {fid}/images/fold{fid}/types.npy')

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_path = self.files[idx]

        if self.dataset == 'kumar':
            mask_path = f'{img_path[:-4].replace("images", "labels")}.npy'
            sub_paths = mask_path.split('/')
            sub_paths[-2] += '_ins'
            mask_path = '/'.join(sub_paths)
        elif self.dataset == 'cpm17':
            mask_path = f'{img_path[:-4].replace("Images", "Labels")}.mat'
        else:
            mask_path = f'{img_path[:-4].replace("Images", "Masks")}.npy'

        img, mask = io.imread(img_path)[..., :3], load_maskfile(mask_path)

        if self.mode != 'train':
            res = self.transform(image=img)

            img, mask = res['image'], torch.as_tensor(mask)
            inst_map, type_map = mask[..., 0], mask[..., 1]
            ori_size = inst_map.shape

            img_name = img_path.split('/')[-1]
            prompt_points = np.load(f'prompts/{self.dataset}/{img_name[:-4]}.npy')
            prompt_points = torch.from_numpy(prompt_points).float()
            prompt_points, prompt_cell_types = prompt_points[..., :2].unsqueeze(1), prompt_points[..., -1]
            prompt_labels = torch.ones(prompt_points.shape[:2], dtype=torch.int)

            return img, inst_map, type_map, prompt_points, prompt_labels, prompt_cell_types, ori_size, idx

        res = self.transform(image=img, mask=mask)
        img, mask = list(res.values())

        inst_map, type_map = mask[..., 0], mask[..., 1]
        unique_pids = np.unique(inst_map)[1:]  # remove zero

        cell_num = len(unique_pids)

        if cell_num:
            all_points = []
            cell_types = []

            for pid in unique_pids:
                mask_single_cell = torch.eq(
                    inst_map,
                    pid
                )

                pt = random.choice(
                    torch.argwhere(mask_single_cell)
                )[None, [1, 0]]

                all_points.append(pt)

                assert type_map[pt[0, 1], pt[0, 0]] > 0
                cell_types.append(type_map[pt[0, 1], pt[0, 0]] - 1)

            all_points = torch.from_numpy(np.concatenate(all_points)).float()

            chosen_pids = np.random.choice(
                unique_pids,
                min(cell_num, self.num_mask_per_img),
                replace=False
            )

            inst_maps = []
            prompt_points = []
            for pid in chosen_pids:
                mask_single_cell = torch.eq(inst_map, pid)

                inst_maps.append(mask_single_cell)
                prompt_points.append(
                    random.choice(
                        torch.argwhere(mask_single_cell)
                    )[None, [1, 0]].float())

            prompt_points = torch.stack(prompt_points, dim=0)
            prompt_labels = torch.ones(prompt_points.shape[:2])
            cell_types = torch.as_tensor(cell_types)

            inst_map = torch.stack(inst_maps, dim=0)

            if self.num_neg_prompt:
                global_indices = [np.where(unique_pids == pid)[0][0] for pid in chosen_pids]

                prompt_points, prompt_labels = add_k_nearest_neg_prompt(
                    prompt_points,
                    global_indices,
                    all_points,
                    k=self.num_neg_prompt
                )
        else:
            prompt_points = torch.empty(0, (self.num_neg_prompt + 1), 2)
            prompt_labels = torch.empty(0, (self.num_neg_prompt + 1))
            all_points = torch.empty(0, 2)
            inst_map = torch.empty(0, 256, 256)
            cell_types = torch.empty(0)

        return img, inst_map.long(), prompt_points, prompt_labels, cell_types, all_points


def load_maskfile(mask_path: str):
    if 'pannuke' in mask_path:
        mask = np.load(mask_path, allow_pickle=True)
        inst_map = mask[()]["inst_map"].astype(np.int32)
        type_map = mask[()]["type_map"].astype(np.int32)

    elif 'cpm17' in mask_path:
        inst_map = scipy.io.loadmat(mask_path)['inst_map']
        type_map = (inst_map.copy() > 0).astype(float)

    else:
        inst_map = np.load(mask_path)
        type_map = (inst_map.copy() > 0).astype(float)

    mask = np.stack([inst_map, type_map], axis=-1)
    return mask


def add_k_nearest_neg_prompt(
        prompt_points,
        global_indices,
        all_points,
        k: int = 1
):
    if len(prompt_points) == 1:
        prompt_points = torch.cat([prompt_points, torch.zeros(1, k, 2)], dim=1)
        prompt_labels = torch.ones(prompt_points.shape[:2], dtype=torch.int)
        prompt_labels[0, 1] = -1
    else:
        all_points = all_points.view(-1, 2)
        dis = torch.cdist(all_points, all_points, p=2.0)
        dis = dis.fill_diagonal_(np.inf)

        available_num = min(k, len(prompt_points) - 1)
        neg_prompt_points = all_points[
                            torch.topk(dis[global_indices], available_num, dim=1, largest=False).indices, :
                            ]
        prompt_points = torch.cat(
            [prompt_points, neg_prompt_points, torch.zeros(len(prompt_points), k - available_num, 2)],
            dim=1
        )

        prompt_labels = torch.ones(prompt_points.shape[:2], dtype=torch.int)
        prompt_labels[:, 1:available_num + 1] = 0
        prompt_labels[:, available_num + 1:] = -1

    return prompt_points, prompt_labels
