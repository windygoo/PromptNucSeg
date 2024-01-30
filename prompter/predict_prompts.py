import os.path

import torch
import numpy as np
from tqdm import tqdm
from skimage import io
from utils import predict, mkdir
from models.dpa_p2pnet import build_model

from main import parse_args
from mmengine.config import Config

args = parse_args()
cfg = Config.fromfile(f'config/{args.config}')

dataset = cfg.data.name
device = torch.device(args.device)

model = build_model(cfg)
# ckpt = torch.load(f'checkpoint/{args.resume}/best.pth', map_location='cpu')
ckpt = torch.load(f'{args.resume}', map_location='cpu')
pretrained_state_dict = ckpt['model']

model.load_state_dict(pretrained_state_dict)
model.eval()
model.to(device)

import albumentations as A
from albumentations.pytorch import ToTensorV2

transform = A.Compose([
    A.PadIfNeeded(min_height=None, min_width=None, pad_height_divisor=cfg.prompter.space,
                  pad_width_divisor=cfg.prompter.space, position="top_left", p=1),
    A.Normalize(),
    ToTensorV2()
], p=1)


def process_files(files):
    for file in tqdm(files):
        img = io.imread(f'../segmentor/{file}')[..., :3]

        image = transform(image=img)['image'].unsqueeze(0).to(device)

        points, scores, classes, _ = predict(
            model,
            image,
            ori_shape=img.shape[:2],
            nms_thr=cfg.test.nms_thr,
            filtering=cfg.test.filtering
        )

        save_content = np.concatenate([points, classes[:, None]], axis=-1)

        np.save(
            f'../segmentor/prompts/{dataset}/{file.split("/")[-1][:-4]}',
            save_content
        )


mkdir(f'../segmentor/prompts/{dataset}')


test_files = np.load(f'../segmentor/datasets/{dataset}_test_files.npy')
process_files(test_files)

try:
    val_files = np.load(f'../segmentor/datasets/{dataset}_val_files.npy')
    process_files(val_files)

except FileNotFoundError:
    pass
