import json
import os

import cv2 as cv
import numpy as np
from skimage import io


def mkdir(path):
    try:
        os.makedirs(path)
    except OSError as e:
        return


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


mkdir('Images')
mkdir('Masks')

for i in range(1, 4):

    images = np.load(f'Fold {i}/images/fold{i}/images.npy')
    masks = np.load(f'Fold {i}/masks/fold{i}/masks.npy')

    for j in range(len(masks)):
        io.imsave(f'Images/{i}_{j}.png', images[j].astype(np.uint8), check_contrast=False)

        mask = masks[j]

        inst_map = np.zeros((256, 256), dtype=np.int32)
        num_nuc = 0
        for k in range(5):
            layer_res = remap_label(mask[:, :, k])
            inst_map = np.where(layer_res != 0, layer_res + num_nuc, inst_map)
            num_nuc = num_nuc + np.max(layer_res)
        inst_map = remap_label(inst_map)

        type_map = np.zeros((256, 256)).astype(np.int32)
        for k in range(5):
            layer_res = ((k + 1) * np.clip(mask[:, :, k], 0, 1)).astype(np.int32)
            type_map = np.where(layer_res != 0, layer_res, type_map)

        outdict = {
            "inst_map": inst_map,
            "type_map": type_map
        }
        np.save(f'Masks/{i}_{j}', outdict)

mkdir('../../../prompter/datasets/pannuke123')
mkdir('../../../prompter/datasets/pannuke213')
mkdir('../../../prompter/datasets/pannuke321')

num_classes = 5

for i in range(1, 4):
    data = {
        'classes': ["Neoplastic", "Inflammatory", "Connective", "Dead", "Epithelial"]
    }

    for file in os.listdir('Masks'):
        if not file.startswith(f'{i}_'):
            continue

        anno = np.load(f'Masks/{file}', allow_pickle=True)[()]

        inst_map = anno['inst_map']
        type_map = anno['type_map']

        points = [[] for _ in range(num_classes)]
        for pid in np.unique(inst_map)[1:]:
            src = ((inst_map == pid) * 255).astype(np.uint8)

            contours = cv.findContours(src, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)[0]
            cnt = max(contours, key=cv.contourArea)

            M = cv.moments(cnt)
            if M['m00']:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
            else:
                dist = cv.distanceTransform(src, cv.DIST_L1, 3)
                cx, cy = np.argwhere(dist == dist.max())[0, [1, 0]].tolist()

            cls = type_map[cy, cx]
            if cls > 0:
                points[cls - 1].append([cx, cy])

        data[f'datasets/pannuke/Images/{file[:-4]}.png'] = points

        if i == 1:

            json.dump(data, open(f'../../../prompter/datasets/pannuke123/train.json', 'w'))
            json.dump(data, open(f'../../../prompter/datasets/pannuke213/val.json', 'w'))
            json.dump(data, open(f'../../../prompter/datasets/pannuke321/test.json', 'w'))

            np.save('../pannuke123_train_files.npy', list(data.keys())[1:])
            np.save('../pannuke213_val_files.npy', list(data.keys())[1:])
            np.save('../pannuke321_test_files.npy', list(data.keys())[1:])

        elif i == 2:
            json.dump(data, open(f'../../../prompter/datasets/pannuke123/val.json', 'w'))
            json.dump(data, open(f'../../../prompter/datasets/pannuke213/train.json', 'w'))
            json.dump(data, open(f'../../../prompter/datasets/pannuke321/val.json', 'w'))

            np.save('../pannuke123_val_files.npy', list(data.keys())[1:])
            np.save('../pannuke213_train_files.npy', list(data.keys())[1:])
            np.save('../pannuke321_val_files.npy', list(data.keys())[1:])

        else:

            json.dump(data, open(f'../../../prompter/datasets/pannuke123/test.json', 'w'))
            json.dump(data, open(f'../../../prompter/datasets/pannuke213/test.json', 'w'))
            json.dump(data, open(f'../../../prompter/datasets/pannuke321/train.json', 'w'))

            np.save('../pannuke123_test_files.npy', list(data.keys())[1:])
            np.save('../pannuke213_test_files.npy', list(data.keys())[1:])
            np.save('../pannuke321_train_files.npy', list(data.keys())[1:])
