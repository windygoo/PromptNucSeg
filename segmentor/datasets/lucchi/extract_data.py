import json
import os
import numpy as np
import cv2 as cv
from PIL import Image

def mkdir(path):
    try:
        os.makedirs(path)
    except OSError as e:
        return


train_anno = {'classes': ['mito']}
test_anno = {'classes': ['mito']}

num_classes = 1

for root, _, files in os.walk('.'):

    for file in files:
        if root[-4:] == 'mask':
            file_path = root + '/' + file
            inst_map = np.asarray(Image.open(file_path), dtype=np.uint8)
            inst_map = cv.connectedComponents(inst_map)[1]

            points = [[] for _ in range(num_classes)]
            print(file)
            for pid in np.unique(inst_map)[1:]:
                src = ((inst_map == pid) * 255).astype(np.uint8)

                contours = cv.findContours(src, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)[0]
                for c in contours:
                    print(c.shape)
                cnt = max(contours, key=cv.contourArea)

                M = cv.moments(cnt)
                if M['m00']:
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])
                else:
                    dist = cv.distanceTransform(src, cv.DIST_L1, 3)
                    cx, cy = np.argwhere(dist == dist.max())[0, [1, 0]].tolist()

                points[0].append([cx, cy])

            if 'train' in root:
                train_anno[f'datasets/lucchi/train/mask/{file}'] = points
            else:
                test_anno[f'datasets/lucchi/test/mask/{file}'] = points

np.save('../lucchi_train_files.npy', list(train_anno.keys())[1:])
np.save('../lucchi_test_files.npy', list(test_anno.keys())[1:])

mkdir('../../../prompter/datasets/lucchi')
json.dump(train_anno, open('../../../prompter/datasets/lucchi/train.json', 'w'))
json.dump(test_anno, open('../../../prompter/datasets/lucchi/test.json', 'w'))
