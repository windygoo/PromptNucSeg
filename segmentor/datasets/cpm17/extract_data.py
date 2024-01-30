import json
import os
import numpy as np
import scipy.io
import cv2 as cv


def mkdir(path):
    try:
        os.makedirs(path)
    except OSError as e:
        return


train_anno = {'classes': ['nuclei']}
test_anno = {'classes': ['nuclei']}

num_classes = 1
for root, _, files in os.walk('.'):
    for file in files:
        if file.endswith('mat'):

            inst_map = scipy.io.loadmat(f'{root}/{file}')['inst_map']

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

                points[0].append([cx, cy])

            if 'train' in root:
                train_anno[f'datasets/cpm17/train/Images/{file}'] = points
            else:
                test_anno[f'datasets/cpm17/test/Images/{file}'] = points

np.save('../cpm17_train_files.npy', list(train_anno.keys())[1:])
np.save('../cpm17_test_files.npy', list(test_anno.keys())[1:])

mkdir('../../../prompter/datasets/cpm17')
json.dump(train_anno, open('../../../prompter/datasets/cpm17/train.json', 'w'))
json.dump(test_anno, open('../../../prompter/datasets/cpm17/test.json', 'w'))
