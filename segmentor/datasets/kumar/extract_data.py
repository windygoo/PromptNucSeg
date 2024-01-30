import json
import os
import numpy as np
import cv2 as cv


def mkdir(path):
    try:
        os.makedirs(path)
    except OSError as e:
        return


train_anno = {'classes': ['nuclei']}
test_anno = {'classes': ['nuclei']}

num_classes = 1

for file in os.listdir('images/train_ori16'):
    inst_map = np.load(f'labels/train_ins_ori16/{file[:-4]}.npy')

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
        train_anno[f'datasets/kumar/images/train_ori16/{file}'] = points

for file in os.listdir('images/test1'):
    inst_map = np.load(f'labels/test1_ins/{file[:-4]}.npy')

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
        test_anno[f'datasets/kumar/images/test1/{file}'] = points

for file in os.listdir('images/test2'):
    inst_map = np.load(f'labels/test2_ins/{file[:-4]}.npy')

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
        test_anno[f'datasets/kumar/images/test2/{file}'] = points

np.save('../kumar_train_files.npy', list(train_anno.keys())[1:])
np.save('../kumar_test_files.npy', list(test_anno.keys())[1:])

mkdir('../../../prompter/datasets/kumar')
json.dump(train_anno, open('../../../prompter/datasets/kumar/train.json', 'w'))
json.dump(test_anno, open('../../../prompter/datasets/kumar/test.json', 'w'))
