import json
import os
from PIL import Image
import torch
from torch.utils import data
import numpy as np
import cv2


def make_dataset(root, seed):
    imgs = []
    img_labels = {}

    # get label dict
    with open(root + '/train.csv', 'r') as f:
        lines = f.readlines()

    for idx in range(1, len(lines)):
        line = lines[idx]
        name, label = line.strip().split(',')
        img_labels[name] = label

    # get image path
    img_names = os.listdir(root + '/image/')
    for i in seed:
        img_name = img_names[i]
        img = os.path.join(root + '/image/', img_name)
        mask = os.path.join(root + '/mask/', img_name)
        imgs.append((img, mask, img_labels[img_name]))

    return imgs


def make_testset(root):
    imgs = []
    with open(root + '/classification.csv', 'r') as f:
        lines = f.readlines()

    for idx in range(1, len(lines)):
        line = lines[idx]
        name, label = line.strip().split(',')
        img = os.path.join(root + '/test_images/', name)
        mask = os.path.join('./data/test_masks/', name)
        imgs.append((img, mask))
    return imgs


def get_bbox(mask):
    mask = cv2.imread(mask)
    mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
    _, labels, stats, centroids = cv2.connectedComponentsWithStats(mask.astype(np.uint8))
    stats = stats[1][:4]
    return stats


class TNSCDataset(data.Dataset):
    def __init__(self, mode='train', transform=None, return_size=False, fold=4):
        self.mode = mode
        self.transform = transform
        self.return_size = return_size

        root = '/home/duadua/TNSC/classifier/data/'
        trainval = json.load(open(root + 'trainval'+str(fold)+'.json', 'r'))  # seeds for k-fold cross validation
        if mode == 'train':
            imgs = make_dataset(root, trainval['train'])
        elif mode == 'val':
            imgs = make_dataset(root, trainval['val'])
        elif mode == 'test':
            imgs = make_testset(root)

        self.imgs = imgs
        self.transform = transform
        self.return_size = return_size

    def __getitem__(self, item):
        if not self.mode == 'test':
            image_path, mask_path, label = self.imgs[item]
        else:
            image_path, mask_path = self.imgs[item]

        assert os.path.exists(image_path), ('{} does not exist'.format(image_path))
        assert os.path.exists(mask_path), ('{} does not exist'.format(mask_path))

        stats = get_bbox(mask_path)
        x, y, w, h = stats
        image = Image.open(image_path).convert('RGB')
        x1 = x - 0.25 * w if x - 0.25 * w >= 0 else 0
        y1 = y - 0.25 * h if y - 0.25 * h >= 0 else 0
        w1 = 1.5 * w if x1 + 1.5 * w <= image.size[0] else image.size[0]
        h1 = 1.5 * h if y1 + 1.5 * h <= image.size[1] else image.size[1]
        image = image.crop((x1, y1, x1 + w1, y1 + h1))

        w, h = image.size
        size = (h, w)
        if not self.mode == 'test':
            sample = {'image': image, 'label': int(label)}
        else:
            sample = {'image': image}

        if self.transform:
            sample = self.transform(sample)
        if self.return_size:
            sample['size'] = torch.tensor(size)

        label_name = os.path.basename(image_path)
        sample['label_name'] = label_name

        return sample

    def __len__(self):
        return len(self.imgs)
