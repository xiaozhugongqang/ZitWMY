import os
import os.path as osp
from PIL import Image

from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np

import random
import math
import torch

IMAGE_PATH = 'path to images'
SPLIT_PATH = 'path to splits'
data_path = '/home/ys/data/wmy'

class TieredImagenet(Dataset):
    """ Usage:
    """

    def __init__(self, setname, args):
        dataset_dir = os.path.join(data_path, 'tiered_ImageNet/')
        if setname == 'train':
            path = osp.join(dataset_dir, 'train')
            label_list = os.listdir(path)
        elif setname == 'test':
            path = osp.join(dataset_dir, 'test')
            label_list = os.listdir(path)
        elif setname == 'val':
            path = osp.join(dataset_dir, 'val')
            label_list = os.listdir(path)
        else:
            raise ValueError('Incorrect set name. Please check!')

        data = []
        label = []

        folders = [osp.join(path, label) for label in label_list if os.path.isdir(osp.join(path, label))]

        for idx, this_folder in enumerate(folders):
            this_folder_images = os.listdir(this_folder)
            for image_path in this_folder_images:
                data.append(osp.join(this_folder, image_path))
                label.append(idx)
        # csv_path = osp.join(SPLIT_PATH, setname + '.csv')
        # # import pdb
        # # pdb.set_trace()
        # lines = [x.strip() for x in open(csv_path, 'r').readlines()][1:]
        #
        # data = []
        # label = []
        # lb = -1
        #
        # self.wnids = []
        #
        # for l in lines:
        #     name, wnid = l.split(',')
        #     path = osp.join(IMAGE_PATH, name)
        #     if wnid not in self.wnids:
        #         self.wnids.append(wnid)
        #         lb += 1
        #     data.append(path)
        #     label.append(lb)

        self.data = data
        self.label = label
        self.num_class = len(set(label))
        self.setname = setname

        image_size = 224  # 112, 128; 144, 168
        self.transform_train = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomResizedCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(np.array([x / 255.0 for x in [120.39586422, 115.59361427, 104.54012653]]),
                                 np.array([x / 255.0 for x in [70.68188272, 68.27635443, 72.54505529]]))
        ])
        self.transform_val_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(np.array([x / 255.0 for x in [120.39586422, 115.59361427, 104.54012653]]),
                                 np.array([x / 255.0 for x in [70.68188272, 68.27635443, 72.54505529]]))
            ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        path, label = self.data[i], self.label[i]
        if self.setname == 'train':
            image = self.transform_train(Image.open(path).convert('RGB'))
        else:
            image = self.transform_val_test(Image.open(path).convert('RGB'))
        return image, label

