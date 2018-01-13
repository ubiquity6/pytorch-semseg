import os
import collections
import torch
import torchvision
import numpy as np
import scipy.misc as m
#import matplotlib.pyplot as plt

from torch.utils import data


class scenenetLoader(data.Dataset):
    ''' Labels:
        0   Other / unlabelled
        1   Bed
        2   Books
        3   Ceiling
        4   Chair
        5   Floor
        6   Furniture
        7   Objects
        8   Picture
        9   Sofa
        10  Table
        11  TV
        12  Wall
        13  Window
    '''
    def __init__(self, root, split="train", is_transform=False, img_size=None):
        assert split in ["val", "train"]
        self.root = root
        self.split = split
        self.img_size = [240, 320]
        self.is_transform = is_transform

        # this will need to be measured for the entire dataset. For now, we comment it out
        # self.mean = np.array([125.91207973, 116.5486107, 110.43807554])
        self.n_classes = 14
        self.files = collections.defaultdict(list)

        for split in ["train", "val"]:
            start = root + '/' + split + '/'
            folder_list = filter(lambda path: len(path) < 3 and os.path.isdir(start + path),
                                 os.listdir(start))
            file_list = []
            for f in folder_list:
                subfolder = start + f + '/'
                subdirs = os.listdir(subfolder)
                for s in subdirs:
                    photo_dir = subfolder + s + '/photo/'
                    images = os.listdir(photo_dir)
                    for image in images:
                        file_list.append(photo_dir + image)
            self.files[split] = file_list
            if split == self.split:
                print('Found %d images.' % len(file_list))

    def __len__(self):
        return len(self.files[self.split])

    def __getitem__(self, index):
        img_path = self.files[self.split][index]
        lbl_path = img_path.replace('/photo/', '/semantic_label/').replace('jpg', 'png')

        img = m.imread(img_path)
        img = np.array(img, dtype=np.uint8)

        lbl = m.imread(lbl_path)
        lbl = np.array(lbl, dtype=np.int32)

        if self.is_transform:
            img, lbl = self.transform(img, lbl)

        return img, lbl

    def transform(self, img, lbl):
        img = img[:, :, ::-1]
        img = img.astype(np.float64)

        # see note above re: mean
        # img -= self.mean

        # images are a fixed size
        # img = m.imresize(img, (self.img_size[0], self.img_size[1]))

        img = img.astype(float) / 255.0
        # NHWC -> NCHW
        img = img.transpose(2, 0, 1)

        lbl = lbl.astype(float)
        # lbl = m.imresize(lbl, (self.img_size[0], self.img_size[1]), 'nearest', mode='F')

        img = torch.from_numpy(img).float()
        lbl = torch.from_numpy(lbl).long()
        return img, lbl
