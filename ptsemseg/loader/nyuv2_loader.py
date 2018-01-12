import os
import collections
import torch
import torchvision
import numpy as np
import scipy.misc as m
#import matplotlib.pyplot as plt

from torch.utils import data


class nyuv2Loader(data.Dataset):
    ''' test source: http://www.doc.ic.ac.uk/~ahanda/nyu_test_rgb.tgz
        train source: http://www.doc.ic.ac.uk/~ahanda/nyu_train_rgb.tgz
        test_labels source: https://github.com/ankurhanda/nyuv2-meta-data/raw/master/test_labels_13/nyuv2_test_class13.tgz
        train_labels source: https://github.com/ankurhanda/nyuv2-meta-data/raw/master/train_labels_13/nyuv2_train_class13.tgz

        Labels:
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
        assert split in ["test", "train"]
        self.root = root
        self.split = split
        self.img_size = [480, 640]
        self.is_transform = is_transform
        self.mean = np.array([123.67608872, 106.15077847, 101.53614841])
        self.n_classes = 14
        self.files = collections.defaultdict(list)

        for split in ["test", "test_labels", "train", "train_labels"]:
            file_list = os.listdir(root + '/' + split)
            self.files[split] = file_list

    def __len__(self):
        return len(self.files[self.split])

    def __getitem__(self, index):
        img_name = self.files[self.split][index]
        img_label_no = img_name[8:12]  # names are of form 'nyu_rgb_0016.png'
        label_name = 'new_nyu_class13_{lbl}.png'.format(lbl=img_label_no)
        img_path = self.root + '/' + self.split + '/' + img_name
        lbl_path = self.root + '/' + self.split + '_labels/' + label_name

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
        img -= self.mean
        img = img.astype(float) / 255.0
        # NHWC -> NCHW
        img = img.transpose(2, 0, 1)

        img = torch.from_numpy(img).float()
        lbl = torch.from_numpy(lbl).long()
        return img, lbl
