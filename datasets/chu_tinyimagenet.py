# Copyright 2022-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from argparse import Namespace
import os
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset
import copy

from backbone.ResNetBlock import resnet18
from backbone.ResNetBlockLayerNorm import resnet18layernorm, resnet34layernorm
from backbone.CNN import CnnLN, CnnBN
from datasets.transforms.denormalization import DeNormalize
from datasets.utils.continual_dataset import (ContinualDataset,
                                              store_chunking_loaders_random, store_chunking_loaders_classes)
from utils import smart_joint
from utils.conf import base_path
from datasets.utils import set_default_from_args


class TinyImagenet(Dataset):
    """Defines the Tiny Imagenet dataset."""

    def __init__(self, root: str, train: bool = True, transform: Optional[nn.Module] = None,
                 target_transform: Optional[nn.Module] = None, download: bool = False) -> None:
        self.not_aug_transform = transforms.Compose([transforms.ToTensor()])
        self.root = root
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.download = download

        if download:
            if os.path.isdir(root) and len(os.listdir(root)) > 0:
                print('Download not needed, files already on disk.')
            else:
                from onedrivedownloader import download

                print('Downloading dataset')
                ln = "https://unimore365-my.sharepoint.com/:u:/g/personal/263133_unimore_it/EVKugslStrtNpyLGbgrhjaABqRHcE3PB_r2OEaV7Jy94oQ?e=9K29aD"
                download(ln, filename=smart_joint(root, 'tiny-imagenet-processed.zip'), unzip=True, unzip_path=root, clean=True)

        self.data = []
        for num in range(20):
            self.data.append(np.load(smart_joint(
                root, 'processed/x_%s_%02d.npy' %
                      ('train' if self.train else 'val', num + 1))))
        self.data = np.concatenate(np.array(self.data))

        self.targets = []
        for num in range(20):
            self.targets.append(np.load(smart_joint(
                root, 'processed/y_%s_%02d.npy' %
                      ('train' if self.train else 'val', num + 1))))
        self.targets = np.concatenate(np.array(self.targets))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(np.uint8(255 * img))
        original_img = img.copy()

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if hasattr(self, 'logits'):
            return img, target, original_img, self.logits[index]

        return img, target


class MyTinyImagenet(TinyImagenet):
    """Overrides the TinyImagenet dataset to change the getitem function."""

    def __init__(self, root: str, train: bool = True, transform: Optional[nn.Module] = None,
                 target_transform: Optional[nn.Module] = None, download: bool = False) -> None:
        super(MyTinyImagenet, self).__init__(
            root, train, transform, target_transform, download)

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(np.uint8(255 * img))
        original_img = img.copy()

        not_aug_img = self.not_aug_transform(original_img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if hasattr(self, 'logits'):
            return img, target, not_aug_img, self.logits[index]

        return img, target, not_aug_img


class ChunkingTinyImagenet(ContinualDataset):
    """The Sequential Tiny Imagenet dataset.

    Args:
        NAME (str): name of the dataset.
        SETTING (str): setting of the dataset.
        N_CLASSES_PER_TASK (int): number of classes per task.
        N_TASKS (int): number of tasks.
        N_CLASSES (int): number of classes.
        SIZE (tuple): size of the images.
        MEAN (tuple): mean of the dataset.
        STD (tuple): standard deviation of the dataset.
        TRANSFORM (torchvision.transforms): transformations to apply to the dataset.
    """

    NAME = 'chu-tinyimg'
    SETTING = 'class-il'
    N_CLASSES_PER_TASK = 20
    N_TASKS = 10
    N_CLASSES = N_CLASSES_PER_TASK * N_TASKS
    MEAN, STD = (0.4802, 0.4480, 0.3975), (0.2770, 0.2691, 0.2821)
    SIZE = (64, 64)
    TRANSFORM = transforms.Compose(
        [transforms.RandomCrop(64, padding=4),
         transforms.RandomHorizontalFlip(),
         transforms.ToTensor(),
         transforms.Normalize(MEAN, STD)])
    
    train_dataset = MyTinyImagenet(base_path() + 'TINYIMG',
                                       train=True, download=True, transform=TRANSFORM)
    
    def __init__(self, args: Namespace) -> None:
        super(ChunkingTinyImagenet, self).__init__(args)
        self.N_TASKS = args.chunks
        self.N_CLASSES_PER_TASK = self.N_CLASSES

        if not isinstance(self.train_dataset.targets, np.ndarray):
            self.train_dataset.targets = np.array(self.train_dataset.targets)
        #randomly permute dataset
        permutation = np.random.permutation(len(self.train_dataset.data))
        self.train_dataset.data = self.train_dataset.data[permutation]
        self.train_dataset.targets = self.train_dataset.targets[permutation]

    def get_data_loaders(self) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
        test_transform = transforms.Compose(
            [transforms.ToTensor(), self.get_normalization_transform()])

        test_dataset = TinyImagenet(base_path() + 'TINYIMG',
                                    train=False, download=True, transform=test_transform)

        train, test = store_chunking_loaders_random(copy.deepcopy(self.train_dataset), test_dataset, self)
        return train, test

    @staticmethod
    def get_backbone(version):
        num_classes = ChunkingTinyImagenet.N_CLASSES_PER_TASK * ChunkingTinyImagenet.N_TASKS
        if version == "0":
            return resnet18layernorm(nclasses = num_classes)
        elif version == '1':
            return CnnLN(16, num_classes)
        elif version == '2':
            return CnnLN(32, num_classes)
        elif version == '3':
            return CnnLN(64, num_classes)
        elif version == '4':
            return CnnLN(112, num_classes)
        elif version == '5':
            return CnnLN(176, num_classes)
        elif version == '6':
            return CnnLN(240, num_classes)
        elif version == '7':
            return CnnLN(368, num_classes)
        elif version == '8':
            return CnnLN(512, num_classes)
        elif version == '10':
            return resnet18(num_classes)
        elif version == '15':
            return CnnBN(176, num_classes)

    @staticmethod
    def get_loss():
        return F.cross_entropy

    def get_transform(self):
        transform = transforms.Compose(
            [transforms.ToPILImage(), self.TRANSFORM])
        return transform

    @staticmethod
    def get_normalization_transform():
        transform = transforms.Normalize(ChunkingTinyImagenet.MEAN, ChunkingTinyImagenet.STD)
        return transform

    @staticmethod
    def get_denormalization_transform():
        transform = DeNormalize(ChunkingTinyImagenet.MEAN, ChunkingTinyImagenet.STD)
        return transform

    @set_default_from_args('n_epochs')
    def get_epochs(self):
        return 100

    @set_default_from_args('batch_size')
    def get_batch_size(self):
        return 32