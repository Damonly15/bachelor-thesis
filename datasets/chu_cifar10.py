# Copyright 2022-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from argparse import Namespace
from typing import Tuple

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from torchvision.datasets import CIFAR10
import numpy as np
import copy

from backbone.ResNetBlock import resnet18
from backbone.ResNetBlockLayerNorm import resnet18layernorm
from backbone.CNN import CnnLN, CnnBN
from datasets.seq_tinyimagenet import base_path
from datasets.transforms.denormalization import DeNormalize
from datasets.utils.continual_dataset import (ContinualDataset,
                                              store_chunking_loaders_random, store_chunking_loaders_classes)
from datasets.utils import set_default_from_args


class TCIFAR10(CIFAR10):
    """Workaround to avoid printing the already downloaded messages."""

    def __init__(self, root, train=True, transform=None,
                 target_transform=None, download=False) -> None:
        self.root = root
        super(TCIFAR10, self).__init__(root, train, transform, target_transform, download=not self._check_integrity())


class MyCIFAR10(CIFAR10):
    """
    Overrides the CIFAR10 dataset to change the getitem function.
    """

    def __init__(self, root, train=True, transform=None,
                 target_transform=None, download=False) -> None:
        self.not_aug_transform = transforms.Compose([transforms.ToTensor()])
        self.root = root
        super(MyCIFAR10, self).__init__(root, train, transform, target_transform, download=not self._check_integrity())

    def __getitem__(self, index: int) -> Tuple[Image.Image, int, Image.Image]:
        """
        Gets the requested element from the dataset.

        Args:
            index: index of the element to be returned

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # to return a PIL Image
        img = Image.fromarray(img, mode='RGB')
        original_img = img.copy()

        not_aug_img = self.not_aug_transform(original_img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if hasattr(self, 'logits'):
            return img, target, not_aug_img, self.logits[index]

        return img, target, not_aug_img


class ChunkingCIFAR10(ContinualDataset):
    """Sequential CIFAR10 Dataset.

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

    NAME = 'chu-cifar10'
    SETTING = 'class-il'
    N_CLASSES_PER_TASK = 2
    N_TASKS = 5
    N_CLASSES = N_CLASSES_PER_TASK * N_TASKS
    SIZE = (32, 32)
    MEAN, STD = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2615)
    TRANSFORM = transforms.Compose(
        [transforms.RandomCrop(32, padding=4),
         transforms.RandomHorizontalFlip(),
         transforms.ToTensor(),
         transforms.Normalize(MEAN, STD)])

    TEST_TRANSFORM = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])

    train_dataset = MyCIFAR10(base_path() + 'CIFAR10', train=True,
                                download=True, transform=TRANSFORM)


    def __init__(self, args: Namespace) -> None:
        super(ChunkingCIFAR10, self).__init__(args)
        self.N_TASKS = args.chunks
        self.N_CLASSES_PER_TASK = self.N_CLASSES

        if not isinstance(self.train_dataset.targets, np.ndarray):
            self.train_dataset.targets = np.array(self.train_dataset.targets)
        #randomly permute dataset
        permutation = np.random.permutation(len(self.train_dataset.data))
        self.train_dataset.data = self.train_dataset.data[permutation]
        self.train_dataset.targets = self.train_dataset.targets[permutation]

    def get_data_loaders(self) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
        """Class method that returns the train and test loaders."""

        test_dataset = TCIFAR10(base_path() + 'CIFAR10', train=False,
                                download=True, transform=self.TEST_TRANSFORM)
        
        train, test = store_chunking_loaders_random(copy.deepcopy(self.train_dataset), test_dataset, self) #distributing chunks randomly
        #train, test = store_chunking_loaders_classes(copy.deepcopy(self.train_dataset), self.test_dataset, self) #distribute chunks according to classes
        return train, test

    @staticmethod
    def get_transform():
        transform = transforms.Compose(
            [transforms.ToPILImage(), ChunkingCIFAR10.TRANSFORM])
        return transform

    @staticmethod
    def get_backbone(version):
        num_classes = ChunkingCIFAR10.N_CLASSES_PER_TASK * ChunkingCIFAR10.N_TASKS
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

    @staticmethod
    def get_normalization_transform():
        transform = transforms.Normalize(ChunkingCIFAR10.MEAN, ChunkingCIFAR10.STD)
        return transform

    @staticmethod
    def get_denormalization_transform():
        transform = DeNormalize(ChunkingCIFAR10.MEAN, ChunkingCIFAR10.STD)
        return transform

    @set_default_from_args('n_epochs')
    def get_epochs(self):
        return 50

    @set_default_from_args('batch_size')
    def get_batch_size(self):
        return 32