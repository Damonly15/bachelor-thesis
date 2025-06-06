# Copyright 2022-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from argparse import Namespace
from typing import Tuple

import torch.nn.functional as F
import torch.optim
import torchvision.transforms as transforms
from PIL import Image
from torchvision.datasets import CIFAR100

from backbone.ResNet18 import resnet18
from backbone.ResNet18LayerNorm import resnet18layernorm
from datasets.transforms.denormalization import DeNormalize
from datasets.utils.continual_dataset import (ContinualDataset,
                                              store_masked_loaders)
# from models.utils.continual_model import ContinualModel
from utils.conf import base_path


class TCIFAR100(CIFAR100):
    """Workaround to avoid printing the already downloaded messages."""

    def __init__(self, root, train=True, transform=None,
                 target_transform=None, download=False) -> None:
        self.root = root
        super(TCIFAR100, self).__init__(root, train, transform, target_transform, download=not self._check_integrity())


class MyCIFAR100(CIFAR100):
    """
    Overrides the CIFAR100 dataset to change the getitem function.
    """

    def __init__(self, root, train=True, transform=None,
                 target_transform=None, download=False, supcon=False) -> None:
        
        self.supcon = supcon
        self.not_aug_transform = transforms.Compose([transforms.ToTensor()])
        self.root = root
        super(MyCIFAR100, self).__init__(root, train, transform, target_transform, not self._check_integrity())

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
            img1 = self.transform(img)
            if self.supcon:
                img2 = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if hasattr(self, 'logits'):
            return img1, target, not_aug_img, self.logits[index]
    
        if self.supcon:
            return img1, target, img2, not_aug_img

        return img1, target, not_aug_img


class SequentialCIFAR100(ContinualDataset):
    """Sequential CIFAR100 Dataset.

    Args:
        NAME (str): name of the dataset.
        SETTING (str): setting of the dataset.
        N_CLASSES_PER_TASK (int): number of classes per task.
        N_TASKS (int): number of tasks.
        N_CLASSES (int): number of classes.
        SIZE (tuple): size of the images.
        MEAN (tuple): mean of the dataset.
        STD (tuple): standard deviation of the dataset.
        TRANSFORM (torchvision.transforms): transformation to apply to the data."""

    NAME = 'seq-cifar100'
    SETTING = 'class-il'
    N_CLASSES_PER_TASK = 10
    N_TASKS = 10
    N_CLASSES = N_CLASSES_PER_TASK * N_TASKS
    SIZE = (32, 32)
    MEAN, STD = (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
    TRANSFORM = transforms.Compose(
        [transforms.RandomCrop(32, padding=4),
         transforms.RandomHorizontalFlip(),
         transforms.ToTensor(),
         transforms.Normalize(MEAN, STD)])

    def get_examples_number(self) -> int:
        train_dataset = MyCIFAR100(base_path() + 'CIFAR10', train=True,
                                   download=True)
        return len(train_dataset.data)

    def get_data_loaders(self) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
        transform = self.TRANSFORM

        test_transform = transforms.Compose(
            [transforms.ToTensor(), self.get_normalization_transform()])
        
        if hasattr(self,"supconaugmentations"):
            transform = transforms.Compose([
                transforms.Resize(size=self.SIZE),
                transforms.RandomResizedCrop(size=self.SIZE, scale=(0.2, 1.)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply([
                    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
                ], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomApply([transforms.GaussianBlur(kernel_size=self.SIZE[0]//20*2+1, sigma=(0.1, 2.0))], p=0.5 if self.SIZE[0]>32 else 0.0),
                transforms.ToTensor(),
                transforms.Normalize(self.MEAN, self.STD)
            ])

        train_dataset = MyCIFAR100(base_path() + 'CIFAR100', train=True,
                                   download=True, transform=transform, supcon=hasattr(self,"supconaugmentations"))
        test_dataset = TCIFAR100(base_path() + 'CIFAR100', train=False,
                                 download=True, transform=test_transform)

        train, test = store_masked_loaders(train_dataset, test_dataset, self)

        return train, test

    @staticmethod
    def get_transform():
        transform = transforms.Compose(
            [transforms.ToPILImage(), SequentialCIFAR100.TRANSFORM])
        return transform

    @staticmethod
    def get_backbone(args, model_compatibility):
        num_classes = SequentialCIFAR100.N_CLASSES_PER_TASK * SequentialCIFAR100.N_TASKS
        if (args.training_setting == 'task-il') and ('task-il' in model_compatibility):
            cpt = SequentialCIFAR100.N_CLASSES_PER_TASK #get backbone with different heads
        else:
            cpt = -1

        bias=True
        if args.model == 'er_wa':
            bias=False
            
        if args.backbone == "ResNet18_LN":
            return resnet18layernorm(nclasses = num_classes, cpt=cpt, bias=bias)
        else: 
            return resnet18(nclasses = num_classes, cpt=cpt, bias=bias)

    @staticmethod
    def get_loss():
        return F.cross_entropy

    @staticmethod
    def get_normalization_transform():
        transform = transforms.Normalize(SequentialCIFAR100.MEAN, SequentialCIFAR100.STD)
        return transform

    @staticmethod
    def get_denormalization_transform():
        transform = DeNormalize(SequentialCIFAR100.MEAN, SequentialCIFAR100.STD)
        return transform

    @staticmethod
    def get_epochs():
        return 50

    @staticmethod
    def get_batch_size():
        return 32

    """@staticmethod
    def get_scheduler(model, args: Namespace) -> torch.optim.lr_scheduler:
        scheduler = ContinualDataset.get_scheduler(model, args)
        if scheduler is None:
            scheduler = torch.optim.lr_scheduler.MultiStepLR(model.opt, [35, 45], gamma=0.1, verbose=False)
        return scheduler"""
