"""
This module implements the simplest form of incremental training, i.e., finetuning.
"""

# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from models.utils.continual_model import ContinualModel
from utils.args import ArgumentParser
from datasets.seq_cifar10 import SequentialCIFAR10
from datasets.seq_cifar100 import SequentialCIFAR100
from datasets.seq_tinyimagenet import SequentialTinyImagenet
from datasets.seq_cub200 import SequentialCUB200


class Sgd(ContinualModel):
    """
    Implementation of the Sgd model for continual learning.
    """

    NAME = 'sgd'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    @staticmethod
    def get_parser() -> ArgumentParser:
        parser = ArgumentParser(description='Finetuning baseline - simple incremental training.')
        return parser

    def __init__(self, backbone, loss, args, transform):
        super(Sgd, self).__init__(backbone, loss, args, transform)
        if args.dataset=='seq-cifar10':
            self.cpt_dataset = SequentialCIFAR10.N_CLASSES_PER_TASK
        elif args.dataset=='seq-cifar100':
            self.cpt_dataset = SequentialCIFAR100.N_CLASSES_PER_TASK
        elif args.dataset=='seq-tinyimg':
            self.cpt_dataset = SequentialTinyImagenet.N_CLASSES_PER_TASK
        elif args.dataset=='seq-cub200':
            self.cpt_dataset = SequentialCUB200.N_CLASSES_PER_TASK

    def observe(self, inputs, labels, not_aug_inputs, epoch=None):
        """
        SGD trains on the current task using the data provided, with no countermeasures to avoid forgetting.
        """
        self.opt.zero_grad()
        if self.args.training_setting == "class-il":
            task_labels=None
        else:
            task_labels = labels // self.cpt_dataset
            labels = labels - (task_labels*self.cpt_dataset)

        outputs = self.net.forward(inputs, task_label=task_labels)
        loss = self.loss(outputs, labels)

        loss.backward()
        self.opt.step()
        return loss.item()
