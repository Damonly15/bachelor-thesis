"""
This module implements the simplest form of incremental training, i.e., finetuning.
"""

# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import timm

from models.utils.continual_model import ContinualModel
from utils.args import ArgumentParser


class Test(ContinualModel):
    """
    Implementation of the Sgd model for continual learning.
    """

    NAME = 'test'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    @staticmethod
    def get_parser() -> ArgumentParser:
        parser = ArgumentParser(description='Finetuning baseline - simple incremental training.')
        parser.add_argument('--teacher', type=str, required=True,
                            help='Penalty weight.')
        return parser

    def __init__(self, backbone, loss, args, transform):
        super(Test, self).__init__(backbone, loss, args, transform)
        self.net = timm.create_model(self.args.teacher, pretrained=True, num_classes=0)
        self.net.to(self.device)
        self.net.eval()

    def observe(self, inputs, labels, not_aug_inputs, epoch=None):

        return 0
