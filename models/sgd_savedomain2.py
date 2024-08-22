"""
This module implements the simplest form of incremental training, i.e., finetuning.
"""

# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import os
import torch

mammoth_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from models.utils.continual_model import ContinualModel
from utils.args import ArgumentParser

class SgdSaveDomain2(ContinualModel):
    """
    Implementation of the Sgd model for continual learning.
    """

    NAME = 'sgd_savedomain2'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    @staticmethod
    def get_parser() -> ArgumentParser:
        parser = ArgumentParser(description='Finetuning baseline - simple incremental training.')
        return parser

    def __init__(self, backbone, loss, args, transform):
        super(SgdSaveDomain2, self).__init__(backbone, loss, args, transform)
        self.true_debug_mode = self.args.debug_mode
        self.args.debug_mode = 1
        self.true_epochs = self.args.n_epochs
        self.args.n_epochs = 1

    def observe(self, inputs, labels, not_aug_inputs, epoch=None):
        """
        SGD trains on the current task using the data provided, with no countermeasures to avoid forgetting.
        """
        if self.current_task + 1 != self.N_TASKS:
            return 0

        self.opt.zero_grad()
        outputs = self.net(inputs)
        loss = self.loss(outputs, labels)
        loss.backward()
        self.opt.step()

        return loss.item()
    
    def begin_task(self, dataset):
        if self.current_task + 1 == self.N_TASKS:
            self.args.n_epochs = self.true_epochs
            self.args.debug_mode = self.true_debug_mode

    def end_task(self, dataset):
        if self.current_task + 1 == self.N_TASKS:
            #save model for later use
            if not os.path.exists(mammoth_path + f"/pretrained_models"):
                os.makedirs(mammoth_path + f"/pretrained_models")
            
            torch.save(self.net.state_dict(), mammoth_path + f"/pretrained_models/perm_mnist_12epochs.pth")
