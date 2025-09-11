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

class SgdSave(ContinualModel):
    """
    Implementation of the Sgd model for continual learning.
    """

    NAME = 'sgd_save'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    @staticmethod
    def get_parser() -> ArgumentParser:
        parser = ArgumentParser(description='Finetuning baseline - simple incremental training.')
        return parser

    def __init__(self, backbone, loss, args, transform):
        super(SgdSave, self).__init__(backbone, loss, args, transform)

    def observe(self, inputs, labels, not_aug_inputs, epoch=None):
        """
        SGD trains on the current task using the data provided, with no countermeasures to avoid forgetting.
        """
        self.opt.zero_grad()
        outputs = self.net(inputs)
        loss = self.loss(outputs, labels)
        loss.backward()
        self.opt.step()

        _, pred = torch.max(outputs.detach()[:, :self.n_seen_classes].data, 1)
        correct = torch.sum(pred == labels).item()
        total = labels.shape[0]
        _wandb_train_acc = correct / total * 100

        return loss.item()

    def end_task(self, dataset):
        #save model for later use
        if not os.path.exists(mammoth_path + f"/pretrained_models"):
            os.makedirs(mammoth_path + f"/pretrained_models")
        
        torch.save(self.net.state_dict(), mammoth_path + f"/pretrained_models/tiny_img_150epochs.pth")