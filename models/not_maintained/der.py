# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import torch
from torch.nn import functional as F

from models.utils.continual_model import ContinualModel
from utils.args import ArgumentParser, add_experiment_args, add_management_args, add_rehearsal_args
from utils.buffer import Buffer


class Der(ContinualModel):
    NAME = 'der'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    @staticmethod
    def get_parser() -> ArgumentParser:
        parser = ArgumentParser(description='Continual learning via'
                                ' Dark Experience Replay.')
        add_rehearsal_args(parser)
        parser.add_argument('--alpha', type=float, required=True,
                            help='Penalty weight.')
        return parser

    def __init__(self, backbone, loss, args, transform):
        super(Der, self).__init__(backbone, loss, args, transform)
        self.buffer = Buffer(self.args.buffer_size)

    def observe(self, inputs, labels, not_aug_inputs, epoch=None):

        self.opt.zero_grad()
        tot_loss = 0

        if self.args.training_setting == 'class-il':
            task_labels = None
        else: 
            task_labels = self.current_task
            labels = labels - (task_labels*self.cpt) 

        outputs = self.net.forward(inputs, task_label=task_labels)
        loss = self.loss(outputs, labels)
        loss.backward()
        tot_loss += loss.item()

        if not self.buffer.is_empty():
            buf_inputs, buf_logits, buf_tasklabels = self.buffer.get_data(
                self.args.minibatch_size, transform=self.transform, device=self.device)
            if self.args.training_setting == 'class-il':
                buf_outputs = self.net.forward(buf_inputs, task_label=None)
            else:
                buf_outputs = self.net.forward(buf_inputs, task_label=buf_tasklabels)

            loss_mse = self.args.alpha * F.mse_loss(buf_outputs, buf_logits)
            loss_mse.backward()
            tot_loss += loss_mse.item()

        self.opt.step()
        self.buffer.add_data(examples=not_aug_inputs, logits=outputs.detach(), task_labels=(torch.ones(not_aug_inputs.shape[0], dtype=torch.int64, device=self.device) * self.current_task))

        return tot_loss
