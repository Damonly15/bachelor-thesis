# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch.nn import functional as F

from models.utils.continual_model import ContinualModel
from utils.args import ArgumentParser, add_experiment_args, add_management_args, add_rehearsal_args
from utils.buffer import Buffer
from utils.metrics import adjust_outputs
from utils.batch_norm import bn_track_stats

class DerTogether(ContinualModel):
    NAME = 'der_together'
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
        super(DerTogether, self).__init__(backbone, loss, args, transform)
        self.buffer = Buffer(self.args.buffer_size)

    def observe(self, inputs, labels, not_aug_inputs, epoch=None):

        self.opt.zero_grad()
        tot_loss = 0

        if self.buffer.is_empty():
            outputs = self.net(inputs)
            loss = self.loss(outputs, labels)
            loss.backward()
            tot_loss += loss.item()

        else:
            buf_inputs, buf_logits = self.buffer.get_data(
                self.args.minibatch_size, transform=self.transform, device=self.device)
            inputs = torch.cat([inputs, buf_inputs])
            outputs = self.net(inputs)
            loss = self.loss(outputs[:labels.shape[0]], labels)
            loss.backward(retain_graph=True)
            tot_loss += loss.item()

            loss_mse = self.args.alpha * F.mse_loss(outputs[labels.shape[0]:], buf_logits)
            loss_mse.backward()
            tot_loss += loss_mse.item()

        self.opt.step()
        self.buffer.add_data(examples=not_aug_inputs, logits=(outputs[:labels.shape[0]]).data)

        return tot_loss