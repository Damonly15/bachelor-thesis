"""
This module implements the simplest form of rehearsal training: Experience Replay. It maintains a buffer
of previously seen examples and uses them to augment the current batch during training.

Example usage:
    model = Er(backbone, loss, args, transform)
    loss = model.observe(inputs, labels, not_aug_inputs, epoch)

"""

# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch.optim import Adam

from models.utils.continual_model import ContinualModel
from utils.args import add_rehearsal_args, ArgumentParser
from utils.buffer import Buffer
from utils.training import evaluate
from utils.feature_forgetting import feature_forgetting_cil

class ErBalanced(ContinualModel):
    NAME = 'er_balanced'
    #this needs task boundaries
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il']

    @staticmethod
    def get_parser() -> ArgumentParser:
        """
        Returns an ArgumentParser object with predefined arguments for the Er model.

        Besides the required `add_management_args` and `add_experiment_args`, this model requires the `add_rehearsal_args` to include the buffer-related arguments.
        """
        parser = ArgumentParser(description='Continual learning via Experience Replay with task boundaries.')
        add_rehearsal_args(parser)
        return parser

    def __init__(self, backbone, loss, args, transform):
        """
        The ER model maintains a buffer of previously seen examples and uses them to augment the current batch during training.
        """
        super(ErBalanced, self).__init__(backbone, loss, args, transform)
        self.buffer = Buffer(self.args.buffer_size)
        self.buffer_nobuffer = Buffer(self.dataset.N_SAMPLES - self.args.buffer_size)

        remainder = self.args.buffer_size % (self.dataset.N_CLASSES)
        ones_indices = torch.randperm(self.dataset.N_CLASSES)[:remainder]
        self.remainder = torch.zeros(self.dataset.N_CLASSES)
        self.remainder[ones_indices] = 1 

        self.overall_batch_size = self.args.batch_size + self.args.minibatch_size
        self.first_task_iterations = 0
        self.current_task_iterations = 0 

    def observe(self, inputs, labels, not_aug_inputs, epoch=None):
        """
        ER trains on the current task using the data provided, but also augments the batch with data from the buffer.
        """
        if inputs.shape[0] != self.dataset.get_batch_size():
            return 0.0
        
        if self.current_task > 0:
            if self.first_task_iterations < self.current_task_iterations:
                return 0
            self.current_task_iterations += 1
        else:
            self.first_task_iterations += 1

        self.opt.zero_grad()

        task_labels = torch.ones(labels.shape[0], dtype=torch.int64, device=self.device) * self.current_task
        if self.args.training_setting == 'task-il':
            labels = labels - (task_labels*self.cpt)

        if not self.buffer.is_empty():
            buf_inputs, buf_labels, buf_tasklabels = self.buffer.get_data(
                self.args.minibatch_size, transform=self.transform, device=self.device)
            
            task_labels = torch.cat((task_labels, buf_tasklabels), dim=0)
            if self.args.training_setting == 'task-il':
                buf_labels = buf_labels - (buf_tasklabels*self.cpt)   
            inputs = torch.cat((inputs, buf_inputs), dim=0)
            labels = torch.cat((labels, buf_labels), dim=0)

        outputs = self.net.forward(inputs, task_label=task_labels)
        loss = self.loss(outputs, labels)
        loss.backward()
                      
        self.opt.step()

        return loss.item()

    def end_task(self, dataset): #Changed this for the paper, it is from xder. It makes sure, that every class has the same amount of samples in the buffer.
        examples_per_class = self.args.buffer_size // dataset.N_CLASSES

        ce = torch.tensor([examples_per_class] * self.cpt) + self.remainder[self.n_past_classes:self.n_seen_classes]

        for data in dataset.train_loader:
            inputs, labels, not_aug_inputs = data

            flags = torch.zeros(len(inputs)).bool()
            flags_nobuffer = torch.zeros(len(inputs)).bool()
            
            for j in range(len(flags)):
                if ce[labels[j] % self.cpt] > 0:
                    flags[j] = True
                    ce[labels[j] % self.cpt] -= 1
                else:
                    flags_nobuffer[j] = True

            if not torch.all(~flags):
                self.buffer.add_data(examples=not_aug_inputs[flags],
                                    labels=labels[flags],
                                    task_labels=(torch.ones(len(flags), dtype=torch.int64) * self.current_task)[flags])
                
            if not torch.all(~flags_nobuffer):
                self.buffer_nobuffer.add_data(examples=not_aug_inputs[flags_nobuffer],
                                    labels=labels[flags_nobuffer],
                                    task_labels=(torch.ones(len(flags), dtype=torch.int64) * self.current_task)[flags_nobuffer])

        self.current_task_iterations = 0
        self.args.batch_size = self.overall_batch_size // (self.current_task+2)
        self.args.minibatch_size = self.overall_batch_size - self.args.batch_size

        return