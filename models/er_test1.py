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
import torch.nn as nn
import math

from models.utils.continual_model import ContinualModel
from utils.args import add_rehearsal_args, ArgumentParser
from utils.buffer import Buffer
from utils.training import evaluate
from utils.feature_forgetting import get_features

class ErTest1(ContinualModel):
    NAME = 'er_test1'
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
        super(ErTest1, self).__init__(backbone, loss, args, transform)
        self.buffer = Buffer(self.args.buffer_size)

        remainder = self.args.buffer_size % (self.dataset.N_CLASSES_PER_TASK*self.dataset.N_TASKS)
        ones_indices = torch.randperm(self.dataset.N_CLASSES_PER_TASK*self.dataset.N_TASKS)[:remainder]
        self.remainder = torch.zeros(self.dataset.N_CLASSES_PER_TASK*self.dataset.N_TASKS)
        self.remainder[ones_indices] = 1 

        self.overall_batch_size = self.args.batch_size + self.args.minibatch_size
        self.args.batch_size = self.overall_batch_size
        self.args.minibatch_size = 0
        self.original_epochs = self.args.n_epochs

    def observe(self, inputs, labels, not_aug_inputs, epoch=None):
        """
        ER trains on the current task using the data provided, but also augments the batch with data from the buffer.
        """

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
        loss = self.loss(outputs[:, :self.n_seen_classes], labels)
        loss.backward()
                      
        self.opt.step()

        return loss.item()

    def end_task(self, dataset): #Changed this for the paper, it is from xder. It makes sure, that every class has the same amount of samples in the buffer.
        examples_per_class = self.args.buffer_size // (dataset.N_CLASSES_PER_TASK * dataset.N_TASKS)  
        ce = torch.tensor([examples_per_class] * self.cpt) + self.remainder[self.current_task*self.cpt:(self.current_task+1)*self.cpt]

        for data in dataset.train_loader:
            inputs, labels, not_aug_inputs = data

            flags = torch.zeros(len(inputs)).bool()
            
            for j in range(len(flags)):
                if ce[labels[j] % self.cpt] > 0:
                    flags[j] = True
                    ce[labels[j] % self.cpt] -= 1

            self.buffer.add_data(examples=not_aug_inputs[flags],
                                labels=labels[flags],
                                task_labels=(torch.ones(len(flags), dtype=torch.int64) * self.current_task)[flags])

        if self.args.buffer_size != 0:
            self.args.batch_size = math.ceil(self.overall_batch_size / (self.current_task+2))
            self.args.minibatch_size = self.overall_batch_size - self.args.batch_size
            self.args.n_epochs = math.ceil(self.original_epochs * (self.args.batch_size / self.overall_batch_size))

        return

    @torch.no_grad()
    def begin_task(self, dataset):
        if (self.current_task == 0):
            return
        buffer_features, buffer_labels, buffer_tasklabels = self.features['buffer']

        if isinstance(self.net.classifier, nn.Linear):
            classifier_weights = self.net.classifier.weight.detach().cpu()[self.n_past_classes:self.n_seen_classes]
        else: 
            classifier_weights = self.net.classifier[self.current_task].weight.detach().cpu()
        
        old_mean_norm = []
        for lab in buffer_labels.unique(sorted=True):
            current_mean = torch.norm(torch.mean(buffer_features[lab == buffer_labels], dim=0), dim=0)
            old_mean_norm.append(current_mean.item())
        old_mean_norm = sum(old_mean_norm) / len(old_mean_norm) 
        
        new_mean_norm = torch.mean(torch.norm(classifier_weights, dim=1), dim=0)
        print(f"Current task: {self.current_task}, old norm: {old_mean_norm}, new norm: {new_mean_norm}")

        gamma = old_mean_norm / new_mean_norm

        if isinstance(self.net.classifier, nn.Linear):
            self.net.classifier.weight[self.n_past_classes:self.n_seen_classes] *= gamma
        else:
            self.net.classifier[self.current_task].weight *= gamma
        return


        