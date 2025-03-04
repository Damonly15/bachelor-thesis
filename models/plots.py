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
import pandas as pd

from models.utils.continual_model import ContinualModel
from utils.args import add_rehearsal_args, ArgumentParser
from utils.buffer import Buffer
from utils.feature_forgetting import get_features
from utils.NC_metrics import evaluate_NC_metrics
from utils.conf import base_path


class Plots(ContinualModel):
    NAME = 'plots'
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
        super(Plots, self).__init__(backbone, loss, args, transform)
        self.buffer = Buffer(self.args.buffer_size)
        self.dataset_object = None
        self.epoch_counter = 0
        self.evaluation_epochs = args.n_epochs//10 #adapt

        column_names = ['task', 'epoch']
        for i in range(10):
            column_names.append(f'within_var_{i+1}')
        for i in range(10): 
            column_names.append(f'between_var_{i+1}')
        self.df_NC = pd.DataFrame(columns=column_names)

    def observe(self, inputs, labels, not_aug_inputs, epoch=None):
        """
        ER trains on the current task using the data provided, but also augments the batch with data from the buffer.
        """
        if self.epoch_counter == epoch:
            if (self.epoch_counter % self.evaluation_epochs) == 0:
                (within_var, between_var), _= evaluate_NC_metrics(self, self.dataset_object, 'train_dataset')
                new_column = [self.current_task+1, self.current_task*self.args.n_epochs + epoch] + within_var + between_var
                print(new_column)
                self.df_NC.loc[len(self.df_NC)] = new_column
            self.epoch_counter+=1

        self.opt.zero_grad()

        if self.args.training_setting == 'class-il':
            task_labels = None
        else: 
            task_labels = torch.ones(labels.shape[0],  dtype=torch.int64, device=self.device) * self.current_task
            labels = labels - (task_labels*self.cpt)

        if not self.buffer.is_empty():
            buf_inputs, buf_labels, buf_tasklabels = self.buffer.get_data(
                self.args.minibatch_size, transform=self.transform, device=self.device)
            
            if self.args.training_setting == 'task-il':
                buf_labels = buf_labels - (buf_tasklabels*self.cpt)
                task_labels = torch.cat((task_labels, buf_tasklabels), dim=0)
            inputs = torch.cat((inputs, buf_inputs), dim=0)
            labels = torch.cat((labels, buf_labels), dim=0)

        outputs = self.net.forward(inputs, task_label=task_labels)
        loss = self.loss(outputs, labels)
        loss.backward()
                      
        self.opt.step()
        return loss.item()
    
    def begin_task(self, dataset):
        self.dataset_object = dataset

    def end_task(self, dataset): #Changed this for the paper, it is from xder. It makes sure, that every class has the same amount of samples in the buffer.
            
        examples_per_class = self.args.buffer_size // ((self.current_task + 1) * self.cpt)
        remainder = self.args.buffer_size % ((self.current_task + 1) * self.cpt)
        ones_indices = torch.randperm(self.n_seen_classes)[:remainder]
        remainder = torch.zeros(self.n_seen_classes)
        if not self.args.buffer_size == dataset.N_CLASSES: #in this case just use one sample per class
            remainder[ones_indices] = 1

        # fdr reduce coreset
        if not self.buffer.is_empty():
            buf_x, buf_lab, buf_tl = self.buffer.get_all_data()
            self.buffer.empty()

            for tl in buf_lab.unique():
                idx = tl == buf_lab
                ex, lab, tasklab = buf_x[idx], buf_lab[idx], buf_tl[idx]
                first = min(ex.shape[0], examples_per_class + int(remainder[tl].item()))
                self.buffer.add_data(
                    examples=ex[:first],
                    labels=lab[:first],
                    task_labels=tasklab[:first]
                )

        # fdr add new task
        ce = torch.tensor([examples_per_class] * self.cpt)
        ce = (ce + remainder[self.n_past_classes:]).int() 

        for data in dataset.train_loader:
            inputs, labels, not_aug_inputs = data
            if all(ce == 0):
                break

            flags = torch.zeros(len(inputs)).bool()
            for j in range(len(flags)):
                if ce[labels[j] % self.cpt] > 0:
                    flags[j] = True
                    ce[labels[j] % self.cpt] -= 1

            self.buffer.add_data(examples=not_aug_inputs[flags],
                                    labels=labels[flags],
                                    task_labels=(torch.ones(len(flags), dtype=torch.int64) * self.current_task)[flags])
        
        self.epoch_counter = 0
        if (self.current_task + 1== dataset.N_TASKS):
            path = base_path() + f'dataframes/nc/{self.args.dataset}_{self.args.buffer_size}_{self.args.seed}.csv'
            self.df_NC.to_csv(path, index=False)
        return
    