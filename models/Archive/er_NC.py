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
from collections import defaultdict

from models.utils.continual_model import ContinualModel
from utils.args import add_rehearsal_args, ArgumentParser
from utils.buffer import Buffer
from utils.NC_metrics import calculate_variance


class ErNC(ContinualModel):
    NAME = 'er_NC'
    #this needs task boundaries
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il']

    @staticmethod
    def get_parser() -> ArgumentParser:
        """
        Returns an ArgumentParser object with predefined arguments for the Er model.

        Besides the required `add_management_args` and `add_experiment_args`, this model requires the `add_rehearsal_args` to include the buffer-related arguments.
        """
        parser = ArgumentParser(description='Continual learning via Experience Replay with task boundaries and neural collapse regularizer.')
        add_rehearsal_args(parser)
        parser.add_argument('--alpha', type=float, required=True,
                            help='Penalty weight.')
        return parser

    def __init__(self, backbone, loss, args, transform):
        """
        The ER model maintains a buffer of previously seen examples and uses them to augment the current batch during training.
        """
        super(ErNC, self).__init__(backbone, loss, args, transform)
        self.buffer = Buffer(self.args.buffer_size)
        self.step_NC = 5
        self.counter = 0
        self.loss_NC = 0.0
        self.inputs_array = []
        self.labels_array = []


    def observe(self, inputs, labels, not_aug_inputs, epoch=None):
        """
        ER trains on the current task using the data provided, but also augments the batch with data from the buffer.
        """
        self.opt.zero_grad()
        tot_loss = 0

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
        tot_loss += loss.item()

        if self.current_task > 0:
            if len(self.inputs_array) < self.step_NC:
                self.inputs_array.append(buf_inputs) 
                self.labels_array.append(buf_labels)
            else:
                self.inputs_array = self.inputs_array[1:]
                self.labels_array = self.labels_array[1:]
                self.inputs_array.append(buf_inputs)  
                self.labels_array.append(buf_labels)

            if self.counter % self.step_NC:
                self.counter=0
                loss_NC1 = self.NC_regularizer(torch.cat(self.inputs_array, dim=0), torch.cat(self.labels_array, dim=0))
                loss_NC2 = self.args.alpha * (1 / loss_NC1)
                loss_NC2.backward()
                tot_loss = loss_NC2.item() 
            else:
                self.counter += 1 

        self.opt.step()

        return tot_loss
    
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
        return

    def NC_regularizer(self, features, labels):
        batch = defaultdict(list)
        class_means = []
        NC_score = 0
    
        # Group features by label
        for feature, label in zip(features, labels):
            label = int(label.item())  # Convert label to an int
            batch[label].append(self.net.forward(feature, returnt='features'))

        #calculate class means
        with torch.no_grad():
            for label, feature_list in batch.items():
                current_batch = torch.stack(feature_list)
                class_means.append(torch.mean(current_batch, dim=0))
            class_means = torch.cat(class_means, dim=0)
            inter_class_var = calculate_variance(class_means)

        for label, feature_list in batch.items():
            current_batch = torch.stack(feature_list)
            NC_score += calculate_variance(current_batch) / inter_class_var
        
        return NC_score

