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
from utils.model_utils import adjust_outputs
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
        self.class_means = defaultdict(lambda: None)
        self.inter_class_var = None

    def observe(self, inputs, labels, not_aug_inputs, epoch=None):
        """
        ER trains on the current task using the data provided, but also augments the batch with data from the buffer.
        """
        batch_size = inputs.shape[0]

        self.opt.zero_grad()
        tot_loss = 0

        if self.args.training_setting == "class-il":
            if not self.buffer.is_empty():
                buf_inputs, buf_labels, _ = self.buffer.get_data(
                    self.args.minibatch_size, transform=self.transform, device=self.device)
                inputs = torch.cat((inputs, buf_inputs))
                labels = torch.cat((labels, buf_labels))

            outputs, features = self.net(inputs, returnt='both')
            loss = self.loss(outputs, labels)
            loss.backward(retain_graph=True)
            tot_loss += loss.item()
        else:
            if not self.buffer.is_empty():
                buf_inputs, buf_labels, buf_tasklabels = self.buffer.get_data(
                    self.args.minibatch_size, transform=self.transform, device=self.device)
                task_labels = torch.cat(((torch.ones(batch_size, dtype=torch.int64, device=self.device) * self.current_task), buf_tasklabels))
                inputs = torch.cat((inputs, buf_inputs))
                labels = torch.cat((labels, buf_labels))
            else:
                task_labels = (torch.ones(batch_size, dtype=torch.int64, device=self.device) * self.current_task)

            outputs, features = self.net(inputs, returnt='both')
            outputs = adjust_outputs(outputs, task_labels, self._cpt)
            adjusted_labels = labels - (task_labels * self._cpt)
            loss = self.loss(outputs, adjusted_labels)
            loss.backward(retain_graph=True)   
            tot_loss += loss.item()

        #NC regularizer 
        with torch.no_grad():
            self.update_running_average(features, labels)
        #no running average, use last batches as means. Accumulate gradients and only update every x steps

        #get different batch?
        if self.current_task>0: #regularizer only on buffer
            nc_loss1 = 1 / self.NC_regularizer(features[batch_size:], labels[batch_size:])
            nc_loss2 = self.args.alpha * nc_loss1
            nc_loss2.backward()
            tot_loss += nc_loss2.item()
        #(intra_class - inter_class)**2, inter_class var as constant?
        self.opt.step()

        return loss.item()
    
    def end_task(self, dataset): #Changed this for the paper, it is from xder. It makes sure, that every class has the same amount of samples in the buffer.
            
        examples_per_class = self.args.buffer_size // ((self.current_task + 1) * self.cpt)
        remainder = self.args.buffer_size % ((self.current_task + 1) * self.cpt)

        # fdr reduce coreset
        if not self.buffer.is_empty():
            buf_x, buf_lab, buf_tl = self.buffer.get_all_data()
            self.buffer.empty()

            for tl in buf_lab.unique():
                idx = tl == buf_lab
                ex, lab, tasklab = buf_x[idx], buf_lab[idx], buf_tl[idx]
                if(remainder > 0):
                    first = min(ex.shape[0], examples_per_class + 1)
                    remainder -= 1
                else:
                    first = min(ex.shape[0], examples_per_class)
                self.buffer.add_data(
                    examples=ex[:first],
                    labels=lab[:first],
                    task_labels=tasklab[:first]
                )

        # fdr add new task
        ce = torch.tensor([examples_per_class] * self.cpt).int()
        for i in range(remainder):
            ce[i] += 1 

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

    def update_running_average(self, features, labels, alpha=0.05):
        batch_means = defaultdict(list)
    
        # Group features by label
        for feature, label in zip(features, labels):
            label = int(label.item())  # Convert label to an int
            batch_means[label].append(feature)

        # Update the running mean for each label
        for label, feature_list in batch_means.items():
            batch_mean = torch.stack(feature_list).mean(dim=0)  # Compute the mean of the current batch for this label
            # Update using exponential moving average
            if self.class_means[label] is None:
                # Initialize the running mean if it doesn't exist
                self.class_means[label] = batch_mean
            else:
                # Update using exponential moving average
                self.class_means[label] = (1 - alpha) * self.class_means[label] + alpha * batch_mean

        # Compute inter class variance
        self.inter_class_var = calculate_variance(torch.stack(list(self.class_means.values())))
        return

    def NC_regularizer(self, features, labels):
        batch = defaultdict(list)
        NC_score = 0
    
        # Group features by label
        for feature, label in zip(features, labels):
            label = int(label.item())  # Convert label to an int
            batch[label].append(feature)

        for label, feature_list in batch.items():
            current_batch = torch.stack(feature_list)
            NC_score += calculate_variance(current_batch, self.class_means[label]) / (self.inter_class_var*len(self.class_means))
        
        return NC_score

