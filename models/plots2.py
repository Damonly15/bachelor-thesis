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
from torch.nn import functional as F
import numpy
import pandas as pd
from sklearn.linear_model import LogisticRegression

from models.utils.continual_model import ContinualModel
from utils.args import add_rehearsal_args, ArgumentParser
from utils.buffer import Buffer
from utils.feature_forgetting import get_features
from utils.conf import base_path


class Plots2(ContinualModel):
    NAME = 'plots2'
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
        super(Plots2, self).__init__(backbone, loss, args, transform)
        self.buffer = Buffer(self.args.buffer_size)
        self.df_bias = pd.DataFrame(columns=['task', 'probability', 'result_type'])

    def observe(self, inputs, labels, not_aug_inputs, epoch=None):
        """
        ER trains on the current task using the data provided, but also augments the batch with data from the buffer.
        """

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
    
    @torch.no_grad
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
        
        #calculate probability
        status = self.net.training
        self.net.eval()

        probability_output = []
        probability_buffer = []
        probability_features = []

        all_features, all_labels, all_tasklabels = get_features(self, dataset, 'train_dataset')
        task_mask = all_tasklabels <= self.current_task
        all_features, all_labels = all_features[task_mask], all_labels[task_mask]
        features_head = LogisticRegression(max_iter=5000, C=10)
        features_head.fit(all_features.numpy(), all_labels.numpy())

        buffer_features, buffer_labels, _ = get_features(self, dataset, 'buffer')
        buffer_head = LogisticRegression(max_iter=5000, C=10)
        buffer_head.fit(buffer_features.numpy(), buffer_labels.numpy())

        for iter in dataset.all_train_loaders[:self.current_task+1]:
            for data in iter:
                if hasattr(iter, 'logits'):
                    inputs, _, _, _ = data
                else:
                    inputs, _, _ = data

                inputs = inputs.to(self.device)
                outputs, features = self.net.forward(inputs, returnt='both')
                outputs, features = outputs.detach().cpu(), features.detach().cpu()

                probability = F.softmax(outputs[:, :self.n_seen_classes], dim=-1)
                probability = probability[:, self.n_past_classes:self.n_seen_classes].sum(dim=1)
                probability_output.append(probability)

                probability = features_head.predict_proba(features.numpy())
                probability = probability[:, self.n_past_classes:self.n_seen_classes].sum(axis=1)
                probability_features.append(probability)

                probability = buffer_head.predict_proba(features.numpy())
                probability = probability[:, self.n_past_classes:self.n_seen_classes].sum(axis=1)
                probability_buffer.append(probability)

        probability_output = torch.cat(probability_output, dim=0)
        probability_output = probability_output.mean(dim=0).item()
        self.df_bias.loc[len(self.df_bias)] = [self.current_task+1, probability_output, 'output']

        probability_features = numpy.concatenate(probability_features, axis=0)
        probability_features = probability_features.mean(axis=0)
        self.df_bias.loc[len(self.df_bias)] = [self.current_task+1, probability_features, 'features']

        probability_buffer = numpy.concatenate(probability_buffer, axis=0)
        probability_buffer = probability_buffer.mean(axis=0)
        self.df_bias.loc[len(self.df_bias)] = [self.current_task+1, probability_buffer, 'buffer']

        print(self.df_bias)
        self.net.train(status)

        if (self.current_task + 1 == dataset.N_TASKS):
            path = base_path() + f'dataframes/bias/{self.args.dataset}_{self.args.buffer_size}_{self.args.seed}.csv'
            self.df_bias.to_csv(path, index=False)
        return