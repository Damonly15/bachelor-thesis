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

class ErBuf(ContinualModel):
    NAME = 'er_buf'
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
        super(ErBuf, self).__init__(backbone, loss, args, transform)
        self.buffer = Buffer(self.args.buffer_size)
        self.buffer_refitting = Buffer(self.args.buffer_size)

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

    def end_task(self, dataset): #Changed this for the paper, it is from xder. It makes sure, that every class has the same amount of samples in the buffer.
    
        examples_per_class = self.args.buffer_size // ((self.current_task + 1) * self.cpt)
        remainder = self.args.buffer_size % ((self.current_task + 1) * self.cpt)
        ones_indices = torch.randperm(self.n_seen_classes)[:remainder]
        remainder = torch.zeros(self.n_seen_classes)
        remainder[ones_indices] = 1

        if self.current_task == 0:
            examples_per_class_refitting = (self.args.buffer_size // ((self.current_task + 1) * self.cpt)) - examples_per_class
        else:
            examples_per_class_refitting = (self.args.buffer_size // (self.current_task * self.cpt)) - examples_per_class

        # fdr reduce coreset
        if not self.buffer.is_empty():
            buf_x, buf_lab, buf_tl = self.buffer.get_all_data()
            self.buffer.empty()
            self.buffer_refitting.empty() #store availible samples in extra buffer which we then use to refit the head.

            for tl in buf_lab.unique():
                idx = tl == buf_lab
                ex, lab, tasklab = buf_x[idx], buf_lab[idx], buf_tl[idx]
                first = min(ex.shape[0], examples_per_class + int(remainder[tl].item()))
                
                self.buffer.add_data(
                    examples=ex[:first],
                    labels=lab[:first],
                    task_labels=tasklab[:first]
                )

                self.buffer_refitting.add_data(
                    examples=ex[first:],
                    labels=lab[first:],
                    task_labels=tasklab[first:]
                )

        # fdr add new task
        ce = torch.tensor([examples_per_class] * self.cpt)
        ce = (ce + remainder[self.n_past_classes:]).int() 
        ce_refitting = torch.tensor([examples_per_class_refitting] * self.cpt).int()

        for data in dataset.train_loader:
            inputs, labels, not_aug_inputs = data
            if all(ce == 0) and all(ce_refitting == 0):
                break

            flags = torch.zeros(len(inputs)).bool()
            flags_refitting = torch.zeros(len(inputs)).bool()
            for j in range(len(flags)):
                if ce[labels[j] % self.cpt] > 0:
                    flags[j] = True
                    ce[labels[j] % self.cpt] -= 1
                elif ce_refitting[labels[j] % self.cpt] > 0:
                    flags_refitting[j] = True
                    ce_refitting[labels[j] % self.cpt] -= 1

            if not torch.all(~flags):
                self.buffer.add_data(examples=not_aug_inputs[flags],
                                    labels=labels[flags],
                                    task_labels=(torch.ones(len(flags), dtype=torch.int64) * self.current_task)[flags])
            
            if not torch.all(~flags_refitting):
                self.buffer_refitting.add_data(examples=not_aug_inputs[flags_refitting],
                                    labels=labels[flags_refitting],
                                    task_labels=(torch.ones(len(flags), dtype=torch.int64) * self.current_task)[flags_refitting])

        return
    
    """
    def refit_head(self, dataset):
        #bias correction
        status = self.net.training
        self.net.eval()

        if self.args.training_setting == 'class-il' and self.current_task > 0:
            self.refitting_opt = Adam(self.net.classifier.parameters(), lr=0.001, weight_decay=self.args.wd_head)

            for l in range(self.args.refitting_iters):
                buf_inputs, buf_labels, _ = self.buffer.get_data(
                    self.args.minibatch_size, transform=self.transform, device=self.device)

                self.refitting_opt.zero_grad()
                self.opt.zero_grad()

                outputs = self.net.forward(buf_inputs)
                loss = self.loss(outputs, buf_labels)
                loss.backward()
                self.refitting_opt.step()
            del self.refitting_opt
        elif self.args.training_setting == 'task-il' and self.current_task > 0:
            for task in range(self.current_task+1):
                buffer_current_task = Buffer(100000)
                buf_x, buf_lab, buf_tl = self.buffer_refitting.get_all_data()
                task_mask = buf_tl == task
                buffer_current_task.add_data(
                    examples=buf_x[task_mask],
                    labels=buf_lab[task_mask],
                    task_labels=buf_tl[task_mask]
                    )
                
                self.refitting_opt = Adam(self.net.classifier[task].parameters(), lr=0.001, weight_decay=self.args.wd_head)

                for l in range(self.args.refitting_iters):
                    buf_inputs, buf_labels, _ = buffer_current_task.get_data(
                        self.args.minibatch_size, transform=self.transform, device=self.device)

                    self.refitting_opt.zero_grad()
                    self.opt.zero_grad()

                    outputs = self.net.forward(buf_inputs, task_label=task)
                    buf_labels = buf_labels - task*self.cpt
                    loss = self.loss(outputs, buf_labels)
                    loss.backward()
                    self.refitting_opt.step()
                del self.refitting_opt

        self.net.train(status)"
        """