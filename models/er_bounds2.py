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

from models.utils.continual_model import ContinualModel
from utils.args import add_rehearsal_args, ArgumentParser
from utils.buffer import Buffer
from utils.model_utils import adjust_outputs
from utils import create_if_not_exists
from utils.conf import base_path


class ErBounds2(ContinualModel):
    NAME = 'er_bounds2'
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
        super(ErBounds2, self).__init__(backbone, loss, args, transform)
        self.buffer = Buffer(self.args.buffer_size)
        self.extra_buffer = Buffer(self.args.buffer_size) #second buffer with different samples


    def observe(self, inputs, labels, not_aug_inputs, epoch=None):
        """
        ER trains on the current task using the data provided, but also augments the batch with data from the buffer.
        """

        self.opt.zero_grad()

        if self.args.training_setting == "class-il":
            if not self.buffer.is_empty():
                buf_inputs, buf_labels, _ = self.buffer.get_data(
                    self.args.minibatch_size, transform=self.transform, device=self.device)
                inputs = torch.cat((inputs, buf_inputs))
                labels = torch.cat((labels, buf_labels))

            outputs = self.net(inputs)
            loss = self.loss(outputs, labels)
            loss.backward()
        else:
            batch_size = labels.shape[0]
            if not self.buffer.is_empty():
                buf_inputs, buf_labels, buf_tasklabels = self.buffer.get_data(
                    self.args.minibatch_size, transform=self.transform, device=self.device)
                task_labels = torch.cat(((torch.ones(batch_size,  dtype=torch.int64, device=self.device) * self.current_task), buf_tasklabels))
                inputs = torch.cat((inputs, buf_inputs))
                labels = torch.cat((labels, buf_labels))
            else:
                task_labels = (torch.ones(batch_size, dtype=torch.int64, device=self.device) * self.current_task)

            outputs = self.net(inputs)
            outputs = adjust_outputs(outputs, task_labels, self._cpt)
            labels = labels - (task_labels * self._cpt)
            loss = self.loss(outputs, labels)
            loss.backward()                
        self.opt.step()

        return loss.item()
    
    def end_task(self, dataset): #Changed this for the paper, it is from xder. It makes sure, that every class has the same amount of samples in the buffer.
        status = self.net.training
        self.net.eval()

        examples_per_class = self.args.buffer_size // ((self.current_task + 1) * self.cpt)
        remainder = self.args.buffer_size % ((self.current_task + 1) * self.cpt)
        remainder2 = remainder

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

            #second buffer
            buf_x, buf_lab, buf_tl = self.extra_buffer.get_all_data()
            self.extra_buffer.empty()

            for tl in buf_lab.unique():
                idx = tl == buf_lab
                ex, lab, tasklab = buf_x[idx], buf_lab[idx], buf_tl[idx]
                if(remainder2 > 0):
                    first = min(ex.shape[0], examples_per_class + 1)
                    remainder2 -= 1
                else:
                    first = min(ex.shape[0], examples_per_class)
                self.extra_buffer.add_data(
                    examples=ex[:first],
                    labels=lab[:first],
                    task_labels=tasklab[:first]
                )

        # fdr add new task
        ce = torch.tensor([examples_per_class] * self.cpt).int()
        for i in range(remainder):
            ce[i] += 1 

        ce2 = torch.tensor([examples_per_class] * self.cpt).int()
        for i in range(remainder2):
            ce2[i] += 1 

        for data in dataset.train_loader:
            inputs, labels, not_aug_inputs = data
            
            if not (all(ce == 0) and all(ce2 == 0)):
                flags = torch.zeros(len(inputs)).bool()
                flags2 = torch.zeros(len(inputs)).bool()

                for j in range(len(inputs)):
                    if ce[labels[j] % self.cpt] > 0:
                        flags[j] = True
                        ce[labels[j] % self.cpt] -= 1
                    elif ce2[labels[j] % self.cpt] > 0:
                        flags2[j] = True
                        ce2[labels[j] % self.cpt] -= 1
                
                if not torch.all(~flags):
                    self.buffer.add_data(examples=not_aug_inputs[flags],
                                        labels=labels[flags],
                                        task_labels=(torch.ones(len(flags), dtype=torch.int64) * self.current_task)[flags])
                if not torch.all(~flags2):
                    self.extra_buffer.add_data(examples=not_aug_inputs[flags2],
                                    labels=labels[flags2],
                                    task_labels=(torch.ones(len(flags2), dtype=torch.int64) * self.current_task)[flags2])  
            else:
                break
        
        self.net.train(status)
        return