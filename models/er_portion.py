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


class ErPortion(ContinualModel):
    NAME = 'er_portion'
    #this needs task boundaries
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il']

    @staticmethod
    def get_parser() -> ArgumentParser:
        """
        Returns an ArgumentParser object with predefined arguments for the Er model.

        Besides the required `add_management_args` and `add_experiment_args`, this model requires the `add_rehearsal_args` to include the buffer-related arguments.
        """
        parser = ArgumentParser(description='Continual learning via Experience Replay with task boundaries. Only uses a portion of the buffer during replay.')
        add_rehearsal_args(parser)
        parser.add_argument('--portion', type=float, required=True,
                            help='Portion of samples used for replay.')
        return parser

    def __init__(self, backbone, loss, args, transform):
        """
        The ER model maintains a buffer of previously seen examples and uses them to augment the current batch during training.
        """
        super(ErPortion, self).__init__(backbone, loss, args, transform)
        self.buffer = Buffer(int(self.args.buffer_size * self.args.portion))
        self.extra_buffer = Buffer(self.args.buffer_size - self.buffer.buffer_size) #second buffer not holding samples not used during replay which can be used to refit the head


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
        status = self.net.training
        self.net.eval()

        examples_per_class = self.buffer.buffer_size // ((self.current_task + 1) * self.cpt)
        remainder = self.buffer.buffer_size % ((self.current_task + 1) * self.cpt)
        ones_indices = torch.randperm(self.n_seen_classes)[:remainder]
        remainder = torch.zeros(self.n_seen_classes)
        remainder[ones_indices] = 1

        examples_per_class_extra = self.extra_buffer.buffer_size // ((self.current_task + 1) * self.cpt)
        remainder_extra = self.extra_buffer.buffer_size % ((self.current_task + 1) * self.cpt)
        ones_indices = torch.randperm(self.n_seen_classes)[:remainder_extra]
        remainder_extra = torch.zeros(self.n_seen_classes)
        remainder_extra[ones_indices] = 1

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

        #second buffer
        if not self.extra_buffer.is_empty():
            buf_x, buf_lab, buf_tl = self.extra_buffer.get_all_data()
            self.extra_buffer.empty()

            for tl in buf_lab.unique():
                idx = tl == buf_lab
                ex, lab, tasklab = buf_x[idx], buf_lab[idx], buf_tl[idx]
                first = min(ex.shape[0], examples_per_class_extra + int(remainder_extra[tl].item()))
                self.extra_buffer.add_data(
                    examples=ex[:first],
                    labels=lab[:first],
                    task_labels=tasklab[:first]
                )

        # fdr add new task
        ce = torch.tensor([examples_per_class] * self.cpt)
        ce = (ce + remainder[self.n_past_classes:]).int()

        ce2 = torch.tensor([examples_per_class_extra] * self.cpt)
        ce2 = (ce2 + remainder_extra[self.n_past_classes:]).int()

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
        
        buf_x, buf_lab, buf_tl = self.buffer.get_all_data()
        buf_x, buf_lab, buf_tl = self.extra_buffer.get_all_data()
        self.net.train(status)
        return