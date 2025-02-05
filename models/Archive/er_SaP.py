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


class ErSaP(ContinualModel):
    NAME = 'er_SaP'
    #this needs task boundaries
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il']

    @staticmethod
    def get_parser() -> ArgumentParser:
        """
        Returns an ArgumentParser object with predefined arguments for the Er model.

        Besides the required `add_management_args` and `add_experiment_args`, this model requires the `add_rehearsal_args` to include the buffer-related arguments.
        """
        parser = ArgumentParser(description='Continual learning via Experience Replay with task boundaries. Shrinks and perturbs the weights to get richer training on new tasks')
        add_rehearsal_args(parser)
        parser.add_argument('--shrink', type=float, required=True,
                            help='Weight shrinking.')
        parser.add_argument('--perturb', type=float, required=True,
                            help='Weight pertubation with gaussian.')
        return parser

    def __init__(self, backbone, loss, args, transform):
        """
        The ER model maintains a buffer of previously seen examples and uses them to augment the current batch during training.
        """
        super(ErSaP, self).__init__(backbone, loss, args, transform)
        self.buffer = Buffer(self.args.buffer_size)

    def observe(self, inputs, labels, not_aug_inputs, epoch=None):
        """
        ER trains on the current task using the data provided, but also augments the batch with data from the buffer.
        """
        if self.current_task==0:
            return 0
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
    
    def begin_task(self, dataset):
        return
        if self.current_task > 0: #shrink and perturb the weights
            with torch.no_grad():  # No gradients required for this operation
                for param in self.net.parameters():
                    if param.requires_grad:  # Ensure the parameter is trainable
                        # Shrink weights
                        param.mul_(self.args.shrink)
                        # Add perturbation noise
                        noise = torch.randn_like(param) * self.args.perturb
                        param.add_(noise)
    
    def end_task(self, dataset): #Changed this for the paper, it is from xder. It makes sure, that every class has the same amount of samples in the buffer.
            
        examples_per_class = self.args.buffer_size // ((self.current_task + 1) * self.cpt)
        remainder = self.args.buffer_size % ((self.current_task + 1) * self.cpt)

        print(examples_per_class)
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
