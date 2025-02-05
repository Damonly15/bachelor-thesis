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


class ErLabelsmoothing(ContinualModel):
    NAME = 'er_labelsmoothing'
    #this needs task boundaries
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il']

    @staticmethod
    def get_parser() -> ArgumentParser:
        """
        Returns an ArgumentParser object with predefined arguments for the Er model.

        Besides the required `add_management_args` and `add_experiment_args`, this model requires the `add_rehearsal_args` to include the buffer-related arguments.
        """
        parser = ArgumentParser(description='Continual learning via Experience Replay.')
        add_rehearsal_args(parser)
        return parser

    def __init__(self, backbone, loss, args, transform):
        """
        The ER model maintains a buffer of previously seen examples and uses them to augment the current batch during training.
        """
        super(ErLabelsmoothing, self).__init__(backbone, loss, args, transform)
        self.buffer = Buffer(self.args.buffer_size)

    def observe(self, inputs, labels, not_aug_inputs, epoch=None):
        """
        ER trains on the current task using the data provided, but also augments the batch with data from the buffer.
        """

        self.opt.zero_grad()
        if not self.buffer.is_empty():
            buf_inputs, buf_labels, _ = self.buffer.get_data(
                self.args.minibatch_size, transform=self.transform, device=self.device)
            inputs = torch.cat((inputs, buf_inputs))
            labels = torch.cat((labels, buf_labels))

        outputs = self.net(inputs)
        loss = self.loss(outputs, labels, label_smoothing=0.1)
        loss.backward()
        self.opt.step()

        return loss.item()
    
    def end_task(self, dataset):
        #make space in the buffer (each task has the same amount of samples in the buffer)
        current_task = self.current_task
        examples_per_task = self.args.buffer_size // (current_task+1)
        remainder = self.args.buffer_size % (current_task+1)

        if(not self.buffer.is_empty()):
            buf_x, buf_lab, buf_tl = self.buffer.get_all_data()
            self.buffer.empty()

            for ttl in buf_tl.unique():
                idx = (buf_tl == ttl)
                ex, lab, tasklab = buf_x[idx], buf_lab[idx], buf_tl[idx]
                if(remainder > 0):
                    first = min(ex.shape[0], examples_per_task + 1)
                    remainder -= 1
                else:
                    first = min(ex.shape[0], examples_per_task)

                self.buffer.add_data(
                    examples=ex[:first],
                    labels=lab[:first],
                    task_labels=tasklab[:first])

        #do some foreward passes to fill up buffer with samples from the current task
        counter = 0
        for data in dataset.train_loader:
            if hasattr(dataset.train_loader.dataset, 'logits'):
                _, labels, not_aug_inputs, _ = data
            else:
                _, labels, not_aug_inputs = data

            not_aug_inputs = not_aug_inputs.to(self.device)
            self.buffer.add_data(examples=not_aug_inputs[:(examples_per_task - counter)],
                                    labels=labels[:(examples_per_task - counter)],
                                    task_labels=(torch.ones(self.args.batch_size, dtype=torch.long) *
                                                current_task)[:(examples_per_task - counter)])
        
            counter += self.args.batch_size
            if examples_per_task - counter <= 0:
                break