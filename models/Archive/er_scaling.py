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
import copy
import torch.optim as optim

from models.utils.continual_model import ContinualModel
from utils.args import add_rehearsal_args, ArgumentParser
from utils.buffer import Buffer
from utils.metrics import adjust_outputs
from utils.feature_forgetting import evaluate_previous
from utils.conf import base_path
from utils import create_if_not_exists


class ErScaling(ContinualModel):
    NAME = 'er_scaling'
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
        super(ErScaling, self).__init__(backbone, loss, args, transform)
        self.buffer = Buffer(self.args.buffer_size)
        self.net2 = None
        self.opt2 = None

    def observe(self, inputs, labels, not_aug_inputs, epoch=None):
        """
        ER trains on the current task using the data provided, but also augments the batch with data from the buffer.
        """
        tot_loss = 0.0
        self.opt.zero_grad()

        if not self.buffer.is_empty():
            buf_inputs, buf_labels, _ = self.buffer.get_data(
                self.args.minibatch_size, transform=self.transform, device=self.device)
            inputs = torch.cat((inputs, buf_inputs))
            labels = torch.cat((labels, buf_labels))

        outputs = self.net(inputs)
        loss = self.loss(outputs, labels)
        loss.backward()              
        self.opt.step()

        if self.net2 is not None:
            self.opt2.zero_grad()
            buf_outputs2 = self.net2(buf_inputs)
            loss2 = self.loss(buf_outputs2, buf_labels)
            loss2.backward()
            tot_loss += loss2.item()
            self.opt2.step()
        return tot_loss + loss.item()
    
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

        if self.current_task + 2 == self.N_TASKS:
            #second last task, copy network
            self.net2 = copy.deepcopy(self.net)
            self.opt2 = optim.SGD(self.net2.parameters(), lr=self.args.lr,
                                    weight_decay=self.args.optim_wd,
                                    momentum=self.args.optim_mom,
                                    nesterov=self.args.optim_nesterov == 1)

        elif self.current_task + 1 == self.N_TASKS:
            #last task, evaluate second model
            accuracy, _ = evaluate_previous(self.net2, dataset, self.device)
            mean_accuracy = sum(accuracy) / len(accuracy)

            wrargs = (vars(self.args)).copy()
            target_folder = base_path() + "results/"

            #here we log to class-il accuracies
            wrargs['accmean_task' + str(dataset.N_TASKS)] = mean_accuracy

            for i, fa in enumerate(accuracy):
                wrargs['accuracy_' + str(i + 1) + '_task' + str(dataset.N_TASKS)] = fa

            create_if_not_exists(target_folder + self.args.training_setting)
            create_if_not_exists(target_folder + self.args.training_setting +
                                "/" + self.args.dataset)
            create_if_not_exists(target_folder + self.args.training_setting +
                                "/" + self.args.dataset + "/er_scaling_buffer")

            path = target_folder + self.args.training_setting + "/" + self.args.dataset\
                + "/er_scaling_buffer/logs.txt"
            print("Logging Class-IL results and arguments in " + path)
            with open(path, 'a') as f:
                f.write(str(wrargs) + '\n')