"""
This module implements the simplest form of incremental training, i.e., finetuning.
"""

# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import os
import numpy as np
import torch
import math
from collections import Counter

mammoth_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from models.utils.continual_model import ContinualModel
from utils.args import ArgumentParser
from utils.training import evaluate
from utils.status import progress_bar

class SgdSaveDomain(ContinualModel):
    """
    Implementation of the Sgd model for continual learning.
    """

    NAME = 'sgd_savedomain'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    @staticmethod
    def get_parser() -> ArgumentParser:
        parser = ArgumentParser(description='Finetuning baseline - simple incremental training.')
        return parser

    def __init__(self, backbone, loss, args, transform):
        super(SgdSaveDomain, self).__init__(backbone, loss, args, transform)
        self.args.debug_mode = 1
        self.true_epochs = self.args.n_epochs
        self.args.n_epochs = 1
        self.old_data = []

    def observe(self, inputs, labels, not_aug_inputs, epoch=None):
        return 0
    
    def end_task(self, dataset):
        """
        This version of joint training simply saves all data from previous tasks and then trains on all data at the end of the last one.
        """

        self.old_data.append(dataset.train_loader)
        # train
        if len(dataset.test_loaders) != dataset.N_TASKS:
            return

        all_inputs = []
        all_labels = []
        for source in self.old_data:
            for x, l, _ in source:
                all_inputs.append(x)
                all_labels.append(l)
        all_inputs = torch.cat(all_inputs)
        all_labels = torch.cat(all_labels)
        bs = self.args.batch_size
        scheduler = dataset.get_scheduler(self, self.args)

        for e in range(self.args.n_epochs):
            order = torch.randperm(len(all_inputs))
            for i in range(int(math.ceil(len(all_inputs) / bs))):
                inputs = all_inputs[order][i * bs: (i + 1) * bs]
                labels = all_labels[order][i * bs: (i + 1) * bs]
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                self.opt.zero_grad()
                outputs = self.net(inputs)
                loss = self.loss(outputs, labels.long())
                loss.backward()
                self.opt.step()
                progress_bar(i, int(math.ceil(len(all_inputs) / bs)), e, 'J', loss.item())

            if scheduler is not None:
                scheduler.step()

            results = evaluate(self, dataset)
            print(results[0])
            print(np.mean(results[0]))
    
    """
    def end_task(self, dataset):
        self.old_data.append(dataset.train_loader)

        # train
        if self.current_task + 1 != self.N_TASKS:
            return
        
        scheduler = dataset.get_scheduler(self, self.args)
        multiplier = 1 / self.N_TASKS
        for e in range(self.true_epochs): #do 20 epochs
            iterators = [iter(current) for current in self.old_data]
            not_finished = True

            while not_finished:
                self.opt.zero_grad()
                for current_iterator in iterators:
                    try:
                        data = next(current_iterator)
                    except StopIteration:
                        not_finished = False
                        break

                    inputs, labels, _ = data
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)

                    outputs = self.net(inputs)
                    loss = multiplier * self.loss(outputs, labels)
                    loss.backward()
                
                self.opt.step()

            if scheduler is not None:
                scheduler.step()

            results = evaluate(self, dataset)
            print(results[0])
            print(np.mean(results[0]))
    """

    """
    def end_task(self, dataset):

        self.old_data.append(dataset.train_loader)
        # train
        if len(dataset.test_loaders) != dataset.N_TASKS:
            return
        
        scheduler = dataset.get_scheduler(self, self.args)
        
        for e in range(self.true_epochs):
            iterators = [iter(current) for current in self.old_data]
            batches_inputs = []
            batches_labels = []
            for current in iterators:
                data = next(current)
                inputs, labels, _ = data
                batches_inputs.append(inputs)
                batches_labels.append(labels)
            choices = [i for i in range(1, 21)]

            while len(batches_inputs) > 0:
                sampled = random.choices(choices, k=self.args.batch_size)
                counts = Counter(sampled)
                results = [counts[i] for i in range(1, 21)]
                current_inputs = None
                current_labels = None
                for pos, r in enumerate(results):
                    if len(batches_inputs[pos]) < r:
                        try:
                            data = next(iterators[pos])
                            inputs, labels, _ = data
                            batches_inputs[pos] = torch.cat([batches_inputs[pos], inputs], dim = 0)
                            batches_labels[pos] = torch.cat([batches_labels[pos], labels], dim = 0)
                        except StopIteration:
                            iterators.pop(pos)
                            batches_inputs.pop(pos)
                            batches_labels.pop(pos)
                    else:
                        if current_inputs== None:
                            current_inputs = batches_inputs[pos][:r]
                            batches_inputs[pos] = batches_inputs[pos][r:]
                            current_labels = batches_labels[pos][:r]
                            batches_labels = batches_labels[pos][r:]
                        else:
                            current_inputs = torch.cat([current_inputs, batches_inputs[pos][:r]], dim=0)
                            batches_inputs[pos] = batches_inputs[pos][r:]
                            current_labels = torch.cat([current_labels, batches_labels[pos][:r]], dim=0)
                            batches_labels[pos] = batches_labels[pos][r:]




        #save model for later use
        if not os.path.exists(mammoth_path + f"/pretrained_models"):
            os.makedirs(mammoth_path + f"/pretrained_models")
        
        torch.save(self.net.state_dict(), mammoth_path + f"/pretrained_models/perm_mnist_15epochs.pth")
    """