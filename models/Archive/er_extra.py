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
from torch.func import vmap, grad, functional_call
from functorch import make_functional

from models.utils.continual_model import ContinualModel
from utils.args import add_rehearsal_args, ArgumentParser
from utils.buffer import Buffer
from utils.training import evaluate
from utils.feature_forgetting import feature_forgetting_cil

class ErExtra(ContinualModel):
    NAME = 'er_extra'
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
        super(ErExtra, self).__init__(backbone, loss, args, transform)
        self.buffer = Buffer(self.args.buffer_size)
        self.buffer_nobuffer = Buffer(self.dataset.N_SAMPLES - self.args.buffer_size)

        remainder = self.args.buffer_size % (self.dataset.N_CLASSES)
        ones_indices = torch.randperm(self.dataset.N_CLASSES)[:remainder]
        self.remainder = torch.zeros(self.dataset.N_CLASSES)
        self.remainder[ones_indices] = 1  

        self.gradient_sv = []
        self.input_storage = []
        self.labels_storage = []
        self.tasklabels_storage = []
        if self.net is not None:
            self.fmodel = self.net  # just reference the module
            self.fparams = dict(self.net.named_parameters())  # get the parameters

    def observe(self, inputs, labels, not_aug_inputs, epoch=None):
        """
        ER trains on the current task using the data provided, but also augments the batch with data from the buffer.
        """
        if inputs.shape[0] != self.dataset.get_batch_size():
            return 0.0

        self.opt.zero_grad()

        task_labels = torch.ones(labels.shape[0], dtype=torch.int64, device=self.device) * self.current_task
        if self.args.training_setting == 'task-il':
            labels = labels - (task_labels*self.cpt)

        if not self.buffer.is_empty():
            buf_inputs, buf_labels, buf_tasklabels = self.buffer.get_data(
                self.args.minibatch_size, transform=self.transform, device=self.device)
            
            task_labels = torch.cat((task_labels, buf_tasklabels), dim=0)
            if self.args.training_setting == 'task-il':
                buf_labels = buf_labels - (buf_tasklabels*self.cpt)  
            inputs = torch.cat((inputs, buf_inputs), dim=0)
            labels = torch.cat((labels, buf_labels), dim=0)


        if epoch + 1 == self.dataset.get_epochs():
            self.input_storage.append(inputs)
            self.labels_storage.append(labels)
            self.tasklabels_storage.append(task_labels)
            if len(self.input_storage) == 1:
                status = self.net.training
                self.net.eval()

                inputs = torch.cat(self.input_storage, dim=0)
                self.input_storage = []
                labels = torch.cat(self.labels_storage, dim=0)
                self.labels_storage = []
                task_labels = torch.cat(self.tasklabels_storage, dim=0)
                self.tasklabels_storage = []

                # Vectorized per-sample gradient computation
                per_sample_grads = vmap(self._grad_per_sample, in_dims=(None, 0, 0, 0))(
                    self.fparams, inputs, labels, task_labels
                )

                batch_size = inputs.shape[0]

                # Flatten each parameter's gradient per sample
                flattened_grads = [g.detach().reshape(batch_size, -1) for g in per_sample_grads.values()]

                # Concatenate all flattened gradients along the feature dimension
                grad_matrix = torch.cat(flattened_grads, dim=1)

                # Compute SVD
                U, S, Vh = torch.linalg.svd(grad_matrix, full_matrices=False)
                self.gradient_sv.append(S.cpu())

                self.net.train(status)
            else:
                return 0

        outputs = self.net(inputs, task_label=task_labels)
        loss = self.loss(outputs, labels)
        loss.backward()
        self.opt.step()

        return loss.item()
        
    
    def _compute_loss(self, params, x, y, task_label):
        preds = functional_call(
            self.fmodel,
            params,
            (x, task_label),  # matches forward(x, task_label)
        )
        loss = self.loss(preds, y)
        return loss

    def _grad_per_sample(self, params, x, y, task_label):
        return grad(self._compute_loss)(params, x.unsqueeze(0), y.unsqueeze(0), task_label.unsqueeze(0))


    def end_task(self, dataset): #Changed this for the paper, it is from xder. It makes sure, that every class has the same amount of samples in the buffer.
        examples_per_class = self.args.buffer_size // dataset.N_CLASSES

        ce = torch.tensor([examples_per_class] * self.cpt) + self.remainder[self.n_past_classes:self.n_seen_classes]

        for data in dataset.train_loader:
            inputs, labels, not_aug_inputs = data

            flags = torch.zeros(len(inputs)).bool()
            flags_nobuffer = torch.zeros(len(inputs)).bool()
            
            for j in range(len(flags)):
                if ce[labels[j] % self.cpt] > 0:
                    flags[j] = True
                    ce[labels[j] % self.cpt] -= 1
                else:
                    flags_nobuffer[j] = True

            if not torch.all(~flags):
                self.buffer.add_data(examples=not_aug_inputs[flags],
                                    labels=labels[flags],
                                    task_labels=(torch.ones(len(flags), dtype=torch.int64) * self.current_task)[flags])
                
            if not torch.all(~flags_nobuffer):
                self.buffer_nobuffer.add_data(examples=not_aug_inputs[flags_nobuffer],
                                    labels=labels[flags_nobuffer],
                                    task_labels=(torch.ones(len(flags), dtype=torch.int64) * self.current_task)[flags_nobuffer])

        return