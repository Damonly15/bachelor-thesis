# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch

from models.utils.continual_model import ContinualModel
from utils.args import add_rehearsal_args, ArgumentParser
from utils.buffer import Buffer


class Fdr(ContinualModel):
    NAME = 'fdr'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    @staticmethod
    def get_parser() -> ArgumentParser:
        parser = ArgumentParser(description='Continual learning via'
                                ' Function Distance Regularization.')
        add_rehearsal_args(parser)
        parser.add_argument('--alpha', type=float, required=True,
                            help='Penalty weight.')
        return parser

    def __init__(self, backbone, loss, args, transform):
        super(Fdr, self).__init__(backbone, loss, args, transform)
        self.buffer = Buffer(self.args.buffer_size)
        self.soft = torch.nn.Softmax(dim=1)
        self.logsoft = torch.nn.LogSoftmax(dim=1)

    def end_task(self, dataset):
        tng = self.net.training
        self.net.train()

        examples_per_task = self.args.buffer_size // self.current_task if self.current_task > 0 else self.args.buffer_size

        if not self.buffer.is_empty():
            buf_x, buf_lab, buf_log, buf_tl = self.buffer.get_all_data()
            self.buffer.empty()

            for ttl in buf_tl.unique():
                idx = (buf_tl == ttl)
                ex, lab, log, tasklab = buf_x[idx], buf_lab[idx], buf_log[idx], buf_tl[idx]
                first = min(ex.shape[0], examples_per_task)
                self.buffer.add_data(
                    examples=ex[:first],
                    labels=lab[:first],
                    logits=log[:first],
                    task_labels=tasklab[:first]
                )
        counter = 0
        with torch.no_grad():
            for i, data in enumerate(dataset.train_loader):
                inputs, labels, not_aug_inputs = data
                inputs = inputs.to(self.device)
                if self.args.training_setting == 'class-il':
                    task_label = None
                else:
                    task_label = self.current_task
                outputs = self.net.forward(inputs, task_label)
                                    
                if examples_per_task - counter < 0:
                    break
                self.buffer.add_data(examples=not_aug_inputs[:(examples_per_task - counter)],
                                    labels=labels[:(examples_per_task - counter)],
                                    logits=(outputs.detach().cpu())[:(examples_per_task - counter)],
                                    task_labels=(torch.ones(self.args.batch_size, dtype=torch.int64) * self.current_task)
                                                [:(examples_per_task - counter)])
                counter += self.args.batch_size

        self.net.train(tng)
        return

    def observe(self, inputs, labels, not_aug_inputs, epoch=None):
        batch_size = labels.shape[0]
        self.opt.zero_grad()
        tot_loss = 0

        if self.args.training_setting == 'class-il':
            task_labels = None
        else: 
            task_labels = torch.ones(batch_size,  dtype=torch.int64, device=self.device) * self.current_task
            labels = labels - (task_labels*self.cpt)

        if not self.buffer.is_empty():
            buf_inputs, _, buf_logits, buf_tasklabels = self.buffer.get_data(self.args.minibatch_size,
                                                            transform=self.transform, device=self.device)
            inputs=torch.cat((inputs, buf_inputs), dim=0)
            if self.args.training_setting == 'task-il':
                task_labels = torch.cat((task_labels, buf_tasklabels), dim=0)

        outputs = self.net.forward(inputs, task_label=task_labels)

        if not self.buffer.is_empty():
            loss_mse = torch.norm(outputs[batch_size:] - buf_logits, 2, 1).mean()
            loss_mse.backward(retain_graph=True)
            tot_loss += loss_mse.item()

        loss = self.loss(outputs[:batch_size], labels)
        loss.backward()
        tot_loss += loss.item()  

        self.opt.step()
        return tot_loss
