# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import torch
from torch.nn import functional as F
import pandas as pd

path = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/plot_dataframes"

from models.utils.continual_model import ContinualModel
from utils.args import ArgumentParser, add_experiment_args, add_management_args, add_rehearsal_args
from utils.buffer import Buffer
from utils.batch_norm import bn_track_stats
from utils.metrics import get_pretrained          

class Plots2(ContinualModel):
    NAME = 'plots2'
    #this needs task boundaries but no test time oracle
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il']

    @staticmethod
    def get_parser() -> ArgumentParser:
        parser = ArgumentParser(description='Continual learning via'
                                ' Dark Experience Replay with KL loss and task boundaries.')
        add_rehearsal_args(parser)
        parser.add_argument('--alpha', type=float, required=True,
                            help='Penalty weight.')
        parser.add_argument('--temperature', type=float, required=True,
                            help='Temperature of softmax.')
        parser.add_argument('--algorithm', type=str, required=True,
                            help='Which algorithm to use.')
        return parser

    def __init__(self, backbone, loss, args, transform):
        super(Plots2, self).__init__(backbone, loss, args, transform)
        self.buffer = Buffer(self.args.buffer_size)
        self.epoch_counter = 0
        self.running_percentage = 1
        self.dataset = None
        self.df_BN = pd.DataFrame(columns=['model', 'train_accuracy', 'percentage', 'epoch', 'task'])

    def observe(self, inputs, labels, not_aug_inputs, epoch=None):
        if self.args.algorithm == "dertempbounds":
            if epoch != self.epoch_counter:
                accs_eval = evaluate(self, self.dataset)
                self.df_BN.loc[len(self.df_BN)] = ["der_tempbounds", accs_eval[0], self.running_percentage, epoch + self.current_task * self.args.n_epochs, self.current_task+1]
                self.epoch_counter += 1
            self.opt.zero_grad()
            tot_loss = 0

            outputs = self.net(inputs)
            loss = self.loss(outputs, labels)
            loss.backward()
            tot_loss += loss.item()

            with torch.no_grad():
                self.net.eval()
                outputs2 = self.net(inputs)
                self.net.train()
                current = calculate(outputs.detach(), outputs2.detach())
                self.running_percentage = 0.95 * self.running_percentage + 0.05 *current

            if not self.buffer.is_empty():
                buf_inputs, buf_logits, _ = self.buffer.get_data(
                    self.args.minibatch_size, transform=self.transform, device=self.device)
                buf_outputs = self.net(buf_inputs)

                if self.args.temperature <= 100:
                    buf_outputs = F.log_softmax(buf_outputs / self.args.temperature, dim=-1)
                    loss_kl = self.args.alpha * self.args.temperature**2 * F.kl_div(buf_outputs, buf_logits, reduction='batchmean', log_target=True)
                    loss_kl.backward()
                    tot_loss += loss_kl.item()
                else:
                    loss_mse = self.args.alpha * F.mse_loss(buf_outputs, buf_logits)
                    loss_mse.backward()
                    tot_loss += loss_mse.item()

            self.opt.step()
            return tot_loss
        elif self.args.algorithm == "der":
            if epoch != self.epoch_counter:
                accs_eval = evaluate(self, self.dataset)
                self.df_BN.loc[len(self.df_BN)] = ["der", accs_eval[0], self.running_percentage, epoch + self.current_task * self.args.n_epochs, self.current_task+1]
                self.epoch_counter += 1

            self.opt.zero_grad()
            tot_loss = 0

            outputs = self.net(inputs)
            loss = self.loss(outputs, labels)
            loss.backward()
            tot_loss += loss.item()

            with torch.no_grad():
                self.net.eval()
                outputs2 = self.net(inputs)
                self.net.train()
                current = calculate(outputs.detach(), outputs2.detach())
                self.running_percentage = 0.95 * self.running_percentage + 0.05 *current

            if not self.buffer.is_empty():
                buf_inputs, buf_logits = self.buffer.get_data(
                    self.args.minibatch_size, transform=self.transform, device=self.device)
                buf_outputs = self.net(buf_inputs)
                loss_mse = self.args.alpha * F.mse_loss(buf_outputs, buf_logits)
                loss_mse.backward()
                tot_loss += loss_mse.item()
            
            self.opt.step()
            self.buffer.add_data(examples=not_aug_inputs, logits=outputs.data)
            return tot_loss
    
    def begin_task(self, dataset):
        self.dataset = dataset

    def end_task(self, dataset):
        if self.args.algorithm == "dertempbounds":
            with torch.no_grad():
                #deactivate running mean and variance of batchnorm layers
                with bn_track_stats(self.net, False):
                    #make space in the buffer (each task has the same amount of samples in the buffer)
                    current_task = self.current_task
                    examples_per_task = self.args.buffer_size // (current_task+1)
                    remainder = self.args.buffer_size % (current_task+1)

                    if(not self.buffer.is_empty()):
                        buf_x, buf_log, buf_tl = self.buffer.get_all_data()
                        self.buffer.empty()

                        for ttl in buf_tl.unique():
                            idx = (buf_tl == ttl)
                            ex, log, tasklab = buf_x[idx], buf_log[idx], buf_tl[idx]
                            if(remainder > 0):
                                first = min(ex.shape[0], examples_per_task + 1)
                                remainder -= 1
                            else:
                                first = min(ex.shape[0], examples_per_task)

                            self.buffer.add_data(
                                examples=ex[:first],
                                logits=log[:first],
                                task_labels=tasklab[:first])

                    #do some foreward passes to fill up buffer with samples from the current task
                    counter = 0
                    for data in dataset.train_loader:
                        if hasattr(dataset.train_loader.dataset, 'logits'):
                            inputs, _, not_aug_inputs, _ = data
                        else:
                            inputs, _, not_aug_inputs = data
                            
                        inputs = inputs.to(self.device)
                        not_aug_inputs = not_aug_inputs.to(self.device)
                        outputs = self.net(inputs)
                        
                        if self.args.temperature <= 100:
                            outputs = F.log_softmax(outputs / self.args.temperature, dim=-1)
                        self.buffer.add_data(examples=not_aug_inputs[:(examples_per_task - counter)],
                                                logits=outputs.detach()[:(examples_per_task - counter)],
                                                task_labels=(torch.ones(self.args.batch_size, dtype=torch.long) *
                                                            current_task)[:(examples_per_task - counter)])

                        counter += self.args.batch_size
                        if examples_per_task - counter <= 0:
                            break
            accs_eval = evaluate(self, dataset)
            self.df_BN.loc[len(self.df_BN)] = ["der_tempbounds", accs_eval[0], self.running_percentage, (self.current_task+1) * self.args.n_epochs, self.current_task+1]
            self.epoch_counter = 0
        elif self.args.algorithm == "der":  
            accs_eval = evaluate(self, dataset)
            self.df_BN.loc[len(self.df_BN)] = ["der_eval", accs_eval[0], self.running_percentage, (self.current_task+1) * self.args.n_epochs, self.current_task+1]
            self.epoch_counter = 0
            
        if self.current_task + 1 == self.N_TASKS:
            self.df_BN.to_csv((path + "/BN/" + self.args.dataset + "/" + self.args.algorithm + "_buffersize_" + 
                            str(self.args.buffer_size) + "_" + str(self.args.seed) + ".txt"), sep='\t', index=False)      

def calculate(outputs, outputs2):
    outputs_correct = torch.argmax(outputs, dim=1) == torch.argmax(outputs2, dim=1)
    buf_logits_accuracy = outputs_correct.float().mean().item()
    return buf_logits_accuracy

@torch.no_grad()
def evaluate(model, dataset, last=True):
        status = model.net.training
        model.net.eval()
        accs_eval = []
        n_classes = dataset.get_offsets()[1]
        for k, test_loader in enumerate(dataset.test_loaders):
            if last and k < len(dataset.test_loaders) - 1:
                continue
            correct_eval, total = 0.0, 0.0
            test_iter = iter(test_loader)

            while True:
                try:
                    data = next(test_iter)
                except StopIteration:
                    break
                inputs, labels = data
                inputs, labels = inputs.to(model.device), labels.to(model.device)
                outputs = model(inputs)

                _, pred = torch.max(outputs[:, :n_classes].data, 1)
                correct_eval += torch.sum(pred == labels).item()
                total += labels.shape[0]

            accs_eval.append(correct_eval / total * 100)
        model.net.train(status)
        
        return accs_eval