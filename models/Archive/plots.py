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

class Plots(ContinualModel):
    NAME = 'plots'
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
        super(Plots, self).__init__(backbone, loss, args, transform)
        self.buffer = Buffer(self.args.buffer_size)
        self.running_buf_train_acc = 1.0
        self.epoch_counter = 0
        self.df_buffer = pd.DataFrame(columns=['model', 'buf_logits_accuracy', 'buf_train_accuracy', 'epoch'])
        self.df_bias = pd.DataFrame(columns=['model', 'current_task_probability', 'task'])
        self.all_dataloaders = []

        self.pretrained_model = get_pretrained(args)
        self.pretrained_model.to(self.device)
        self.pretrained_model.eval()

    @torch.no_grad()
    def evaluate_buffer(self, buf_labels, buf_logits):
        #evaluate logits accuracy
        buf_logits_correct = torch.argmax(buf_logits, dim=1) == buf_labels
        buf_logits_accuracy = buf_logits_correct.float().mean().item()
        return buf_logits_accuracy
    
    @torch.no_grad()
    def calculate_probability(self, network):
        probabilities = []
        with bn_track_stats(network, False):
            for iter in self.all_dataloaders:
                for data in iter:
                    if hasattr(iter, 'logits'):
                        inputs, _, _, _ = data
                    else:
                        inputs, _, _ = data

                    inputs = inputs.to(self.device)
                    outputs = network(inputs).detach().cpu()
                    probability = F.softmax(outputs, dim=-1)
                    probability = probability[:, self.n_past_classes:self.n_seen_classes].sum(dim=1)
                    probabilities.append(probability)

        probabilities = torch.cat(probabilities)
        return probabilities.mean(dim=0).item()

    def observe(self, inputs, labels, not_aug_inputs, epoch=None):
        if self.args.algorithm == "dertempbounds":
            if not self.buffer.is_empty() and epoch != self.epoch_counter:
                _, eval_labels, eval_logits, _ = self.buffer.get_all_data()
                buf_logits_accuracy = self.evaluate_buffer(eval_labels, eval_logits)
                self.df_buffer.loc[len(self.df_buffer)] = ["der_tempbounds", buf_logits_accuracy, self.running_buf_train_acc,  epoch + self.current_task * self.args.n_epochs]

                self.epoch_counter += 1

            self.opt.zero_grad()
            tot_loss = 0

            outputs = self.net(inputs)
            loss = self.loss(outputs, labels)
            loss.backward()
            tot_loss += loss.item()

            if not self.buffer.is_empty():
                buf_inputs, buf_labels, buf_logits, _ = self.buffer.get_data(
                    self.args.minibatch_size, transform=self.transform, device=self.device)
                buf_outputs = self.net(buf_inputs)
                buf_accuracy = self.evaluate_buffer(buf_labels, buf_outputs.detach())
                self.running_buf_train_acc = 0.95 * self.running_buf_train_acc + 0.05 * buf_accuracy

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
            if not self.buffer.is_empty() and epoch != self.epoch_counter:
                _, eval_labels, eval_logits = self.buffer.get_all_data()
                buf_logits_accuracy = self.evaluate_buffer(eval_labels, eval_logits)
                self.df_buffer.loc[len(self.df_buffer)] = ["der", buf_logits_accuracy, self.running_buf_train_acc,  epoch + self.current_task * self.args.n_epochs]

                self.epoch_counter += 1

            self.opt.zero_grad()
            tot_loss = 0

            outputs = self.net(inputs)
            loss = self.loss(outputs, labels)
            loss.backward()
            tot_loss += loss.item()

            if not self.buffer.is_empty():
                buf_inputs, buf_labels, buf_logits = self.buffer.get_data(
                    self.args.minibatch_size, transform=self.transform, device=self.device)
                buf_outputs = self.net(buf_inputs)
                loss_mse = self.args.alpha * F.mse_loss(buf_outputs, buf_logits)
                loss_mse.backward()
                tot_loss += loss_mse.item()

                buf_accuracy = self.evaluate_buffer(buf_labels, buf_outputs.detach())
                self.running_buf_train_acc = 0.95 * self.running_buf_train_acc + 0.05 * buf_accuracy
            
            self.opt.step()
            self.buffer.add_data(examples=not_aug_inputs, labels=labels, logits=outputs.data)

            return tot_loss

        elif self.args.algorithm == "derpretrained":
            if not self.buffer.is_empty() and epoch != self.epoch_counter:
                _, eval_labels, eval_logits, _ = self.buffer.get_all_data()
                buf_logits_accuracy = self.evaluate_buffer(eval_labels, eval_logits)
                self.df_buffer.loc[len(self.df_buffer)] = ["der_pretrained", buf_logits_accuracy, self.running_buf_train_acc,  epoch + self.current_task * self.args.n_epochs]

                self.epoch_counter += 1

            self.opt.zero_grad()
            tot_loss = 0

            outputs = self.net(inputs)
            loss = self.loss(outputs, labels)
            loss.backward()
            tot_loss += loss.item()

            if not self.buffer.is_empty():
                buf_inputs, buf_labels, buf_logits, _ = self.buffer.get_data(
                    self.args.minibatch_size, transform=self.transform, device=self.device)
                buf_outputs = self.net(buf_inputs)
                buf_accuracy = self.evaluate_buffer(buf_labels, buf_outputs.detach())
                self.running_buf_train_acc = 0.95 * self.running_buf_train_acc + 0.05 * buf_accuracy

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
        else:
            return 0
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
                        buf_x, buf_lab, buf_log, buf_tl = self.buffer.get_all_data()
                        self.buffer.empty()

                        for ttl in buf_tl.unique():
                            idx = (buf_tl == ttl)
                            ex, lab, log, tasklab = buf_x[idx], buf_lab[idx], buf_log[idx], buf_tl[idx]
                            if(remainder > 0):
                                first = min(ex.shape[0], examples_per_task + 1)
                                remainder -= 1
                            else:
                                first = min(ex.shape[0], examples_per_task)

                            self.buffer.add_data(
                                examples=ex[:first],
                                labels=lab[:first],
                                logits=log[:first],
                                task_labels=tasklab[:first])

                    #do some foreward passes to fill up buffer with samples from the current task
                    counter = 0
                    for data in dataset.train_loader:
                        if hasattr(dataset.train_loader.dataset, 'logits'):
                            inputs, labels, not_aug_inputs, _ = data
                        else:
                            inputs, labels, not_aug_inputs = data
                            
                        inputs = inputs.to(self.device)
                        not_aug_inputs = not_aug_inputs.to(self.device)
                        outputs = self.net(inputs)

                        if self.args.temperature <= 100:
                            outputs = F.log_softmax(outputs / self.args.temperature, dim=-1)
                        self.buffer.add_data(examples=not_aug_inputs[:(examples_per_task - counter)],
                                                labels=labels[:(examples_per_task - counter)],
                                                logits=outputs.detach()[:(examples_per_task - counter)],
                                                task_labels=(torch.ones(self.args.batch_size, dtype=torch.long) *
                                                            current_task)[:(examples_per_task - counter)])

                        counter += self.args.batch_size
                        if examples_per_task - counter <= 0:
                            break

            if not self.buffer.is_empty():
                _, eval_labels, eval_logits, _ = self.buffer.get_all_data()
                buf_logits_accuracy = self.evaluate_buffer(eval_labels, eval_logits)
                self.df_buffer.loc[len(self.df_buffer)] = ["der_tempbounds", buf_logits_accuracy, self.running_buf_train_acc,  (self.current_task+1) * self.args.n_epochs]

                self.epoch_counter = 0
            
            self.all_dataloaders.append(dataset.train_loader)
            probability = self.calculate_probability(self.net)
            self.df_bias.loc[len(self.df_bias)] = ["der_tempbounds", probability, (self.current_task+1)]
        elif self.args.algorithm == "der":  
            if not self.buffer.is_empty():
                _, eval_labels, eval_logits = self.buffer.get_all_data()
                buf_logits_accuracy = self.evaluate_buffer(eval_labels, eval_logits)
                self.df_buffer.loc[len(self.df_buffer)] = ["der", buf_logits_accuracy, self.running_buf_train_acc,  (self.current_task+1) * self.args.n_epochs]

                self.epoch_counter = 0
        elif self.args.algorithm == "derpretrained":
            with torch.no_grad():

                current_task = self.current_task
                examples_per_task = self.args.buffer_size // (current_task+1)
                remainder = self.args.buffer_size % (current_task+1)

                if(not self.buffer.is_empty()):
                    buf_x, buf_lab, buf_log, buf_tl = self.buffer.get_all_data()
                    self.buffer.empty()

                    for ttl in buf_tl.unique():
                        idx = (buf_tl == ttl)
                        ex, lab, log, tasklab = buf_x[idx], buf_lab[idx], buf_log[idx], buf_tl[idx]
                        if(remainder > 0):
                            first = min(ex.shape[0], examples_per_task + 1)
                            remainder -= 1
                        else:
                            first = min(ex.shape[0], examples_per_task)

                        self.buffer.add_data(
                            examples=ex[:first],
                            labels=lab[:first],
                            logits=log[:first],
                            task_labels=tasklab[:first])

                #do some foreward passes to fill up buffer with samples from the current task
                counter = 0
                for data in dataset.train_loader:
                    if hasattr(dataset.train_loader.dataset, 'logits'):
                        inputs, labels, not_aug_inputs, _ = data
                    else:
                        inputs, labels, not_aug_inputs = data
                        
                    inputs = inputs.to(self.device)
                    not_aug_inputs = not_aug_inputs.to(self.device)
                    outputs = self.pretrained_model(inputs)

                    if self.args.temperature <= 100:
                        outputs = F.log_softmax(outputs / self.args.temperature, dim=-1)
                    self.buffer.add_data(examples=not_aug_inputs[:(examples_per_task - counter)],
                                            labels=labels[:(examples_per_task - counter)],
                                            logits=outputs.detach()[:(examples_per_task - counter)],
                                            task_labels=(torch.ones(self.args.batch_size, dtype=torch.long) *
                                                        current_task)[:(examples_per_task - counter)])
                
                    counter += self.args.batch_size
                    if examples_per_task - counter <= 0:
                        break
                
            if not self.buffer.is_empty():
                _, eval_labels, eval_logits, _ = self.buffer.get_all_data()
                buf_logits_accuracy = self.evaluate_buffer(eval_labels, eval_logits)
                self.df_buffer.loc[len(self.df_buffer)] = ["der_pretrained", buf_logits_accuracy, self.running_buf_train_acc,  (self.current_task+1) * self.args.n_epochs]

                self.epoch_counter = 0

            self.all_dataloaders.append(dataset.train_loader)
            probability = self.calculate_probability(self.pretrained_model)
            self.df_bias.loc[len(self.df_bias)] = ["der_pretrained", probability, (self.current_task+1)]
        elif self.args.algorithm == "derpermuted":
            with torch.no_grad():
                self.all_dataloaders.append(dataset.train_loader)
                probabilities = []
                for iter in self.all_dataloaders:
                    for data in iter:
                        if hasattr(iter, 'logits'):
                            inputs, labels, _, _ = data
                        else:
                            inputs, labels, _ = data

                        inputs = inputs.to(self.device)
                        outputs = self.pretrained_model(inputs).detach()

                        #permute outputs
                        for pos in range(outputs.shape[0]):
                            fixed_pos = labels[pos].item()
                            sub_tensor = outputs[pos]
                            elements_to_permute = torch.cat((sub_tensor[:fixed_pos], sub_tensor[fixed_pos+1:]))
                            permuted_elements = elements_to_permute[torch.randperm(elements_to_permute.size(0))]
                            new_sub_tensor = torch.cat((permuted_elements[:fixed_pos], sub_tensor[fixed_pos:fixed_pos+1], permuted_elements[fixed_pos:]))
                            outputs[pos] = new_sub_tensor
                        outputs = outputs.cpu()

                        probability = F.softmax(outputs, dim=-1)
                        probability = probability[:, self.n_past_classes:self.n_seen_classes].sum(dim=1)
                        probabilities.append(probability)

                probabilities = torch.cat(probabilities)
                probability = probabilities.mean(dim=0).item()
                self.df_bias.loc[len(self.df_bias)] = ["der_permuted", probability, (self.current_task+1)]
        elif self.args.algorithm == "derpretrained2":
            with torch.no_grad():
                self.all_dataloaders.append(dataset.train_loader)
                probabilities = []
                for iter in self.all_dataloaders:
                    for data in iter:
                        if hasattr(iter, 'logits'):
                            inputs, labels, _, _ = data
                        else:
                            inputs, labels, _ = data

                        inputs = inputs.to(self.device)
                        intermediate = self.pretrained_model(inputs).detach()
                        
                        outputs = torch.zeros_like(intermediate)
                        outputs[:, self.n_past_classes:self.n_seen_classes] = intermediate[:, self.n_past_classes:self.n_seen_classes]
                        outputs = outputs.cpu()

                        probability = F.softmax(outputs, dim=-1)
                        probability = probability[:, self.n_past_classes:self.n_seen_classes].sum(dim=1)
                        probabilities.append(probability)

                probabilities = torch.cat(probabilities)
                probability = probabilities.mean(dim=0).item()
                self.df_bias.loc[len(self.df_bias)] = ["der_pretrained2", probability, (self.current_task+1)]
        elif self.args.algorithm == "erlabelsmoothing":
            self.all_dataloaders.append(dataset.train_loader)
            probabilities = []
            for iter in self.all_dataloaders:
                for data in iter:
                    if hasattr(iter, 'logits'):
                        _, labels, _, _ = data
                    else:
                        _, labels, _ = data

                    labels_smoothed = F.one_hot(labels, num_classes=dataset.N_CLASSES).float()
                    true_label = 1.0 - 0.1
                    other_label = 0.1 / dataset.N_CLASSES

                    labels_smoothed *= true_label
                    labels_smoothed += other_label
                    probability = labels_smoothed[:, self.n_past_classes:self.n_seen_classes].sum(dim=1)
                    probabilities.append(probability)

            probabilities = torch.cat(probabilities)
            probability = probabilities.mean(dim=0).item()
            self.df_bias.loc[len(self.df_bias)] = ["er_labelsmoothing", probability, (self.current_task+1)]

        if self.current_task + 1 == self.N_TASKS:
            self.df_buffer.to_csv((path + "/buffer_accuracy/" + self.args.dataset + "/" + self.args.algorithm + "_buffersize_" + 
                            str(self.args.buffer_size) + "_" + str(self.args.seed) + ".txt"), sep='\t', index=False)  
            self.df_bias.to_csv((path + "/bias/" + self.args.dataset + "/" + self.args.algorithm + "_buffersize_" + 
                            str(self.args.buffer_size) + "_" + str(self.args.seed) + ".txt"), sep='\t', index=False)    

        
        