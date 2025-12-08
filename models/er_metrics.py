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
import math
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd

from models.utils.continual_model import ContinualModel
from utils.args import add_rehearsal_args, ArgumentParser
from utils.buffer import Buffer
from utils.training import evaluate
from utils.feature_forgetting import get_features
from utils.conf import base_path
from utils import create_if_not_exists

class ErMetrics(ContinualModel):
    NAME = 'er_metrics'
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
        super(ErMetrics, self).__init__(backbone, loss, args, transform)
        self.buffer = Buffer(self.args.buffer_size)

        remainder = self.args.buffer_size % (self.dataset.N_CLASSES_PER_TASK*self.dataset.N_TASKS)
        ones_indices = torch.randperm(self.dataset.N_CLASSES_PER_TASK*self.dataset.N_TASKS)[:remainder]
        self.remainder = torch.zeros(self.dataset.N_CLASSES_PER_TASK*self.dataset.N_TASKS)
        self.remainder[ones_indices] = 1 

        self.overall_batch_size = self.args.batch_size + self.args.minibatch_size
        self.args.batch_size = self.overall_batch_size
        self.args.minibatch_size = 0
        self.original_epochs = self.args.n_epochs

        self.iters_counter = 0
        self.current_iters = 0
        if self.dataset.N_SAMPLES == 50000 and self.args.optimizer=='adamw':
            self.eval_iters = 10
        elif self.dataset.N_SAMPLES == 50000:
            self.eval_iters = 1000
        elif self.dataset.N_SAMPLES == 100000:
            self.eval_iters = 500
        elif self.dataset.N_SAMPLES == 12000:
            self.eval_iters = 50

        self.within_var = pd.DataFrame(columns=['iter', 'training_task'] + [f'task{i}' for i in range(3)])
        self.between_var = pd.DataFrame(columns=['iter', 'training_task'] + [f'task{i}' for i in range(3)])
        self.norm_train = pd.DataFrame(columns=['iter', 'training_task'] + [f'task{i}' for i in range(3)])
        self.norm_test = pd.DataFrame(columns=['iter', 'training_task'] + [f'task{i}' for i in range(3)])

        self.NC2_off_diagonal = pd.DataFrame(columns=['iter', 'training_task'] + [f'task{i}' for i in range(3)])
        self.var_off_diagonal = pd.DataFrame(columns=['iter', 'training_task'] + [f'task{i}' for i in range(3)])
        self.NC2_between_tasks = pd.DataFrame(columns=['iter', 'training_task'] + ['value'])
        self.var_between_tasks = pd.DataFrame(columns=['iter', 'training_task'] + ['value'])
        self.NC3 = pd.DataFrame(columns=['iter', 'training_task'] + [f'task{i}' for i in range(3)])

        self.hyperspher_task_mean = pd.DataFrame(columns=['iter', 'training_task'] + [f'task{i}' for i in range(3)])
        self.hyperspher_task_var = pd.DataFrame(columns=['iter', 'training_task'] + [f'task{i}' for i in range(3)])
        self.hyperspher_total_mean = pd.DataFrame(columns=['iter', 'training_task'] + ['value'])
        self.hyperspher_total_var = pd.DataFrame(columns=['iter', 'training_task'] + ['value'])

        self.dataset = None

    def observe(self, inputs, labels, not_aug_inputs, epoch=None):
        """
        ER trains on the current task using the data provided, but also augments the batch with data from the buffer.
        """

        if self.current_iters % self.eval_iters == 0:
            self.log_NC_metrics()
        self.current_iters += 1
        self.iters_counter += 1

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

        outputs = self.net.forward(inputs, task_label=task_labels)
        loss = self.loss(outputs[:, :self.n_seen_classes], labels)
        loss.backward()
                      
        self.opt.step()

        return loss.item()

    def end_task(self, dataset): #Changed this for the paper, it is from xder. It makes sure, that every class has the same amount of samples in the buffer.
        examples_per_class = self.args.buffer_size // (dataset.N_CLASSES_PER_TASK * dataset.N_TASKS)  
        ce = torch.tensor([examples_per_class] * self.cpt) + self.remainder[self.current_task*self.cpt:(self.current_task+1)*self.cpt]

        for data in dataset.train_loader:
            inputs, labels, not_aug_inputs = data

            flags = torch.zeros(len(inputs)).bool()
            
            for j in range(len(flags)):
                if ce[labels[j] % self.cpt] > 0:
                    flags[j] = True
                    ce[labels[j] % self.cpt] -= 1

            self.buffer.add_data(examples=not_aug_inputs[flags],
                                labels=labels[flags],
                                task_labels=(torch.ones(len(flags), dtype=torch.int64) * self.current_task)[flags])


        if self.args.buffer_size != 0:
            self.args.batch_size = math.ceil(self.overall_batch_size / (self.current_task+2))
            self.args.minibatch_size = self.overall_batch_size - self.args.batch_size
            self.args.n_epochs = math.ceil(self.original_epochs * (self.args.batch_size / self.overall_batch_size))

        self.current_iters = 0

        if self.current_task+1 == self.args.stop_after:
            if self.args.training_setting == 'task-il':
                target_folder = base_path() + 'results/task-il/' + self.args.dataset + '/' + self.args.model + '/dataframes'
            else:
                target_folder = base_path() + 'results/' + dataset.SETTING + '/' + self.args.dataset + '/' + self.args.model + '/dataframes'
            create_if_not_exists(target_folder)            

            self.within_var.to_csv((target_folder + f'/within_var_{self.args.buffer_size}_{self.args.seed}'), sep='\t', index=False)  
            self.between_var.to_csv((target_folder + f'/between_var_{self.args.buffer_size}_{self.args.seed}'), sep='\t', index=False) 
            self.norm_train.to_csv((target_folder + f'/norm_train_{self.args.buffer_size}_{self.args.seed}'), sep='\t', index=False)
            self.norm_test.to_csv((target_folder + f'/norm_test_{self.args.buffer_size}_{self.args.seed}'), sep='\t', index=False)


            self.NC2_off_diagonal.to_csv((target_folder + f'/NC2_off_diagonal_{self.args.buffer_size}_{self.args.seed}'), sep='\t', index=False) 
            self.var_off_diagonal.to_csv((target_folder + f'/var_off_diagonal_{self.args.buffer_size}_{self.args.seed}'), sep='\t', index=False) 
            self.NC2_between_tasks.to_csv((target_folder + f'/NC2_between_tasks_{self.args.buffer_size}_{self.args.seed}'), sep='\t', index=False)
            self.var_between_tasks.to_csv((target_folder + f'/var_between_tasks_{self.args.buffer_size}_{self.args.seed}'), sep='\t', index=False)
            self.NC3.to_csv((target_folder + f'/NC3_{self.args.buffer_size}_{self.args.seed}'), sep='\t', index=False)  

            self.hyperspher_task_mean.to_csv((target_folder + f'/hyperspher_task_mean_{self.args.buffer_size}_{self.args.seed}'), sep='\t', index=False) 
            self.hyperspher_task_var.to_csv((target_folder + f'/hyperspher_task_var_{self.args.buffer_size}_{self.args.seed}'), sep='\t', index=False) 
            self.hyperspher_total_mean.to_csv((target_folder + f'/hyperspher_total_mean_{self.args.buffer_size}_{self.args.seed}'), sep='\t', index=False)  
            self.hyperspher_total_var.to_csv((target_folder + f'/hyperspher_total_var_{self.args.buffer_size}_{self.args.seed}'), sep='\t', index=False)  
        return
    
    def begin_task(self, dataset):
        self.dataset = dataset
        return
    
    @torch.no_grad
    def log_NC_metrics(self):
        train_features, train_labels, train_tasklabels = get_features(self, self.dataset, 'train_dataset', 2)
        test_features, test_labels, test_tasklabels = get_features(self, self.dataset, 'test_dataset', 2)

        if self.current_task > 0 and self.args.buffer_size >= self.dataset.N_CLASSES_PER_TASK * self.dataset.N_TASKS:
            buffer_features, buffer_labels, buffer_tasklabels = get_features(self, self.dataset, 'buffer', self.current_task)
            buffer_features = torch.cat((buffer_features, train_features[train_tasklabels == self.current_task]), dim=0)
            buffer_labels = torch.cat((buffer_labels, train_labels[train_tasklabels == self.current_task]), dim=0)
            buffer_tasklabels = torch.cat((buffer_tasklabels, train_tasklabels[train_tasklabels == self.current_task]), dim=0)
        else:
            buffer_features = train_features[train_tasklabels == self.current_task]
            buffer_labels = train_labels[train_tasklabels == self.current_task]
            buffer_tasklabels = train_tasklabels[train_tasklabels == self.current_task]

        within_var = []
        between_var = []

        norm_train = []
        norm_test = []

        NC2_off_diagonal = []
        var_off_diagonal = []
        NC3 = []

        hyperspher_task_mean = []
        hyperspher_task_var = []
        

        all_class_means = []
        for task in buffer_tasklabels.unique(sorted=True):
            if self.dataset.SETTING != 'domain-il':
                start_label = task*self.cpt
                end_label = (task+1)*self.cpt
            else:
                start_label = 0
                end_label = self.dataset.N_CLASSES

            current_within_var = []
            for lab in range(start_label, end_label):
                idx = (lab == buffer_labels) & (task == buffer_tasklabels) #evaluate metrics for every class
                current_features = buffer_features[idx]

                mean_feature = torch.mean(current_features, dim=0)
                all_class_means.append(mean_feature.unsqueeze(0))

                current_within_var.append(calculate_variance(current_features, mean_feature).item())
         
            within_var.append(sum(current_within_var) / len(current_within_var))

        all_class_means = torch.cat(all_class_means, dim=0)
        U = all_class_means

        U_tilde = torch.zeros_like(U)
        if (((self.args.training_setting == 'class-il') and (self.dataset.SETTING == 'class-il')) 
            or (self.args.buffer_size < self.dataset.N_CLASSES_PER_TASK * self.dataset.N_TASKS)):
            U_tilde = U - torch.mean(U, dim=0)
        else:
            for lab in range(0, (self.current_task+1)*self.cpt, self.cpt):
                current_U = U[lab:lab+self.cpt]
                U_tilde[lab:lab+self.cpt] = (current_U - torch.mean(current_U, dim=0))   
        U_tilde = U_tilde.T

        all_train_means = []
        all_test_means = []
        for task in range(3):
            if self.dataset.SETTING != 'domain-il':
                start_label = task*self.cpt
                end_label = (task+1)*self.cpt
            else:
                start_label = 0
                end_label = self.dataset.N_CLASSES

            for lab in range(start_label, end_label):
                current_train_mean = torch.mean(train_features[(lab == train_labels) & (task == train_tasklabels)], dim=0)
                all_train_means.append((current_train_mean).unsqueeze(0))

                current_test_mean = torch.mean(test_features[(lab == test_labels) & (task == test_tasklabels)], dim=0)
                all_test_means.append((current_test_mean).unsqueeze(0))

        all_train_means = torch.cat(all_train_means, dim=0)
        all_test_means = torch.cat(all_test_means, dim=0)

        Q, _ = torch.linalg.qr(U_tilde, mode='reduced')
        projection = Q @ Q.T

        for lab in range(0, 3*self.cpt, self.cpt):
            current_train_means = all_train_means[lab:lab+self.cpt] - torch.mean(all_train_means[lab:lab+self.cpt], dim=0)
            current_test_means = all_test_means[lab:lab+self.cpt] - torch.mean(all_test_means[lab:lab+self.cpt], dim=0) 

            norm_train.append(torch.norm((projection @ current_train_means.T).T, dim=1, p=2).mean().item())
            norm_test.append(torch.norm((projection @ current_test_means.T).T, dim=1, p=2).mean().item())
        

        self.norm_train.loc[len(self.norm_train)] = {'iter': self.iters_counter, 'training_task': self.current_task, **{f'task{i}': norm_train[i] for i in range(len(norm_train))}}
        self.norm_test.loc[len(self.norm_test)] = {'iter': self.iters_counter, 'training_task': self.current_task, **{f'task{i}': norm_test[i] for i in range(len(norm_test))}}
        if self.args.buffer_size < self.dataset.N_CLASSES_PER_TASK * self.dataset.N_TASKS:
            return

        for lab in range(0, (self.current_task+1)*self.cpt, self.cpt):
            if ((self.args.training_setting == 'class-il') and (self.dataset.SETTING == 'class-il')):
                between_var.append(calculate_variance(all_class_means[lab:lab+self.cpt], torch.mean(all_class_means, dim=0)).item())
            else:
                between_var.append(calculate_variance(all_class_means[lab:lab+self.cpt]).item())
            

        U_tilde_normalized = U_tilde / U_tilde.norm(dim=0, keepdim=True, p=2)
        UT_U_tilde_normalized = U_tilde_normalized.T @ U_tilde_normalized  

        hyperspher_total = []
        for lab in range(0, (self.current_task+1)*self.cpt, self.cpt):
            start = lab
            end = lab + self.cpt
            
            hyperspher_task = []
            for class1 in range(start, end):
                for class2 in range(start):
                    hyperspher_total.append(calculate_hyperspher(U_tilde_normalized[:, class1], U_tilde_normalized[:, class2]))

            for class1 in range(start, end):
                for class2 in range(start, class1):
                    hyperspher_task.append(calculate_hyperspher(U_tilde_normalized[:, class1], U_tilde_normalized[:, class2]))
                
            for class1 in range(start, end):
                for class2 in range(end, (self.current_task+1)*self.cpt):
                    hyperspher_total.append(calculate_hyperspher(U_tilde_normalized[:, class1], U_tilde_normalized[:, class2]))

            hyperspher_task = torch.stack(hyperspher_task)
            hyperspher_task_mean.append(hyperspher_task.mean().item())
            hyperspher_task_var.append(calculate_variance(hyperspher_task).item())
        if (self.current_task > 0):
            hyperspher_total = torch.stack(hyperspher_total)
            hyperspher_total_mean = hyperspher_total.mean().item()
            hyperspher_total_var = calculate_variance(hyperspher_total).item()    
        else:
            hyperspher_total_mean = 0.0
            hyperspher_total_var = 0.0

        between_mask = torch.ones_like(UT_U_tilde_normalized, dtype=torch.bool)

        for lab in range(0, (self.current_task+1)*self.cpt, self.cpt):
            if ((self.args.training_setting == 'class-il') and (self.dataset.SETTING == 'class-il')):
                U_tilde_normalized_block = UT_U_tilde_normalized
                current_block = ~torch.eye(UT_U_tilde_normalized.shape[0], dtype=torch.bool)
                current_block[:lab, :] = False
                current_block[lab + self.cpt:, :] = False
            else:
                U_tilde_normalized_block = UT_U_tilde_normalized[lab:lab+self.cpt, lab:lab+self.cpt]
                current_block = ~torch.eye(U_tilde_normalized_block.shape[0], dtype=torch.bool)
            
            NC2_off_diagonal.append(U_tilde_normalized_block[current_block].mean().item())
            var_off_diagonal.append(calculate_variance(U_tilde_normalized_block[current_block]).item())

            between_mask[lab:lab+self.cpt, lab:lab+self.cpt] = False

        NC2_between_tasks = UT_U_tilde_normalized[between_mask].mean().item()
        var_between_tasks = calculate_variance(UT_U_tilde_normalized[between_mask]).item()

        if isinstance(self.net.classifier, nn.Linear):
            classifier_weights = (self.net.classifier.weight.detach().cpu()[:self.n_seen_classes]).T
        else: 
            weights = [layer.weight.detach().cpu() for layer in self.net.classifier]
            classifier_weights = (torch.cat(weights, dim=0)[:self.n_seen_classes]).T

        classifier_weights = classifier_weights / classifier_weights.norm(dim=0, keepdim=True, p=2)
        for lab in range(0, (self.current_task+1)*self.cpt, self.cpt):
            if self.dataset.SETTING != 'domain-il':
                block = classifier_weights.T @ U_tilde_normalized 
                NC3.append(torch.diag(block)[lab: lab+self.cpt].mean().item())
            else:
                block = classifier_weights.T @ U_tilde_normalized[:, lab:lab+self.cpt]
                NC3.append(torch.diag(block).mean().item())

        self.within_var.loc[len(self.within_var)] = {'iter': self.iters_counter, 'training_task': self.current_task, **{f'task{i}': within_var[i] for i in range(len(within_var))}}
        self.between_var.loc[len(self.between_var)] = {'iter': self.iters_counter, 'training_task': self.current_task, **{f'task{i}': between_var[i] for i in range(len(between_var))}}
        
        self.NC2_off_diagonal.loc[len(self.NC2_off_diagonal)] = {'iter': self.iters_counter, 'training_task': self.current_task, **{f'task{i}': NC2_off_diagonal[i] for i in range(len(NC2_off_diagonal))}}
        self.var_off_diagonal.loc[len(self.var_off_diagonal)] = {'iter': self.iters_counter, 'training_task': self.current_task, **{f'task{i}': var_off_diagonal[i] for i in range(len(var_off_diagonal))}}
        self.NC2_between_tasks.loc[len(self.NC2_between_tasks)] = {'iter': self.iters_counter, 'training_task': self.current_task, 'value': NC2_between_tasks}
        self.var_between_tasks.loc[len(self.var_between_tasks)] = {'iter': self.iters_counter, 'training_task': self.current_task, 'value': var_between_tasks}
        self.NC3.loc[len(self.NC3)] = {'iter': self.iters_counter, 'training_task': self.current_task, **{f'task{i}': NC3[i] for i in range(len(NC3))}}

        self.hyperspher_task_mean.loc[len(self.hyperspher_task_mean)] = {'iter': self.iters_counter, 'training_task': self.current_task, **{f'task{i}': hyperspher_task_mean[i] for i in range(len(hyperspher_task_mean))}}
        self.hyperspher_task_var.loc[len(self.hyperspher_task_var)] = {'iter': self.iters_counter, 'training_task': self.current_task, **{f'task{i}': hyperspher_task_var[i] for i in range(len(hyperspher_task_var))}}
        self.hyperspher_total_mean.loc[len(self.hyperspher_total_mean)] = {'iter': self.iters_counter, 'training_task': self.current_task, 'value': hyperspher_total_mean}
        self.hyperspher_total_var.loc[len(self.hyperspher_total_var)] = {'iter': self.iters_counter, 'training_task': self.current_task, 'value': hyperspher_total_var}
        return

@torch.no_grad
def calculate_variance(features, mean=None):
    if features.shape[0] <= 1:
        return torch.tensor(0.0)
    
    bias_correction = 0
    if mean is None:
        mean = torch.mean(features, dim=0)
        bias_correction = 0

    if features.ndim == 2:
        features = torch.norm(features - mean, dim=1, p=2) ** 2
        variance = features.sum() / (features.shape[0] + bias_correction)
    else:
        features = features - mean
        variance = features.pow(2).sum() / (features.shape[0] + bias_correction)

    return variance

@torch.no_grad
def calculate_hyperspher(class1, class2):
    norm_diff = torch.norm(class1 - class2, dim=0, p=2)
    return torch.log(1 / norm_diff)