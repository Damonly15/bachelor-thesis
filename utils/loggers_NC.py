import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import avg_pool2d

from models.utils.continual_model import ContinualModel
from datasets.utils.continual_dataset import ContinualDataset
from utils.feature_forgetting import get_features
from utils.conf import base_path
from utils import create_if_not_exists

@torch.no_grad
def calculate_variance(features, mean=None):
    if features.shape[0] <= 1:
        return torch.tensor(0.0)

    if mean is None:
        mean = torch.mean(features, dim=0)
        
    norms = torch.norm(features - mean, dim=1, p=2) ** 2
    variance = norms.sum() / (norms.shape[0])
    return variance

class LoggerVersion:
    def __init__(self, version) -> None:
        self.within_var = []
        self.between_var = []
        self.global_var = []
        self.snr = []

        self.entropy = []
        
        self.version = version

    @torch.no_grad
    def log(self, dataset :ContinualDataset, model: ContinualModel):
        within_var = []
        between_var = []
        global_var = []
        snr = []
            
        if self.version == 'buffer':
            max_task = model.current_task
            max_class = model.n_past_classes
        else:
            max_task = dataset.N_TASKS
            max_class = model.n_seen_classes

        all_features, all_labels, all_tasklabels = model.features[self.version]

        all_class_means = []
        for task in range(max_task):
            if dataset.SETTING != 'domain-il':
                start_label = task*model.cpt
                end_label = (task+1)*model.cpt
            else:
                start_label = 0
                end_label = dataset.N_CLASSES

            current_within_var = []
            current_class_means = []
            for lab in range(start_label, end_label):
                idx = (lab == all_labels) & (task == all_tasklabels) #evaluate metrics for every class
                current_features = all_features[idx]

                mean_feature = torch.mean(current_features, dim=0)
                current_class_means.append(mean_feature)
                all_class_means.append(mean_feature.unsqueeze(0))

                current_within_var.append(calculate_variance(current_features).item())

            current_snr = []
            for class1 in range(0, end_label-start_label):
                for class2 in range(0, class1):
                    current_snr.append((torch.norm(current_class_means[class1]-current_class_means[class2], dim=0, p=2)**2
                                       / (current_within_var[class1] + current_within_var[class2])).item())
                    
            within_var.append(sum(current_within_var) / len(current_within_var))
            snr.append(sum(current_snr) / len(current_snr))

        if len(all_class_means) > 0:
            all_class_means = torch.cat(all_class_means, dim=0)
        else:
            all_class_means = torch.empty(0, model.net.feature_dim)

        for task in range(max_task):
            task_idx = task == all_tasklabels
            taskwise_features = all_features[task_idx]

            global_var.append(calculate_variance(taskwise_features).item())

            current_classes = all_class_means[task*model.cpt: (task+1)*model.cpt]
            between_var.append(calculate_variance(current_classes).item())
            
        self.within_var.append(within_var)
        self.between_var.append(between_var)
        self.global_var.append(global_var)
        self.snr.append(snr)

        class_means = []
        for lab in range(max_class):
            mean_feature = torch.mean(all_features[all_labels == lab], dim=0)
            class_means.append(mean_feature.unsqueeze(0))
        
        if len(class_means) > 0:
            class_means = torch.cat(class_means, dim=0)
        else:
            class_means = torch.empty(0, model.net.feature_dim)

        return all_class_means, class_means
    
    def log_classifier(self, dataset :ContinualDataset, model: ContinualModel):
        entropy = []

        max_logit = model.n_seen_classes
        if self.version == 'buffer':
            max_task = model.current_task
        else: 
            max_task = dataset.N_TASKS

        all_logits, all_labels, all_tasklabels = get_features(model, dataset, self.version, dataset.N_TASKS-1, "logits")

        for task in range(max_task):
            task_idx = all_tasklabels == task
            prob = F.softmax(all_logits[task_idx, :max_logit], dim=1)
            log_prob = F.log_softmax(all_logits[task_idx, :max_logit], dim=1)

            entropy.append((-torch.sum(prob * log_prob, dim=1)).mean().item())

        self.entropy.append(entropy)
        return
        
class LoggerNC:
    def __init__(self, model: ContinualModel) -> None:
        if hasattr(model, 'buffer') and model.args.buffer_size >= model.dataset.N_CLASSES_PER_TASK * model.dataset.N_TASKS:
            self.versions = ['buffer']
        else:
            self.versions = []

        self.versions = self.versions + ['train_dataset', 'test_dataset']     

        self.all_loggers = {
            i: LoggerVersion(i) for i in self.versions
        }

        self.mean_shift = []

        self.NC2_diagonal = []
        self.NC2_off_diagonal = []
        self.NC2_between_tasks = []
        self.NC3 = []

        self.norm = []
        self.norm_complement = []
    
    @torch.no_grad
    def log(self, dataset: ContinualDataset, model: ContinualModel):
        mean_shift = []

        NC2_diagonal = []
        NC2_off_diagonal = []
        NC3 = []

        norm = []
        norm_complement = []

        """
        all_features, all_labels, all_tasklabels = model.features['train_dataset']

        global_mean = torch.mean(all_features, dim=0)
        global_mean_gpu = global_mean.unsqueeze(0).to(model.device)

        if model.args.training_setting == "class-il":
            mean_prediction = model.net.classifier(global_mean_gpu).squeeze(0).cpu()
        else:
            mean_prediction = model.net.classifier[0](global_mean_gpu).squeeze(0).cpu()
            for task in range(1, dataset.N_TASKS):
                current_mean_prediction = model.net.classifier[task](global_mean_gpu).squeeze(0).cpu()
                mean_prediction = torch.cat((mean_prediction, current_mean_prediction), dim=0)
        """
        if isinstance(model.net.classifier, nn.Linear):
            classifier_weights = (model.net.classifier.weight.detach().cpu()[:model.n_seen_classes]).T
        else: 
            weights = [layer.weight.detach().cpu() for layer in model.net.classifier]
            classifier_weights = (torch.cat(weights, dim=0)[:model.n_seen_classes]).T

        all_train_means, train_means = self.all_loggers['train_dataset'].log(dataset, model)
        self.all_loggers['train_dataset'].log_classifier(dataset, model)

        all_test_means, tests_means = self.all_loggers['test_dataset'].log(dataset, model)
        self.all_loggers['test_dataset'].log_classifier(dataset, model)

        if model.args.buffer_size >= dataset.N_CLASSES_PER_TASK * dataset.N_TASKS:
            all_buffer_means, buffer_means = self.all_loggers['buffer'].log(dataset, model)
            self.all_loggers['buffer'].log_classifier(dataset, model)

            delta_mean = torch.norm(all_buffer_means - all_train_means[:model.cpt * model.current_task], dim=1, p=2)

            for task in range(model.current_task):
                mean_shift.append(torch.mean(delta_mean[task*model.cpt:(task+1)*model.cpt]).item())
            
            U = torch.cat((buffer_means, train_means[-model.cpt:]), dim=0)
        else:
            for task in range(model.current_task):
                mean_shift.append(0)

            U = train_means[-model.cpt:]
            classifier_weights = (classifier_weights[:, -model.cpt:])

        if model.dataset.SETTING == 'domain-il':
            train_features, train_labels, train_tasklabels = model.features['train_dataset']
            train_features = train_features[train_tasklabels == model.current_task]
            train_labels = train_labels[train_tasklabels == model.current_task]

            buffer_features, buffer_labels, buffer_tasklabels = model.features['buffer']
            buffer_features = buffer_features[buffer_tasklabels < model.current_task]
            buffer_labels = buffer_labels[buffer_tasklabels < model.current_task]

            all_features = torch.cat((train_features, buffer_features), dim=0)
            all_labels = torch.cat((train_labels, buffer_labels), dim=0)

            U = []
            for lab in range(model.cpt):
                mean_feature = torch.mean(all_features[all_labels == lab], dim=0)
                U.append(mean_feature.unsqueeze(0))
            U = torch.cat(U, dim=0)
               
        U_tilde = (U - torch.mean(U, dim=0)).T
        U_tilde_normalized = U_tilde / U_tilde.norm(dim=0, keepdim=True, p=2)
        UT_U = U_tilde_normalized.T @ U_tilde_normalized
        #print(UT_U)

        keep_mask = torch.tril(torch.ones_like(UT_U, dtype=torch.bool), diagonal=-1)

        for lab in range(0, model.n_seen_classes, model.cpt):
            block = UT_U[lab:lab+model.cpt, lab:lab+model.cpt]

            NC2_diagonal.append(torch.diag(block).mean().item())

            tril_vals = torch.tril(block, diagonal=-1)
            NC2_off_diagonal.append(tril_vals.mean().item())

            keep_mask[lab:lab+model.cpt, lab:lab+model.cpt] = False

        self.NC2_between_tasks.append(UT_U[keep_mask].mean().item())

        classifier_weights = classifier_weights / classifier_weights.norm(dim=0, keepdim=True, p=2)
        classifier_weights = classifier_weights.T @ U_tilde_normalized

        for lab in range(0, model.n_seen_classes, model.cpt):
            NC3.append(torch.diag(classifier_weights)[lab: lab+model.cpt].mean().item())
        
        train_features, train_labels, train_tasklabels = model.features['train_dataset']
        model.projection = U_tilde @ torch.inverse(U_tilde.T @ U_tilde) @ U_tilde.T
        projection = model.projection
        complement_projection = torch.eye(projection.shape[0]) - projection

        for lab in range(dataset.N_TASKS):
            idx = train_tasklabels == lab
            current_features = train_features[idx]

            projected_features = (projection @ current_features.T).T
            norm.append(torch.norm(projected_features, dim=0, p=2).mean().item() / (U_tilde.shape[1]-1))
            print((U_tilde.shape[1]-1))

            complement_features = (complement_projection @ current_features.T).T
            norm_complement.append(torch.norm(complement_features, dim=0, p=2).mean().item() / (U_tilde.shape[0]-(U_tilde.shape[1]-1)))
            print((U_tilde.shape[0]-(U_tilde.shape[1]-1)))
              
        self.mean_shift.append(mean_shift)

        self.NC2_diagonal.append(NC2_diagonal)
        self.NC2_off_diagonal.append(NC2_off_diagonal)
        self.NC3.append(NC3)

        self.norm.append(norm)
        self.norm_complement.append(norm_complement)
        return

    def write(self, model: ContinualModel): 
        for key, value in self.all_loggers.items():

            wrargs = (vars(model.args)).copy()
            wrargs['result_type'] = key

            if 'class_order' in wrargs:
                del wrargs['class_order'] #don't need how we permuted the classes in the log file. This can get very long if we have many classes.

            target_folder = base_path() + "results/"

            for i, fa in enumerate(value.within_var):
                for j, var in enumerate(fa):
                    wrargs['within_var_' + str(j + 1) + '_task' + str(i+1)] = var
                
            for i, fa in enumerate(value.between_var):
                for j, var in enumerate(fa):
                    wrargs['between_var_' + str(j + 1) + '_task' + str(i+1)] = var

            for i, fa in enumerate(value.global_var):
                for j, var in enumerate(fa):
                    wrargs['global_var_' + str(j + 1) + '_task' + str(i+1)] = var

            for i, fa in enumerate(value.snr):
                for j, var in enumerate(fa):
                    wrargs['snr_' + str(j + 1) + '_task' + str(i+1)] = var

            for i, fa in enumerate(value.entropy):
                for j, var in enumerate(fa):
                    wrargs['entropy_' + str(j + 1) + '_task' + str(i+1)] = var
            
            if key == 'train_dataset':
                for i, fa in enumerate(self.mean_shift):
                    for j, var in enumerate(fa):
                        wrargs['mean_shift_' + str(j + 1) + '_task' + str(i+1)] = var

                for i, fa in enumerate(self.NC2_diagonal):
                    for j, var in enumerate(fa):
                        wrargs['NC2_diagonal_' + str(j + 1) + '_task' + str(i+1)] = var

                for i, fa in enumerate(self.NC2_off_diagonal):
                    for j, var in enumerate(fa):
                        wrargs['NC2_off_diagonal_' + str(j + 1) + '_task' + str(i+1)] = var

                for i, fa in enumerate(self.NC2_between_tasks):
                    wrargs['NC2_between_tasks_task' + str(i+1)] = fa
                
                for i, fa in enumerate(self.NC3):
                    for j, var in enumerate(fa):
                        wrargs['NC3_' + str(j + 1) + '_task' + str(i+1)] = var

                for i, fa in enumerate(self.norm):
                    for j, var in enumerate(fa):
                        wrargs['norm_' + str(j + 1) + '_task' + str(i+1)] = var
                    
                for i, fa in enumerate(self.norm_complement):
                    for j, var in enumerate(fa):
                        wrargs['norm_complement_' + str(j + 1) + '_task' + str(i+1)] = var


            create_if_not_exists(target_folder + model.args.training_setting)
            create_if_not_exists(target_folder + model.args.training_setting +
                                "/" + model.args.dataset)
            create_if_not_exists(target_folder + model.args.training_setting +
                                "/" + model.args.dataset + "/" + model.args.model)

            pre_path = target_folder + model.args.training_setting + "/" + model.args.dataset\
                + "/" + model.args.model
            path = pre_path + "/logs_NC.txt"
            print("Logging NC metrics in " + path)
            with open(path, 'a') as f:
                f.write(str(wrargs) + '\n')
