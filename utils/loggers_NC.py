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
    
    bias_correction = 0
    if mean is None:
        mean = torch.mean(features, dim=0)
        bias_correction = -1

    features = torch.norm(features - mean, dim=1, p=2) ** 2
    variance = norms.sum() / (norms.shape[0] + bias_correction)
    
    return variance

class LoggerVersion:
    def __init__(self, version) -> None:
        self.within_var = []
        self.within_var_together = []

        self.between_var = []
        self.between_var_together = []

        self.features_norm = []
        self.features_norm_together = []

        self.snr = []
        self.snr_together = []

        self.entropy = []
        self.entropy_together = []
        
        self.version = version

    @torch.no_grad
    def log(self, dataset :ContinualDataset, model: ContinualModel):
        within_var = []
        between_var = []
        features_norm = []
        snr = []

        all_features, all_labels, all_tasklabels = model.features[self.version]

        all_class_means = []
        for task in range(all_tasklabels.max() + 1):
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
            features_norm.append(torch.norm(torch.stack(current_class_means), p=2, dim=1).mean().item())
            snr.append(sum(current_snr) / len(current_snr))

        all_class_means = torch.cat(all_class_means, dim=0)
        for task in range(all_tasklabels.max() + 1):
            current_classes = all_class_means[task*model.cpt: (task+1)*model.cpt]
            between_var.append(calculate_variance(current_classes).item())
            
        self.within_var.append(within_var)
        self.between_var.append(between_var)
        self.features_norm.append(features_norm)
        self.snr.append(snr)

        
        class_means = []
        current_within_var = []
        current_class_means = []
        
        for lab in range(model.n_seen_classes):
            mean_feature = torch.mean(all_features[(all_labels == lab) & (all_tasklabels <= model.current_task)], dim=0)
            current_class_means.append(mean_feature)
            class_means.append(mean_feature.unsqueeze(0))
            
            current_within_var.append(calculate_variance(all_features[all_labels == lab]).item())
        
        self.within_var_together.append(sum(current_within_var) / len(current_within_var))
        class_means = torch.cat(class_means, dim=0)
        self.features_norm_together.append(torch.norm(class_means, p=2, dim=1).mean().item())
        self.between_var_together.append(calculate_variance(class_means).item())

        current_snr = []
        for class1 in range(model.n_seen_classes):
            for class2 in range(0, class1):
                current_snr.append((torch.norm(current_class_means[class1]-current_class_means[class2], dim=0, p=2)**2
                                    / (current_within_var[class1] + current_within_var[class2])).item())
        self.snr_together.append(sum(current_snr) / len(current_snr))

        return all_class_means, class_means
    
    def log_classifier(self, dataset :ContinualDataset, model: ContinualModel):
        entropy = []

        if self.version == 'buffer':
            all_logits, all_labels, all_tasklabels = get_features(model, dataset, 'train_dataset', dataset.N_TASKS-1, "logits")
            mask = all_tasklabels == model.current_task
            all_logits = all_logits[mask]
            all_labels = all_labels[mask]
            all_tasklabels = all_tasklabels[mask]

            if model.args.buffer_size >= (dataset.N_CLASSES_PER_TASK * dataset.N_TASKS) and (not model.current_task==0):
                buffer_logits, buffer_labels, buffer_tasklabels = get_features(model, dataset, self.version, dataset.N_TASKS-1, "logits")
                all_logits = torch.cat((buffer_logits, all_logits), dim=0)
                all_labels = torch.cat((buffer_labels, all_labels), dim=0)
                all_tasklabels = torch.cat((buffer_tasklabels, all_tasklabels), dim=0)
        else:
            all_logits, all_labels, all_tasklabels = get_features(model, dataset, self.version, dataset.N_TASKS-1, "logits")

        for task in range(all_tasklabels.max() + 1):
            if model.args.training_setting == 'task-il':
                min_logit = task*model.cpt
                max_logit = (task+1)*model.cpt
            else:
                min_logit = 0
                max_logit = model.n_seen_classes

            task_idx = all_tasklabels == task
            prob = F.softmax(all_logits[task_idx, min_logit:max_logit], dim=1)
            log_prob = F.log_softmax(all_logits[task_idx, min_logit:max_logit], dim=1)

            entropy.append((-torch.sum(prob * log_prob, dim=1)).mean().item())

        self.entropy.append(entropy)

        all_logits = all_logits[all_tasklabels <= model.current_task]

        max_logit = model.n_seen_classes
        prob = F.softmax(all_logits[:, :max_logit], dim=1)
        log_prob = F.log_softmax(all_logits[:, :max_logit], dim=1)
        self.entropy_together.append((-torch.sum(prob * log_prob, dim=1)).mean().item())

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

        self.mean_norm = []

        self.mean_shift = []
        self.mean_shift_together = []

        self.orth_diagonal = []
        self.orth_diagonal_together = []
        self.NC2_diagonal = []
        self.NC2_diagonal_together = []

        self.orth_off_diagonal = []
        self.orth_off_diagonal_together = []
        self.NC2_off_diagonal = []
        self.NC2_off_diagonal_together = []
        
        self.orth_between_tasks = []
        self.orth_between_tasks_together = []
        self.NC2_between_tasks = []
        self.NC2_between_tasks_together = []

        self.NC2_all_off_diagonal = []
        self.NC2_all_off_diagonal_together = []

        self.NC3 = []
        self.NC3_together = []

        self.rank = []
        self.rank_together = []

        self.norm = []
        self.svd = []
        self.norm_together = []
        self.svd_together = []
        
        self.norm_complement = []
        self.svd_complement = []
        self.norm_complement_together = []
        self.svd_complement_together = []
    
    @torch.no_grad
    def log(self, dataset: ContinualDataset, model: ContinualModel):
        mean_shift = []

        orth_diagonal = []
        NC2_diagonal = []
        
        orth_off_diagonal = []
        NC2_off_diagonal = []
        
        NC3 = []

        norm = []
        svd = []

        norm_complement = []
        svd_complement = []

        all_train_means, train_means = self.all_loggers['train_dataset'].log(dataset, model)
        self.all_loggers['train_dataset'].log_classifier(dataset, model)

        all_test_means, tests_means = self.all_loggers['test_dataset'].log(dataset, model)
        self.all_loggers['test_dataset'].log_classifier(dataset, model)

        all_buffer_means, buffer_means = self.all_loggers['buffer'].log(dataset, model)
        self.all_loggers['buffer'].log_classifier(dataset, model)

        self.mean_norm.append(torch.norm(train_means, dim=1).mean().item())

        if isinstance(model.net.classifier, nn.Linear):
            classifier_weights = (model.net.classifier.weight.detach().cpu()[:model.n_seen_classes]).T
        else: 
            weights = [layer.weight.detach().cpu() for layer in model.net.classifier]
            classifier_weights = (torch.cat(weights, dim=0)[:model.n_seen_classes]).T
        classifier_weights = (classifier_weights[:, -buffer_means.shape[0]:])

        delta_mean = torch.norm(all_buffer_means - all_train_means[:model.cpt * (model.current_task+1)], dim=1, p=2)

        for task in range(model.current_task+1):
            mean_shift.append(torch.mean(delta_mean[task*model.cpt:(task+1)*model.cpt]).item())
        self.mean_shift.append(mean_shift)

        self.mean_shift_together.append(torch.norm(buffer_means - train_means[:model.n_seen_classes], dim=1, p=2).mean().item())

        #For TIL and CIL I am doing the same computation twice. However, for DIL it makes a differnce, as U is calculated differently.
        U = all_buffer_means

        if (dataset.SETTING == 'class-il') and (model.args.training_setting == 'class-il'):
            U_tilde = (U - torch.mean(U, dim=0))
        else:
            U_tilde = torch.zeros_like(U)
            for task in range(model.current_task+1):
                current_U = U[task*model.cpt:(task+1)*model.cpt]
                U_tilde[task*model.cpt:(task+1)*model.cpt] = (current_U - torch.mean(current_U, dim=0))
        U_tilde = U_tilde.T
        UT_U_tilde = U_tilde.T @ U_tilde
        self.rank.append(torch.linalg.matrix_rank(U_tilde).item())

        U_tilde_normalized = U_tilde / U_tilde.norm(dim=0, keepdim=True, p=2)
        UT_U_tilde_normalized = U_tilde_normalized.T @ U_tilde_normalized  

        block_mask = torch.zeros_like(UT_U_tilde, dtype=torch.bool)
        between_mask = torch.ones_like(UT_U_tilde, dtype=torch.bool)
        train_features, train_labels, train_tasklabels = model.features['train_dataset']

        for lab in range(0, (model.current_task+1)*model.cpt, model.cpt):
            U_tilde_normalized_block = UT_U_tilde_normalized[lab:lab+model.cpt, lab:lab+model.cpt]
            U_tilde_block = UT_U_tilde[lab:lab+model.cpt, lab:lab+model.cpt]
            
            NC2_diagonal.append(torch.diag(U_tilde_normalized_block).mean().item())
            orth_diagonal.append(torch.diag(U_tilde_block).mean().item())

            current_block = ~torch.eye(U_tilde_block.shape[0], dtype=torch.bool)
            NC2_off_diagonal.append(U_tilde_normalized_block[current_block].mean().item())
            orth_off_diagonal.append(U_tilde_block[current_block].mean().item())

            block_mask[lab:lab+model.cpt, lab:lab+model.cpt] = current_block
            between_mask[lab:lab+model.cpt, lab:lab+model.cpt] = False

            current_features = train_features[train_tasklabels == (lab // model.cpt)]
            if dataset.SETTING != 'domain-il':
                projection = U_tilde @ torch.linalg.pinv(U_tilde)
            else:
                projection = U_tilde[:, lab:lab+model.cpt] @ torch.linalg.pinv(U_tilde[:, lab:lab+model.cpt])
            complement_projection = torch.eye(projection.shape[0]) - projection

            projected_features = (projection @ current_features.T).T
            norm.append(torch.norm(projected_features, dim=1, p=2).mean().item())
            U, S, Vh = torch.linalg.svd(projected_features, full_matrices=False)
            svd.append(S[:model.n_seen_classes-1].mean().item())
            
            complement_features = (complement_projection @ current_features.T).T
            norm_complement.append(torch.norm(complement_features, dim=1, p=2).mean().item())
            U, S, Vh = torch.linalg.svd(complement_features, full_matrices=False)
            svd_complement.append(S[:-(model.n_seen_classes-1)].mean().item())
 
        self.NC2_between_tasks.append(UT_U_tilde_normalized[between_mask].mean().item())
        self.orth_between_tasks.append(UT_U_tilde[between_mask].mean().item())

        self.NC2_all_off_diagonal.append(UT_U_tilde_normalized[~torch.eye(UT_U_tilde_normalized.shape[0], dtype=torch.bool)].mean().item())

        classifier_weights = classifier_weights / classifier_weights.norm(dim=0, keepdim=True, p=2)

        for lab in range(0, (model.current_task+1)*model.cpt, model.cpt):
            if dataset.SETTING != 'domain-il':
                block = classifier_weights.T @ U_tilde_normalized 
                NC3.append(torch.diag(block)[lab: lab+model.cpt].mean().item())
            else:
                block = classifier_weights.T @ U_tilde_normalized[:, lab:lab+model.cpt]
                NC3.append(torch.diag(block).mean().item())

        self.orth_diagonal.append(orth_diagonal)
        self.NC2_diagonal.append(NC2_diagonal)

        self.orth_off_diagonal.append(orth_off_diagonal)
        self.NC2_off_diagonal.append(NC2_off_diagonal)

        self.NC3.append(NC3)

        self.norm.append(norm)
        self.svd.append(svd)

        self.norm_complement.append(norm_complement)
        self.svd_complement.append(svd_complement)
        

        U = buffer_means

        U_tilde = (U - torch.mean(U, dim=0)).T
        UT_U_tilde = U_tilde.T @ U_tilde
        self.rank_together.append(torch.linalg.matrix_rank(U_tilde).item())

        U_tilde_normalized = U_tilde / U_tilde.norm(dim=0, keepdim=True, p=2)
        UT_U_tilde_normalized = U_tilde_normalized.T @ U_tilde_normalized 


        if dataset.SETTING == 'domain-il':
            block_mask = ~torch.eye(UT_U_tilde.shape[0], dtype=torch.bool)
            between_mask= torch.zeros_like(UT_U_tilde, dtype=torch.bool)

        self.NC2_diagonal_together.append(torch.diag(UT_U_tilde_normalized).mean().item())
        self.orth_diagonal_together.append(torch.diag(UT_U_tilde).mean().item())

        self.NC2_off_diagonal_together.append(UT_U_tilde_normalized[block_mask].mean().item())
        self.orth_off_diagonal_together.append(UT_U_tilde[block_mask].mean().item())

        self.NC2_between_tasks_together.append(UT_U_tilde_normalized[between_mask].mean().item())
        self.orth_between_tasks_together.append(UT_U_tilde[between_mask].mean().item())

        self.NC2_all_off_diagonal_together.append(UT_U_tilde_normalized[~torch.eye(UT_U_tilde_normalized.shape[0], dtype=torch.bool)].mean().item())

        classifier_weights = classifier_weights.T @ U_tilde_normalized
        self.NC3_together.append(torch.diag(classifier_weights).mean().item())

        
        train_features = train_features[train_tasklabels <= model.current_task]
        model.projection = U_tilde @ torch.linalg.pinv(U_tilde)
        projection = model.projection
        complement_projection = torch.eye(projection.shape[0]) - projection
              
        projected_features = (projection @ train_features.T).T
        self.norm_together.append(torch.norm(projected_features, dim=1, p=2).mean().item())
        U, S, Vh = torch.linalg.svd(projected_features, full_matrices=False)
        self.svd_together.append(S[:model.n_seen_classes-1].mean().item())

        complement_features = (complement_projection @ current_features.T).T
        self.norm_complement_together.append(torch.norm(complement_features, dim=1, p=2).mean().item())
        U, S, Vh = torch.linalg.svd(complement_features, full_matrices=False)
        svd_complement.append(S[:-(model.n_seen_classes-1)].mean().item())

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
            
            for i, fa in enumerate(value.within_var_together):
                wrargs['within_var_task' + str(i+1)] = fa
                

            for i, fa in enumerate(value.between_var):
                for j, var in enumerate(fa):
                    wrargs['between_var_' + str(j + 1) + '_task' + str(i+1)] = var

            for i, fa in enumerate(value.between_var_together):
                wrargs['between_var_task' + str(i+1)] = fa


            for i, fa in enumerate(value.features_norm):
                for j, var in enumerate(fa):
                    wrargs['features_norm_' + str(j + 1) + '_task' + str(i+1)] = var

            for i, fa in enumerate(value.features_norm_together):
                wrargs['features_norm_task' + str(i+1)] = fa    


            for i, fa in enumerate(value.snr):
                for j, var in enumerate(fa):
                    wrargs['snr_' + str(j + 1) + '_task' + str(i+1)] = var

            for i, fa in enumerate(value.snr_together):
                wrargs['snr_task' + str(i+1)] = fa


            for i, fa in enumerate(value.entropy):
                for j, var in enumerate(fa):
                    wrargs['entropy_' + str(j + 1) + '_task' + str(i+1)] = var

            for i, fa in enumerate(value.entropy_together):
                wrargs['entropy_task' + str(i+1)] = fa


            if key == 'train_dataset':
                for i, fa in enumerate(self.mean_norm):
                    wrargs['mean_norm_task' + str(i+1)] = fa

                for i, fa in enumerate(self.mean_shift):
                    for j, var in enumerate(fa):
                        wrargs['mean_shift_' + str(j + 1) + '_task' + str(i+1)] = var

                for i, fa in enumerate(self.mean_shift_together):
                    wrargs['mean_shift_task' + str(i+1)] = fa


                for i, fa in enumerate(self.orth_diagonal):
                    for j, var in enumerate(fa):
                        wrargs['orth_diagonal_' + str(j + 1) + '_task' + str(i+1)] = var
                
                for i, fa in enumerate(self.orth_diagonal_together):
                    wrargs['orth_diagonal_task' + str(i+1)] = fa

                for i, fa in enumerate(self.NC2_diagonal):
                    for j, var in enumerate(fa):
                        wrargs['NC2_diagonal_' + str(j + 1) + '_task' + str(i+1)] = var
                
                for i, fa in enumerate(self.NC2_diagonal_together):
                    wrargs['NC2_diagonal_task' + str(i+1)] = fa


                for i, fa in enumerate(self.orth_off_diagonal):
                    for j, var in enumerate(fa):
                        wrargs['orth_off_diagonal_' + str(j + 1) + '_task' + str(i+1)] = var
                
                for i, fa in enumerate(self.orth_off_diagonal_together):
                    wrargs['orth_off_diagonal_task' + str(i+1)] = fa

                for i, fa in enumerate(self.NC2_off_diagonal):
                    for j, var in enumerate(fa):
                        wrargs['NC2_off_diagonal_' + str(j + 1) + '_task' + str(i+1)] = var
                
                for i, fa in enumerate(self.NC2_off_diagonal_together):
                    wrargs['NC2_off_diagonal_task' + str(i+1)] = fa


                for i, fa in enumerate(self.orth_between_tasks):
                    wrargs['orth_between_tasks_task' + str(i+1)] = fa
                
                for i, fa in enumerate(self.orth_between_tasks_together):
                    wrargs['orth_between_tasks_together_task' + str(i+1)] = fa

                for i, fa in enumerate(self.NC2_between_tasks):
                    wrargs['NC2_between_tasks_task' + str(i+1)] = fa
                
                for i, fa in enumerate(self.NC2_between_tasks_together):
                    wrargs['NC2_between_tasks_together_task' + str(i+1)] = fa

                for i, fa in enumerate(self.NC2_all_off_diagonal):
                    wrargs['NC2_all_off_diagonal_task' + str(i+1)] = fa

                for i, fa in enumerate(self.NC2_all_off_diagonal_together):
                    wrargs['NC2_all_off_diagonal_together_task' + str(i+1)] = fa
                
                
                for i, fa in enumerate(self.rank):
                    wrargs['rank_task' + str(i+1)] = fa
                
                for i, fa in enumerate(self.rank_together):
                    wrargs['rank_together_task' + str(i+1)] = fa
                

                for i, fa in enumerate(self.NC3):
                    for j, var in enumerate(fa):
                        wrargs['NC3_' + str(j + 1) + '_task' + str(i+1)] = var
                
                for i, fa in enumerate(self.NC3_together):
                    wrargs['NC3_task' + str(i+1)] = fa


                for i, fa in enumerate(self.norm):
                    for j, var in enumerate(fa):
                        wrargs['norm_' + str(j + 1) + '_task' + str(i+1)] = var

                for i, fa in enumerate(self.norm_together):
                    wrargs['norm_task' + str(i+1)] = fa

                for i, fa in enumerate(self.svd):
                    for j, var in enumerate(fa):
                        wrargs['svd_' + str(j + 1) + '_task' + str(i+1)] = var

                for i, fa in enumerate(self.svd_together):
                    wrargs['svd_task' + str(i+1)] = fa

                    
                for i, fa in enumerate(self.norm_complement):
                    for j, var in enumerate(fa):
                        wrargs['norm_complement_' + str(j + 1) + '_task' + str(i+1)] = var
                
                for i, fa in enumerate(self.norm_complement_together):
                    wrargs['norm_complement_task' + str(i+1)] = fa

                for i, fa in enumerate(self.svd_complement):
                    for j, var in enumerate(fa):
                        wrargs['svd_complement_' + str(j + 1) + '_task' + str(i+1)] = var
                
                for i, fa in enumerate(self.svd_complement_together):
                    wrargs['svd_complement_task' + str(i+1)] = fa


            if model.args.training_setting == "task-il":
                create_if_not_exists(target_folder + "task-il")
                create_if_not_exists(target_folder + "task-il" +
                                    "/" + model.args.dataset)
                create_if_not_exists(target_folder + "task-il" +
                                    "/" + model.args.dataset + "/" + model.args.model)
                path = target_folder + "task-il" + "/" + model.args.dataset\
                    + "/" + model.args.model + "/logsNC.txt"
            else:
                create_if_not_exists(target_folder + model.dataset.SETTING)
                create_if_not_exists(target_folder +  model.dataset.SETTING +
                                    "/" + model.args.dataset)
                create_if_not_exists(target_folder + model.dataset.SETTING +
                                    "/" + model.args.dataset + "/" + model.args.model)
                path = target_folder + model.dataset.SETTING + "/" + model.args.dataset\
                    + "/" + model.args.model + "/logsNC.txt"

            print("Logging NC metrics in " + path)
            with open(path, 'a') as f:
                f.write(str(wrargs) + '\n')
