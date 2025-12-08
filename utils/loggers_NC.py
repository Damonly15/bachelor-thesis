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
        bias_correction = 0

    if features.ndim == 2:
        features = torch.norm(features - mean, dim=1, p=2) ** 2
        variance = features.sum() / (features.shape[0] + bias_correction)
    else:
        features = features - mean
        variance = features.pow(2).sum() / (features.shape[0] + bias_correction)

    return variance

class LoggerVersion:
    def __init__(self, version) -> None:
        self.within_var = []
        self.within_var_together = []

        self.between_var = []
        self.between_var_together = []

        self.global_means_norm = []
        self.features_norm = []

        self.snr = []
        self.snr_together = []

        self.norm = []
        self.norm_together = []
        self.norm_complement = []
        self.norm_complement_together = []
        
        self.version = version

    @torch.no_grad
    def log(self, dataset: ContinualDataset, model: ContinualModel):
        within_var = []
        between_var = []
        features_norm = []
        global_means_norm = []
        snr = []

        all_features, all_labels, all_tasklabels = model.features[self.version]

        all_class_means = []
        for task in range(model.current_task+1):
            if dataset.SETTING != 'domain-il':
                start_label = task*model.cpt
                end_label = (task+1)*model.cpt
            else:
                start_label = 0
                end_label = model.cpt

            current_within_var = []
            current_class_means = []
            for lab in range(start_label, end_label):
                idx = (lab == all_labels) & (task == all_tasklabels) #evaluate metrics for every class
                current_features = all_features[idx]

                mean_feature = torch.mean(current_features, dim=0)
                current_class_means.append(mean_feature)
                all_class_means.append(mean_feature.unsqueeze(0))

                current_within_var.append(calculate_variance(current_features, mean_feature).item())

            current_snr = []
            for class1 in range(0, end_label-start_label):
                for class2 in range(0, class1):
                    current_snr.append((torch.norm(current_class_means[class1]-current_class_means[class2], dim=0, p=2)**2
                        / (current_within_var[class1] + current_within_var[class2])).item())
                    
            within_var.append(sum(current_within_var) / len(current_within_var))
            snr.append(sum(current_snr) / len(current_snr))

        all_class_means = torch.cat(all_class_means, dim=0)    
        self.within_var.append(within_var)
        self.snr.append(snr)

        class_means = [] 
        current_within_var = []
        for lab in range(model.n_seen_classes):
            current_features = all_features[(all_labels == lab) & (all_tasklabels <= model.current_task)]
            mean_feature = torch.mean(current_features, dim=0)
            class_means.append(mean_feature.unsqueeze(0))

            current_within_var.append(calculate_variance(current_features, mean_feature).item())
    
        self.within_var_together.append(sum(current_within_var) / len(current_within_var))
        class_means = torch.cat(class_means, dim=0)

        for lab in range(0, (model.current_task+1)*model.cpt, model.cpt):
            global_mean = torch.mean(all_class_means[lab:lab+model.cpt], dim=0)

            between_var.append(calculate_variance(all_class_means[lab:lab+model.cpt], global_mean).item())
            features_norm.append(torch.norm(all_class_means[lab:lab+model.cpt] - global_mean, p=2, dim=1).mean().item())
            global_means_norm.append((torch.norm(global_mean, dim=0, p=2).item()))
        

        current_snr = []
        if model.args.training_setting == 'class-il':
            self.between_var_together.append(calculate_variance(class_means).item())

            for class1 in range(model.n_seen_classes):
                for class2 in range(class1):
                    current_snr.append((torch.norm(class_means[class1]-class_means[class2], dim=0, p=2)**2
                        / (current_within_var[class1] + current_within_var[class2])).item())
        else:
            current_between_var = []
            for lab in range(0, (model.current_task+1)*model.cpt, model.cpt):
                current_between_var.append(calculate_variance(class_means[lab:lab+model.cpt]).item())

                for class1 in range(lab, lab+model.cpt):
                    for class2 in range(lab, class1):
                        current_snr.append((torch.norm(class_means[class1]-class_means[class2], dim=0, p=2)**2
                            / (current_within_var[class1] + current_within_var[class2])).item())

            self.between_var_together.append(sum(current_between_var) / len(current_between_var))
        
        self.snr_together.append(sum(current_snr) / len(current_snr))
        self.between_var.append(between_var)
        self.features_norm.append(features_norm)
        self.global_means_norm.append(global_means_norm)
        return all_class_means, class_means
    
    def log_projection(self, model, projection_taskwise, projection_global, all_means, means):
        norm = []
        norm_complement = []
        for lab in range(0, (model.current_task+1)*model.cpt, model.cpt):
            taskwise_means = all_means[lab:lab+model.cpt]
            taskwise_means = taskwise_means - torch.mean(taskwise_means, dim=0) 

            projected_features = (projection_taskwise @ taskwise_means.T).T
            norm.append(torch.norm(projected_features, dim=1, p=2).mean().item())
            
            complement_features = taskwise_means - projected_features
            norm_complement.append(torch.norm(complement_features, dim=1, p=2).mean().item())

        self.norm.append(norm)
        self.norm_complement.append(norm_complement)

        global_means = means
        if (model.args.training_setting == 'class-il'):
            global_means = global_means - torch.mean(global_means, dim=0)
        else:
            for lab in range(0, (model.current_task+1)*model.cpt, model.cpt):
                global_means[lab:lab+model.cpt] = global_means[lab:lab+model.cpt] - torch.mean(global_means[lab:lab+model.cpt], dim=0)

        projected_features = (projection_global @ global_means.T).T
        self.norm_together.append(torch.norm(projected_features, dim=1, p=2).mean().item())

        complement_features = global_means - projected_features
        self.norm_complement_together.append(torch.norm(complement_features, dim=1, p=2).mean().item())

        return

    def log_classifier(self, dataset :ContinualDataset, model: ContinualModel):
        entropy = []

        all_features, all_labels, all_tasklabels = model.features[self.version]
        with torch.no_grad():
            device = next(model.net.parameters()).device  # get device from model
            all_logits = model.net.final_layer(
                all_features.to(device),
                all_tasklabels.to(device)
            ).detach().cpu()

        for task in range(all_tasklabels.max() + 1):
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

        self.var_diagonal = []
        self.var_diagonal_together = []
        self.NC2_diagonal = []
        self.NC2_diagonal_together = []
        self.beta = []
        self.beta_together = []

        self.var_off_diagonal = []
        self.var_off_diagonal_together = []
        self.NC2_off_diagonal = []
        self.NC2_off_diagonal_together = []
        
        self.var_between_tasks = []
        self.var_between_tasks_together = []
        self.NC2_between_tasks = []
        self.NC2_between_tasks_together = []

        self.var_all_off_diagonal = []
        self.NC2_all_off_diagonal = []
        self.var_all_off_diagonal_together = []
        self.NC2_all_off_diagonal_together = []

        self.NC3 = []
        self.NC3_together = []

        self.rank = []

    @torch.no_grad
    def log(self, dataset: ContinualDataset, model: ContinualModel):

        var_diagonal = []
        NC2_diagonal = []
        beta = []
        
        var_off_diagonal = []
        NC2_off_diagonal = []
        
        NC3 = []

        all_train_means, train_means = self.all_loggers['train_dataset'].log(dataset, model)
        all_test_means, test_means = self.all_loggers['test_dataset'].log(dataset, model)

        if not ('buffer' in self.all_loggers):
            buffer_means = all_train_means[model.current_task*model.cpt:(model.current_task+1)*model.cpt]

            U = buffer_means
            U_tilde = U - torch.mean(U, dim=0)
            U_tilde = U_tilde.T

            q, r = torch.linalg.qr(U_tilde, mode='reduced')
            projection_taskwise = q @ q.T
            projection_global = q @ q.T
        else:
            all_buffer_means, buffer_means = self.all_loggers['buffer'].log(dataset, model)

            if isinstance(model.net.classifier, nn.Linear):
                classifier_weights = (model.net.classifier.weight.detach().cpu()[:model.n_seen_classes]).T
            else: 
                weights = [layer.weight.detach().cpu() for layer in model.net.classifier]
                classifier_weights = (torch.cat(weights, dim=0)[:model.n_seen_classes]).T

            #Here we do the task wise computation
            U = all_buffer_means

            U_tilde = torch.zeros_like(U)
            if (model.args.training_setting == 'class-il') and (dataset.SETTING == 'class-il'):
                U_tilde = U - torch.mean(U, dim=0)
            else:
                for lab in range(0, (model.current_task+1)*model.cpt, model.cpt):
                    current_U = U[lab:lab+model.cpt]
                    U_tilde[lab:lab+model.cpt] = current_U - torch.mean(current_U, dim=0)

            U_tilde = U_tilde.T
            UT_U_tilde = U_tilde.T @ U_tilde
            print(UT_U_tilde)
            q, r = torch.linalg.qr(U_tilde, mode='reduced')
            projection_taskwise = q @ q.T

            try:
                print(torch.linalg.matrix_rank(U_tilde).item())
            except:
                pass

            U_tilde_normalized = U_tilde / U_tilde.norm(dim=0, keepdim=True, p=2)
            UT_U_tilde_normalized = U_tilde_normalized.T @ U_tilde_normalized  

            block_mask = torch.zeros_like(UT_U_tilde, dtype=torch.bool)
            between_mask = torch.ones_like(UT_U_tilde, dtype=torch.bool)

            for lab in range(0, (model.current_task+1)*model.cpt, model.cpt):
                U_tilde_normalized_block = UT_U_tilde_normalized[lab:lab+model.cpt, lab:lab+model.cpt]
                U_tilde_block = UT_U_tilde[lab:lab+model.cpt, lab:lab+model.cpt]
                
                NC2_diagonal.append(torch.diag(U_tilde_normalized_block).mean().item())
                beta.append(torch.sqrt(torch.diag(U_tilde_block)).mean().item())
                var_diagonal.append(calculate_variance(torch.diag(U_tilde_block)).item())

                current_block = ~torch.eye(U_tilde_block.shape[0], dtype=torch.bool)
                NC2_off_diagonal.append(U_tilde_normalized_block[current_block].mean().item())
                var_off_diagonal.append(calculate_variance(U_tilde_normalized_block[current_block]).item())

                block_mask[lab:lab+model.cpt, lab:lab+model.cpt] = current_block
                between_mask[lab:lab+model.cpt, lab:lab+model.cpt] = False
    
            self.NC2_between_tasks.append(UT_U_tilde_normalized[between_mask].mean().item())
            self.var_between_tasks.append(calculate_variance(UT_U_tilde_normalized[between_mask]).item())

            self.NC2_all_off_diagonal.append(UT_U_tilde_normalized[~torch.eye(UT_U_tilde_normalized.shape[0], dtype=torch.bool)].mean().item())
            self.var_all_off_diagonal.append(calculate_variance(UT_U_tilde_normalized[~torch.eye(UT_U_tilde_normalized.shape[0], dtype=torch.bool)]).item())

            classifier_weights = classifier_weights / classifier_weights.norm(dim=0, keepdim=True, p=2)

            for lab in range(0, (model.current_task+1)*model.cpt, model.cpt):
                if dataset.SETTING != 'domain-il':
                    block = classifier_weights.T @ U_tilde_normalized 
                    NC3.append(torch.diag(block)[lab: lab+model.cpt].mean().item())
                else:
                    block = classifier_weights.T @ U_tilde_normalized[:, lab:lab+model.cpt]
                    NC3.append(torch.diag(block).mean().item())

            self.var_diagonal.append(var_diagonal)
            self.NC2_diagonal.append(NC2_diagonal)
            self.beta.append(beta)

            self.var_off_diagonal.append(var_off_diagonal)
            self.NC2_off_diagonal.append(NC2_off_diagonal)

            self.NC3.append(NC3)

            #here we do the global computation
            U = buffer_means

            U_tilde = torch.zeros_like(U)
            if model.args.training_setting == 'class-il':
                U_tilde = U - torch.mean(U, dim=0)
            else:
                for lab in range(0, (model.current_task+1)*model.cpt, model.cpt):
                    current_U = U[lab:lab+model.cpt]
                    U_tilde[lab:lab+model.cpt] = current_U - torch.mean(current_U, dim=0)

            U_tilde = U_tilde.T
            UT_U_tilde = U_tilde.T @ U_tilde
            q, r = torch.linalg.qr(U_tilde, mode='reduced')
            projection_global = q @ q.T

            U_tilde_normalized = U_tilde / U_tilde.norm(dim=0, keepdim=True, p=2)
            UT_U_tilde_normalized = U_tilde_normalized.T @ U_tilde_normalized 

            if dataset.SETTING == 'domain-il':
                block_mask = ~torch.eye(UT_U_tilde.shape[0], dtype=torch.bool)
                between_mask= torch.zeros_like(UT_U_tilde, dtype=torch.bool)

            self.NC2_diagonal_together.append(torch.diag(UT_U_tilde_normalized).mean().item())
            self.beta_together.append(torch.sqrt(torch.diag(UT_U_tilde)).mean().item())
            self.var_diagonal_together.append(calculate_variance(torch.diag(UT_U_tilde)).item())

            self.NC2_off_diagonal_together.append(UT_U_tilde_normalized[block_mask].mean().item())
            self.var_off_diagonal_together.append(calculate_variance(UT_U_tilde_normalized[block_mask]).item())

            self.NC2_between_tasks_together.append(UT_U_tilde_normalized[between_mask].mean().item())
            self.var_between_tasks_together.append(calculate_variance(UT_U_tilde_normalized[between_mask]).item())

            self.NC2_all_off_diagonal_together.append(UT_U_tilde_normalized[~torch.eye(UT_U_tilde_normalized.shape[0], dtype=torch.bool)].mean().item())
            self.var_all_off_diagonal_together.append(calculate_variance(UT_U_tilde_normalized[~torch.eye(UT_U_tilde_normalized.shape[0], dtype=torch.bool)]).item())

            classifier_weights = classifier_weights.T @ U_tilde_normalized
            self.NC3_together.append(torch.diag(classifier_weights).mean().item())

        self.rank.append(torch.linalg.matrix_rank(U_tilde).item())
        self.all_loggers['train_dataset'].log_projection(model, projection_taskwise, projection_global, all_train_means, train_means)
        self.all_loggers['test_dataset'].log_projection(model, projection_taskwise, projection_global, all_test_means, test_means)
    
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
                wrargs['within_var_task' + str(i + 1)] = fa     


            for i, fa in enumerate(value.between_var):
                for j, var in enumerate(fa):
                    wrargs['between_var_' + str(j + 1) + '_task' + str(i+1)] = var
            
            for i, fa in enumerate(value.between_var_together):
                wrargs['between_var_task' + str(i + 1) ] = fa


            for i, fa in enumerate(value.global_means_norm):
                for j, var in enumerate(fa):
                    wrargs['global_norm_' + str(j + 1) + '_task' + str(i+1)] = var

            for i, fa in enumerate(value.features_norm):
                for j, var in enumerate(fa):
                    wrargs['features_norm_' + str(j + 1) + '_task' + str(i+1)] = var


            for i, fa in enumerate(value.snr):
                for j, var in enumerate(fa):
                    wrargs['snr_' + str(j + 1) + '_task' + str(i+1)] = var
            
            for i, fa in enumerate(value.snr_together):
                wrargs['snr_task' + str(i + 1) ] = fa

            
            for i, fa in enumerate(value.norm):
                for j, var in enumerate(fa):
                    wrargs['norm_' + str(j + 1) + '_task' + str(i+1)] = var
            
            for i, fa in enumerate(value.norm_together):
                wrargs['norm_task' + str(i + 1) ] = fa


            for i, fa in enumerate(value.norm_complement):
                for j, var in enumerate(fa):
                    wrargs['norm_complement_' + str(j + 1) + '_task' + str(i+1)] = var
            
            for i, fa in enumerate(value.norm_complement_together):
                wrargs['norm_complement_task' + str(i + 1) ] = fa
        

            if key == 'train_dataset':
                for i, fa in enumerate(self.var_diagonal):
                    for j, var in enumerate(fa):
                        wrargs['var_diagonal_' + str(j + 1) + '_task' + str(i+1)] = var
                
                for i, fa in enumerate(self.var_diagonal_together):
                    wrargs['var_diagonal_task' + str(i+1)] = fa

                for i, fa in enumerate(self.NC2_diagonal):
                    for j, var in enumerate(fa):
                        wrargs['NC2_diagonal_' + str(j + 1) + '_task' + str(i+1)] = var
                
                for i, fa in enumerate(self.NC2_diagonal_together):
                    wrargs['NC2_diagonal_task' + str(i+1)] = fa

                for i, fa in enumerate(self.beta):
                    for j, var in enumerate(fa):
                        wrargs['beta_' + str(j + 1) + '_task' + str(i+1)] = var
                
                for i, fa in enumerate(self.beta_together):
                    wrargs['beta_task' + str(i+1)] = fa


                for i, fa in enumerate(self.var_off_diagonal):
                    for j, var in enumerate(fa):
                        wrargs['var_off_diagonal_' + str(j + 1) + '_task' + str(i+1)] = var
                
                for i, fa in enumerate(self.var_off_diagonal_together):
                    wrargs['var_off_diagonal_task' + str(i+1)] = fa

                for i, fa in enumerate(self.NC2_off_diagonal):
                    for j, var in enumerate(fa):
                        wrargs['NC2_off_diagonal_' + str(j + 1) + '_task' + str(i+1)] = var
                
                for i, fa in enumerate(self.NC2_off_diagonal_together):
                    wrargs['NC2_off_diagonal_task' + str(i+1)] = fa


                for i, fa in enumerate(self.var_between_tasks):
                    wrargs['var_between_tasks_task' + str(i+1)] = fa
                
                for i, fa in enumerate(self.var_between_tasks_together):
                    wrargs['var_between_tasks_together_task' + str(i+1)] = fa

                for i, fa in enumerate(self.NC2_between_tasks):
                    wrargs['NC2_between_tasks_task' + str(i+1)] = fa
                
                for i, fa in enumerate(self.NC2_between_tasks_together):
                    wrargs['NC2_between_tasks_together_task' + str(i+1)] = fa


                for i, fa in enumerate(self.var_all_off_diagonal):
                    wrargs['var_all_off_diagonal_task' + str(i+1)] = fa

                for i, fa in enumerate(self.var_all_off_diagonal_together):
                    wrargs['var_all_off_diagonal_together_task' + str(i+1)] = fa

                for i, fa in enumerate(self.NC2_all_off_diagonal):
                    wrargs['NC2_all_off_diagonal_task' + str(i+1)] = fa

                for i, fa in enumerate(self.NC2_all_off_diagonal_together):
                    wrargs['NC2_all_off_diagonal_together_task' + str(i+1)] = fa
                

                for i, fa in enumerate(self.NC3):
                    for j, var in enumerate(fa):
                        wrargs['NC3_' + str(j + 1) + '_task' + str(i+1)] = var
                
                for i, fa in enumerate(self.NC3_together):
                    wrargs['NC3_task' + str(i+1)] = fa

            
                for i, fa in enumerate(self.rank):
                    wrargs['rank_task' + str(i+1)] = fa
                

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
