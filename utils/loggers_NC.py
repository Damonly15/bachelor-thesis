import torch
import torch.nn.functional as F
from torch.nn.functional import avg_pool2d

from models.utils.continual_model import ContinualModel
from datasets.utils.continual_dataset import ContinualDataset
from utils.feature_forgetting import get_features
from utils.conf import base_path
from utils import create_if_not_exists

@torch.no_grad
def calculate_variance(features, mean=None):
    bias_correction = 0
    if features.shape[0] <= 1:
        return torch.tensor(0.0)

    if mean is None:
        mean = torch.mean(features, dim=0)
        bias_correction = -1 #if the mean is not provided, then we need bias correction
        
    norms = torch.norm(features - mean, dim=1, p=2) ** 2
    variance = norms.sum() / (norms.shape[0] + bias_correction)
    return variance

class LoggerVersion:
    def __init__(self, version) -> None:
        self.within_var = []
        self.between_var = []
        self.global_var = []
        self.dist_OCS = []

        self.version = version

    @torch.no_grad
    def log(self, dataset :ContinualDataset, model: ContinualModel, global_mean):
        within_var = []
        between_var = []
        global_var = []
            
        if model.current_task <= 0 and self.version in ['buffer', 'nobuffer']:
            max_class = 0
        elif self.version in ['buffer', 'nobuffer']:
            max_class = model.n_past_classes
        else:
            max_class = dataset.N_CLASSES 

        all_features, all_labels, all_tasklabels = model.features[self.version]

        current_within_var = []
        class_means = []

        for lab in range(max_class):
            idx = lab == all_labels #evaluate metrics for every class
            current_features = all_features[idx]

            mean_feature = torch.mean(current_features, dim=0)
            class_means.append(mean_feature.unsqueeze(0))

            current_within_var.append(calculate_variance(current_features).item())

            if((lab%model.cpt)==(model.cpt-1)): #if it is last class of a task
                within_var.append(sum(current_within_var) / len(current_within_var))
                current_within_var = []
        
        if self.version == 'nobuffer':
            all_features2, all_labels2, all_tasklabels2 = model.features['train_dataset']

            for lab in range(max_class, dataset.N_CLASSES):
                idx = lab == all_labels2 #evaluate metrics for every class
                current_features = all_features2[idx]

                mean_feature = torch.mean(current_features, dim=0)
                class_means.append(mean_feature.unsqueeze(0))

                current_within_var.append(calculate_variance(current_features).item())

                if((lab%model.cpt)==(model.cpt-1)): #if it is last class of a task
                    within_var.append(sum(current_within_var) / len(current_within_var))
                    current_within_var = []

        if len(class_means) > 0:
            class_means = torch.cat(class_means, dim=0)
        else:
            class_means = torch.empty((0,))
        for lab in range(0, class_means.shape[0], model.cpt):
            if lab<max_class:
                task_idx = lab//model.cpt == all_tasklabels
                taskwise_features = all_features[task_idx]
            else:
                task_idx = lab//model.cpt == all_tasklabels2
                taskwise_features = all_features2[task_idx]

            global_var.append(calculate_variance(taskwise_features, global_mean).item())

            current_classes = class_means[lab: lab+model.cpt]
            between_var.append(calculate_variance(current_classes, global_mean).item())

        self.within_var.append(within_var)
        self.between_var.append(between_var)
        self.global_var.append(global_var)

        return class_means[:model.n_seen_classes]
    
    def log_classifier(self, dataset :ContinualDataset, model: ContinualModel, mean_prediction):
        dist_OCS = []
        max_logit = model.n_seen_classes

        if model.current_task <= 0 and ['buffer', 'nobuffer']:
            max_task = 0
        elif self.version in ['buffer', 'nobuffer']:
            max_task = model.current_task
            if self.version == 'nobuffer' and model.args.buffer_size == dataset.N_SAMPLES: #we have no hold ou samples then
                all_logits, all_labels, all_tasklabels = get_features(model, dataset, 'train_dataset', model.current_task, "logits") 
            else:
                all_logits, all_labels, all_tasklabels = get_features(model, dataset, self.version, model.current_task, "logits") 
        else:
            all_logits, all_labels, all_tasklabels = get_features(model, dataset, self.version, dataset.N_TASKS-1, "logits") 
            max_task = all_tasklabels.max().item() + 1

        for task in range(max_task):
            task_idx = all_tasklabels == task
            prob = F.softmax(all_logits[task_idx, :max_logit], dim=1)

            log_prob = F.log_softmax(all_logits[task_idx, :max_logit], dim=1)

            if model.args.training_setting == 'class-il':
                log_mean_prediction = F.log_softmax(mean_prediction[:max_logit], dim=0)
            else:
                log_mean_prediction = F.log_softmax(mean_prediction[task*model.cpt : task*model.cpt + model.cpt], dim=0)
            log_mean_prediction = log_mean_prediction.unsqueeze(0).expand_as(log_prob)

            kl_div = torch.sum(prob * (log_prob - log_mean_prediction), dim=1)  # per sample
            dist_OCS.append(kl_div.mean().item())

        if self.version == 'nobuffer':
            all_logits2, all_labels2, all_tasklabels2 = get_features(model, dataset, 'train_dataset', dataset.N_TASKS-1, "logits")

            for task in range(max_task, all_tasklabels2.max().item() + 1):
                task_idx = all_tasklabels2 == task
                prob = F.softmax(all_logits2[task_idx, :max_logit], dim=1)

                log_prob = F.log_softmax(all_logits2[task_idx, :max_logit], dim=1)

                if model.args.training_setting == 'class-il':
                    log_mean_prediction = F.log_softmax(mean_prediction[:max_logit], dim=0)
                else:
                    log_mean_prediction = F.log_softmax(mean_prediction[task*model.cpt : task*model.cpt + model.cpt], dim=0)
                log_mean_prediction = log_mean_prediction.unsqueeze(0).expand_as(log_prob)

                kl_div = torch.sum(prob * (log_prob - log_mean_prediction), dim=1)  # per sample
                dist_OCS.append(kl_div.mean().item())

        self.dist_OCS.append(dist_OCS)
        return
        
class LoggerNC:
    def __init__(self, model: ContinualModel) -> None:
        if hasattr(model, 'buffer') and model.args.buffer_size >= model.N_CLASSES:
            self.versions = ['buffer']
        else:
            self.versions = []

        self.versions = self.versions + ['nobuffer', 'test_dataset']     

        self.all_loggers = {
            i: LoggerVersion(i) for i in self.versions
        }

        self.model_weights = []
        self.global_mean_feature_norm = []
        self.mean_prediction = []

        self.gradient_sv = []
        self.features_sv = []
        
        self.past = {
            'b' : [],
            'cov' : [],
            'pred' : []
        }

        self.current = {
            'b' : [],
            'cov' : [],
            'pred' : []
        }

        self.future = {
            'b' : [],
            'cov' : [],
            'pred' : []
        }

    
    @torch.no_grad
    def log(self, dataset: ContinualDataset, model: ContinualModel):
        status = model.net.training
        model.net.eval()

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

        if model.NAME == "er_extra":
            model.gradient_sv = torch.stack(model.gradient_sv)
            self.gradient_sv.append(torch.mean(model.gradient_sv, dim=0))
            model.gradient_sv = []
        
            task_mask = all_tasklabels == model.current_task
            U, singular_values, Vh = torch.linalg.svd(all_features[task_mask], full_matrices=False)
            self.features_sv.append(singular_values)
        
            self.mean_prediction.append(mean_prediction)

        nobuffer_means = self.all_loggers['nobuffer'].log(dataset, model, global_mean)
        self.all_loggers['nobuffer'].log_classifier(dataset, model, global_mean)

        test_means = self.all_loggers['test_dataset'].log(dataset, model, global_mean)
        self.all_loggers['test_dataset'].log_classifier(dataset, model, global_mean)

        if model.args.buffer_size >= model.N_CLASSES:
            buffer_means = self.all_loggers['buffer'].log(dataset, model, global_mean)
            self.all_loggers['buffer'].log_classifier(dataset, model, global_mean)

        if model.current_task > 0 and model.current_task+1 < dataset.N_TASKS:
            if model.args.buffer_size >= model.N_CLASSES:
                U_tilde = (torch.cat([buffer_means, nobuffer_means[model.n_past_classes: model.n_seen_classes]], dim=0)).T# - global_mean).T
            else:
                U_tilde = (nobuffer_means[model.n_past_classes: model.n_seen_classes]).T# - global_mean).T
            projection = torch.inverse(U_tilde.T @ U_tilde) @ U_tilde.T

            for current_class, my_dictionary in [(0, self.past), (model.n_past_classes, self.current), (dataset.N_CLASSES-model.cpt, self.future)]:
                if current_class == 0:
                    all_features, all_labels, _ = model.features['nobuffer']
                else:
                    all_features, all_labels, _ = model.features['train_dataset']

                idx = all_labels == current_class
                current_features = all_features[idx]
                b = projection @ current_features.T

                my_dictionary['b'].append(torch.mean(b, dim=1))
                print(my_dictionary['b'])

                b_centered = b - b.mean(dim=1, keepdim=True) 
                my_dictionary['cov'] = (b_centered @ b_centered.T) / (b.shape[1] - 1) 
                print(my_dictionary['cov'])

                current_feature = torch.mean(current_features, dim=0).to(model.device)
                my_dictionary['mean_prediction'] = model.net.final_layer(current_feature, torch.ones(current_feature.shape[0], dtype=torch.int64, device=model.device) * current_class)[:model.n_seen_classes]
                print(my_dictionary['mean_prediction'])

        model_weights = 0.0
        for param in model.net.parameters():
            if param.requires_grad:  # Only count trainable parameters
                model_weights += torch.norm(param, p=2) ** 2
        self.model_weights.append(model_weights.sqrt().item()) 
        
        self.global_mean_feature_norm.append(torch.norm(global_mean, p=2, dim=0).item())
        
        model.net.train(status)

    def write(self, model: ContinualModel): 
        for key, value in self.all_loggers.items():

            wrargs = (vars(model.args)).copy()
            wrargs['result_type'] = key if key != 'nobuffer' else 'train_dataset'

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

            for i, fa in enumerate(value.dist_OCS):
                for j, var in enumerate(fa):
                    wrargs['dist_OCS_' + str(j + 1) + '_task' + str(i+1)] = var
            
            if key == 'nobuffer':
                for i, fa in enumerate(self.global_mean_feature_norm):
                    wrargs['feature_norm_task' + str(i+1)] = fa

                for i, fa in enumerate(self.model_weights):
                    wrargs['model_weights_task' + str(i+1)] = fa 


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

        if model.NAME == "er_extra":
            current_path = pre_path + "/mean_prediction"
            create_if_not_exists(current_path)
            with open(current_path + f'/{wrargs["buffer_size"]}_{wrargs["seed"]}.txt', "w") as f:
                for tensor in self.mean_prediction:
                    line = ' '.join([f"{v:.5f}" for v in tensor.tolist()])
                    f.write(line + "\n")

            current_path = pre_path + "/gradient_sv"
            create_if_not_exists(current_path)
            with open(current_path + f'/{wrargs["buffer_size"]}_{wrargs["seed"]}.txt', "w") as f:
                for tensor in self.gradient_sv:
                    line = ' '.join([f"{v:.5f}" for v in tensor.tolist()])
                    f.write(line + "\n")
            
            current_path = pre_path + "/features_sv"
            create_if_not_exists(current_path)
            with open(current_path + f'/{wrargs["buffer_size"]}_{wrargs["seed"]}.txt', "w") as f:
                for tensor in self.features_sv:
                    line = ' '.join([f"{v:.5f}" for v in tensor.tolist()])
                    f.write(line + "\n")


