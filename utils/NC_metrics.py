import torch

from utils import create_if_not_exists
from utils.conf import base_path
from utils.buffer import Buffer
from utils.feature_forgetting import get_features

@torch.no_grad
def evaluate_NC_metrics(model, dataset, version):
    status = model.net.training
    model.net.eval()

    intra_class_var = []
    inter_class_var = []
    class_means = []

    all_features, all_labels, _ = get_features(model, dataset, version)

    current_intra_class_var = []
    for lab in range(all_labels.max().item() + 1):
        idx = lab == all_labels #evaluate metrics for every class
        current_features = all_features[idx]

        mean_feature = torch.mean(current_features, dim=0)
        class_means.append(mean_feature.unsqueeze(0))

        if(current_features.shape[0] > 1):
            current_intra_class_var.append(calculate_variance(current_features).item())
        else:
            current_intra_class_var.append(0.0)

        if((lab%model.cpt)==(model.cpt-1)): #if it is last class of a task
            intra_class_var.append(sum(current_intra_class_var) / len(current_intra_class_var))
            current_intra_class_var = []

            if model.args.training_setting == 'task-il': #calculate inter_class_var separately for the dataset of each task if we are training task il
                current_class_means = torch.cat(class_means[-model.cpt:], dim=0)
                inter_class_var.append(calculate_variance(current_class_means).item())

    class_means = torch.cat(class_means, dim=0)
    if model.args.training_setting == 'class-il':
        inter_class_var = [calculate_variance(class_means).item()] * (dataset.N_TASKS)
       
    model.net.train(status)
    return (intra_class_var, inter_class_var), class_means

def calculate_variance(features, mean=None):
    bias_correction = 0
    if mean is None:
        mean = torch.mean(features, dim=0)
        bias_correction = -1 #if the mean is not provided, then we need bias correction
        
    norms = torch.norm(features - mean, dim=1, p=2) ** 2
    variance = norms.sum() / (norms.size(0) + bias_correction)
    return variance

def calculate_mean_distance(a, b, cpt, version):
    if version=='norm':
        norm = torch.norm(a-b, dim=1, p=2) ** 2
    elif version=='cos':
        norm = torch.nn.functional.cosine_similarity(a, b, dim=1)
    else:
        raise Exception("Something went wrong when calculating the norm")

    norm = norm.view(-1, cpt).mean(dim=1)
    norm = [x.item() for x in norm]
    return norm

def log_NC(model, result_type, NC_metrics):
    intra_class_var = []
    inter_class_var = []
    l2_distance = []
    cos_distance = []
    for (c_intra_class_var, c_inter_class_var, c_l2_distance, c_cos_distance) in NC_metrics:
        intra_class_var.append(c_intra_class_var)
        inter_class_var.append(c_inter_class_var)
        l2_distance.append(c_l2_distance)
        cos_distance.append(c_cos_distance)
        
    wrargs = (vars(model.args)).copy()
    wrargs['result_type'] = result_type
    if 'class_order' in wrargs:
        del wrargs['class_order'] #don't need how we permuted the classes in the log file. This can get very long if we have many classes.

    target_folder = base_path() + "results/"

    for i, fa in enumerate(intra_class_var):
        for j, var in enumerate(fa):
            wrargs['within_var_' + str(j + 1) + '_task' + str(i + 1)] = var
        
    for i, fa in enumerate(inter_class_var):
        for j, var in enumerate(fa):
            wrargs['between_var_' + str(j + 1) + '_task' + str(i + 1)] = var

    for i, fa in enumerate(l2_distance):
        for j, var in enumerate(fa):
            wrargs['l2_distance_' + str(j + 1) + '_task' + str(i + 1)] = var

    for i, fa in enumerate(cos_distance):
        for j, var in enumerate(fa):
            wrargs['cos_distance_' + str(j + 1) + '_task' + str(i + 1)] = var

    create_if_not_exists(target_folder + model.args.training_setting)
    create_if_not_exists(target_folder + model.args.training_setting +
                        "/" + model.args.dataset)
    create_if_not_exists(target_folder + model.args.training_setting +
                        "/" + model.args.dataset + "/" + model.args.model)

    path = target_folder + model.args.training_setting + "/" + model.args.dataset\
        + "/" + model.args.model + "/logs_NC.txt"
    print("Logging NC metrics in " + path)
    with open(path, 'a') as f:
        f.write(str(wrargs) + '\n')
