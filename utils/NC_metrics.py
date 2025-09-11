import torch

from utils import create_if_not_exists
from utils.conf import base_path
from utils.feature_forgetting import get_features

@torch.no_grad
def evaluate_NC_metrics(model, dataset, version):
    status = model.net.training
    model.net.eval()

    intra_class_var = []
    inter_class_var_task = []
    inter_class_var_overall = 0.0
    inter_class_var_previous = 0.0
    inter_class_var_current = 0.0
    global_variance = []
    features_norm = []
    class_means = []

    if(model.args.model == 'plot_nc'):
        all_features, all_labels, all_tasklabels = get_features(model, dataset, version, 9)
    else:
        all_features, all_labels, all_tasklabels = get_features(model, dataset, version, model.current_task)

    current_intra_class_var = []
    current_features_norm = []
    for lab in range(all_labels.max().item() + 1):
        idx = lab == all_labels #evaluate metrics for every class
        current_features = all_features[idx]

        mean_feature = torch.mean(current_features, dim=0)
        class_means.append(mean_feature.unsqueeze(0))

        if(current_features.shape[0] > 1):
            current_intra_class_var.append(calculate_variance(current_features).item())
        else:
            current_intra_class_var.append(0.0)

        current_features_norm.append(torch.norm(current_features, dim=1, p=2).mean(dim=0).item())

        if((lab%model.cpt)==(model.cpt-1)): #if it is last class of a task
            intra_class_var.append(sum(current_intra_class_var) / len(current_intra_class_var))
            current_intra_class_var = []

            features_norm.append(sum(current_features_norm) / len(current_features_norm))
            current_features_norm = []

            current_class_means = torch.cat(class_means[-model.cpt:], dim=0)
            inter_class_var_task.append(calculate_variance(current_class_means).item())

    class_means = torch.cat(class_means, dim=0)

    inter_class_var_overall = calculate_variance(class_means).item()
    inter_class_var_current = inter_class_var_task[-1]
    if model.current_task > 0:
        inter_class_var_previous = calculate_variance(class_means[:-model.cpt]).item()
    else:
        inter_class_var_previous = 0.0

    for lab in range(all_labels.max().item() + 1):
        idx = lab == all_labels #evaluate metrics for every class
        current_features = all_features[idx]
        if model.args.training_setting == 'class-il':
            overall_mean = torch.mean(class_means, dim=0)
        else:
            start = (lab // model.cpt) * model.cpt
            end = (lab // model.cpt) * model.cpt + model.cpt
            overall_mean = torch.mean(class_means[start:end], dim=0)
        global_variance.append(calculate_variance(current_features, overall_mean).item())

    global_variance = sum(global_variance) / len(global_variance)

    model.net.train(status)
    return (intra_class_var, inter_class_var_task, inter_class_var_overall, inter_class_var_current, inter_class_var_previous, global_variance, features_norm), class_means

def calculate_variance(features, mean=None):
    bias_correction = 0
    if mean is None:
        mean = torch.mean(features, dim=0)
        bias_correction = -1 #if the mean is not provided, then we need bias correction
        
    norms = torch.norm(features - mean, dim=1, p=2) ** 2
    variance = norms.sum() / (norms.shape[0] + bias_correction)
    return variance

def calculate_mean_distance(a, b, cpt, version):
    if version=='norm':
        norm = torch.norm(a-b, dim=1, p=2)
    elif version=='cos':
        norm = torch.nn.functional.cosine_similarity(a, b, dim=1)
    else:
        raise Exception("Something went wrong when calculating the norm")

    norm = norm.view(-1, cpt).mean(dim=1)
    norm = [x.item() for x in norm]
    return norm

def calculate_class_distance(a, b, cpt, setting):
    result = []
    current_result = []
    for i in range(a.shape[0]):
        current_distances = []
        if setting == 'class-il':
            start = 0
            end = a.shape[0]
        else:
            start = (i // cpt) * cpt
            end = (i // cpt) * cpt + cpt
        for j in range(start, end):
            if i==j:
                continue
            current_distances.append(torch.nn.functional.cosine_similarity(a[j].unsqueeze(0), b[i].unsqueeze(0)).item())
        current_result.append(sum(current_distances) / len(current_distances))

        if((i%cpt)==(cpt-1)):
            result.append(sum(current_result) / len(current_result))
            current_result=[]
    return result
        
def log_NC(model, result_type, NC_metrics):
    intracv = []
    intercv = []
    intercv_overall = []
    intercv_current = []
    intercv_previous = []
    global_variance = []
    features_norm = []
    cos_movement = []
    cos_mean_distance = []

    for (c_intracv, c_intercv, c_intercv_overall, c_intercv_current, c_intercv_previous, c_global_variance, c_features_norm, c_cos_movement, c_cos_mean_distance) in NC_metrics:
        intracv.append(c_intracv)
        intercv.append(c_intercv)
        intercv_overall.append(c_intercv_overall)
        intercv_current.append(c_intercv_current)
        intercv_previous.append(c_intercv_previous)
        global_variance.append(c_global_variance)
        features_norm.append(c_features_norm)
        cos_movement.append(c_cos_movement)
        cos_mean_distance.append(c_cos_mean_distance)
        
    wrargs = (vars(model.args)).copy()
    wrargs['result_type'] = result_type
    if 'class_order' in wrargs:
        del wrargs['class_order'] #don't need how we permuted the classes in the log file. This can get very long if we have many classes.

    target_folder = base_path() + "results/"

    for i, fa in enumerate(intracv):
        for j, var in enumerate(fa):
            wrargs['within_var_' + str(j + 1) + '_task' + str(i + 1)] = var
        
    for i, fa in enumerate(intercv):
        for j, var in enumerate(fa):
            wrargs['between_var_' + str(j + 1) + '_task' + str(i + 1)] = var
    
    for i, var in enumerate(intercv_overall):
        wrargs['between_var_overall_task' + str(i + 1)] = var

    for i, var in enumerate(intercv_current):
        wrargs['between_var_current_task' + str(i + 1)] = var

    for i, var in enumerate(intercv_previous):
        wrargs['between_var_previous_task' + str(i + 1)] = var   

    for i, var in enumerate(global_variance):
        wrargs['global_variance_task' + str(i + 1)] = var  

    for i, fa in enumerate(features_norm):
        for j, var in enumerate(fa):
            wrargs['features_norm_' + str(j + 1) + '_task' + str(i + 1)] = var

    for i, fa in enumerate(cos_movement):
        for j, var in enumerate(fa):
            wrargs['cos_movement_' + str(j + 1) + '_task' + str(i + 1)] = var

    for i, fa in enumerate(cos_mean_distance):
        for j, var in enumerate(fa):
            wrargs['cos_mean_distance_' + str(j + 1) + '_task' + str(i + 1)] = var

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



"""
if args.buffer_size != 0:
                buffer_metrics, buffer_means = evaluate_NC_metrics(model, dataset, 'buffer') #replay buffer
            train_metrics, train_means = evaluate_NC_metrics(model, dataset, 'train_dataset') #train dataset
            test_metrics, test_means = evaluate_NC_metrics(model, dataset, 'test_dataset') #test dataset

            if args.buffer_size != 0:
                NC_metrics[0].append(buffer_metrics + (calculate_mean_distance(buffer_means, test_means, model.cpt, 'cos'), calculate_class_distance(buffer_means, test_means, model.cpt, args.training_setting)))
            NC_metrics[1].append(train_metrics + (calculate_mean_distance(train_means, test_means, model.cpt, 'cos'), calculate_class_distance(train_means, test_means, model.cpt, args.training_setting)))
            if args.buffer_size != 0:
                NC_metrics[2].append(test_metrics + (calculate_mean_distance(buffer_means, train_means, model.cpt, 'cos'), calculate_class_distance(buffer_means, train_means, model.cpt, args.training_setting)))
            else:
                NC_metrics[2].append(test_metrics + (calculate_mean_distance(test_means, test_means, model.cpt, 'cos'), calculate_class_distance(test_means, test_means, model.cpt, args.training_setting)))

"""