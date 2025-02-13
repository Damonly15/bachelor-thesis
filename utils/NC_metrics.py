import torch

from utils import create_if_not_exists
from utils.conf import base_path
from utils.buffer import Buffer

@torch.no_grad
def evaluate_NC_metrics(model, buffer, fixed_mean=None, augmentation=True):
    status = model.net.training
    model.net.eval()

    ranks = []
    intra_class_var = []
    class_means = []
    
    if augmentation:
        transform = model.transform
    else: #if samples are already augmented, like with the buffer containing the test samples, don't apply any augmentation when retrieving the samples
        transform = None

    if isinstance(buffer, tuple):
        buf_x, buf_lab, buf_tl = buffer[0].get_all_data(transform=transform)
        extra_buf_x, extra_buf_lab, extra_buf_tl = buffer[1].get_all_data(transform=transform)
        buf_x = torch.cat([buf_x, extra_buf_x], dim=0)
        buf_lab = torch.cat([buf_lab, extra_buf_lab], dim=0)
        buf_tl = torch.cat([buf_tl, extra_buf_tl], dim=0)
    else:
       buf_x, buf_lab, buf_tl = buffer.get_all_data(transform=transform)

    '''
    max_samples = min(model.args.buffer_size // model.N_TASKS, 512)
    for tl in buf_tl.unique():
        idx = tl == buf_tl #evaluate metrics for the dataset of every task
        current_buf_x = buf_x[idx]
        current_buf_x = current_buf_x[:max_samples]

        all_features = []
        for i in range(0, current_buf_x.size(0), model.args.batch_size):
            inputs = current_buf_x[i: i+model.args.batch_size]
            inputs = inputs.to(model.device)
            features = model.net.forward(inputs, returnt="features").detach()
            all_features.append(features)
        all_features = torch.cat(all_features, dim=0)

        ranks.append(torch.linalg.matrix_rank(all_features).item())
        '''
    current_intra_class_var = []
    for lab in buf_lab.unique():
        idx = lab == buf_lab #evaluate metrics for every class
        current_buf_x = buf_x[idx]

        all_features = []
        for i in range(0, current_buf_x.shape[0], model.args.batch_size):
            inputs = current_buf_x[i: i+model.args.batch_size]
            inputs = inputs.to(model.device)
            features = model.net.forward(inputs, returnt="features").detach().cpu()
            all_features.append(features)
        all_features = torch.cat(all_features, dim=0)

        mean_feature = torch.mean(all_features, dim=0)
        class_means.append(mean_feature.unsqueeze(0))

        if fixed_mean is None:
            current_intra_class_var.append(calculate_variance(all_features).item())
        else:
            current_intra_class_var.append(calculate_variance(all_features, fixed_mean[lab]).item())

        if((lab%model.cpt)==(model.cpt-1)): #if it is last class of a task
            intra_class_var.append(sum(current_intra_class_var) / len(current_intra_class_var))
            current_intra_class_var = []

    if fixed_mean is None:
        class_means = torch.cat(class_means, dim=0)
        inter_class_var = calculate_variance(class_means).item()
    else:
        inter_class_var = calculate_variance(fixed_mean).item()

    model.net.train(status)
    return (intra_class_var, inter_class_var), class_means

def calculate_variance(features, mean=None):
    bias_correction = 0
    if mean is None:
        mean = torch.mean(features, dim=0)
        bias_correction = -1 #if the mean is not provided, then we need bias correction
        
    norms = torch.norm(features - mean, dim=1) ** 2
    variance = norms.sum() / (norms.size(0) + bias_correction)
    return variance

def get_test_buffer(model, test_dataloaders):
    buffer = Buffer(100000) #make sure buffer fits complete test dataset

    for (k, dataloader) in enumerate(test_dataloaders):
        for (inputs, labels) in dataloader: #test dataloader does not shuffle, therefore we always get the same samples in the buffer
                buffer.add_data(examples=inputs,
                                    labels=labels,
                                    task_labels=(torch.ones(labels.shape[0], dtype=torch.int64) * k))
    return buffer     

def log_NC(model, result_type, NC_metrics):
    rank = []
    intra_class_var = []
    inter_class_var = []
    for (c_intra_class_var, c_inter_class_var) in NC_metrics:
        intra_class_var.append(c_intra_class_var)
        inter_class_var.append(c_inter_class_var)
        
    wrargs = (vars(model.args)).copy()
    wrargs['result_type'] = result_type
    if 'class_order' in wrargs:
        del wrargs['class_order'] #don't need how we permuted the classes in the log file. This can get very long if we have many classes.

    target_folder = base_path() + "results/"

    for i, fa in enumerate(intra_class_var):
        for j, var in enumerate(fa):
            wrargs['intra_class_var_' + str(j + 1) + '_task' + str(i + 1)] = var
        
    for i, var in enumerate(inter_class_var):
        wrargs['inter_class_var_task_' + str(i + 1)] = var

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
