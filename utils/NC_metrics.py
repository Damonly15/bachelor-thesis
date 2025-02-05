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
        buf_x, buf_lab, buf_tl = buffer.get_all_data(transform=model.transform)
    else: #if samples are already augmented, like with the buffer containing the test samples, don't apply any augmentation when retrieving the samples
        buf_x, buf_lab, buf_tl = buffer.get_all_data()

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
    max_samples = model.args.buffer_size // model.N_CLASSES
    current_intra_class_var = []
    for tl in buf_lab.unique():
        idx = tl == buf_lab #evaluate metrics for every class
        current_buf_x = buf_x[idx]
        current_buf_x = current_buf_x[:max_samples]

        all_features = []
        for i in range(0, current_buf_x.size(0), model.args.batch_size):
            inputs = current_buf_x[i: i+model.args.batch_size]
            inputs = inputs.to(model.device)
            features = model.net.forward(inputs, returnt="features").detach().cpu()
            all_features.append(features)
        all_features = torch.cat(all_features, dim=0)

        mean_feature = torch.mean(all_features, dim=0)
        class_means.append(mean_feature.unsqueeze(0))

        if fixed_mean is None:
            current_intra_class_var.append(calculate_variance(all_features, mean_feature).item())
        else:
            current_intra_class_var.append(calculate_variance(all_features, fixed_mean[tl]).item())

        if((tl%model.cpt)==(model.cpt-1)): #if it is last class of a task
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
    if mean is None:
        mean = torch.mean(features, dim=0)
    norms = torch.norm(features - mean, dim=1) ** 2
    variance = norms.sum() / norms.size(0)
    return variance

def get_test_buffer(model, test_dataloaders):
    buffer = Buffer(model.args.buffer_size)

    examples_per_class = model.args.buffer_size // ((model.current_task + 1) * model.cpt)
    remainder = model.args.buffer_size % ((model.current_task + 1) * model.cpt)

    ce = torch.tensor([examples_per_class] * (model.cpt * (model.current_task + 1))).int()
    for i in range(remainder):
        ce[i] += 1 

    for (k, dataloader) in enumerate(test_dataloaders):
        start_pos = model.cpt * k
        end_pos = model.cpt * (k+1)
        current_ce = ce[start_pos:end_pos]
        for (inputs, labels) in dataloader: #test dataloader does not shuffle, therefore we always get the same samples in the buffer
            if not all(current_ce[:] == 0):
                flags = torch.zeros(len(inputs)).bool()
                for j in range(len(flags)):
                    if current_ce[labels[j] % model.cpt] > 0:
                        flags[j] = True
                        current_ce[labels[j] % model.cpt] -= 1

                buffer.add_data(examples=inputs[flags],
                                    labels=labels[flags],
                                    task_labels=(torch.ones(len(flags), dtype=torch.int64) * k)[flags])
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
    print("Logging Class-IL results and arguments in " + path)
    with open(path, 'a') as f:
        f.write(str(wrargs) + '\n')
