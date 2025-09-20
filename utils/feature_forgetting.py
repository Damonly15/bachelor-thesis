import torch
from torch.nn import functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from typing import Tuple

def feature_forgetting(model, dataset, version):
    if version=='class-il':
        return feature_forgetting_cil(model, dataset, 'train_dataset')
    else:
        return feature_forgetting_til(model, dataset, 'train_dataset')
    
def buffer_forgetting(model, dataset, version):
    if version=='class-il':
        return feature_forgetting_cil(model, dataset, 'buffer')
    else:
        return feature_forgetting_til(model, dataset, 'buffer')
    
def clustering(model, dataset, version, num_iters=100):
    if version=='class-il':
        return clustering_cil(model, dataset, num_iters)
    else:
        return clustering_til(model, dataset, num_iters)

def feature_forgetting_til(model, dataset, version):
    heads = []    #Separate head for every task
    all_features, all_labels, all_tasklabels = model.features[version]

    for k in range(model.current_task+1):
        task_mask = all_tasklabels == k
        current_features = all_features[task_mask]
        current_labels = all_labels[task_mask] - k*model.cpt

        logreg_model = LogisticRegression(max_iter=5000, C=10)
        logreg_model.fit(current_features.numpy(), current_labels.numpy())
        heads.append(logreg_model)

    accuracy = evaluate_til(model, dataset, heads)
    
    return accuracy

def evaluate_til(model, dataset, heads) -> Tuple[list, list]:
    test_features, test_labels, test_tasklabels = model.features['test_dataset']

    accs = []
    for task in range(model.current_task+1):
        current_features = test_features[test_tasklabels == task]
        current_labels = test_labels[test_tasklabels == task]

        outputs = heads[task].predict_proba(current_features.numpy())
        outputs = torch.from_numpy(outputs)
        _, pred = torch.max(outputs, 1)

        current_labels = current_labels - (task*model.cpt)
        accs.append((pred == current_labels).float().mean().item() * 100)
    return accs

def feature_forgetting_cil(model, dataset, version):
    all_features, all_labels, all_tasklabels = model.features[version]
    current_features = all_features[all_tasklabels <= model.current_task]
    current_labels = all_labels[all_tasklabels <= model.current_task]

    logreg_model = LogisticRegression(max_iter=5000, C=10)
    logreg_model.fit(current_features.numpy(), current_labels.numpy())

    accuracy = evaluate_cil(model, dataset, logreg_model)
    return accuracy

def evaluate_cil(model, dataset, head) -> Tuple[list, list]:
    test_features, test_labels, test_tasklabels = model.features['test_dataset']

    accs = []
    for task in range(model.current_task+1):
        current_features = test_features[test_tasklabels == task]
        current_labels = test_labels[test_tasklabels == task]

        outputs = head.predict_proba(current_features.numpy())
        outputs = torch.from_numpy(outputs)
        _, pred = torch.max(outputs, 1)

        accs.append((pred == current_labels).float().mean().item() * 100)     
    return accs

def clustering_cil(model, dataset, num_iters):   
    buffer_features, buffer_labels, buffer_tasklabels = model.features['buffer']
    seen_mask = buffer_tasklabels <= model.current_task
    buffer_features = buffer_features[seen_mask]
    buffer_labels = buffer_labels[seen_mask]
    buffer_tasklabels = buffer_tasklabels[seen_mask]
    buffer_features = (model.projection @ buffer_features.T).T

    test_features, test_labels, test_tasklabels = model.features['test_dataset']
    seen_mask = test_tasklabels <= model.current_task
    test_features = test_features[seen_mask]
    test_labels = test_labels[seen_mask]
    test_tasklabels = test_tasklabels[seen_mask]
    test_features = (model.projection @ test_features.T).T
 
    # --- Initialization: cluster means = mean of buffer_features per label ---
    cluster_means = []
    for label in range(model.n_seen_classes):
        cluster_means.append(buffer_features[buffer_labels == label].mean(dim=0))
    cluster_means = torch.stack(cluster_means) # (K, D)

    for _ in range(num_iters):
        # --- E-step: assign test_features to nearest cluster ---
        dists = torch.cdist(test_features, cluster_means) # (M, K)
        test_assignments = dists.argmin(dim=1) # (M,)

        # --- M-step: update cluster means using test_features ---
        new_means = []
        for k in range(model.n_seen_classes):
            assigned = test_features[test_assignments == k]
            if len(assigned) > 0:
                new_means.append(assigned.mean(dim=0))
            else:
            # if no test feature assigned, keep old mean
                new_means.append(cluster_means[k])
                print("fallback was used")
        cluster_means = torch.stack(new_means)

    # --- Final label assignment: use buffer_features to map clusters -> labels ---
    dists_buf = torch.cdist(buffer_features, cluster_means) # (N, K)
    buffer_assignments = dists_buf.argmin(dim=1) # (N,)

    cluster_to_label = {}
    for k in range(model.n_seen_classes):
        #labels_k = buffer_labels[buffer_assignments == k]
        #if len(labels_k) > 0:
        # majority vote
        #    majority_label = labels_k.mode().values.item()
        #    cluster_to_label[k] = majority_label
        #else:
            # fallback: map directly from initialization
        cluster_to_label[k] = k
        #    print("fallback was used")

    acc = []
    for task in range(model.current_task+1):
        dists = torch.cdist(test_features[task == test_tasklabels], cluster_means)  # (M, K)
        test_assignments = dists.argmin(dim=1)             # (M,)

        # Map cluster indices → predicted labels
        pred_labels = torch.tensor(
            [cluster_to_label[int(c.item())] for c in test_assignments])

        # Accuracy
        acc.append((pred_labels == test_labels[task == test_tasklabels]).float().mean().item() * 100)
    return acc

def clustering_til(model, dataset, num_iters):
    buffer_features, buffer_labels, buffer_tasklabels = model.features['buffer']
    buffer_features = (model.projection @ buffer_features.T).T

    test_features, test_labels, test_tasklabels = model.features['test_dataset']
    test_features = (model.projection @ test_features.T).T

    # --- Initialization: cluster means = mean of buffer_features per label ---
    acc = []
    for task in range(model.current_task+1):
        current_buffer_features = buffer_features[buffer_tasklabels == task]
        current_buffer_labels = buffer_labels[buffer_tasklabels == task] - (task*model.cpt)

        current_test_features = test_features[test_tasklabels == task]
        current_test_labels = test_labels[test_tasklabels == task] - (task*model.cpt)

        cluster_means = []
        for label in range(model.cpt):
            cluster_means.append(current_buffer_features[current_buffer_labels == label].mean(dim=0))
        cluster_means = torch.stack(cluster_means) # (K, D)

        for _ in range(num_iters):
            # --- E-step: assign test_features to nearest cluster ---
            dists = torch.cdist(current_test_features, cluster_means) # (M, K)
            test_assignments = dists.argmin(dim=1) # (M,)

            # --- M-step: update cluster means using test_features ---
            new_means = []
            for k in range(model.cpt):
                assigned = current_test_features[test_assignments == k]
                if len(assigned) > 0:
                    new_means.append(assigned.mean(dim=0))
                else:
                # if no test feature assigned, keep old mean
                    new_means.append(cluster_means[k])
                    print("fallback was used")
                    
            cluster_means = torch.stack(new_means)

        # --- Final label assignment: use buffer_features to map clusters -> labels ---
        dists_buf = torch.cdist(current_buffer_features, cluster_means) # (N, K)
        buffer_assignments = dists_buf.argmin(dim=1) # (N,)

        cluster_to_label = {}
        for k in range(model.cpt):
            #labels_k = current_buffer_labels[buffer_assignments == k]
            #if len(labels_k) > 0:
            # majority vote
            #    majority_label = labels_k.mode().values.item()
            #    cluster_to_label[k] = majority_label
            #else:
                # fallback: map directly from initialization
            cluster_to_label[k] = k
            #    print("fallback was used")
    
        print(cluster_to_label)

        dists = torch.cdist(current_test_features, cluster_means)  # (M, K)
        test_assignments = dists.argmin(dim=1)             # (M,)

        # Map cluster indices → predicted labels
        pred_labels = torch.tensor(
            [cluster_to_label[int(c.item())] for c in test_assignments])

        # Accuracy
        acc.append((pred_labels == current_test_labels).float().mean().item() * 100)
    return acc


@torch.no_grad
def get_features(model, dataset, version, max_task, feat_or_log="features"):
    model_status = model.net.training
    model.net.eval()

    if version == 'buffer':        
        buf_x, all_labels, all_tasklabels = model.buffer.get_all_data(transform=model.transform)

        all_features = []
        for i in range(0, buf_x.shape[0], model.args.batch_size):
            inputs = buf_x[i: i+model.args.batch_size]
            current_labels = all_tasklabels[i: i+model.args.batch_size]
            inputs = inputs.to(model.device)

            if feat_or_log=="features":
                features = model.net.forward(inputs, returnt="features").detach().cpu()
            elif feat_or_log=="logits":
                features = model.net.forward(inputs, task_label=current_labels).detach().cpu()

            all_features.append(features)
            
        all_features = torch.cat(all_features, dim=0)
    elif version == 'train_dataset' or version == 'test_dataset':
        all_features, all_labels, all_tasklabels = [], [], []
        
        if version == 'train_dataset':
            dataloader = dataset.all_train_loaders
        else:
            dataloader = dataset.all_test_loaders

        for current_task, source in enumerate(dataloader):
            if current_task > max_task:
                break

            for data in source:
                if version == 'train_dataset':
                    if hasattr(dataloader[current_task].dataset, 'logits'):
                        inputs, labels, _, _ = data
                    else:
                        inputs, labels, _ = data
                else:
                    inputs, labels = data

                inputs = inputs.to(model.device)

                if feat_or_log=="features":
                    features = model.net.forward(inputs, returnt="features").detach().cpu()
                elif feat_or_log=="logits":
                    task_label = torch.ones(labels.shape[0], dtype=torch.int64, device=model.device) * current_task
                    features = model.net.forward(inputs, task_label=task_label).detach().cpu()

                all_features.append(features)
                all_labels.append(labels)
                all_tasklabels.append(torch.ones(labels.shape[0], dtype=torch.int64) * current_task)

        all_features = torch.cat(all_features, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        all_tasklabels = torch.cat(all_tasklabels, dim=0)        
    else:
        raise Exception("Something went wrong when getting the features")
    
    model.net.train(model_status)
    return all_features, all_labels, all_tasklabels    

#old code
'''
@torch.no_grad
def extra_metric(model: ContinualModel, head):
    probabilities_network = []
    probabilities_refitted = []
    for iter in all_train_loaders:
        for data in iter:
            if hasattr(iter, 'logits'):
                inputs, _, _, _ = data
            else:
                inputs, _, _ = data

            inputs = inputs.to(model.device)
            outputs_network, features = model.net.forward(inputs, returnt='both')
            outputs_network, features = outputs_network.detach().cpu(), features.detach().cpu()

            probability = F.softmax(outputs_network, dim=-1)
            probability = probability[:, model.n_past_classes:model.n_seen_classes].sum(dim=1)
            probabilities_network.append(probability)

            probability = head.predict_proba(features.numpy())
            probability = torch.from_numpy(probability)
            probability = probability[:, model.n_past_classes:model.n_seen_classes].sum(dim=1)
            probabilities_refitted.append(probability)


    probabilities_network = torch.cat(probabilities_network, dim=0)
    probabilities_network = torch.mean(probabilities_network, dim=0)
    print(f'network probabilities: {probabilities_network}')

    probabilities_refitted = torch.cat(probabilities_refitted, dim=0)
    probabilities_refitted = torch.mean(probabilities_refitted, dim=0)
    print(f'refitted probabilities: {probabilities_refitted}')

    return 

@torch.no_grad()
def evaluate_feature_forgetting_task(model: ContinualModel, dataset: ContinualDataset):
    all_train_loaders.append(dataset.train_loader)

    heads = []
    model_status = model.net.training
    model.net.eval()
    for k, source in enumerate(all_train_loaders):
        #fit head for every task
        all_features = []
        all_labels = []
        label_subtraction =  (k*dataset.N_CLASSES_PER_TASK)
        for data in source:
            if hasattr(dataset.train_loader.dataset, 'logits'):
                inputs, labels, _, _ = data
            else:
                inputs, labels, _ = data
            inputs = inputs.to(model.device)
            features = model.net.forward(inputs, returnt="features")
            all_features.append(features.detach())
            all_labels.append(labels - label_subtraction)

        all_features = torch.cat(all_features, dim=0).cpu()
        all_labels = torch.cat(all_labels, dim=0).cpu()

        logreg_model = LogisticRegression(penalty=None, max_iter=5000)
        logreg_model.fit(all_features, all_labels)
        heads.append(logreg_model)

    model.net.train(model_status)

    return evaluate_task(model, dataset, heads)

@torch.no_grad()
def evaluate_knn_forgetting(model: ContinualModel, dataset: ContinualDataset):
    all_outputs = []

    status = model.net.training
    model.net.eval()
    buf_x, buf_lab, _, _ = model.buffer.get_all_data(transform=model.transform)

    for i in range(0, model.args.buffer_size, model.args.batch_size):
        inputs = buf_x[i: i+model.args.batch_size]
        inputs = inputs.to(model.device)
        outputs = model.net.forward(inputs, returnt="features").detach().cpu()
        all_outputs.append(outputs)
    model.net.train(status)

    all_outputs = torch.cat(all_outputs, dim=0)

    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(all_outputs, buf_lab)

    return evaluate(model, dataset, knn)


def mask_classes(outputs: torch.Tensor, dataset: ContinualDataset, k: int) -> None:
    """
    Given the output tensor, the dataset at hand and the current task,
    masks the former by setting the responses for the other tasks at -inf.
    It is used to obtain the results for the task-il setting.

    Args:
        outputs: the output tensor
        dataset: the continual dataset
        k: the task index
    """
    outputs[:, 0:k * dataset.N_CLASSES_PER_TASK] = -float('inf')
    outputs[:, (k + 1) * dataset.N_CLASSES_PER_TASK:
            dataset.N_TASKS * dataset.N_CLASSES_PER_TASK] = -float('inf')

@torch.no_grad()
def evaluate_task(model: ContinualModel, dataset: ContinualDataset, heads, last=False) -> Tuple[list, list]:
    """
    Evaluates the accuracy of the model for each past task.

    The accuracy is evaluated for all the tasks up to the current one, only for the total number of classes seen so far.

    Args:
        model: the model to be evaluated
        dataset: the continual dataset at hand
        last: a boolean indicating whether to evaluate only the last task

    Returns:
        a tuple of lists, containing the class-il and task-il accuracy for each task. If return_loss is True, the loss is also returned as a third element.
    """
    status = model.net.training
    model.net.eval()
    accs, accs2 = [], []
    for k, test_loader in enumerate(dataset.test_loaders):
        if last and k < len(dataset.test_loaders) - 1:
            continue
        correct, total = 0.0, 0.0
        test_iter = iter(test_loader)
        i = 0
        while True:
            try:
                data = next(test_iter)
            except StopIteration:
                break
            if model.args.debug_mode and i > model.get_debug_iters():
                break
            inputs, labels = data
            inputs = inputs.to(model.device)
            
            #do proper forward pass
            features = model.net.forward(inputs, returnt="features")
            features = features.detach().cpu()
            outputs = heads[k].predict_proba(features)
            outputs = torch.from_numpy(outputs)

            _, pred = torch.max(outputs, 1)
            labels = labels - (k*dataset.N_CLASSES_PER_TASK)
            correct += torch.sum(pred == labels).item()
            total += labels.shape[0]
            i += 1

        accs.append(correct / total * 100)
        accs2.append(0.0)

    model.net.train(status)
    return accs, accs2

@torch.no_grad()
def evaluate_previous(net, dataset: ContinualDataset, device)  -> Tuple[list, list]:
    """
    Evaluates the accuracy of the model only for past tasks.

    Args:
        model: the model to be evaluated
        dataset: the continual dataset at hand
        last: a boolean indicating whether to evaluate only the last task
        return_loss: a boolean indicating whether to return the loss in addition to the accuracy

    Returns:
        a tuple of lists, containing the class-il and task-il accuracy for each task. If return_loss is True, the loss is also returned as a third element.
    """
    status = net.training
    net.eval()
    accs, accs2 = [], []
    n_classes = dataset.get_offsets()[1] - dataset.N_CLASSES_PER_TASK
    for k, test_loader in enumerate(dataset.test_loaders):
        if k >= len(dataset.test_loaders) - 1:
            continue
        correct, total = 0.0, 0.0
        test_iter = iter(test_loader)
        while True:
            try:
                data = next(test_iter)
            except StopIteration:
                break
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)

            _, pred = torch.max(outputs[:, :n_classes].data, 1)
            correct += torch.sum(pred == labels).item()
            total += labels.shape[0]

        accs.append(correct / total * 100)
        accs2.append(0.0)
        
    net.train(status)
    return accs, accs2
'''