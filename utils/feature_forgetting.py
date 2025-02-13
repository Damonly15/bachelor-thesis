import torch
from torch.nn import functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from typing import Tuple

from datasets import ContinualDataset
from models import ContinualModel

all_train_loaders = []

def feature_forgetting(model: ContinualModel, dataset: ContinualDataset, version):
    all_train_loaders.append(dataset.train_loader)
    if version=='class-il':
        return feature_forgetting_cil(model, dataset)
    else:
        return feature_forgetting_til(model, dataset)
    
def buffer_forgetting(model: ContinualModel, dataset: ContinualDataset, version):
    if version=='class-il':
        return buffer_forgetting_cil(model, dataset)
    else:
        return buffer_forgetting_til(model, dataset)
    


@torch.no_grad()
def feature_forgetting_til(model: ContinualModel, dataset: ContinualDataset):
    """
    Evaluate the feature quality at four different layers with a separate head
    """
    
    model_status = model.net.training
    model.net.eval()

    heads = []    #Separate head for every task
    for k, source in enumerate(all_train_loaders):
        all_features = []
        all_labels = []
        label_subtraction = (k*dataset.N_CLASSES_PER_TASK)
        for data in source:
            if hasattr(dataset.train_loader.dataset, 'logits'):
                inputs, labels, _, _ = data
            else:
                inputs, labels, _ = data

            mask = (labels >= label_subtraction)
            inputs, labels = inputs[mask], labels[mask] #this is needed because icarl adds buffer samples to datastream
            
            inputs = inputs.to(model.device)
            current_features = (model.net.forward(inputs, returnt="features")).detach().cpu()
            all_features.append(current_features)
            all_labels.append(labels - label_subtraction)

        all_features = torch.cat(all_features, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        
        logreg_model = LogisticRegression(max_iter=5000, C=10)
        logreg_model.fit(all_features.numpy(), all_labels.numpy())
        heads.append(logreg_model)

    accuracy = evaluate_til(model, dataset, heads)
    model.net.train(model_status)
    return accuracy

@torch.no_grad()
def buffer_forgetting_til(model: ContinualModel, dataset: ContinualDataset):
    status = model.net.training
    model.net.eval()
    
    buf_x, buf_lab, buf_tl = model.buffer.get_all_data(transform=model.transform)
    if hasattr(model, 'extra_buffer') and (not model.extra_buffer.is_empty()): #if we have two buffers, use both to fit the head
        extra_buf_x, extra_buf_lab, extra_buf_tl = model.extra_buffer.get_all_data(transform=model.transform)
        buf_x = torch.cat([buf_x, extra_buf_x], dim=0)
        buf_lab = torch.cat([buf_lab, extra_buf_lab], dim=0)
        buf_tl = torch.cat([buf_tl, extra_buf_tl], dim=0)

    heads = []
    for k in range(torch.max(buf_tl, dim=0).values + 1):
        task_mask = buf_tl == k
        current_buf_x = buf_x[task_mask]

        all_features = []
        all_labels = buf_lab[task_mask] - (k*dataset.N_CLASSES_PER_TASK)

        for i in range(0, current_buf_x.shape[0], model.args.batch_size):
            inputs = current_buf_x[i: i+model.args.batch_size]
            inputs = inputs.to(model.device)
            features = model.net.forward(inputs, returnt="features").detach().cpu()
            all_features.append(features)

        all_features = torch.cat(all_features, dim=0)

        logreg_model = LogisticRegression(max_iter=5000, C=10)
        logreg_model.fit(all_features.numpy(), all_labels.numpy())
        heads.append(logreg_model)

    accuracy = evaluate_til(model, dataset, heads)
    model.net.train(status)
    return accuracy

@torch.no_grad()
def evaluate_til(model: ContinualModel, dataset: ContinualDataset, heads) -> Tuple[list, list]:
    """
    Evaluates the accuracy of the model for each past task in TIL paradigm with a fitted head.
    """
    accs = []
    for k, test_loader in enumerate(dataset.test_loaders):
        correct, total = 0.0, 0.0
        test_iter = iter(test_loader)
        while True:
            try:
                data = next(test_iter)
            except StopIteration:
                break
            inputs, labels = data
            inputs = inputs.to(model.device)
            
            #do proper forward pass
            features = model.net.forward(inputs, returnt="features").detach().cpu()
            outputs = heads[k].predict_proba(features.numpy())
            outputs = torch.from_numpy(outputs)

            _, pred = torch.max(outputs, 1)
            labels = labels - (k*dataset.N_CLASSES_PER_TASK)
            correct += torch.sum(pred == labels).item()
            total += labels.shape[0]

        accs.append(correct / total * 100)
    return accs

@torch.no_grad()
def feature_forgetting_cil(model: ContinualModel, dataset: ContinualDataset):
    """
    Evaluate the feature quality at four different layers with a common head
    """
    
    model_status = model.net.training
    model.net.eval()

    all_features = []
    all_labels = []
    for k, source in enumerate(all_train_loaders):
        for data in source:
            if hasattr(dataset.train_loader.dataset, 'logits'):
                inputs, labels, _, _ = data
            else:
                inputs, labels, _ = data

            inputs = inputs.to(model.device)
            current_features = model.net.forward(inputs, returnt="features").detach().cpu()
            all_features.append(current_features)
            all_labels.append(labels)

    all_features = torch.cat(all_features, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    logreg_model = LogisticRegression(max_iter=5000, C=10)
    logreg_model.fit(all_features.numpy(), all_labels.numpy())

    accuracy = evaluate_cil(model, dataset, logreg_model)
    model.net.train(model_status)
    return accuracy

@torch.no_grad()
def buffer_forgetting_cil(model: ContinualModel, dataset: ContinualDataset):
    status = model.net.training
    model.net.eval()
    
    buf_x, buf_lab, _ = model.buffer.get_all_data(transform=model.transform)
    if hasattr(model, 'extra_buffer') and (not model.extra_buffer.is_empty()): #if we have two buffers, use both to fit the head
        extra_buf_x, extra_buf_lab, _ = model.extra_buffer.get_all_data(transform=model.transform)
        buf_x = torch.cat([buf_x, extra_buf_x], dim=0)
        buf_lab = torch.cat([buf_lab, extra_buf_lab], dim=0)

    all_features = []
    for i in range(0, buf_x.shape[0], model.args.batch_size):
        inputs = buf_x[i: i+model.args.batch_size]
        inputs = inputs.to(model.device)
        features = model.net.forward(inputs, returnt="features").detach().cpu()
        all_features.append(features)

    all_features = torch.cat(all_features, dim=0)

    logreg_model = LogisticRegression(max_iter=5000, C=10)
    logreg_model.fit(all_features.numpy(), buf_lab.numpy())
    
    accuracy = evaluate_cil(model, dataset, logreg_model)

    model.net.train(status)
    return accuracy

@torch.no_grad()
def evaluate_cil(model: ContinualModel, dataset: ContinualDataset, head) -> Tuple[list, list]:
    """
    Evaluates the accuracy of the model for each past task in CIL paradigm with a fitted head.
    """
    accs = []
    for k, test_loader in enumerate(dataset.test_loaders):
        correct, total = 0.0, 0.0
        test_iter = iter(test_loader)
        while True:
            try:
                data = next(test_iter)
            except StopIteration:
                break
            inputs, labels = data
            inputs = inputs.to(model.device)
            
            #do proper forward pass
            features = (model.net.forward(inputs, returnt="features")).detach().cpu()
            outputs = head.predict_proba(features.numpy())
            outputs = torch.from_numpy(outputs)

            _, pred = torch.max(outputs, 1)
            correct += torch.sum(pred == labels).item()
            total += labels.shape[0]

        accs.append(correct / total * 100)
    return accs

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
#old code
'''
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