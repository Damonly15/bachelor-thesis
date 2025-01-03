import torch
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from typing import Tuple
from torch.nn.functional import avg_pool2d
from utils.conf import base_path
from utils import create_if_not_exists

from datasets import ContinualDataset
from models import ContinualModel

all_train_loaders = []

@torch.no_grad()
def feature_forgetting_til(model: ContinualModel, dataset: ContinualDataset, number_layers):
    """
    Evaluate the feature quality at four different layers with a separate head
    """
    all_train_loaders.append(dataset.train_loader)
    
    model_status = model.net.training
    model.net.eval()

    heads = [[], [], [], [], []]    #Fit a separate linear head at all layers
    for k, source in enumerate(all_train_loaders):
        all_features = [[], [], [], [], []]
        all_labels = []
        label_subtraction = (k*dataset.N_CLASSES_PER_TASK)
        for data in source:
            if hasattr(dataset.train_loader.dataset, 'logits'):
                inputs, labels, _, _ = data
            else:
                inputs, labels, _ = data

            inputs = inputs.to(model.device)
            current_features = (model.net.forward(inputs, returnt="full"))[1]
            all_labels.append(labels - label_subtraction)

            for layer in range(0, number_layers):
                current = current_features[layer].detach()
                current = avg_pool2d(current, current.shape[2])
                current = current.view(current.size(0), -1)
                all_features[layer].append(current)

        for layer in range(0, number_layers):
            all_features[layer] = torch.cat(all_features[layer], dim=0).cpu()
        all_labels = torch.cat(all_labels, dim=0).cpu()

        for layer in range(0, number_layers):
            current_features = all_features[layer]
            logreg_model = LogisticRegression(max_iter=7000, C=10) #Use a small regularizer, this makes optimizer converge much faster
            logreg_model.fit(current_features, all_labels)
            heads[layer].append(logreg_model)


    full_accuracies = []
    for layer in range(0, number_layers):
        full_accuracies.append(evaluate_til(model, dataset, heads[layer], layer))
    model.net.train(model_status)
    return full_accuracies

@torch.no_grad()
def evaluate_til(model: ContinualModel, dataset: ContinualDataset, heads, layer) -> Tuple[list, list]:
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
            features = (model.net.forward(inputs, returnt="full"))[1][layer]
            features = avg_pool2d(features, features.shape[2])
            features = features.view(features.size(0), -1)

            features = features.detach().cpu()
            outputs = heads[k].predict_proba(features)
            outputs = torch.from_numpy(outputs)

            _, pred = torch.max(outputs, 1)
            labels = labels - (k*dataset.N_CLASSES_PER_TASK)
            correct += torch.sum(pred == labels).item()
            total += labels.shape[0]

        accs.append(correct / total * 100)
    return accs

@torch.no_grad()
def feature_forgetting_cil(model: ContinualModel, dataset: ContinualDataset, number_layers):
    """
    Evaluate the feature quality at four different layers with a common head
    """
    all_train_loaders.append(dataset.train_loader)
    
    model_status = model.net.training
    model.net.eval()

    heads = [] #Fit a linear head at all layers
    all_features = [[], [], [], [], []]
    all_labels = []
    for k, source in enumerate(all_train_loaders):
        for data in source:
            if hasattr(dataset.train_loader.dataset, 'logits'):
                inputs, labels, _, _ = data
            else:
                inputs, labels, _ = data

            inputs = inputs.to(model.device)
            current_features = (model.net.forward(inputs, returnt="full"))[1]
            all_labels.append(labels)

            for layer in range(0, number_layers):
                current = current_features[layer].detach()
                current = avg_pool2d(current, current.shape[2])
                current = current.view(current.size(0), -1)
                all_features[layer].append(current)

    for layer in range(0, number_layers):
        all_features[layer] = torch.cat(all_features[layer], dim=0).cpu()
    all_labels = torch.cat(all_labels, dim=0).cpu()

    for layer in range(0, number_layers):
        current_features = all_features[layer]
        logreg_model = LogisticRegression(max_iter=7000, C=10) #Use a small regularizer, this makes optimizer converge much faster
        logreg_model.fit(current_features, all_labels)
        heads.append(logreg_model)

    full_accuracies = []
    for layer in range(0, number_layers):
        full_accuracies.append(evaluate_cil(model, dataset, heads[layer], layer))
    model.net.train(model_status)
    return full_accuracies

@torch.no_grad()
def feature_forgetting_buffer(model: ContinualModel, dataset: ContinualDataset):
    all_outputs = []

    status = model.net.training
    model.net.eval()
    buf_x, buf_lab, _ = model.buffer.get_all_data(transform=model.transform)

    for i in range(0, model.args.buffer_size, model.args.batch_size):
        inputs = buf_x[i: i+model.args.batch_size]
        inputs = inputs.to(model.device)
        outputs = model.net.forward(inputs, returnt="features").detach().cpu()
        all_outputs.append(outputs)
    

    all_outputs = torch.cat(all_outputs, dim=0)

    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(all_outputs, buf_lab)
    
    accuracy = evaluate_cil(model, dataset, knn, 1)
    model.net.train(status)
    return accuracy

@torch.no_grad()
def evaluate_cil(model: ContinualModel, dataset: ContinualDataset, head, layer) -> Tuple[list, list]:
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
            features = (model.net.forward(inputs, returnt="full"))[1][layer]
            features = avg_pool2d(features, features.shape[2])
            features = features.view(features.size(0), -1)

            features = features.detach().cpu()
            outputs = head.predict_proba(features)
            outputs = torch.from_numpy(outputs)

            _, pred = torch.max(outputs, 1)
            labels = labels
            correct += torch.sum(pred == labels).item()
            total += labels.shape[0]

        accs.append(correct / total * 100)
    return accs

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