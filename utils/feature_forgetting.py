import torch
from sklearn.linear_model import LogisticRegression
from typing import Tuple

from datasets import ContinualDataset
from models import ContinualModel

all_train_loaders = []

@torch.no_grad()
def evaluate_feature_forgetting(model: ContinualModel, dataset: ContinualDataset):
    all_train_loaders.append(dataset.train_loader)

    #extract features
    all_features = []
    all_labels = []

    model_status = model.net.training
    model.net.eval()
    for source in all_train_loaders:
        for data in source:
            if hasattr(dataset.train_loader.dataset, 'logits'):
                inputs, labels, _, _ = data
            else:
                inputs, labels, _ = data
            inputs = inputs.to(model.device)
            features = model.net.forward(inputs, returnt="features")
            all_features.append(features.detach())
            all_labels.append(labels)
    model.net.train(model_status)

    all_features = torch.cat(all_features, dim=0).cpu()
    all_labels = torch.cat(all_labels, dim=0).cpu()

    #fit logistic regression model, with multiomial loss. Set max iter really high, to make sure the optimizer converges.
    logreg_model = LogisticRegression(penalty=None, max_iter=5000)
    logreg_model.fit(all_features, all_labels)

    return evaluate(model, dataset, logreg_model)

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
def evaluate(model: ContinualModel, dataset: ContinualDataset, logreg_model, last=False) -> Tuple[list, list]:
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
    accs, accs_mask_classes = [], []
    n_classes = dataset.get_offsets()[1]
    for k, test_loader in enumerate(dataset.test_loaders):
        if last and k < len(dataset.test_loaders) - 1:
            continue
        correct, correct_mask_classes, total = 0.0, 0.0, 0.0
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
            outputs = logreg_model.predict_proba(features)
            outputs = torch.from_numpy(outputs)

            _, pred = torch.max(outputs[:, :n_classes], 1)
            correct += torch.sum(pred == labels).item()
            total += labels.shape[0]
            i += 1

            if dataset.SETTING == 'class-il':
                mask_classes(outputs, dataset, k)
                _, pred = torch.max(outputs.data, 1)
                correct_mask_classes += torch.sum(pred == labels).item()

        accs.append(correct / total * 100)
        accs_mask_classes.append(correct_mask_classes / total * 100)

    model.net.train(status)
    return accs, accs_mask_classes


