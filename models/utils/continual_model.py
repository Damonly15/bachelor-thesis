"""
This is the base class for all models. It provides some useful methods and defines the interface of the models.

The `observe` method is the most important one: it is called at each training iteration and it is responsible for computing the loss and updating the model's parameters.

The `begin_task` and `end_task` methods are called before and after each task, respectively.

The `get_parser` method returns the parser of the model. Additional model-specific hyper-parameters can be added by overriding this method.

The `get_debug_iters` method returns the number of iterations to be used for debugging. Default: 3.

The `get_optimizer` method returns the optimizer to be used for training. Default: SGD.

The `load_buffer` method is called when a buffer is loaded. Default: do nothing.

The `meta_observe`, `meta_begin_task` and `meta_end_task` methods are wrappers for `observe`, `begin_task` and `end_task` methods, respectively. They take care of updating the internal counters.
"""

# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from abc import abstractmethod
from argparse import ArgumentParser, Namespace
from typing import List

import torch
import torch.nn as nn
import torch.optim as optim
from datasets import get_dataset
from datasets.utils.continual_dataset import ContinualDataset

from utils.conf import get_device, base_path
from utils.kornia_utils import to_kornia_transform
from torchvision import transforms
from utils.feature_forgetting import get_features
from utils import create_if_not_exists


class ContinualModel(nn.Module):
    """
    Continual learning model.
    """
    NAME: str
    COMPATIBILITY: List[str]
    AVAIL_OPTIMS = ['sgd', 'adam', 'adamw']

    @staticmethod
    def get_parser() -> Namespace:
        """
        Returns the parser of the model.

        Additional model-specific hyper-parameters can be added by overriding this method.

        Returns:
            the parser of the model
        """
        parser = ArgumentParser(description='Base CL model')
        return parser

    @property
    def current_task(self):
        """
        Returns the index of current task.
        """
        return self._current_task

    @property
    def n_classes_current_task(self):
        """
        Returns the number of classes in the current task.
        Returns -1 if task has not been initialized yet.
        """
        if hasattr(self, '_n_classes_current_task'):
            return self._n_classes_current_task
        else:
            return -1

    @property
    def n_seen_classes(self):
        """
        Returns the number of classes seen so far.
        Returns -1 if task has not been initialized yet.
        """
        if hasattr(self, '_n_seen_classes'):
            return self._n_seen_classes
        else:
            return -1

    @property
    def n_remaining_classes(self):
        """
        Returns the number of classes remaining to be seen.
        Returns -1 if task has not been initialized yet.
        """
        if hasattr(self, '_n_remaining_classes'):
            return self._n_remaining_classes
        else:
            return -1

    @property
    def n_past_classes(self):
        """
        Returns the number of classes seen up to the PAST task.
        Returns -1 if task has not been initialized yet.
        """
        if hasattr(self, '_n_past_classes'):
            return self._n_past_classes
        else:
            return -1

    @property
    def cpt(self):
        """
        Returns the raw number of classes per task.
        Warning: return value might be either an integer or a list of integers.
        """
        return self._cpt

    @cpt.setter
    def cpt(self, value):
        """
        Sets the number of classes per task.
        """
        self._cpt = value

    def __init__(self, backbone: nn.Module, loss: nn.Module,
                 args: Namespace, transform: nn.Module) -> None:
        super(ContinualModel, self).__init__()
        print("Using {} as backbone".format(backbone.__class__.__name__))
        self.net = backbone
        self.loss = loss
        self.args = args
        self.transform = transform
        self.dataset = get_dataset(self.args)
        self.N_CLASSES = self.dataset.N_CLASSES
        self.num_classes = self.N_CLASSES
        self.N_TASKS = self.dataset.N_TASKS
        self.n_tasks = self.N_TASKS
        self.SETTING = self.dataset.SETTING
        self._cpt = self.dataset.N_CLASSES_PER_TASK
        self._current_task = 0

        self.features = {
            'train_features': {},
            'test_features': {},
            'buffer_features': {},
            'nobuffer_features': {}
        }

        try:
            self.weak_transform = to_kornia_transform(transform.transforms[-1].transforms)
            self.normalization_transform = to_kornia_transform(self.dataset.get_normalization_transform())
        except BaseException:
            print("Warning: could not initialize kornia transforms.")
            self.weak_transform = transforms.Compose([transforms.ToPILImage(), self.transform])
            self.normalization_transform = transforms.Compose([transforms.ToPILImage(), self.dataset.TEST_TRANSFORM]) if hasattr(
                self.dataset, 'TEST_TRANSFORM') else transforms.Compose([transforms.ToPILImage(), transforms.ToTensor(), self.dataset.get_normalization_transform()])

        if self.net is not None:
            self.opt = self.get_optimizer()
        else:
            print("Warning: no default model for this dataset. You will have to specify the optimizer yourself.")
            self.opt = None
        self.device = get_device()

        if not self.NAME or not self.COMPATIBILITY:
            raise NotImplementedError('Please specify the name and the compatibility of the model.')

        if self.args.label_perc != 1 and 'cssl' not in self.COMPATIBILITY:
            print('WARNING: label_perc is not explicitly supported by this model -> training may break')

    def to(self, device):
        """
        Captures the device to be used for training.
        """
        self.device = device
        return super().to(device)

    def load_buffer(self, buffer, version='normal'):
        """
        Default way to handle load buffer.
        """
        self.buffer = buffer

    def get_parameters(self):
        """
        Returns the parameters of the model.
        """
        return self.net.parameters()

    def get_optimizer(self):
        # check if optimizer is in torch.optim
        supported_optims = {optim_name.lower(): optim_name for optim_name in dir(optim) if optim_name.lower() in self.AVAIL_OPTIMS}
        opt = None
        if self.args.optimizer.lower() in supported_optims:
            if self.args.optimizer.lower() == 'sgd':
                opt = getattr(optim, supported_optims[self.args.optimizer.lower()])(self.get_parameters(), lr=self.args.lr,
                                                                                    weight_decay=self.args.optim_wd,
                                                                                    momentum=self.args.optim_mom,
                                                                                    nesterov=self.args.optim_nesterov == 1)
            elif self.args.optimizer.lower() == 'adam' or self.args.optimizer.lower() == 'adamw':
                opt = getattr(optim, supported_optims[self.args.optimizer.lower()])(self.get_parameters(), lr=self.args.lr,
                                                                                    weight_decay=self.args.optim_wd)

        if opt is None:
            raise ValueError('Unknown optimizer: {}'.format(self.args.optimizer))
        return opt

    def _compute_offsets(self, task):
        cpt = self.N_CLASSES // self.N_TASKS
        offset1 = task * cpt
        offset2 = (task + 1) * cpt
        return offset1, offset2

    def get_debug_iters(self):
        """
        Returns the number of iterations to be used for debugging.
        Default: 3
        """
        return 5

    def begin_task(self, dataset: ContinualDataset) -> None:
        """
        Prepares the model for the current task.
        Executed before each task.
        """
        pass

    def end_task(self, dataset: ContinualDataset) -> None:
        """
        Prepares the model for the next task.
        Executed after each task.
        """
        pass


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes a forward pass.

        Args:
            x: batch of inputs
            task_label: some models require the task label

        Returns:
            the result of the computation
        """
        return self.net(x)

    def meta_observe(self, *args, **kwargs):
        """
        Wrapper for `observe` method.

        Takes care of dropping unlabeled data if not supported by the model.

        Args:
            inputs: batch of inputs
            labels: batch of labels
            not_aug_inputs: batch of inputs without augmentation
            kwargs: some methods could require additional parameters

        Returns:
            the value of the loss function
        """

        if 'cssl' not in self.COMPATIBILITY:  # drop unlabeled data if not supported
            labeled_mask = args[1] != -1
            if labeled_mask.sum() == 0:
                return 0
            args = [arg[labeled_mask] if isinstance(arg, torch.Tensor) and arg.shape[0] == args[0].shape[0] else arg for arg in args]
        ret = self.observe(*args, **kwargs)
        self.task_iteration += 1
        return ret

    def meta_begin_task(self, dataset):
        """
        Wrapper for `begin_task` method.

        Takes care of updating the internal counters.

        Args:
            dataset: the current task's dataset
        """
        self.task_iteration = 0
        self._n_classes_current_task = self._cpt if isinstance(self._cpt, int) else self._cpt[self._current_task]
        if self.SETTING != 'domain-il':
            self._n_seen_classes = self._cpt * (self._current_task + 1) if isinstance(self._cpt, int) else sum(self._cpt[:self._current_task + 1])
            self._n_remaining_classes = self.N_CLASSES - self._n_seen_classes
            self._n_past_classes = self._cpt * self._current_task if isinstance(self._cpt, int) else sum(self._cpt[:self._current_task])
        else:
            self._n_seen_classes = self._cpt
            self._n_remaining_classes = self._cpt
            self._n_past_classes = self._cpt
        self.begin_task(dataset)

    def meta_end_task(self, dataset):
        """
        Wrapper for `end_task` method.

        Takes care of updating the internal counters.
        No this is only done at the end of a task.

        Args:
            dataset: the current task's dataset
        """
        self.end_task(dataset)
        self._current_task = self._current_task + 1

    def store_features(self, dataset):
        if self.NAME in ['er_balanced', 'er_metrics']:
            versions = ['train_dataset', 'test_dataset', 'buffer']
        else:
            versions = ['train_dataset', 'test_dataset']

        for version in versions:
            if version == 'buffer':
                train_features, train_labels, train_tasklabels = self.features['train_dataset']
                mask = train_tasklabels == self.current_task
                train_features = train_features[mask]
                train_labels = train_labels[mask]
                train_tasklabels = train_tasklabels[mask]

                if self.args.buffer_size < (dataset.N_CLASSES_PER_TASK * dataset.N_TASKS) or self.current_task==0:
                    self.features[version] = train_features, train_labels, train_tasklabels
                else:
                    buffer_features, buffer_labels, buffer_tasklabels = get_features(self, dataset, version, self.current_task)
                    buffer_features = torch.cat((buffer_features, train_features), dim=0)
                    buffer_labels = torch.cat((buffer_labels, train_labels), dim=0)
                    buffer_tasklabels = torch.cat((buffer_tasklabels, train_tasklabels), dim=0)
            
                    self.features[version] = buffer_features, buffer_labels, buffer_tasklabels
            else:
                self.features[version] = get_features(self, dataset, version, self.current_task)

        if (self.args.store_features) and (self.current_task + 1 == dataset.N_TASKS):
            path = base_path() + f'features/{self.args.dataset}'
            create_if_not_exists(path)
            path = path + f'/bs{self.args.buffer_size}_s{self.args.seed}.pth'

            train_features, train_labels, train_tasklabels = self.features['train_dataset']
            test_features, test_labels, test_tasklabels = self.features['test_dataset']
            buffer_features, buffer_labels, buffer_tasklabels = self.features['buffer']

            torch.save({"train_features": train_features, "train_labels": train_labels, "train_tasklabels": train_tasklabels,
                        "test_features": test_features, "test_labels": test_labels, "test_tasklabels": test_tasklabels,
                        "buffer_features": buffer_features, "buffer_labels": buffer_labels, "buffer_tasklabels": buffer_tasklabels}, path)


    @abstractmethod
    def observe(self, inputs: torch.Tensor, labels: torch.Tensor,
                not_aug_inputs: torch.Tensor, epoch: int = None) -> float:
        """
        Compute a training step over a given batch of examples.

        Args:
            inputs: batch of examples
            labels: ground-truth labels
            kwargs: some methods could require additional parameters

        Returns:
            the value of the loss function
        """
        raise NotImplementedError
