# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
This module contains the Logger class and related functions for logging accuracy values and other metrics.
"""

from contextlib import suppress
import sys
from typing import Any, Dict

import numpy as np

from utils import create_if_not_exists, smart_joint
from utils.conf import base_path
from utils.metrics import backward_transfer, forward_transfer, forgetting
with suppress(ImportError):
    import wandb


def log_accs(args, logger, accs, t, setting, epoch=None, prefix="RESULT"):
    """
    Logs the accuracy values and other metrics.

    All metrics are prefixed with `RESULT_` to be logged on wandb.

    Args:
        args: The arguments for logging.
        logger: The Logger object.
        accs: The accuracy values.
        t: The task index.
        setting: The setting of the benchmark (e.g., `class-il`).
        epoch: The epoch number (optional).
        prefix: The prefix for the metrics (default="RESULT").
    """
    t += 1
    mean_acc = print_mean_accuracy(accs, t, setting, epoch=epoch)

    if not args.disable_log:
        logger.log(mean_acc)
        logger.log_fullacc(accs)

    if not args.nowand:
        postfix = "" #"" if epoch is None else f"_epoch_{epoch}"
        if setting == 'general-continual':
            d2 = {f'{prefix}_domain_mean_accs{postfix}': mean_acc,
                **{f'{prefix}_domain_acc_{i}{postfix}': a for i, a in enumerate(accs)},
                'Task': t}
        else:
            d2 = {f'{prefix}_class_mean_accs{postfix}': mean_acc,
                **{f'{prefix}_class_acc_{i}{postfix}': a for i, a in enumerate(accs)},
                'Task': t}

        wandb.log(d2)


def print_mean_accuracy(accs: np.ndarray, task_number: int,
                        setting: str, epoch=None) -> None:
    """
    Prints the mean accuracy on stderr.

    Args:
        accs: accuracy values per task
        task_number: task index
        setting: the setting of the benchmark
        joint: whether it's joint accuracy or not
        epoch: the epoch number (optional)

    Returns:
        The mean accuracy value.
    """
    mean_acc = np.mean(accs, axis=0)

    """ if joint:
        prefix = "Joint Accuracy" if epoch is None else f"Joint Accuracy (epoch {epoch})"
        if setting == 'domain-il' or setting == 'general-continual':
            mean_acc, _ = mean_acc
            print('\n{}: \t [Domain-IL]: {} %'.format(prefix, round(mean_acc, 2), file=sys.stderr))
            print('\tRaw accuracy values: Domain-IL {}'.format(accs[0]), file=sys.stderr)
        else:
            mean_acc_class_il, mean_acc_task_il = mean_acc
            print('\n{}: \t [Class-IL]: {} % \t [Task-IL]: {} %'.format(prefix, round(
                mean_acc_class_il, 2), round(mean_acc_task_il, 2)), file=sys.stderr)
            print('\tRaw accuracy values: Class-IL {} | Task-IL {}'.format(accs[0], accs[1]), file=sys.stderr)
    else:"""
    prefix = "Accuracy" if epoch is None else f"Accuracy (epoch {epoch})"
    if setting == 'general-continual':
        print('\n{} for {} task(s): [Domain-IL]: {} %'.format(prefix,
                                                                task_number, round(mean_acc, 2)), file=sys.stderr)
        print('\tRaw accuracy values: Domain-IL {}'.format(accs), file=sys.stderr)
    else:
        print('\n{} for {} task(s): \t [Class-IL]: {}'.format(prefix, task_number, round(mean_acc, 2), file=sys.stderr))
        print('\tRaw accuracy values: Class-IL {}'.format(accs,), file=sys.stderr)

    return mean_acc


class Logger:
    def __init__(self, setting_str: str, dataset_str: str,
                 model_str: str) -> None:
        """
        Initializes a Logger object. This will take track and log the accuracy values and other metrics in the default path (`data/results`).

        Args:
            setting_str: The setting of the benchmark.
            dataset_str: The dataset used.
            model_str: The model used.
        """
        self.accs = []
        self.fullaccs = []
        self.setting = setting_str
        self.dataset = dataset_str
        self.model = model_str
        self.forgetting = None

    def dump(self):
        """
        Dumps the state of the logger in a dictionary.

        Returns:
            A dictionary containing the logged values.
        """
        dic = {
            'accs': self.accs,
            'fullaccs': self.fullaccs,
            'forgetting': self.forgetting,
        }

        return dic

    def load(self, dic):
        """
        Loads the state of the logger from a dictionary.

        Args:
            dic: The dictionary containing the logged values.
        """
        self.accs = dic['accs']
        self.fullaccs = dic['fullaccs']
        self.forgetting = dic['forgetting']

    def rewind(self, num):
        """
        Rewinds the logger by a given number of values.

        Args:
            num: The number of values to rewind.
        """
        self.accs = self.accs[:-num]
        self.fullaccs = self.fullaccs[:-num]
        with suppress(BaseException):
            self.forgetting = self.forgetting[:-num]

    def add_fwt(self, results, accs, results_mask_classes, accs_mask_classes):
        """
        Adds forward transfer values.

        Args:
            results: The results.
            accs: The accuracy values.
            results_mask_classes: The results for masked classes.
            accs_mask_classes: The accuracy values for masked classes.
        """
        self.fwt = forward_transfer(results, accs)
        if self.setting == 'class-il':
            self.fwt_mask_classes = forward_transfer(results_mask_classes, accs_mask_classes)

    def add_bwt(self, results, results_mask_classes):
        """
        Adds backward transfer values.

        Args:
            results: The results.
            results_mask_classes: The results for masked classes.
        """
        self.bwt = backward_transfer(results)
        self.bwt_mask_classes = backward_transfer(results_mask_classes)

    def add_forgetting(self, results):
        """
        Adds forgetting values.

        Args:
            results: The results.
            results_mask_classes: The results for masked classes.
        """
        self.forgetting = forgetting(results)
        #self.forgetting_mask_classes = forgetting(results_mask_classes)

    def log(self, mean_acc: np.ndarray) -> None:
        """
        Logs a mean accuracy value.

        Args:
            mean_acc: mean accuracy value
        """
        self.accs.append(mean_acc)

    def log_fullacc(self, accs):
        """
        Logs all the accuracy of the classes from the current and past tasks.

        Args:
            accs: the accuracy values
        """
        self.fullaccs.append(accs)

    def write(self, args: Dict[str, Any], result_type) -> None:
        """
        Writes out the logged value along with its arguments in the default path (`data/results`).

        Args:
            args: the namespace of the current experiment
        """
        wrargs = args.copy()
        wrargs['result_type'] = result_type
        if 'class_order' in wrargs:
            del wrargs['class_order'] #don't need how we permuted the classes in the log file. This can get very long if we have many classes.

        target_folder = base_path() + "results/"

        for i, acc in enumerate(self.accs):
            wrargs['accmean_task' + str(i + 1)] = acc

        for i, fa in enumerate(self.fullaccs):
            for j, acc in enumerate(fa):
                wrargs['accuracy_' + str(j + 1) + '_task' + str(i + 1)] = acc

        wrargs['forgetting'] = self.forgetting

        if args["training_setting"] == "task-il":
            create_if_not_exists(target_folder + "task-il")
            create_if_not_exists(target_folder + "task-il" +
                                "/" + self.dataset)
            create_if_not_exists(target_folder + "task-il" +
                                "/" + self.dataset + "/" + self.model)
            path = target_folder + "task-il" + "/" + self.dataset\
                + "/" + self.model + "/logs.txt"
        else:
            create_if_not_exists(target_folder + self.setting)
            create_if_not_exists(target_folder + self.setting +
                                "/" + self.dataset)
            create_if_not_exists(target_folder + self.setting +
                                "/" + self.dataset + "/" + self.model)
            path = target_folder + self.setting + "/" + self.dataset\
                + "/" + self.model + "/logs.txt"
        
        print("Logging results and arguments in " + path)
        with open(path, 'a') as f:
            f.write(str(wrargs) + '\n')