# Copyright 2022-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from copy import deepcopy
import math
import sys
from argparse import Namespace
from typing import Tuple

import torch
from datasets import get_dataset
from datasets.utils.continual_dataset import ContinualDataset
from datasets.utils.gcl_dataset import GCLDataset
from models.utils.continual_model import ContinualModel

from utils import random_id
from utils.checkpoints import mammoth_load_checkpoint
from utils.loggers import *
from utils.status import ProgressBar
from utils.feature_forgetting import feature_forgetting, clustering, buffer_forgetting
from utils.loggers_NC import LoggerNC
from utils import create_if_not_exists

try:
    import wandb
except ImportError:
    wandb = None


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
def evaluate(model: ContinualModel, dataset: ContinualDataset, last=False) -> Tuple[list, list]:
    """
    Evaluates the accuracy of the model for each past task.

    The accuracy is evaluated for all the tasks up to the current one, only for the total number of classes seen so far.

    Args:
        model: the model to be evaluated
        dataset: the continual dataset at hand

    Returns:
        a tuple of lists, containing the class-il and task-il accuracy for each task
    """
    status = model.net.training
    model.net.eval()
    accs = []
    n_classes = dataset.get_offsets()[1]
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
            inputs, labels = inputs.to(model.device), labels.to(model.device)
            if (model.args.training_setting == 'task-il') and ('task-il' in model.COMPATIBILITY):
                task_label = torch.ones(labels.shape[0], dtype=torch.int64, device=model.device) * k
                outputs = model.net.forward(inputs, task_label=task_label)
                labels = labels - (k*dataset.N_CLASSES_PER_TASK)
            elif (model.args.training_setting == 'task-il'):
                outputs = model(inputs)
                mask_classes(outputs, dataset, k)
            else:
                outputs = model(inputs)

            _, pred = torch.max(outputs[:, :n_classes].data, 1)
            correct += torch.sum(pred == labels).item()
            total += labels.shape[0]
            i += 1

        accs.append(correct / total * 100
                    if 'class-il' in model.COMPATIBILITY or 'general-continual' in model.COMPATIBILITY else 0)

    model.net.train(status)
    return accs


def initialize_wandb(args: Namespace) -> None:
    """
    Initializes wandb, if installed.

    Args:
        args: the arguments of the current execution
    """
    assert wandb is not None, "Wandb not installed, please install it or run without wandb"
    run_name = args.wandb_name if args.wandb_name is not None else args.model

    run_id = random_id(5)
    name = f'{run_name}_{run_id}'
    wandb.init(project=args.wandb_project, entity=args.wandb_entity, config=vars(args), name=name)
    args.wandb_url = wandb.run.get_url()


def train(model: ContinualModel, dataset: ContinualDataset,
          args: Namespace) -> None:
    """
    The training process, including evaluations and loggers.

    Args:
        model: the module to be trained
        dataset: the continual dataset at hand
        args: the arguments of the current execution
    """
    print(args)

    if not args.nowand:
        initialize_wandb(args)


    model.net.to(model.device)
    checkpoint_path = f'./checkpoints'
    results = []

    dataset_copy = get_dataset(args)
    all_train_loaders = []
    all_test_loaders = []
    for i in range(dataset.N_TASKS):
        train_loader, test_loader = dataset_copy.get_data_loaders()
        all_train_loaders.append(train_loader)
        all_test_loaders.append(test_loader)
    
    dataset.all_train_loaders = all_train_loaders
    dataset.all_test_loaders = all_test_loaders

    logger = Logger(dataset.SETTING, dataset.NAME, model.NAME)
    if (args.log_feature_forgetting):
        feature_forgetting_loggers = []
        feature_forgetting_loggers.append(Logger(dataset.SETTING, dataset.NAME, model.NAME))
        if dataset.SETTING != 'domain-il':
            feature_forgetting_loggers.append(Logger(dataset.SETTING, dataset.NAME, model.NAME))   
        if (model.NAME in ["er", "er_balanced"] and args.buffer_size >= dataset.N_CLASSES_PER_TASK * dataset.N_TASKS):
            clustering_forgetting_logger = Logger(dataset.SETTING, dataset.NAME, model.NAME)

    if args.log_NC_metrics:
        logger_NC = LoggerNC(model)

    if args.loadcheck is not None:
        args.loadcheck = checkpoint_path + f'/{args.ckpt_name}_{args.start_from}.pt'
        model, past_res = mammoth_load_checkpoint(args, model)

        for t in range(start_task):
            train_loader, test_loader = dataset.get_data_loaders()
            model.meta_begin_task(dataset)
            model.meta_end_task(dataset)

        print('Checkpoint Loaded!')

    progress_bar = ProgressBar(joint=args.joint, verbose=not args.non_verbose)

    print(file=sys.stderr)
    start_task = args.start_from
    end_task = dataset.N_TASKS if args.stop_after is None else args.stop_after

    torch.cuda.empty_cache()

    for t in range(start_task, end_task):
        model.net.train()
        train_loader, test_loader = dataset.get_data_loaders()
        model.meta_begin_task(dataset)

        if (not args.inference_only) and (not (args.joint and t != end_task-1)) and (t >= args.start_from): #if joint training last task contains all samples

            scheduler = dataset.get_scheduler(model, args) if not hasattr(model, 'scheduler') else model.scheduler
            for epoch in range(model.args.n_epochs):
                train_iter = iter(train_loader)
                data_len = None
                if not isinstance(dataset, GCLDataset):
                    data_len = len(train_loader)
                i = 0
                while True:
                    try:
                        data = next(train_iter)
                    except StopIteration:
                        break
                    if args.debug_mode and i > model.get_debug_iters():
                        break
                    if hasattr(dataset.train_loader.dataset, 'logits'):
                        inputs, labels, not_aug_inputs, logits = data
                        inputs = inputs.to(model.device)
                        labels = labels.to(model.device, dtype=torch.long)
                        not_aug_inputs = not_aug_inputs.to(model.device)
                        logits = logits.to(model.device)
                        loss = model.meta_observe(inputs, labels, not_aug_inputs, logits, epoch=epoch)
                    else:
                        inputs, labels, not_aug_inputs = data
                        inputs, labels = inputs.to(model.device), labels.to(model.device, dtype=torch.long)
                        not_aug_inputs = not_aug_inputs.to(model.device)
                        loss = model.meta_observe(inputs, labels, not_aug_inputs, epoch=epoch)
                    assert not math.isnan(loss)
                    progress_bar.prog(i, data_len, epoch, t, loss)
                    i += 1

                if scheduler is not None:
                    scheduler.step()

                if args.eval_epochs is not None and epoch % args.eval_epochs == 0 and epoch < model.args.n_epochs - 1:
                    epoch_accs = evaluate(model, dataset)

                    disable_log_state = args.disable_log
                    args.disable_log = 1
                    log_accs(args, logger, epoch_accs, t, dataset.SETTING, epoch=epoch)
                    args.disable_log = disable_log_state

        model.store_features(dataset)

        if args.log_NC_metrics:  
            logger_NC.log(dataset, model)    

        accs = evaluate(model, dataset)
        results.append(accs)

        log_accs(args, logger, accs, t, dataset.SETTING)

        if(args.log_feature_forgetting):
            full_accuracies = feature_forgetting(model, dataset, 'class-il')
            log_accs(args, feature_forgetting_loggers[0], full_accuracies, t, dataset.SETTING)
            if dataset.SETTING != 'domain-il':
                full_accuracies = feature_forgetting(model, dataset, 'task-il')
                log_accs(args, feature_forgetting_loggers[1], full_accuracies, t, dataset.SETTING)   
            if (model.NAME in ["er", "er_balanced"] and args.buffer_size >= dataset.N_CLASSES_PER_TASK * dataset.N_TASKS):
                full_accuracies = clustering(model, dataset, args.training_setting)
                log_accs(args, clustering_forgetting_logger, full_accuracies, t, dataset.SETTING)    

        if args.savecheck:
            save_obj = {
                'model': model.state_dict(),
                'args': args,
                'results': [results, logger.dump()],
                'optimizer': model.opt.state_dict() if hasattr(model, 'opt') else None,
                'scheduler': scheduler.state_dict() if scheduler is not None else None,
            }
            if 'buffer_size' in model.args:
                save_obj['buffer'] = deepcopy(model.buffer).to('cpu')

            # Saving model checkpoint
            checkpoint_name = checkpoint_path + f'/{args.ckpt_name}_{t}.pt'
            create_if_not_exists(checkpoint_path)
            torch.save(save_obj, checkpoint_name)  

        model.meta_end_task(dataset)
        

    if args.validation:
        del dataset
        args.validation = None

        final_dataset = get_dataset(args)
        for _ in range(final_dataset.N_TASKS):
            final_dataset.get_data_loaders()
        accs = evaluate(model, final_dataset)
        log_accs(args, logger, accs, t, final_dataset.SETTING, prefix="FINAL")

    if not args.disable_log and args.enable_other_metrics:
        logger.add_forgetting()

    if not args.disable_log:
        logger.write(vars(args), 'output')
        if not args.nowand:
            d = logger.dump()
            d['wandb_url'] = wandb.run.get_url()
            wandb.log(d)

    if args.log_feature_forgetting:
        if args.enable_other_metrics:
            feature_forgetting_loggers[0].add_forgetting()
        feature_forgetting_loggers[0].write(vars(args), 'features_cil')

        if dataset.SETTING != 'domain-il':
            if args.enable_other_metrics:
                feature_forgetting_loggers[1].add_forgetting()
            feature_forgetting_loggers[1].write(vars(args), 'features_til')

        if (model.NAME in ["er", "er_balanced"] and args.buffer_size >= dataset.N_CLASSES_PER_TASK * dataset.N_TASKS):
            if args.enable_other_metrics:
                clustering_forgetting_logger.add_forgetting()
            clustering_forgetting_logger.write(vars(args), 'clustering')

    if args.log_NC_metrics:
        logger_NC.write(model)

    if not args.nowand:
        wandb.finish()
