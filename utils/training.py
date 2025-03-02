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
from utils.feature_forgetting import feature_forgetting, buffer_forgetting
from utils.NC_metrics import evaluate_NC_metrics, calculate_mean_distance, log_NC

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
                outputs = model.net.forward(inputs, task_label=k)
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
    return accs, accs


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
    results, results_mask_classes = [], []

    if (args.log_feature_forgetting != 'output') or args.log_NC_metrics or (args.model in ['plots', 'plots2']):
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
    if (args.log_feature_forgetting == 'features') or (args.log_feature_forgetting == 'buffer'):
        feature_forgetting_loggers = []
        feature_forgetting_loggers.append(Logger(dataset.SETTING, dataset.NAME, model.NAME))
    elif (args.log_feature_forgetting == 'all'):
        feature_forgetting_loggers = []
        feature_forgetting_loggers.append(Logger(dataset.SETTING, dataset.NAME, model.NAME))
        feature_forgetting_loggers.append(Logger(dataset.SETTING, dataset.NAME, model.NAME))

    if args.log_NC_metrics:
        NC_metrics = [[], [], []]

    if args.start_from is not None:
        for i in range(args.start_from):
            train_loader, _ = dataset.get_data_loaders()
            model.meta_begin_task(dataset)
            model.meta_end_task(dataset)

    if args.loadcheck is not None:
        model, past_res = mammoth_load_checkpoint(args, model)

        if not args.disable_log and past_res is not None:
            (results, results_mask_classes, csvdump) = past_res
            logger.load(csvdump)

        print('Checkpoint Loaded!')

    progress_bar = ProgressBar(joint=args.joint, verbose=not args.non_verbose)

    if args.enable_other_metrics:
        dataset_copy = get_dataset(args)
        for t in range(dataset.N_TASKS):
            model.net.train()
            _, _ = dataset_copy.get_data_loaders()
        if model.NAME != 'icarl' and model.NAME != 'pnn':
            random_results_class, random_results_task = evaluate(model, dataset_copy)

    print(file=sys.stderr)
    start_task = 0 if args.start_from is None else args.start_from
    end_task = dataset.N_TASKS if args.stop_after is None else args.stop_after

    torch.cuda.empty_cache()
    for t in range(start_task, end_task):
        model.net.train()
        train_loader, test_loader = dataset.get_data_loaders()

        model.meta_begin_task(dataset)

        if (not args.inference_only) and (not (args.joint and t != end_task-1)): #if joint training last task contains all samples
            if t and args.enable_other_metrics:
                accs = evaluate(model, dataset, last=True)
                results[t - 1] = results[t - 1] + accs[0]
                if dataset.SETTING == 'class-il':
                    results_mask_classes[t - 1] = results_mask_classes[t - 1] + accs[1]

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
                    elif hasattr(dataset, 'supconaugmentations'): 
                        inputs1, labels, inputs2, not_aug_img = data
                        inputs = torch.cat([inputs1, inputs2], dim=0)
                        inputs = inputs.to(model.device)
                        labels = labels.to(model.device, dtype=torch.long)
                        loss = model.meta_observe(inputs, labels, not_aug_img, epoch=epoch)
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

        model.meta_end_task(dataset)

        accs = evaluate(model, dataset)
        results.append(accs[0])
        results_mask_classes.append(accs[1])

        log_accs(args, logger, accs, t, dataset.SETTING)

        if(args.log_feature_forgetting == 'features'):
            full_accuracies = feature_forgetting(model, dataset, args.training_setting)
            log_accs(args, feature_forgetting_loggers[0], (full_accuracies, full_accuracies), t, dataset.SETTING)
        elif(args.log_feature_forgetting == 'buffer'):
            full_accuracies = buffer_forgetting(model, dataset, args.training_setting)
            log_accs(args, feature_forgetting_loggers[0], (full_accuracies, full_accuracies), t, dataset.SETTING)
        elif(args.log_feature_forgetting == 'all'):
            full_accuracies = feature_forgetting(model, dataset, args.training_setting)
            log_accs(args, feature_forgetting_loggers[0], (full_accuracies, full_accuracies), t, dataset.SETTING)
            full_accuracies = buffer_forgetting(model, dataset, args.training_setting)
            log_accs(args, feature_forgetting_loggers[1], (full_accuracies, full_accuracies), t, dataset.SETTING)

        if args.log_NC_metrics:  
            if args.buffer_size != 0:
                buffer_metrics, buffer_means = evaluate_NC_metrics(model, dataset, 'buffer') #replay buffer
            train_metrics, train_means = evaluate_NC_metrics(model, dataset, 'train_dataset') #train dataset
            test_metrics, test_means = evaluate_NC_metrics(model, dataset, 'test_dataset') #test dataset

            if args.buffer_size != 0:
                NC_metrics[0].append(buffer_metrics + (calculate_mean_distance(buffer_means, test_means[:model.n_seen_classes], model.cpt, 'norm'), calculate_mean_distance(buffer_means, test_means[:model.n_seen_classes], model.cpt, 'cos')))
            NC_metrics[1].append(train_metrics + (calculate_mean_distance(train_means, test_means, model.cpt, 'norm'), calculate_mean_distance(train_means, test_means, model.cpt, 'cos')))
            if args.buffer_size != 0:
                NC_metrics[2].append(test_metrics + (calculate_mean_distance(buffer_means, train_means[:model.n_seen_classes], model.cpt, 'norm'), calculate_mean_distance(buffer_means, train_means[:model.n_seen_classes], model.cpt, 'cos')))
            else:
                NC_metrics[2].append(test_metrics + (calculate_mean_distance(test_means, test_means, model.cpt, 'norm'), calculate_mean_distance(test_means, test_means, model.cpt, 'cos')))

        if args.savecheck:
            save_obj = {
                'model': model.state_dict(),
                'args': args,
                'results': [results, results_mask_classes, logger.dump()],
                'optimizer': model.opt.state_dict() if hasattr(model, 'opt') else None,
                'scheduler': scheduler.state_dict() if scheduler is not None else None,
            }
            if 'buffer_size' in model.args:
                save_obj['buffer'] = deepcopy(model.buffer).to('cpu')

            # Saving model checkpoint
            checkpoint_name = f'/cluster/scratch/dammeier/mammoth_checkpoints/{args.ckpt_name}_{t}.pt'
            torch.save(save_obj, checkpoint_name)

        #increase this at the end of a task    
        model._current_task = model._current_task + 1

    if args.validation:
        del dataset
        args.validation = None

        final_dataset = get_dataset(args)
        for _ in range(final_dataset.N_TASKS):
            final_dataset.get_data_loaders()
        accs = evaluate(model, final_dataset)
        log_accs(args, logger, accs, t, final_dataset.SETTING, prefix="FINAL")

    if not args.disable_log and args.enable_other_metrics:
        logger.add_bwt(results, results_mask_classes)
        logger.add_forgetting(results, results_mask_classes)
        if model.NAME != 'icarl' and model.NAME != 'pnn':
            logger.add_fwt(results, random_results_class,
                           results_mask_classes, random_results_task)

    if not args.disable_log:
        logger.write(vars(args), 'output')
        if not args.nowand:
            d = logger.dump()
            d['wandb_url'] = wandb.run.get_url()
            wandb.log(d)

    if((args.log_feature_forgetting == 'features') or (args.log_feature_forgetting == 'all')):
        feature_forgetting_loggers[0].write(vars(args), 'features')

        if(args.log_feature_forgetting == 'all'):
            feature_forgetting_loggers[1].write(vars(args), 'buffer')
    elif(args.log_feature_forgetting == 'buffer'):
        feature_forgetting_loggers[0].write(vars(args), 'buffer')


    if args.log_NC_metrics:
        if args.buffer_size != 0:
            log_NC(model, "buffer", NC_metrics[0])
        log_NC(model, "train_dataset", NC_metrics[1])
        log_NC(model, "test_dataset", NC_metrics[2])

    if not args.nowand:
        wandb.finish()
