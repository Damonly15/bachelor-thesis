# Copyright 2022-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from argparse import ArgumentParser
from datasets import get_dataset_names
from models import get_all_models
from models.utils.continual_model import ContinualModel
from utils import custom_str_underscore


def add_experiment_args(parser: ArgumentParser) -> None:
    """
    Adds the arguments used by all the models.

    Args:
        parser: the parser instance

    Returns:
        None
    """
    parser.add_argument('--dataset', type=str, required=True,
                        choices=get_dataset_names(),
                        help='Which dataset to perform experiments on.')
    parser.add_argument('--training_setting', type=str, required=False, default="class-il",
                           choices=["class-il", "task-il"],
                           help='Use class or task incremental training. Please use class-il for domain-il setting')
    parser.add_argument('--model', type=custom_str_underscore, required=True,
                        help='Model name.', choices=list(get_all_models().keys()))

    parser.add_argument('--lr', type=float, required=True,
                        help='Learning rate.')

    parser.add_argument('--optimizer', type=str, default='sgd',
                        choices=ContinualModel.AVAIL_OPTIMS,
                        help='Optimizer.')
    parser.add_argument('--optim_wd', type=float, default=0.,
                        help='optimizer weight decay.')
    parser.add_argument('--optim_mom', type=float, default=0.,
                        help='optimizer momentum.')
    parser.add_argument('--optim_nesterov', type=int, default=0,
                        help='optimizer nesterov momentum.')

    parser.add_argument('--lr_scheduler', type=str, help='Learning rate scheduler.')
    parser.add_argument('--lr_milestones', type=int, nargs='+', default=[],
                        help='Learning rate scheduler milestones (used if `lr_scheduler=multisteplr`).')
    parser.add_argument('--sched_multistep_lr_gamma', type=float, default=0.1,
                        help='Learning rate scheduler gamma (used if `lr_scheduler=multisteplr`).')

    parser.add_argument('--n_epochs', type=int,
                        help='Number of epochs.')
    parser.add_argument('--batch_size', type=int,
                        help='Batch size.')

    parser.add_argument('--distributed', type=str, default='no', choices=['no', 'dp', 'ddp'],
                        help='Enable distributed training?')
    parser.add_argument('--stop_after', type=int, default=None, help="Task limit")

    parser.add_argument('--joint', type=int, choices=[0, 1], default=0,
                        help='Train model on Joint (single task)?')
    parser.add_argument('--label_perc', type=float, default=1,
                        help='Percentage in (0-1] of labeled examples per task.')


def add_management_args(parser: ArgumentParser) -> None:
    """
    Adds the management arguments.

    Args:
        parser: the parser instance

    Returns:
        None
    """
    parser.add_argument('--seed', type=int, default=None,
                        help='The random seed.')
    parser.add_argument('--permute_classes', type=int, choices=[0, 1], default=0,
                        help='Permute classes before splitting tasks (applies seed before permute if seed is present)?')
    parser.add_argument('--base_path', type=str, default="./data/",
                        help='The base path where to save datasets, logs, results.')
    parser.add_argument('--notes', type=str, default=None,
                        help='Notes for this run.')

    parser.add_argument('--non_verbose', default=1, choices=[0, 1], type=int, help='Make progress bars non verbose')
    parser.add_argument('--disable_log', default=0, choices=[0, 1], type=int, help='Disable logging?')
    parser.add_argument('--num_workers', type=int, default=None, help='Number of workers for the dataloaders (default=infer from number of cpus).')

    parser.add_argument('--validation', type=int, help='Percentage of validation set drawn from the training set.')
    parser.add_argument('--enable_other_metrics', default=1, choices=[0, 1], type=int,
                        help='Enable computing additional metrics: forward and backward transfer.')
    parser.add_argument('--debug_mode', type=int, default=0, choices=[0, 1], help='Run only a few forward steps per epoch')

    parser.add_argument('--eval_epochs', type=int, default=None,
                        help='Perform inference intra-task at every `eval_epochs`.')
    parser.add_argument('--inference_only', action="store_true",
                        help='Perform inference only for each task (no training).')
    
    parser.add_argument('--log_feature_forgetting', type=int, default=0, choices=[0, 1], required=False,
                            help='Additionally evaluate performance witha fitted head.')
    parser.add_argument('--log_NC_metrics', type=int, default=0, choices=[0, 1], required=False,
                            help='Additionally evaluate NC metrics. It evaluate NC metrics on the replay buffer, a second buffer not used during replay and samples from the test set')
    parser.add_argument('--store_features', type=int, default=0, choices=[0, 1], required=False,
                            help='Store features at the end of training')


def add_rehearsal_args(parser: ArgumentParser) -> None:
    """
    Adds the arguments used by all the rehearsal-based methods

    Args:
        parser: the parser instance

    Returns:
        None
    """
    parser.add_argument('--buffer_size', type=int, required=True,
                        help='The size of the memory buffer.')
    parser.add_argument('--minibatch_size', type=int,
                        help='The batch size of the memory buffer.')
