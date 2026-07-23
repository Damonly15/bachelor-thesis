"""
This script is the main entry point for the Mammoth project. It contains the main function `main()` that orchestrates the training process.

The script performs the following tasks:
- Imports necessary modules and libraries.
- Sets up the necessary paths and configurations.
- Parses command-line arguments.
- Initializes the dataset, model, and other components.
- Trains the model using the `train()` function.

To run the script, execute it directly or import it as a module and call the `main()` function.
"""
# Copyright 2022-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# needed (don't change it)
import numpy  # noqa
import importlib
import os
import socket
import sys
import datetime
import uuid
from argparse import ArgumentParser
import torch

mammoth_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(mammoth_path)
sys.path.append(mammoth_path + '/datasets')
sys.path.append(mammoth_path + '/backbone')
sys.path.append(mammoth_path + '/models')

from utils import create_if_not_exists, custom_str_underscore
from utils.args import add_management_args, add_experiment_args
from utils.conf import base_path
from utils.distributed import make_dp
from utils.conf import set_random_seed


def lecun_fix():
    # Yann moved his website to CloudFlare. You need this now
    from six.moves import urllib  # pyright: ignore
    opener = urllib.request.build_opener()
    opener.addheaders = [('User-agent', 'Mozilla/5.0')]
    urllib.request.install_opener(opener)


def parse_args():
    """
    Parse command line arguments for the mammoth program and sets up the `args` object.

    Returns:
        args (argparse.Namespace): Parsed command line arguments.
    """
    from models import get_all_models, get_model_class
    from datasets import get_dataset_names, get_dataset_class

    parser = ArgumentParser(description='mammoth', allow_abbrev=False, add_help=False)
    parser.add_argument('--model', type=custom_str_underscore, help='Model name.', choices=list(get_all_models().keys()))

    args = parser.parse_known_args()[0]
    models_dict = get_all_models()
    if args.model is None:
        print('No model specified. Please specify a model with --model to see all other options.')
        print('Available models are: {}'.format(list(models_dict.keys())))
        sys.exit(1)

    parser = get_model_class(args).get_parser()
    add_management_args(parser)
    add_experiment_args(parser)
    args = parser.parse_args()

    tmp_dset_class = get_dataset_class(args)
    n_epochs = tmp_dset_class.get_epochs()
    if (args.model == 'er-balanced' or args.model == 'er-metrics') and args.n_epochs is None:
        args.n_epochs = n_epochs * 2 #compensate for the fact that er balanced has a bigger batch size for the first task
    elif args.n_epochs is None:
        args.n_epochs = n_epochs
    else:
        if args.n_epochs != n_epochs:
            print('Warning: n_epochs set to {} instead of {}.'.format(args.n_epochs, n_epochs), file=sys.stderr)

    args.model = models_dict[args.model]

    if args.lr_scheduler is not None:
        print('Warning: lr_scheduler set to {}, overrides default from dataset.'.format(args.lr_scheduler), file=sys.stderr)

    if args.seed is not None:
        set_random_seed(args.seed)

    assert 0 < args.label_perc <= 1, "label_perc must be in (0, 1]"

    return args


def main(args=None):
    from models import get_model
    from datasets import ContinualDataset, get_dataset
    from utils.training import train

    lecun_fix()
    if args is None:
        args = parse_args()

    # set base path
    base_path(args.base_path)

    os.putenv("MKL_SERVICE_FORCE_INTEL", "1")
    os.putenv("NPY_MKL_FORCE_INTEL", "1")

    # Add uuid, timestamp and hostname for logging
    args.conf_jobnum = str(uuid.uuid4())
    args.conf_timestamp = str(datetime.datetime.now())
    args.conf_host = socket.gethostname()
    dataset = get_dataset(args)

    if args.batch_size is None:
        args.batch_size = dataset.get_batch_size()
        if hasattr(importlib.import_module('models.' + args.model), 'Buffer') and (not hasattr(args, 'minibatch_size') or args.minibatch_size is None):
            args.minibatch_size = dataset.get_minibatch_size()
    else:
        args.minibatch_size = args.batch_size

    model_compatibility = get_model(args, None, None, None).COMPATIBILITY
    backbone = dataset.get_backbone(args, model_compatibility)
    loss = dataset.get_loss()
    model = get_model(args, backbone, loss, dataset.get_transform())

    if args.distributed == 'dp':
        if args.batch_size < torch.cuda.device_count():
            raise Exception(f"Batch too small for DataParallel (Need at least {torch.cuda.device_count()}).")

        model.net = make_dp(model.net)
        model.to('cuda:0')
        args.conf_ngpus = torch.cuda.device_count()
    elif args.distributed == 'ddp':
        # DDP breaks the buffer, it has to be synchronized.
        raise NotImplementedError('Distributed Data Parallel not supported yet.')

    if args.debug_mode:
        print('Debug mode enabled: running only a few forward steps per epoch.')

    try:
        import setproctitle
        # set job name
        setproctitle.setproctitle('{}_{}_{}'.format(args.model, args.buffer_size if 'buffer_size' in args else 0, args.dataset))
    except Exception:
        pass

    if dataset.SETTING == "domain-il" and args.training_setting == "task-il":
        raise Exception("Task-IL training method is not compatible with a Domain-IL dataset. Please use Class-IL training with a Domain-IL dataset")

    train(model, dataset, args)


if __name__ == '__main__':
    main()
