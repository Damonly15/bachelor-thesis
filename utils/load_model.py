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
import time
import importlib
import os
import socket
import sys
import datetime
import uuid
from argparse import ArgumentParser
import torch
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.lines as mlines

mammoth_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(mammoth_path)
sys.path.append(mammoth_path + '/datasets')
sys.path.append(mammoth_path + '/backbone')
sys.path.append(mammoth_path + '/models')

from utils import create_if_not_exists, custom_str_underscore
from utils.args import add_management_args, add_experiment_args
from utils.conf import base_path
from utils.distributed import make_dp
from utils.best_args import best_args
from utils.conf import set_random_seed
from utils.checkpoints import mammoth_load_checkpoint
from utils.training import evaluate
from utils.feature_forgetting import get_features


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
    parser.add_argument('--load_best_args', action='store_true',
                        help='Loads the best arguments for each method, '
                             'dataset and memory buffer.')

    args = parser.parse_known_args()[0]
    models_dict = get_all_models()
    if args.model is None:
        print('No model specified. Please specify a model with --model to see all other options.')
        print('Available models are: {}'.format(list(models_dict.keys())))
        sys.exit(1)

    mod = importlib.import_module('models.' + models_dict[args.model])

    if args.load_best_args:
        parser.add_argument('--dataset', type=str, required=True,
                            choices=get_dataset_names(),
                            help='Which dataset to perform experiments on.')
        if hasattr(mod, 'Buffer'):
            parser.add_argument('--buffer_size', type=int, required=True,
                                help='The size of the memory buffer.')
        args = parser.parse_args()
        if args.model == 'joint':
            best = best_args[args.dataset]['sgd']
        else:
            best = best_args[args.dataset][args.model]
        if hasattr(mod, 'Buffer'):
            best = best[args.buffer_size]
        else:
            best = best[-1]

        parser = get_model_class(args).get_parser()
        add_management_args(parser)
        add_experiment_args(parser)
        to_parse = sys.argv[1:] + ['--' + k + '=' + str(v) for k, v in best.items()]
        to_parse.remove('--load_best_args')
        args = parser.parse_args(to_parse)
        if args.model == 'joint' and args.dataset == 'mnist-360':
            args.model = 'joint_gcl'
    else:
        parser = get_model_class(args).get_parser()
        add_management_args(parser)
        add_experiment_args(parser)
        args = parser.parse_args()

    tmp_dset_class = get_dataset_class(args)
    n_epochs = tmp_dset_class.get_epochs()
    if args.n_epochs is None:
        args.n_epochs = n_epochs
    else:
        if args.n_epochs != n_epochs:
            print('Warning: n_epochs set to {} instead of {}.'.format(args.n_epochs, n_epochs), file=sys.stderr)

    args.model = models_dict[args.model]

    if args.lr_scheduler is not None:
        print('Warning: lr_scheduler set to {}, overrides default from dataset.'.format(args.lr_scheduler), file=sys.stderr)

    if args.seed is not None:
        set_random_seed(args.seed)

    if args.savecheck:
        assert args.inference_only == 0, "Should not save checkpoint in inference only mode"

        now = time.strftime("%Y%m%d-%H%M%S")
        extra_ckpt_name = "" if args.ckpt_name is None else f"{args.ckpt_name}_"
        args.ckpt_name = f"{extra_ckpt_name}_{args.dataset}_{args.training_setting}_{args.model}_{args.buffer_size if hasattr(args, 'buffer_size') else 0}_{args.seed}"
        print("Saving checkpoint into", args.ckpt_name, file=sys.stderr)

    if args.joint:
        assert args.start_from is None and args.stop_after is None, "Joint training does not support start_from and stop_after"
        assert args.enable_other_metrics == 0, "Joint training does not support other metrics"

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

    if args.n_epochs is None and isinstance(dataset, ContinualDataset):
        args.n_epochs = dataset.get_epochs()
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
        print('Debug mode enabled: running only a few forward steps per epoch with W&B disabled.')
        args.nowand = 1

    if args.wandb_entity is None or args.wandb_project is None:
        print('Warning: wandb_entity and wandb_project not set. Disabling wandb.')
        args.nowand = 1
    else:
        print('Logging to wandb: {}/{}'.format(args.wandb_entity, args.wandb_project))
        args.nowand = 0

    try:
        import setproctitle
        # set job name
        setproctitle.setproctitle('{}_{}_{}'.format(args.model, args.buffer_size if 'buffer_size' in args else 0, args.dataset))
    except Exception:
        pass

    if dataset.SETTING == "domain-il" and args.training_setting == "task-il":
        raise Exception("Task-IL training method is not compatible with a Domain-IL dataset. Please use Class-IL training with a Domain-IL dataset")

    if args.log_feature_forgetting == 'all' and args.buffer_size == 0:
        args.log_feature_forgetting = 'features'
    elif args.log_feature_forgetting == 'buffer' and args.buffer_size == 0:
        args.log_feature_forgetting = 'output'

    #get all dataset
    dataset_copy = get_dataset(args)
    all_train_loaders = []
    all_test_loaders = []
    for i in range(dataset.N_TASKS):
        train_loader, test_loader = dataset_copy.get_data_loaders()
        all_train_loaders.append(train_loader)
        all_test_loaders.append(test_loader)
    
    dataset.all_train_loaders = all_train_loaders
    dataset.all_test_loaders = all_test_loaders

    load_model(model, dataset, args)

def load_model(model, dataset, args):
    palette = sns.color_palette("deep")[:2]

    fig, axes = plt.subplots(1, 5, figsize=(15, 3.1), dpi=800, sharey=True, sharex=True)

    for i in range(dataset.N_TASKS):
        args.loadcheck = f'/cluster/scratch/dammeier/mammoth_checkpoints/{args.ckpt_name}_{i}.pt'
        model, past_res = mammoth_load_checkpoint(args, model)
        model.net.eval()

        """
        _, _ = dataset.get_data_loaders()
        dataset.train_loader = dataset.all_trains_loaders[:i+1]
        print(evaluate(model, dataset))
        """
        if(i>=2):
            dataset_samples=50
        else:
            dataset_samples=100
        
        all_features, all_labels, all_tasklabels = get_features(model, dataset, 'train_dataset')
        task_mask = 2 == all_tasklabels
        current_features = all_features[task_mask][:dataset_samples]
        current_labels = all_labels[task_mask][:dataset_samples] - 4

        buffer_features, buffer_labels, buffer_tasklabels = get_features(model, dataset, 'buffer')
        buffer_mask = 2 == buffer_tasklabels
        buffer_features, buffer_labels = buffer_features[buffer_mask], (buffer_labels[buffer_mask] - 4)
        permuted_indices = torch.randperm(buffer_labels.size(0))
        buffer_features, buffer_labels = buffer_features[permuted_indices][:50], buffer_labels[permuted_indices][:50]

        pca = PCA(n_components=2)
        reduced_features = pca.fit_transform(torch.cat([current_features, buffer_features], dim=0))

        # Separate train and buffer features
        train_reduced_features = reduced_features[:len(current_features)]
        buffer_reduced_features = reduced_features[len(current_features):]

        # Separate train and buffer features
        train_reduced_features = reduced_features[:len(current_features)]
        buffer_reduced_features = reduced_features[len(current_features):]

        # Plot
        # Train dataset (dots) - Use palette colors (removed c argument)
        axes[i].scatter(train_reduced_features[:, 0], train_reduced_features[:, 1], 
                        alpha=0.7, label="Train", marker='o', 
                        color=[palette[0] if label == 0 else palette[1] for label in current_labels])
        
        # Buffer dataset (triangles) - Use palette colors (removed c argument)
        axes[i].scatter(buffer_reduced_features[:, 0], buffer_reduced_features[:, 1], 
                        alpha=0.7, label="Buffer", marker='^', 
                        color=[palette[0] if label == 0 else palette[1] for label in buffer_labels])
        
        if i == 2:
            handles = [
            mlines.Line2D([], [], marker='o', color='w', markerfacecolor=palette[1], markersize=10, label='Class 4'),
            mlines.Line2D([], [], marker='^', color='w', markerfacecolor=palette[1], markersize=10, label='Buffer class 4'),
            mlines.Line2D([], [], marker='o', color='w', markerfacecolor=palette[0], markersize=10, label='Class 5'),
            mlines.Line2D([], [], marker='^', color='w', markerfacecolor=palette[0], markersize=10, label='Buffer class 5')
            ]

            # Place legend at the top-center of the plot
            axes[i].legend(handles=handles, loc='upper center', ncol=2)
            # Set axis labels
        
        axes[i].set_xlabel(f'PCA1')
        if i == 0:
            axes[i].set_ylabel(f'PCA2')
        
    fig.tight_layout()
    fig.savefig(mammoth_path + f"/PCA_task.pdf", dpi=800)  # Save the plot
    fig.clf()
        
    

if __name__ == '__main__':
    main()
