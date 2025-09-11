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
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import BaggingClassifier
import pandas as pd
import re

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
from utils.feature_forgetting import get_features, evaluate_til, evaluate_cil
from utils.NC_metrics import evaluate_NC_metrics
from sklearn.base import BaseEstimator, ClassifierMixin
from scipy.spatial.distance import cdist

class NearestMeanTaskAwareClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, mean_features, task_variances=None, classes_per_task=1):
        """
        mean_features: (num_classes, feature_dim) numpy array of class mean features
        task_variances: (num_tasks,) numpy array containing variance for each task, or None
        classes_per_task: Number of classes per task (integer)
        """
        self.mean_features = mean_features
        self.class_to_task = numpy.arange(len(self.mean_features)) // classes_per_task
        
        # Compute the number of tasks based on total classes
        self.num_tasks = len(mean_features) // classes_per_task
        self.task_variances = task_variances if task_variances is not None else numpy.ones(self.num_tasks)

    def fit(self, X, y):
        """Dummy fit method to comply with sklearn API (not needed for fixed means)."""
        return self

    def predict(self, X):
        """Compute normalized distances and return the nearest class."""
        distances = cdist(X, self.mean_features, metric='euclidean')  # Compute distances
        
        # Normalize distances by task standard deviation
        task_std_devs = numpy.sqrt(self.task_variances[self.class_to_task])  # Get std per class
        normalized_distances = distances / task_std_devs  # Normalize distances

        return numpy.argmin(normalized_distances, axis=1)  # Return class index with min distance

    def predict_proba(self, X):
        """Return one-hot encoded probabilities where the closest class gets probability 1."""
        predictions = self.predict(X)  # Get closest class indices
        proba = numpy.zeros((X.shape[0], len(self.mean_features)))  # Initialize probability matrix
        proba[numpy.arange(X.shape[0]), predictions] = 1  # Assign probability 1 to closest class
        return proba

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

    load_model_pca(model, dataset, args)

def load_model_pca(model, dataset, args):
    start_from = -1
    end = 4
    classes = [1, 2, 11, 12, 21, 22, 31, 32]
    uninitialized_model = model

    dictionary = dict()

    for i in range(start_from, end):
        if i>=0:
            args.loadcheck = f'/cluster/scratch/dammeier/mammoth_checkpoints/{args.ckpt_name}_{i}.pt'
            model, past_res = mammoth_load_checkpoint(args, model)
            model._current_task = i
        else:
            model=uninitialized_model
            model.net.to(model.device)
        model.net.eval()


        if i>=0:
            model.NAME='er_nobuf'
            features, labels, tasklabels = get_features(model, dataset, 'buffer', end-1)
            for c in classes[:(i+1)*2]:
                mask = labels==c
                dictionary[f'task{i+1}_class{c}_buffer'] = (features[mask], torch.mean(features[mask], dim=0))
                
            features, labels, tasklabels = get_features(model, dataset, 'train_dataset', end-1)
            for c in classes[:(i+1)*2]:
                mask = labels==c
                dictionary[f'task{i+1}_class{c}_nobuffer'] = (features[mask], torch.mean(features[mask], dim=0))

            model.NAME='er'   
            features, labels, tasklabels = get_features(model, dataset, 'train_dataset', end-1)
            for c in classes[(i+1)*2:]:
                mask = labels==c
                dictionary[f'task{i+1}_class{c}_nobuffer'] = (features[mask], torch.mean(features[mask], dim=0)) 
        else:
            model.NAME='er'
            features, labels, tasklabels = get_features(model, dataset, 'train_dataset', end-1)
            for c in classes:
                mask = labels==c
                dictionary[f'task{i+1}_class{c}_nobuffer'] = (features[mask], torch.mean(features[mask], dim=0))
           
    all_features = []

    for key, value in dictionary.items():
        print(key)
        all_features.append(value[0])
    
    pca = PCA(n_components=2)
    _ = pca.fit_transform(torch.cat(all_features, dim=0))

    results = pd.DataFrame(columns=["Version", "Training Task", "Class", "Feature Mean"])
    for i, (key, value) in enumerate(dictionary.items()):
        match = re.match(r"task(\d+)_class(\d+)_(nobuffer|buffer)$", key)
        task = int(match.group(1))  # subtract 1 to get the original i
        clas = int(match.group(2))
        version = match.group(3)

        results.loc[i] = [version, task, clas, pca.transform(value[1].unsqueeze(0))]
    

    fig, axes = plt.subplots(1, 8, figsize=(24, 3), sharex=True, sharey=True)

    color_palette = sns.color_palette("tab10")
    marker_styles = ['o', 'D'] 

    task_to_color = {task: color_palette[i % len(color_palette)] for i, task in enumerate(range(0, end+1))}
    version_to_marker = {version: marker_styles[i % len(marker_styles)] for i, version in enumerate(["buffer", "nobuffer"])}


    for idx, c in enumerate(classes):
        ax = axes[idx]
        subset = results[results["Class"] == c]

        for _, row in subset.iterrows():
            features = row["Feature Mean"]
            color = task_to_color[row["Training Task"]]
            marker = version_to_marker[row["Version"]]
            
            ax.scatter(features[:, 0], features[:, 1], color=color, marker=marker, label=f'{row["Version"]}-{row["Training Task"]}', alpha=0.7)

        ax.set_title(f'Class {c}')
        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')

        # Avoid duplicate labels in legend
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc='best', fontsize='small')

    fig.tight_layout()
    fig.savefig(mammoth_path + f"/PCA_new.pdf", dpi=800)  # Save the plot
    fig.clf()

def load_model_NN(model, dataset, args):
    start_from = dataset.N_TASKS - 1
    total_samples = args.buffer_size
    for i in range(dataset.N_TASKS):
        dataset.get_data_loaders()
    args.loadcheck = f'/cluster/scratch/dammeier/mammoth_checkpoints/{args.ckpt_name}_{start_from}.pt'
    model, past_res = mammoth_load_checkpoint(args, model)
    model.net.eval()
    model._current_task = start_from
    
    (within_var, between_var), mean_features = evaluate_NC_metrics(model, dataset, 'buffer')
    buffer_features, buffer_labels, buffer_tasklabels = get_features(model, dataset, 'buffer')
    print(buffer_features.shape[0])

    nn_predictor = NearestMeanTaskAwareClassifier(mean_features=mean_features.numpy(), task_variances=None, classes_per_task=dataset.N_CLASSES_PER_TASK)

    results = evaluate_cil(model, dataset, nn_predictor)

    print(results)
    print(sum(results) / len(results))

def load_model_plot(model, dataset, args):
    C = 1
    TASK = 4
    palette = sns.color_palette("cool", C+1)
    # palette = {0: '#225ea8', 1: '#cb181d', 2: 'darkorange'}
    # Set the font and style for the plot
    plt.rcParams.update({
            'font.family': 'serif',
            'font.serif': ['Times New Roman'],
            'font.size': 10,
            'axes.titlesize': 12,
            'axes.labelsize': 10,
            'xtick.labelsize': 8,
            'ytick.labelsize': 8,
            'legend.fontsize': 8,
            'figure.titlesize': 12
    })

    sns.set_theme(style="whitegrid", context="paper")
    fig, axes = plt.subplots(1, dataset.N_TASKS, figsize=(dataset.N_TASKS*2, 2), dpi=800, sharey=True, sharex=True)
    for i in range(0, dataset.N_TASKS):
        dataset.get_data_loaders()
        args.loadcheck = f'/cluster/scratch/dammeier/mammoth_checkpoints/{args.ckpt_name}_{i}.pt'
        model, past_res = mammoth_load_checkpoint(args, model)
        model.net.eval()
        model._current_task = i

        all_features, all_labels, all_tasklabels = get_features(model, dataset, 'train_dataset', TASK)
        mask = (all_tasklabels == TASK) & ((all_labels == 40) | (all_labels == 41))
        all_features, all_labels = all_features[mask], all_labels[mask]
        permutation = torch.randperm(all_features.shape[0])
        all_features, all_labels = all_features[permutation][:30], all_labels[permutation][:30]

        buffer_features, buffer_labels, buffer_tasklabels = get_features(model, dataset, 'buffer', TASK)
        mask = (buffer_tasklabels == TASK) & ((buffer_labels == 40) | (buffer_labels == 41))
        buffer_features, buffer_labels = buffer_features[mask], buffer_labels[mask]
        permutation = torch.randperm(buffer_features.shape[0])
        buffer_features, buffer_labels = buffer_features[permutation][:10], buffer_labels[permutation][:10]

        current_features_selected = all_features
        current_labels_selected = all_labels
        buffer_features_selected = buffer_features
        buffer_labels_selected = buffer_labels

        pca = PCA(n_components=2)
        reduced_features = pca.fit_transform(torch.cat([current_features_selected, buffer_features_selected], dim=0))
        # Separate train and buffer features
        train_reduced_features = reduced_features[:len(current_features_selected)]
        buffer_reduced_features = reduced_features[len(current_features_selected):]
        # Separate train and buffer features
        train_reduced_features = reduced_features[:len(current_features_selected)]
        buffer_reduced_features = reduced_features[len(current_features_selected):]

        
        # Plot
        # Train dataset (dots) - Use palette colors (removed c argument)
        axes[i].scatter(train_reduced_features[:, 0], train_reduced_features[:, 1],
                        label="Train", marker='o', edgecolor='white', s=50,
                        color=[palette[label.item()-40] for label in current_labels_selected])
        # Buffer dataset (triangles) - Use palette colors (removed c argument)


        axes[i].scatter(buffer_reduced_features[:, 0], buffer_reduced_features[:, 1],
                        label="Buffer", marker='o', edgecolor='black', s=50, linewidths=1.,
                        color=[palette[label.item()-40] for label in buffer_labels_selected])
        # if i == 0:
        #     # handles = [
        #     # mlines.Line2D([], [], marker='o', color='w', markerfacecolor=palette['DB'], markersize=10, label='4'),
        #     # mlines.Line2D([], [], marker='^', color='w', markerfacecolor=palette['LB'], markersize=10, label='Buffer 4'),
        #     # mlines.Line2D([], [], marker='o', color='w', markerfacecolor=palette['DR'], markersize=10, label='5'),
        #     # mlines.Line2D([], [], marker='^', color='w', markerfacecolor=palette['LR'], markersize=10, label='Buffer 5')
        #     # ]
        #     # Place legend at the top-center of the plot

        #     axes[i].legend(handles=handles, loc='upper center', ncol=2)
            # Set axis labels
        
        # Add dotted grid lines
        axes[i].set_title(f'Task {i+1}')
        axes[i].grid(True, linestyle='--', linewidth=1.0, axis='y', zorder=2)

        # Remove the boxes around each plot
        axes[i].spines['top'].set_visible(False)
        axes[i].spines['right'].set_visible(False)
        axes[i].spines['left'].set_visible(True)
        axes[i].spines['bottom'].set_visible(False)
        axes[i].set_xlabel(f'PC1')
        if i == 0:
            axes[i].set_ylabel(f'PC2')
    fig.tight_layout()
    fig.savefig(mammoth_path + f"/PCA_task.pdf", dpi=800)  # Save the plot
    fig.clf()

def load_model3(model, dataset, args):
    start_from = 9
    total_samples = 20
    for i in range(start_from+1):
        dataset.get_data_loaders()
    args.loadcheck = f'/cluster/scratch/dammeier/mammoth_checkpoints/{args.ckpt_name}_{start_from}.pt'
    model, past_res = mammoth_load_checkpoint(args, model)
    model.net.eval()
    model._current_task = start_from
    
    (within_var, between_var), _ = evaluate_NC_metrics(model, dataset, 'buffer')

    buffer_features, buffer_labels, buffer_tasklabels = get_features(model, dataset, 'buffer')
    print(buffer_features.shape[0])

    rations = [1.0, 0.75, 0.5, 0.25, 0.1, 0.05, 0.03, 0.01]
    for i in rations:
        all_features = []
        all_labels = []
        heads = []

        for task in range(dataset.N_TASKS):
            variance = within_var[task] * i
            task_mask = buffer_tasklabels == task
            current_features, current_labels = buffer_features[task_mask], buffer_labels[task_mask]
            N, D = current_features.shape
            
            noise = torch.randn(N, total_samples, D) * variance**0.5 
            sampled_features = current_features.unsqueeze(1) + noise  # Shape: (N, num_samples, D)
            sampled_features = sampled_features.view(N * total_samples, D)
            sampled_labels = current_labels.unsqueeze(1).expand(N, total_samples).reshape(-1)  # Shape: (N * num_samples,)

            if args.training_setting == 'task-il': 
                logreg_model = LogisticRegression(max_iter=5000, C=10)
                logreg_model.fit(sampled_features.numpy(), sampled_labels.numpy())
                heads.append(logreg_model)
            else:
                all_features.append(sampled_features)
                all_labels.append(sampled_labels)
        
        # Concatenate original features and labels
        if args.training_setting == 'task-il':
            results = evaluate_til(model, dataset, heads)
        else:
            all_features = torch.cat(all_features + [buffer_features], dim=0)  # Shape: ((N * (num_samples+1)), D)
            all_labels = torch.cat(all_labels + [buffer_labels], dim=0)  # Shape: ((N * (num_samples+1)),)

            logreg_model = LogisticRegression(max_iter=5000, C=1)
            logreg_model.fit(all_features.numpy(), all_labels.numpy())

            results = evaluate_cil(model, dataset, logreg_model)

        print(results)
        print(sum(results) / len(results))

def load_model2(model, dataset, args):
    start_from = 9
    for i in range(start_from+1):
        dataset.get_data_loaders()
    args.loadcheck = f'/cluster/scratch/dammeier/mammoth_checkpoints/{args.ckpt_name}_{start_from}.pt'
    model, past_res = mammoth_load_checkpoint(args, model)
    model.net.eval()
    model._current_task = start_from

    #normal buffer
    buf_x, buffer_labels, buffer_tasklabels = model.buffer.get_all_data(transform=model.transform)
    buffer_features = []
    for i in range(0, buf_x.shape[0], model.args.batch_size):
        inputs = buf_x[i: i+model.args.batch_size]
        inputs = inputs.to(model.device)
        features = model.net.forward(inputs, returnt="features").detach().cpu()
        buffer_features.append(features)
    buffer_features = torch.cat(buffer_features, dim=0)

    #upsample hold out samples
    hold_x, hold_labels, hold_tasklabels = [], [], []
    for i in range(4):
        c_x, c_labels, c_tasklabels = model.extra_buffer.get_all_data(transform=model.transform)
        hold_x.append(c_x)
        hold_labels.append(c_labels)
        hold_tasklabels.append(c_tasklabels)
    hold_x, hold_labels, hold_tasklabels = torch.cat(hold_x,dim=0), torch.cat(hold_labels,dim=0), torch.cat(hold_tasklabels, dim=0)

    hold_features = []
    for i in range(0, hold_x.shape[0], model.args.batch_size):
        inputs = hold_x[i: i+model.args.batch_size]
        inputs = inputs.to(model.device)
        features = model.net.forward(inputs, returnt="features").detach().cpu()
        hold_features.append(features)
    hold_features = torch.cat(hold_features, dim=0)

    all_features, all_labels, all_tasklabels = torch.cat([buffer_features, hold_features], dim=0), torch.cat([buffer_labels, hold_labels], dim=0), torch.cat([buffer_tasklabels, hold_labels],dim=0)
    if args.training_setting == 'task-il':
        heads = []
        for i in range(dataset.N_TASKS):
            if i+1==dataset.N_TASKS:
                task_mask = i == all_tasklabels
                current_features, current_labels = all_features[task_mask], current_labels[task_mask]

            bagging_clf = BaggingClassifier(
                estimator=LogisticRegression(max_iter=5000, C=10),  # Logistic Regression as base estimator
                n_estimators=10,  # Number of base models
                max_samples=0.8,  # Each model sees 80% of the data
                bootstrap=True,   # Sample with replacement
                n_jobs=-1,        # Use all available CPU cores for parallel training
                random_state=args.seed)

            bagging_clf.fit(all_features, all_labels, dim=0)
            heads.append(bagging_clf)   
    else:
        bagging_clf = BaggingClassifier(
            estimator=LogisticRegression(max_iter=5000, C=1),  # Logistic Regression as base estimator
            n_estimators=10,  # Number of base models
            max_samples=0.8,  # Each model sees 80% of the data
            bootstrap=True,   # Sample with replacement
            n_jobs=-1,        # Use all available CPU cores for parallel training
            random_state=args.seed)

        bagging_clf.fit(all_features, all_labels)
        results = evaluate_cil(model, dataset, bagging_clf)
        print(sum(results) / len(results))
        print(results)


def load_model(model, dataset, args):
    start_from = dataset.N_TASKS - 1
    total_samples = args.buffer_size
    for i in range(dataset.N_TASKS):
        dataset.get_data_loaders()
    args.loadcheck = f'/cluster/scratch/dammeier/mammoth_checkpoints/{args.ckpt_name}_{start_from}.pt'
    model, past_res = mammoth_load_checkpoint(args, model)
    model.net.eval()
    model._current_task = start_from
    
    (within_var, between_var), mean_features = evaluate_NC_metrics(model, dataset, 'train_dataset')

    buffer_features, buffer_labels, buffer_tasklabels = get_features(model, dataset, 'buffer')

    #mean_features, mean_labels, all_tasklabels = get_features(model, dataset, 'train_dataset')
    #permuted_indices = torch.randperm(mean_labels.size(0))
    #mean_features, mean_labels = mean_features[permuted_indices][:1000], mean_labels[permuted_indices][:1000]
    #unique_labels = mean_labels.unique(sorted=True)
    #mean_features = torch.stack([mean_features[mean_labels == label].mean(dim=0) for label in unique_labels])
    #mean_labels = unique_labels
    mean_labels = buffer_labels.unique(sorted=True)


    for i in range(1, 21, 2):
        current_samples = total_samples // dataset.N_CLASSES
        all_features = []
        all_labels = []
        heads = []

        for task in range(dataset.N_TASKS):
            variance = within_var[task] / i
            N, D = mean_features[task*dataset.N_CLASSES_PER_TASK:(task+1)*dataset.N_CLASSES_PER_TASK].shape
            
            noise = torch.randn(N, current_samples, D) * variance**0.5 
            sampled_features = mean_features[task*dataset.N_CLASSES_PER_TASK:(task+1)*dataset.N_CLASSES_PER_TASK].unsqueeze(1) + noise  # Shape: (N, num_samples, D)
            sampled_features = sampled_features.view(N * current_samples, D)
            sampled_labels = mean_labels[task*dataset.N_CLASSES_PER_TASK:(task+1)*dataset.N_CLASSES_PER_TASK].unsqueeze(1).expand(N, current_samples).reshape(-1)  # Shape: (N * num_samples,)

            if args.training_setting == 'task-il':
                buffer_mask = task == buffer_tasklabels
                current_features = torch.cat([sampled_features, buffer_features[buffer_mask]], dim=0)
                current_labels = torch.cat([sampled_labels, buffer_labels[buffer_mask]], dim=0) 
                logreg_model = LogisticRegression(max_iter=5000, C=10)
                logreg_model.fit(current_features.numpy(), current_labels.numpy())
                heads.append(logreg_model)
            else:
                all_features.append(sampled_features)
                all_labels.append(sampled_labels)
        
        # Concatenate original features and labels
        if args.training_setting == 'task-il':
            results = evaluate_til(model, dataset, heads)
        else:
            all_features = torch.cat(all_features, dim=0)  # Shape: ((N * (num_samples+1)), D)
            all_labels = torch.cat(all_labels, dim=0)  # Shape: ((N * (num_samples+1)),)

            logreg_model = LogisticRegression(max_iter=5000, C=10)
            logreg_model.fit(all_features.numpy(), all_labels.numpy())

            results = evaluate_cil(model, dataset, logreg_model)

        print(results)
        print(sum(results) / len(results))
        
    

if __name__ == '__main__':
    main()
