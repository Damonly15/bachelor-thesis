import os
import itertools
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='Prepare grid')
parser.add_argument('--begin_string', type=str, default="", help='Initial string common for all experiments')
parser.add_argument('--job_folder', type=str, default="data/jobs/", help='Folder to save the job list')
args = parser.parse_args()

grid_combinations = [
    {
        'name':'seq_cifar100_supcon_all',
        'combos': {
            "dataset": ['seq-cifar100'],
            "model": ["supcon"],
            "buffer_size": [100, 500], 
            "lr": [0.01],
            "alpha": [1.],
            "temperature": [0.5],
            #"beta": [2],
            "seed": [1000, 2000, 3000],
            "enable_other_metrics" : [1],
            "log_feature_forgetting": ["all"],
            "log_NC_metrics": [1],
            "training_setting": ["task-il","class-il"],
            "asym": [""],
            "n_epochs": [100],
            #"backbone": ["ResNet18_LN"],
            #"optimizer": ["adamw"],
            #"optim_wd": [0.00005],
            #"joint": [1],
            #"eval_epochs": [1],
            #"permute_classes": [1],
            #"portion": [0, 0.5, 0.7, 0.8, 0.9, 1.0]
            "savecheck": [True]
        },
    },
]

configs = []
all_configs = []

for experiment in grid_combinations:
    filenam, combos = experiment['name'], experiment['combos']
    configs = list(itertools.product(*combos.values()))

    print(filenam, len(configs), 'items')

    begin = args.begin_string
    folder = args.job_folder
    os.makedirs(folder, exist_ok=True, mode=0o777)

    clines = 0
    print(f'{folder}list_{filenam}.txt')
    with open(f'{folder}list_{filenam}.txt', 'w') as f:
        for c in configs:
            ll = begin
            for k, v in zip(combos.keys(), c):
                if v is None:
                    continue
                if type(k) == tuple:
                    for i in range(len(k)):
                        ll += f" --{k[i]}={v[i]}"
                else:
                    ll += f" --{k}={v}"
            f.write(ll+'\n')
            all_configs.append(ll)

            clines += 1

    print(f"Total ({filenam}):",clines)

print(f'{folder}list_all_grid.txt')
clines = 0
with open(f'{folder}list_all_grid.txt', 'w') as f:
    for ll in all_configs:
        f.write(ll + '\n')
        clines += 1

print("Total (all):",clines)
print('')
