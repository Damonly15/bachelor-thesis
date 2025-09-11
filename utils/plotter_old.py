import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import ast
import re
import itertools
from argparse import ArgumentParser
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import numpy as np
import matplotlib.ticker as ticker

from conf import base_path

path = base_path() + "results_version1/"
palette = sns.color_palette("deep")

def preprocess_line(line):
       # Remove the 'device' entry
        line = re.sub(r"'device': device\(.*?\),?", "", line)
        line = re.sub(r"array\((\[.*?\])\)", r"\1", line)
        return line

def get_data(path, scenario, dataset, model, logs="/logs.txt"):
        filepath = path + scenario + "/" + dataset + "/" + model
        with open(filepath + logs, "r") as file:
                dictlist = list()
                for line in file:
                        line = preprocess_line(line)
                        dictlist.append(ast.literal_eval(line))
                return dictlist, filepath

def get_arguments(dictionary):
        base = ["dataset", "model"]
        keep = ["dataset", "model"]
        combine = []
        if "buffer_size" in dictionary:
                base.append("buffer_size")
                keep.append("buffer_size")
        if "lr" in dictionary:
                base.append("lr")
                combine.append("lr")
        if "alpha" in dictionary:
                base.append("alpha")
                combine.append("alpha")
        if "beta" in dictionary:
                base.append("beta")
                combine.appen("beta")
        if "temperature" in dictionary:
                base.append("temperature")
                combine.append("temperature")
        if "result_type" in dictionary:
                base.append("result_type")
                keep.append("result_type")
        if "portion" in dictionary:
                base.append("portion")
                keep.append("portion")
        
        return base, keep, combine

def get_dataframe(dictlist, keep_arguments):
        df = pd.DataFrame(dictlist)
        df = df.drop(columns=[col for col in df.columns if col not in (keep_arguments)])

        #if the temperature is above 100, we used the mse approximation
        if "temperature" in keep_arguments:
                index = df['temperature'] > 100
                df["temperature"] = df["temperature"].astype("str")
                df.loc[index, 'temperature'] = 'mse'
        return df

def get_arrays(dictionary, pre_string1='accuracy', pre_string2='accmean_task'):
        dataset = dictionary['dataset']
        accmean_columns = list()
        acctask_columns = list()

        amount_task = 0
        if(any(substring in dataset for substring in ["seq-tinyimg", "seq-cifar100", 'seq-cub200'])):
                amount_task = 10
        elif(any(substring in dataset for substring in ["rot-mnist", "perm-mnist"])):
                amount_task = 20
        elif(any(substring in dataset for substring in ["seq-mnist", "seq-cifar10"])):
                amount_task = 5
        for i in range(1, amount_task+1):
                accmean_columns.append(pre_string2 + f"{i}")
                for j in range(1, amount_task+1):
                        acctask_columns.append(pre_string1 + f"_{j}_task{i}")
        result_accuracy = accmean_columns[-1]

        return accmean_columns, acctask_columns, result_accuracy

def plotting_name(logging_name):
        if logging_name == 'er_bounds':
                return 'ER'
        elif logging_name == 'er_buf':
                return 'ER'
        elif logging_name == 'sgd':
                return 'JOINT'
        elif logging_name == 'fdr':
                return 'FDR'
        elif logging_name == 'der':
                return 'DER'
        elif logging_name == 'icarl':
                return 'iCaRL'
        elif logging_name == 'er_bic':
                return 'ER-BiC'
        elif logging_name == 'er_wa':
                return 'ER-WA'
        elif logging_name == 'seq-cifar10':
                return 'Cifar10'
        elif logging_name == 'seq-cifar100' or logging_name == 'seq-cifar100-224':
                return 'Cifar100'
        elif logging_name == 'seq-tinyimg' or logging_name == 'seq-tinyimg-224':
                return 'TinyIMG'
        elif logging_name == 'seq-cub200':
                return 'CUB200'
        elif logging_name == 'class-il':
                return 'single-head'
        elif logging_name == 'task-il':
                return 'multi-head'
        else:
                return logging_name

def replace(dataframe, key, new_value):
        dataframe[key] = new_value

def group_df(df, filter, keep_arguments, group_arguments, result_accuracy):
        #First filter
        grouped = df
        if filter is not None:
                for key, value in filter.items():
                        grouped = grouped[grouped[key] == value]
        #Group according to all arguments
        grouped = grouped.groupby(keep_arguments + group_arguments, as_index=False).agg({result_accuracy: 'mean'})
        #print(grouped)
        #Calculate max of mean for arguments we should keep for plotting
        best = grouped.groupby(keep_arguments, as_index=False).agg({result_accuracy: 'max'})
        best = grouped.merge(best[result_accuracy], on=[result_accuracy], how='inner')
        #print(best)
        #Filter dataframe according to best arguments
        best = best.drop(result_accuracy, axis=1)
        result = df.merge(best, on=(keep_arguments + group_arguments), how='inner')
        #print(result)
        return result

# Group the dataframe by 'seed' and filter based on result_type
def subtract_buffer_train(group):
    if len(group) == 2:  # We expect two rows for each seed (one for buffer, one for train)
        buffer_row = group[group['result_type'] == 'buffer']
        train_row = group[group['result_type'] == 'test_dataset']

        # Ensure there is exactly one buffer and one train row
        if not buffer_row.empty and not train_row.empty:
            # Subtract values in all other columns except 'seed' and 'result_type'
            cols_to_subtract = [col for col in group.columns if col not in ['seed', 'result_type']]

            for col in cols_to_subtract:
                # Perform the subtraction
                group[col] = train_row[col].values - buffer_row[col].values

    return group

def figure2():

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
        # Create the dictionary with the updated assignments
        colors = sns.color_palette("viridis", 6)  # Requesting 6 colors to access cyan
        color_dict = {
        "ER": colors[1],  # blue
        "ER-OR": "black",  # cyan
        "DER": colors[2],  # green
        "FDR": colors[3],  # violet
        "iCaRL": colors[4],  # orange
        "JOINT": "#A0A0A0",  # black
        "RESET": "#A0A0A0",
        "RANDOM": "#A0A0A0",
        }

        datasets = ['seq-cifar100', 'seq-tinyimg']
        training_setting = ['task-il','class-il']
        models = ['joint', 'reset', 'random', 'er_buf', 'der', 'fdr', 'icarl']

        fig, axes = plt.subplots(1, len(datasets) * len(training_setting), figsize=(8, 3), dpi=800)
        for i, setting in enumerate(list(itertools.product(datasets, training_setting))):
                data_buffer = []
                data_nobuffer = []

                if setting[0] == 'seq-cifar10':
                        infinite_size = 2500
                elif setting[0] == 'seq-cifar100':
                        infinite_size = 6000
                elif setting[0] == 'seq-tinyimg':
                        infinite_size = 12000

                for model in models:
                        dictlist, filepath = get_data(path, setting[1], setting[0], model)
                        arguments_base, arguments_keep, arguments_combine = get_arguments(dictlist[0])
                        accmean_columns, acctask_columns, result_accuracy = get_arrays(dictlist[0])

                        df = get_dataframe(dictlist, arguments_base + [result_accuracy])
                        if setting[1] == "class-il":
                                features_version = "features_cil"
                        else:
                                features_version = "features_til"
                        if model in ['er_buf']:
                                df = df[df['result_type'].isin(list(['output', features_version]))]
                                df.loc[df['result_type'] == features_version, 'model'] = 'ER-OR'
                                buffer_sizes = df['buffer_size'].sort_values().tolist()
                                df["buffer_size"] = df["buffer_size"].replace(buffer_sizes[-1], infinite_size)
                        elif model in ['der', 'fdr']:
                                if model == 'der' and setting[1] == 'task-il':
                                        df[result_accuracy]= df[result_accuracy] - 0.5
                                df = df[df['result_type'].isin(list(['output']))]
                                buffer_sizes = df['buffer_size'].sort_values().tolist()
                                df["buffer_size"] = df["buffer_size"].replace(buffer_sizes[-1], infinite_size)
                                df = df[df['buffer_size'] > 0]
                        else:
                                if model == 'icarl':
                                        df = df[df['result_type'].isin(list(['output']))]
                                if model == 'joint':
                                        df['model'] = 'JOINT'
                                if model == 'random':
                                        df = df[df['result_type'].isin(list([features_version]))]
                                        df['model'] = 'RANDOM'
                                if model == 'reset':
                                        df = df[df['result_type'].isin(list([features_version]))]
                                        df['model'] = 'RESET'

                        df['model'] = df.apply(lambda row: plotting_name(row['model']), axis=1)
                        if 'buffer_size' in arguments_base:
                                df = df[df['buffer_size'] < 50000]
                                data_buffer.append(df)
                        else:
                                data_nobuffer.append(df)   
                
                data_buffer = pd.concat(data_buffer)
                data_nobuffer = pd.concat(data_nobuffer)

                all_buffer_sizes = data_buffer['buffer_size'].unique()
                df_buffer_sizes = pd.DataFrame({'buffer_size': all_buffer_sizes})
                data_nobuffer = data_nobuffer.merge(df_buffer_sizes, how='cross')

                data_buffer_OR = data_buffer[data_buffer['model'] == 'ER-OR']
                data_buffer_rest = data_buffer[data_buffer['model'] != 'ER-OR']

                sns.lineplot(data=data_nobuffer[data_nobuffer['model'] == 'JOINT'], x='buffer_size', y=result_accuracy, hue='model', marker='', legend=False, palette=color_dict, ax=axes[i], linewidth=1.0, errorbar=None)
                sns.lineplot(data=data_nobuffer[data_nobuffer['model'] == 'RESET'], x='buffer_size', y=result_accuracy, hue='model', marker='', linestyle='-.', legend=False, palette=color_dict, ax=axes[i], linewidth=1.0, errorbar=None)
                sns.lineplot(data=data_nobuffer[data_nobuffer['model'] == 'RANDOM'], x='buffer_size', y=result_accuracy, hue='model', marker='', linestyle='--', legend=False, palette=color_dict, ax=axes[i], linewidth=1.0, errorbar=None)
                sns.lineplot(data=data_buffer_rest, x='buffer_size', y=result_accuracy, hue='model',  marker='o',  markersize=5, linestyle='--', legend=False, palette=color_dict, ax=axes[i])
                sns.lineplot(data=data_buffer_OR, x='buffer_size', y=result_accuracy, hue='model',  marker='D',  markersize=5, linestyle=':', legend=False, palette=color_dict, ax=axes[i])

                # Remove the boxes around each plot
                axes[i].spines['top'].set_visible(False)
                axes[i].spines['right'].set_visible(False)
                axes[i].spines['left'].set_visible(False)
                axes[i].spines['bottom'].set_visible(False)


                #y_min = min(data_buffer[result_accuracy].min(), data_nobuffer[result_accuracy].min(), data_buffer_OR[result_accuracy].min())
                #y_max = max(data_buffer[result_accuracy].max(), data_nobuffer[result_accuracy].max(), data_buffer_OR[result_accuracy].max())
                #axes[i].set_ylim(y_min - 5, y_max)

                #current_yticks = axes[i].get_yticks()
                #y_min, y_max = axes[i].get_ylim()  # Store exact limits
                #axes[i].set_yticks(current_yticks)  # Freeze ticks
                #axes[i].set_ylim((y_min, y_max))  # Reset exact limits

                if setting[0] == 'seq-cifar10':
                        xticks = [0, 200, 500, 1000, 1500, 2000, 2500]
                elif setting[0] == 'seq-cifar100':
                        xticks = [0, 1000, 2000, 3000, 4000, 5000, 6000]
                        dataset_size = 50000
                elif setting[0] == 'seq-tinyimg':
                        xticks = [0, 2000, 4000, 6000, 8000, 10000, 12000]
                        dataset_size = 100000
                
                axes[i].set_xticks(xticks)  # Ensure tick positions remain the same
                xticks = [(100*tick / dataset_size) for tick in xticks]
                xticklabels = [str(int(tick)) if tick != max(xticks) else '100' for tick in xticks]  # Replace max tick with 'inf'
                axes[i].set_xticklabels(xticklabels)  # Set the new labels

                #Make the fontsize of the axis smaller
                for tick in axes[i].get_xticklabels():
                        tick.set_rotation(45)
                axes[i].tick_params(axis='both', which='major', labelsize=8)
                
                
                # Customize the grid  
                axes[i].grid(visible=True, which='major', axis='x', linestyle=':', linewidth=0.5, color='lightgrey')  # Dashed grid for x-axis
                axes[i].grid(visible=True, which='major', axis='y', linestyle='--', linewidth=0.5, color='lightgrey')   

                axes[i].set_title(f'{plotting_name(setting[0])} ({plotting_name(setting[1])})', fontsize=10)
                axes[i].set_xlabel("Buffer Size [%]")

                if i==0:
                        axes[i].set_ylabel("Test Accuracy [%]", fontsize=8)
                else:
                        axes[i].set_ylabel("")
                
        legend_handles = []
        for label, color in color_dict.items():
                if label == "JOINT":
                        legend_handles.append(mlines.Line2D([0], [0], color=color, label=label))
                elif label == "RANDOM":
                        legend_handles.append(mlines.Line2D([0], [0], color=color, label=label, linestyle='--'))
                elif label == "RESET":
                        legend_handles.append(mlines.Line2D([0], [0], color=color, label=label, linestyle='-.'))
                elif label == "ER-OR":
                        legend_handles.append(mlines.Line2D([0], [0], color=color, marker='D', linestyle=':', markersize=4, label=label))
                else:
                        legend_handles.append(mlines.Line2D([0], [0], color=color, marker='o', markersize=4, label=label))
        labels = color_dict.keys()
        # Move ER-OR to the first position in the legend
        legend_handles = [legend_handles[1]] + legend_handles[:1] + legend_handles[2:]
        labels = ['ER-OR'] + [label for label in labels if label != 'ER-OR']
        fig.legend(legend_handles, labels, loc='center', bbox_to_anchor=(0.5, 0.05), ncol=8, title_fontsize=8, fontsize=8)
        #fig.legend(legend_handles, labels, loc='center', bbox_to_anchor=(1.05, 0.5), ncol=1, title_fontsize=8, fontsize=8)
        fig.tight_layout(rect=[0, 0.05, 1, 1])
        fig.savefig(path + f"/figure2.pdf", dpi=800)
        fig.clf() 

def figure2_extra():
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
        # Create the dictionary with the updated assignments
        colors = sns.color_palette("viridis", 6)  # Requesting 6 colors to access cyan
        #colors = sns.color_palette("mako", 6)
        color_dict = {
                #"ER": colors[1],  # blue
                #"DER": colors[2],  # green
                #"FDR": colors[3],  # violet
                "iCaRL": colors[4],  # orange
                "ER-OR": "black",  # cyan
                #"ER-BUF": colors[3],  # orange
                #"ER-WA": colors[4],
                #"ER-BiC": colors[5],
                #"JOINT": "silver"  # black
        }
        datasets = ['seq-cifar100', 'seq-tinyimg']
        training_setting = ['task-il', 'class-il']
        models = ['icarl']

        fig, axes = plt.subplots(1, len(datasets) * len(training_setting), figsize=(8, 3), dpi=800)
        for i, setting in enumerate(list(itertools.product(datasets, training_setting))):
                data_buffer = []
                data_nobuffer = []

                if setting[0] == 'seq-cifar10':
                        infinite_size = 2500
                elif setting[0] == 'seq-cifar100':
                        infinite_size = 6000
                elif setting[0] == 'seq-tinyimg':
                        infinite_size = 12000
                elif setting[0] == 'seq-cub200':
                        infinite_size = 2500

                for model in models:
                        dictlist, filepath = get_data(path, setting[1], setting[0], model)
                        arguments_base, arguments_keep, arguments_combine = get_arguments(dictlist[0])
                        accmean_columns, acctask_columns, result_accuracy = get_arrays(dictlist[0])

                        df = get_dataframe(dictlist, arguments_base + [result_accuracy])
                        if setting[1] == "class-il":
                                features_version = "features_cil"
                        else:
                                features_version = "features_til"
                        if model in ['er_buf']:
                                df = df[df['result_type'].isin(list(['output', features_version, 'buffer']))]
                                df.loc[df['result_type'] == features_version, 'model'] = 'ER-OR'
                                df.loc[df['result_type'] == 'buffer', 'model'] = 'ER-BUF'
                                buffer_sizes = df['buffer_size'].sort_values().tolist()

                                df = df[df['buffer_size'] < infinite_size]
                                df = df[df['buffer_size'] > 0]
                                #df["buffer_size"] = df["buffer_size"].replace(buffer_sizes[-1], infinite_size)
                        elif model in ['der', 'fdr']:
                                df = df[df['result_type'].isin(list(['output', features_version]))]
                                df.loc[df['result_type'] == features_version, 'model'] = 'ER-OR'

                                #df = df[df['result_type'].isin(list(['output']))]
                                buffer_sizes = df['buffer_size'].sort_values().tolist()
                                df["buffer_size"] = df["buffer_size"].replace(buffer_sizes[-1], infinite_size)
                                df = df[df['buffer_size'] > 0]
                        else:
                                df = df[df['result_type'].isin(list(['output', features_version]))]
                                df.loc[df['result_type'] == features_version, 'model'] = 'ER-OR'
                                
                                #df = df[df['result_type'].isin(list(['output']))]

                        df['model'] = df.apply(lambda row: plotting_name(row['model']), axis=1)
                        if 'buffer_size' in arguments_base:
                                df = df[df['buffer_size'] < 50000]
                                data_buffer.append(df)
                        else:
                                data_nobuffer.append(df)   
                
                data_buffer = pd.concat(data_buffer)
                #data_nobuffer = pd.concat(data_nobuffer)

                #all_buffer_sizes = data_buffer['buffer_size'].unique()
                #df_buffer_sizes = pd.DataFrame({'buffer_size': all_buffer_sizes})
                #data_nobuffer = data_nobuffer.merge(df_buffer_sizes, how='cross')

                data_buffer_OR = data_buffer[data_buffer['model'] == 'ER-OR' ]
                data_buffer_rest = data_buffer[data_buffer['model'] != 'ER-OR']

                sns.lineplot(data=data_buffer_rest, x='buffer_size', y=result_accuracy, hue='model',  marker='o',  markersize=5, linestyle='--', legend=False, palette=color_dict, ax=axes[i])
                #sns.lineplot(data=data_nobuffer, x='buffer_size', y=result_accuracy, hue='model', marker='',  linestyle='-', legend=False, palette=color_dict, ax=axes[i], linewidth=0.5)
                sns.lineplot(data=data_buffer_OR, x='buffer_size', y=result_accuracy, hue='model',  marker='D',  markersize=5, linestyle=':', legend=False, palette=color_dict, ax=axes[i])

                # Remove the boxes around each plot
                axes[i].spines['top'].set_visible(False)
                axes[i].spines['right'].set_visible(False)
                axes[i].spines['left'].set_visible(False)
                axes[i].spines['bottom'].set_visible(False)


                #y_min = min(data_buffer[result_accuracy].min(), data_nobuffer[result_accuracy].min(), data_buffer_OR[result_accuracy].min())
                #y_max = max(data_buffer[result_accuracy].max(), data_nobuffer[result_accuracy].max(), data_buffer_OR[result_accuracy].max())
                #axes[i].set_ylim(y_min - 5, y_max)

                #current_yticks = axes[i].get_yticks()
                #y_min, y_max = axes[i].get_ylim()  # Store exact limits
                #axes[i].set_yticks(current_yticks)  # Freeze ticks
                #axes[i].set_ylim((y_min, y_max))  # Reset exact limits

                if setting[0] == 'seq-cifar10':
                        xticks = [0, 200, 500, 1000, 1500, 2000]#, 2500]
                elif setting[0] == 'seq-cifar100':
                        xticks = [0, 1000, 2000, 3000, 4000, 5000]#, 6000]
                        dataset_size = 50000
                elif setting[0] == 'seq-tinyimg':
                        xticks = [0, 2000, 4000, 6000, 8000, 10000]#, 12000]
                        dataset_size = 100000
                elif setting[0] == 'seq-cub200':
                        xticks = [0, 400, 800, 1200, 1600, 2000]#, 2500]
                        dataset_size = 12000
                
                axes[i].set_xticks(xticks)  # Ensure tick positions remain the same
                if setting[0] != 'seq-cub200':
                        xticklabels = [str(int(100*tick / dataset_size)) if tick != max(xticks) else '10' for tick in xticks]
                else:
                        xticklabels = [str(round((100*tick / dataset_size), 1)) for tick in xticks]
                  # Replace max tick with 'inf'
                axes[i].set_xticklabels(xticklabels)  # Set the new labels

                #Make the fontsize of the axis smaller
                for tick in axes[i].get_xticklabels():
                        tick.set_rotation(45)
                axes[i].tick_params(axis='both', which='major', labelsize=8)
                
                
                # Customize the grid  
                axes[i].grid(visible=True, which='major', axis='x', linestyle=':', linewidth=0.5, color='lightgrey')  # Dashed grid for x-axis
                axes[i].grid(visible=True, which='major', axis='y', linestyle='--', linewidth=0.5, color='lightgrey')   

                axes[i].set_title(f'{plotting_name(setting[0])} ({plotting_name(setting[1])})', fontsize=10)
                axes[i].set_xlabel("Buffer Size [%]")

                if i==0:
                        axes[i].set_ylabel("Test Accuracy [%]", fontsize=8)
                else:
                        axes[i].set_ylabel("")
                
        legend_handles = []
        for label, color in color_dict.items():
                if label == "JOINT":
                        legend_handles.append(mlines.Line2D([0], [0], color=color, label=label))
                elif label == "ER-OR":
                        legend_handles.append(mlines.Line2D([0], [0], color=color, marker='D', linestyle=':', markersize=4, label=label))
                else:
                        legend_handles.append(mlines.Line2D([0], [0], color=color, marker='o', markersize=4, label=label))
        labels = color_dict.keys()
        # Move ER-OR to the first position in the legend
        legend_handles = [legend_handles[1]] + legend_handles[:1] + legend_handles[2:] 
        labels = ['iCaRL-OR'] + [label for label in labels if label != 'ER-OR']
        fig.legend(legend_handles, labels, loc='center', bbox_to_anchor=(0.18, 0.42), ncol=1, title_fontsize=8, fontsize=8) 
        #fig.legend(legend_handles, labels, loc='center', bbox_to_anchor=(0.89, 0.42), ncol=1, title_fontsize=8, fontsize=8) 
        fig.tight_layout()
        fig.savefig(path + f"/figure2_extra.pdf", dpi=800)
        fig.clf() 

def figure3():
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
        # Create the dictionary with the updated assignments
        colors = sns.color_palette("viridis", 6)  # Requesting 6 colors to access cyan
        color_dict = {
                "ER": colors[1],  # blue
                "ER-OR": "black",  # cyan
        }
        datasets = ['seq-cifar100-224', 'seq-tinyimg-224']#, 'seq-cub200']
        training_setting = ['task-il','class-il']
        models = ['er_buf']

        fig, axes = plt.subplots(1, len(datasets) * len(training_setting), figsize=(8, 3), dpi=800)
        for i, setting in enumerate(list(itertools.product(datasets, training_setting))):
                data_buffer = []

                if setting[0] == 'seq-cifar10':
                        infinite_size = 2500
                        gap = [2200, 2300]
                elif setting[0] in ['seq-cifar100', 'seq-cifar100-224']:
                        infinite_size = 6000
                        dataset_size = 50000
                        gap = [5400, 5600]
                elif setting[0] in ['seq-tinyimg', 'seq-tinyimg-224']:
                        infinite_size = 12000
                        dataset_size = 100000
                        gap = [10800, 11200]
                elif setting[0] == 'seq-cub200':
                        infinite_size = 2500
                        dataset_size = 12000

                for model in models:
                        dictlist, filepath = get_data(path, setting[1], setting[0], model)
                        arguments_base, arguments_keep, arguments_combine = get_arguments(dictlist[0])
                        accmean_columns, acctask_columns, result_accuracy = get_arrays(dictlist[0])

                        df = get_dataframe(dictlist, arguments_base + [result_accuracy])

                        if setting[1] == "class-il":
                                features_version = "features_cil"
                        else:
                                features_version = "features_til"

                        df = df[df['result_type'].isin(list(['output', features_version]))]
                        df.loc[df['result_type'] == features_version, 'model'] = 'ER-OR'
                        buffer_sizes = df['buffer_size'].sort_values().tolist()
                        
                        #df["buffer_size"] = df["buffer_size"].replace(buffer_sizes[-1], infinite_size)
                        df["buffer_size"] = (100 * df["buffer_size"]) / dataset_size
                        
                        df['model'] = df.apply(lambda row: plotting_name(row['model']), axis=1)

                        data_buffer.append(df)
 
                
                data_buffer = pd.concat(data_buffer)

                data_buffer_OR = data_buffer[data_buffer['model'] == 'ER-OR']
                data_buffer_rest = data_buffer[data_buffer['model'] != 'ER-OR']

                sns.lineplot(data=data_buffer_rest, x='buffer_size', y=result_accuracy, hue='model',  marker='o',  markersize=5, linestyle='--', legend=False, palette=color_dict, ax=axes[i])
                #sns.lineplot(data=data_nobuffer, x='buffer_size', y=result_accuracy, hue='model', marker='',  linestyle='-', legend=False, palette=color_dict, ax=axes[i], linewidth=0.5)
                sns.lineplot(data=data_buffer_OR, x='buffer_size', y=result_accuracy, hue='model',  marker='D',  markersize=5, linestyle=':', legend=False, palette=color_dict, ax=axes[i])

                # Remove the boxes around each plot
                axes[i].spines['top'].set_visible(False)
                axes[i].spines['right'].set_visible(False)
                axes[i].spines['left'].set_visible(False)
                axes[i].spines['bottom'].set_visible(False)


                y_min = min(data_buffer[result_accuracy].min(),  data_buffer_OR[result_accuracy].min())
                y_max = max(data_buffer[result_accuracy].max(),  data_buffer_OR[result_accuracy].max())
                #axes[i].set_ylim(y_min - 5, y_max)

                #current_yticks = axes[i].get_yticks()
                #y_min, y_max = axes[i].get_ylim()  # Store exact limits
                #axes[i].set_yticks(current_yticks)  # Freeze ticks
                #axes[i].set_ylim((y_min, y_max))  # Reset exact limits

                if setting[0] in ['seq-cifar100', 'seq-cifar100-224']:
                        #xticks = [0, 1000, 2000, 3000, 4000, 5000, 6000]
                        axes[i].axvline(x=10000/dataset_size, color='purple', linestyle='--', lw=1, alpha=0.7)
                        axes[i].axhline(y=94.7 if setting[1] == 'task-il' else 73, color='purple', linestyle='--', lw=1, alpha=0.7)
                        axes[i].margins(x=0.1)
                        axes[i].set_xscale('symlog')
                elif setting[0] in ['seq-tinyimg', 'seq-tinyimg-224']:
                        #xticks = [0, 2000, 4000, 6000, 8000, 10000, 12000]
                        axes[i].axvline(x=20000/dataset_size, color='purple', linestyle='--', lw=1, alpha=0.7)
                        axes[i].axhline(y=83.5 if setting[1] == 'task-il' else 52, color='purple', linestyle='--', lw=1, alpha=0.7)
                        axes[i].margins(x=0.1)
                        axes[i].set_xscale('symlog')
                elif setting[0] == 'seq-cub200':
                        #xticks = [0, 500, 1000, 1500, 2000, 2500]
                        axes[i].axvline(x=20000/dataset_size, color='purple', linestyle='--', lw=1, alpha=0.7)
                        axes[i].axhline(y=89.5 if setting[1] == 'task-il' else 64, color='purple', linestyle='--', lw=1, alpha=0.7)
                        axes[i].margins(x=0.1)
                        axes[i].set_xscale('symlog')
                        
                
                #axes[i].set_xticks(xticks)  # Ensure tick positions remain the same
                #xticks = [(100*tick / dataset_size) for tick in xticks]
                #xticklabels = [str(int(tick)) if tick != max(xticks) else '1' for tick in xticks]  # Replace max tick with 'inf'
                #axes[i].set_xticklabels(xticklabels)  # Set the new labels

                
                #Make the fontsize of the axis smaller
                for tick in axes[i].get_xticklabels():
                        tick.set_rotation(45)
                axes[i].tick_params(axis='both', which='major', labelsize=8)
                
                
                # Customize the grid  
                axes[i].grid(visible=True, which='major', axis='x', linestyle=':', linewidth=0.5, color='lightgrey')  # Dashed grid for x-axis
                axes[i].grid(visible=True, which='major', axis='y', linestyle='--', linewidth=0.5, color='lightgrey') 

                axes[i].set_title(f'{plotting_name(setting[0])} ({plotting_name(setting[1])})', fontsize=10)
                axes[i].set_xlabel("Buffer Size [%]")

                if i==0:
                        axes[i].set_ylabel("Test Accuracy [%]", fontsize=8)
                else:
                        axes[i].set_ylabel("")
  
        legend_handles = []
        for label, color in color_dict.items():
                if label == "JOINT":
                        legend_handles.append(mlines.Line2D([0], [0], color=color, label=label))
                elif label == "ER-OR":
                        legend_handles.append(mlines.Line2D([0], [0], color=color, marker='D', linestyle=':', markersize=4, label=label))
                else:
                        legend_handles.append(mlines.Line2D([0], [0], color=color, marker='o', markersize=4, label=label))
        labels = color_dict.keys()
        # Move ER-OR to the first position in the legend
        legend_handles = [legend_handles[1]] + legend_handles[:1] + legend_handles[2:]
        labels = ['ER-OR'] + [label for label in labels if label != 'ER-OR']
        #fig.legend(legend_handles, labels, loc='center', bbox_to_anchor=(0.135, 0.5), ncol=1, title_fontsize=8, fontsize=8)
        fig.legend(legend_handles, labels, loc='center', bbox_to_anchor=(0.19, 0.5), ncol=1, title_fontsize=8, fontsize=8)
        fig.tight_layout()
        fig.savefig(path + f"/figure3.pdf", dpi=800)
        fig.clf() 

def figure5():
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
        # Create the dictionary with the updated assignments
        palette = sns.color_palette("tab10", 3)  # Requesting 6 colors to access cyan
        palette = {
                'buffer': palette[0],
                'train_dataset': palette[1],
                'test_dataset': palette[2]
        }

        model = 'er_buf'
        datasets = ['seq-cifar100', 'seq-tinyimg', 'seq-cub200']
        training_setting = ['task-il', 'class-il']
        fig, axes = plt.subplots(1, 6, figsize=(15, 3), dpi=800)

        total = 0

        for i, (ax, setting) in enumerate(zip(axes, list(itertools.product(datasets, training_setting)))):
                dictlist, _ = get_data(path, setting[1], setting[0], model, "/logs_NC.txt")
                arguments_base, _, _ = get_arguments(dictlist[0])

                tasks, within_var, _ = get_arrays(dictlist[0], 'within_var')
                _, between_var, _ = get_arrays(dictlist[0], 'between_var')
                between_var_overall, _, _ = get_arrays(dictlist[0], pre_string2='between_var_overall_task')
                between_var_current, _, _ = get_arrays(dictlist[0], pre_string2='between_var_current_task')
                between_var_previous, _, _ = get_arrays(dictlist[0], pre_string2='between_var_previous_task')
                _, cos_movement, _ = get_arrays(dictlist[0], 'cos_movement')
                df = get_dataframe(dictlist, arguments_base + ['seed'] + within_var + between_var + between_var_overall + between_var_current + between_var_previous + cos_movement)
                # df = df[df['result_type'].isin(list(['train_dataset']))]
                if setting[0] == 'seq-cub200':
                        df = df[df['buffer_size'] == 600]
                elif setting[0] == 'seq-tinyimg':
                        df = df[df['buffer_size'] == 5000]
                else:
                        df = df[df['buffer_size'] == 2000]


                ax = axes[total] #attention

                results = []
                for i in range(1, len(tasks) + 1):
                        for j in range(1, i):
                                if setting[1] == "class-il":
                                        current_result = pd.DataFrame({'training_task': i, 'evaluating_task': j, 
                                                'NC': (df[f'within_var_{j}_task{i}'] / df[f'between_var_overall_task{i}']), 'result_type': df['result_type'], 'buffer_size': df['buffer_size']})
                                else: 
                                        current_result = pd.DataFrame({'training_task': i, 'evaluating_task': j, 
                                                'NC': (df[f'within_var_{j}_task{i}'] / df[f'between_var_{j}_task{i}']), 'result_type': df['result_type'], 'buffer_size': df['buffer_size']})
                                results.append(current_result) 

                results = pd.concat(results)
                # sns.lineplot(data=results, x='training_task', y='accuracy', hue='evaluating_task', style='result_type', marker='o', legend=True, ax=ax, palette=palette)
                sns.lineplot(data=results, x='training_task', y='NC', hue='result_type', ax=ax, errorbar=('ci', 95), palette=palette, style='result_type', marker='o', dashes=False)
                
                ax.set_title(f'{plotting_name(setting[0])} ({plotting_name(setting[1])})')
                ax.set_xlabel("Training Task")
                if total==0:
                        ax.set_ylabel(f"Neural Collapse")
                else:
                        ax.set_ylabel("")

                # Add dotted grid lines
                ax.grid(visible=True, which='major', axis='x', linestyle=':', linewidth=0.5, color='lightgrey')  # Dashed grid for x-axis
                ax.grid(visible=True, which='major', axis='y', linestyle='--', linewidth=0.5, color='lightgrey')   

                # Remove the boxes around each plot
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['left'].set_visible(False)
                ax.spines['bottom'].set_visible(False)

                ax.legend_.remove()
                total += 1
                        
        handles, labels = axes[0].get_legend_handles_labels()
        handles = [handles[2], handles[1], handles[0]]
        labels = ['Test','Train','Buffer']
        fig.legend(handles, labels, loc='center', bbox_to_anchor=(0.135, 0.51), ncol=1, title_fontsize=8, fontsize=8)
        fig.tight_layout()
        #fig.legend(handles, labels, bbox_to_anchor=(1.04, 0.9), ncol=1)
        #fig.tight_layout()
        fig.savefig(path + f"/figure5.pdf", dpi=800)#, bbox_inches='tight')
        fig.clf() 

def figure5_extra():
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
        # Create the dictionary with the updated assignments
        palette = (sns.color_palette("tab10", 3)).as_hex()[2:]  # Requesting 6 colors to access cyan
        #palette = {
        #        'buffer': palette[0],
        #        'train_dataset': palette[1],
        #        'test_dataset': palette[2]
        #}

        #palette = ['black']
        model = 'er_buf'
        datasets = ['seq-cifar100', 'seq-tinyimg', 'seq-cub200']
        training_setting = ['task-il', 'class-il']
        fig, axes = plt.subplots(1, 6, figsize=(15, 3), dpi=800)

        total = 0

        for i, (ax, setting) in enumerate(zip(axes, list(itertools.product(datasets, training_setting)))):
                dictlist, _ = get_data(path, setting[1], setting[0], model, "/logs_NC.txt")
                arguments_base, _, _ = get_arguments(dictlist[0])

                tasks, within_var, _ = get_arrays(dictlist[0], 'within_var')
                _, between_var, _ = get_arrays(dictlist[0], 'between_var')
                between_var_overall, _, _ = get_arrays(dictlist[0], pre_string2='between_var_overall_task')
                between_var_current, _, _ = get_arrays(dictlist[0], pre_string2='between_var_current_task')
                between_var_previous, _, _ = get_arrays(dictlist[0], pre_string2='between_var_previous_task')
                _, cos_movement, _ = get_arrays(dictlist[0], 'cos_movement')
                _, feature_norm, _ = get_arrays(dictlist[0], 'features_norm')
                global_variance, _, _ = get_arrays(dictlist[0], pre_string2='global_variance_task')

                df = get_dataframe(dictlist, arguments_base + ['seed'] + within_var + between_var + between_var_overall + between_var_current + between_var_previous + cos_movement + feature_norm + global_variance)
                df = df[df['result_type'].isin(list(['train_dataset']))]
                if setting[0] == 'seq-cub200':
                        df = df[df['buffer_size'] == 0]
                elif setting[0] == 'seq-tinyimg':
                        df = df[df['buffer_size'] == 0]
                else:
                        df = df[df['buffer_size'] == 0]

                ax = axes[total] #attention

                results = []
                for i in range(1, len(tasks) + 1):
                        for j in range(i, i+1):
                                if setting[1] == "class-il":
                                        current_result = pd.DataFrame({'training_task': i, 'evaluating_task': j, 
                                                'NC': (df[f'global_variance_task{i}']), 'result_type': df['result_type'], 'buffer_size': df['buffer_size']})
                                else: 
                                        current_result = pd.DataFrame({'training_task': i, 'evaluating_task': j, 
                                                'NC': (df[f'global_variance_task{i}']), 'result_type': df['result_type'], 'buffer_size': df['buffer_size']})
                                results.append(current_result) 

                results = pd.concat(results)
                # sns.lineplot(data=results, x='training_task', y='accuracy', hue='evaluating_task', style='result_type', marker='o', legend=True, ax=ax, palette=palette)
                sns.lineplot(data=results, x='training_task', y='NC', hue='result_type', ax=ax, errorbar=('ci', 95), palette=palette, style='result_type', marker='o', dashes=False)
                
                ax.set_title(f'{plotting_name(setting[0])} ({plotting_name(setting[1])})')
                ax.set_xlabel("Training Task")
                if total==0:
                        ax.set_ylabel(f"Global Variance")
                else:
                        ax.set_ylabel("")

                # Add dotted grid lines
                ax.grid(visible=True, which='major', axis='x', linestyle=':', linewidth=0.5, color='lightgrey')  # Dashed grid for x-axis
                ax.grid(visible=True, which='major', axis='y', linestyle='--', linewidth=0.5, color='lightgrey')   

                # Remove the boxes around each plot
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['left'].set_visible(False)
                ax.spines['bottom'].set_visible(False)

                ax.xaxis.set_major_locator(ticker.MultipleLocator(2))

                ax.legend_.remove()
                total += 1
        
        for i in [0, 2, 4]:
                y_min = 0
                y_max = max(axes[i].get_ylim()[1], axes[i+1].get_ylim()[1])
                axes[i].set_ylim(y_min, y_max)
                axes[i+1].set_ylim(y_min, y_max)    
        #handles, labels = axes[0].get_legend_handles_labels()
        #handles = [handles[2], handles[1], handles[0]]
        #labels = ['Test','Train','Buffer']
        #fig.legend(handles, labels, loc='center', bbox_to_anchor=(0.135, 0.51), ncol=1, title_fontsize=8, fontsize=8)
        fig.tight_layout()
        #fig.legend(handles, labels, bbox_to_anchor=(1.04, 0.9), ncol=1)
        #fig.tight_layout()
        fig.savefig(path + f"/figure5_extra.pdf", dpi=800)#, bbox_inches='tight')
        fig.clf() 

def figure6():
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
        # Create the dictionary with the updated assignments
        palette = sns.color_palette("tab10", 3)  # Requesting 6 colors to access cyan

        palette = {
                'buffer': palette[0],
                'train_dataset': palette[1],
                'test_dataset': palette[2]
        }

        model = 'er_buf'
        datasets = ['seq-cifar100', 'seq-tinyimg', 'seq-cub200']
        training_setting = ['task-il', 'class-il']
        fig, axes = plt.subplots(1, 6, figsize=(15, 3), dpi=800)
        #plt.rcParams['text.usetex'] = True # TeX rendering

        total = 0

        for i, (ax, setting) in enumerate(zip(axes, list(itertools.product(datasets, training_setting)))):
                dictlist, _ = get_data(path, setting[1], setting[0], model, "/logs_NC.txt")
                arguments_base, _, _ = get_arguments(dictlist[0])

                tasks, within_var, _ = get_arrays(dictlist[0], 'within_var')
                _, between_var, _ = get_arrays(dictlist[0], 'between_var')
                between_var_overall, _, _ = get_arrays(dictlist[0], pre_string2='between_var_overall_task')
                between_var_current, _, _ = get_arrays(dictlist[0], pre_string2='between_var_current_task')
                between_var_previous, _, _ = get_arrays(dictlist[0], pre_string2='between_var_previous_task')
                _, cos_movement, _ = get_arrays(dictlist[0], 'cos_movement')
                df = get_dataframe(dictlist, arguments_base + ['seed'] + within_var + between_var + between_var_overall + between_var_current + between_var_previous + cos_movement)
                # df = df[df['result_type'].isin(list(['train_dataset']))]

                if setting[0] == 'seq-cifar100':
                        infinite_size = 6000
                elif setting[0] == 'seq-tinyimg':
                        infinite_size = 12000
                elif setting[0] == 'seq-cub200':
                        infinite_size = 2500

                buffer_sizes = df['buffer_size'].sort_values().tolist()
                df["buffer_size"] = df["buffer_size"].replace(buffer_sizes[-1], infinite_size)

                ax = axes[total] #attention

                results = []
                for i in range(1, len(tasks) + 1):
                        for j in range(1, i):
                                if setting[1] == "class-il":
                                        current_result = pd.DataFrame({'training_task': i, 'evaluating_task': j, 
                                                'NC': (df[f'within_var_{j}_task{i}'] / df[f'between_var_overall_task{i}']), 'result_type': df['result_type'], 'buffer_size': df['buffer_size']})
                                else: 
                                        current_result = pd.DataFrame({'training_task': i, 'evaluating_task': j, 
                                                'NC': (df[f'within_var_{j}_task{i}'] / df[f'between_var_{j}_task{i}']), 'result_type': df['result_type'], 'buffer_size': df['buffer_size']})
                                results.append(current_result) 

                results = pd.concat(results)
                # sns.lineplot(data=results, x='training_task', y='accuracy', hue='evaluating_task', style='result_type', marker='o', legend=True, ax=ax, palette=palette)
                sns.lineplot(data=results, x='buffer_size', y='NC', hue='result_type', ax=ax, errorbar=('ci', 95), palette=palette, style='result_type', marker='o', dashes=False)
                
                if setting[0] == 'seq-cifar10':
                        xticks = [0, 200, 500, 1000, 1500, 2000, 2500]
                elif setting[0] == 'seq-cifar100':
                        xticks = [0, 1000, 2000, 3000, 4000, 5000, 6000]
                        dataset_size = 50000
                elif setting[0] == 'seq-tinyimg':
                        xticks = [0, 2000, 4000, 6000, 8000, 10000, 12000]
                        dataset_size = 100000
                elif setting[0] == 'seq-cub200':
                        xticks = [0, 400, 800, 1200, 1600, 2000, 2500]
                        dataset_size = 12000

                ax.set_xticks(xticks)  # Ensure tick positions remain the same
                if setting[0] != 'seq-cub200':
                        xticklabels = [str(int(100*tick / dataset_size)) if tick != max(xticks) else '100' for tick in xticks]
                else:
                        xticklabels = [str(round((100*tick / dataset_size), 1)) if tick != max(xticks) else '100' for tick in xticks]
                ax.set_xticklabels(xticklabels)  # Set the new labels

                #Make the fontsize of the axis smaller
                for tick in ax.get_xticklabels():
                        tick.set_rotation(45)

                ax.set_title(f'{plotting_name(setting[0])} ({plotting_name(setting[1])})')
                ax.set_xlabel("Buffer Size [%]")
                if total==0:
                        ax.set_ylabel(f"Neural Collapse")
                else:
                        ax.set_ylabel("")

                # Add dotted grid lines
                ax.grid(visible=True, which='major', axis='x', linestyle=':', linewidth=0.5, color='lightgrey')  # Dashed grid for x-axis
                ax.grid(visible=True, which='major', axis='y', linestyle='--', linewidth=0.5, color='lightgrey')   

                # Remove the boxes around each plot
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['left'].set_visible(False)
                ax.spines['bottom'].set_visible(False)

                ax.legend_.remove()
                total += 1
                        
        # labels = ["1", "3", "5", "7", "9"]
        # legend_handles = [mpatches.Patch(color=palette[i]) for i in range(5)]
        #fig.legend(legend_handles, labels, loc='center', bbox_to_anchor=(0.5, 0.05), ncol=6)
        handles, labels = axes[0].get_legend_handles_labels()
        handles = [handles[1], handles[0], handles[2]]
        labels = ['Test', 'Train','Buffer']
        fig.legend(handles, labels, loc='center', bbox_to_anchor=(0.135, 0.77), ncol=1, title_fontsize=8, fontsize=8)
        fig.tight_layout()
        fig.savefig(path + f"/figure6.pdf", dpi=800)
        fig.clf() 

def figure6_extra():
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
        # Create the dictionary with the updated assignments
        palette = sns.color_palette("tab10", 3)  # Requesting 6 colors to access cyan

        palette = {
                'buffer': palette[2],
                'train_dataset': palette[1],
                'test_dataset': palette[1]
        }

        model = 'er_buf'
        datasets = ['seq-cifar100', 'seq-tinyimg', 'seq-cub200']
        training_setting = ['task-il', 'class-il']
        fig, axes = plt.subplots(1, 6, figsize=(15, 3), dpi=800)
        #plt.rcParams['text.usetex'] = True # TeX rendering

        total = 0

        for i, (ax, setting) in enumerate(zip(axes, list(itertools.product(datasets, training_setting)))):
                dictlist, _ = get_data(path, setting[1], setting[0], model, "/logs_NC.txt")
                arguments_base, _, _ = get_arguments(dictlist[0])

                tasks, within_var, _ = get_arrays(dictlist[0], 'within_var')
                _, between_var, _ = get_arrays(dictlist[0], 'between_var')
                between_var_overall, _, _ = get_arrays(dictlist[0], pre_string2='between_var_overall_task')
                between_var_current, _, _ = get_arrays(dictlist[0], pre_string2='between_var_current_task')
                between_var_previous, _, _ = get_arrays(dictlist[0], pre_string2='between_var_previous_task')
                _, cos_movement, _ = get_arrays(dictlist[0], 'cos_movement')
                df = get_dataframe(dictlist, arguments_base + ['seed'] + within_var + between_var + between_var_overall + between_var_current + between_var_previous + cos_movement)
                df = df[df['result_type'].isin(list(['test_dataset', 'buffer']))]

                if setting[0] == 'seq-cifar100':
                        df = df[(df['buffer_size'] > 0)]
                        infinite_size = 6000
                elif setting[0] == 'seq-tinyimg':
                        df = df[(df['buffer_size'] > 0)]
                        infinite_size = 12000
                elif setting[0] == 'seq-cub200':
                        df = df[(df['buffer_size'] > 0)]
                        infinite_size = 2500

                buffer_sizes = df['buffer_size'].sort_values().tolist()
                df["buffer_size"] = df["buffer_size"].replace(buffer_sizes[-1], infinite_size)

                ax = axes[total] #attention

                results = []
                for i in range(1, len(tasks) + 1):
                        for j in range(1, i):
                                if setting[1] == "class-il":
                                        current_result = pd.DataFrame({'training_task': i, 'evaluating_task': j, 
                                                'NC': (df[f'cos_movement_{j}_task{i}']), 'result_type': df['result_type'], 'buffer_size': df['buffer_size']})
                                else: 
                                        current_result = pd.DataFrame({'training_task': i, 'evaluating_task': j, 
                                                'NC': (df[f'cos_movement_{j}_task{i}']), 'result_type': df['result_type'], 'buffer_size': df['buffer_size']})
                                results.append(current_result) 

                results = pd.concat(results)
                # sns.lineplot(data=results, x='training_task', y='accuracy', hue='evaluating_task', style='result_type', marker='o', legend=True, ax=ax, palette=palette)
                sns.lineplot(data=results, x='buffer_size', y='NC', hue='result_type', ax=ax, errorbar=('ci', 95), palette=palette, zorder=0, style='result_type', marker='o', dashes=False)
                
                if setting[0] == 'seq-cifar10':
                        xticks = [0, 200, 500, 1000, 1500, 2000, 2500]
                elif setting[0] == 'seq-cifar100':
                        xticks = [0, 1000, 2000, 3000, 4000, 5000, 6000]
                        dataset_size = 50000
                elif setting[0] == 'seq-tinyimg':
                        xticks = [0, 2000, 4000, 6000, 8000, 10000, 12000]
                        dataset_size = 100000
                elif setting[0] == 'seq-cub200':
                        xticks = [0, 400, 800, 1200, 1600, 2000, 2500]
                        dataset_size = 12000

                ax.set_xticks(xticks)  # Ensure tick positions remain the same
                if setting[0] != 'seq-cub200':
                        xticklabels = [str(int(100*tick / dataset_size)) if tick != max(xticks) else '100' for tick in xticks]
                else:
                        xticklabels = [str(round((100*tick / dataset_size), 1)) if tick != max(xticks) else '100' for tick in xticks]
                ax.set_xticklabels(xticklabels)  # Set the new labels

                #Make the fontsize of the axis smaller
                for tick in ax.get_xticklabels():
                        tick.set_rotation(45)

                ax.set_title(f'{plotting_name(setting[0])} ({plotting_name(setting[1])})')
                ax.set_xlabel("Buffer Size [%]")
                if total==0:
                        ax.set_ylabel(f"")
                else:
                        ax.set_ylabel("")

                # Add dotted grid lines
                ax.grid(visible=True, which='major', axis='x', linestyle=':', linewidth=0.5, color='lightgrey')  # Dashed grid for x-axis
                ax.grid(visible=True, which='major', axis='y', linestyle='--', linewidth=0.5, color='lightgrey')   

                # Remove the boxes around each plot
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['left'].set_visible(False)
                ax.spines['bottom'].set_visible(False)

                ax.legend_.remove()
                total += 1
                        
        # labels = ["1", "3", "5", "7", "9"]
        # legend_handles = [mpatches.Patch(color=palette[i]) for i in range(5)]
        #fig.legend(legend_handles, labels, loc='center', bbox_to_anchor=(0.5, 0.05), ncol=6)
        #fig.suptitle("Cosine Similarity class means", fontsize=10)
        handles, labels = axes[0].get_legend_handles_labels()
        handles = [handles[1], handles[0]]
        labels = ['Train', 'Test']
        fig.legend(handles, labels, loc='center', bbox_to_anchor=(0.125, 0.38), ncol=1, title_fontsize=8, fontsize=8)
        fig.tight_layout()
        fig.savefig(path + f"/figure6_extra.pdf", dpi=800)
        fig.clf() 

def figure7_1():
        datasets = {'seq-cifar100': [500], 'seq-tinyimg': [1000], 'seq-cub200': [600]}
        model = 'er_buf'
        training_settings = ['class-il', 'task-il']

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
        # Create the dictionary with the updated assignments
        palette = sns.color_palette("Set1") 

        fig, axes = plt.subplots(1, 3, figsize=(8, 3), dpi=800, sharey=False)

        for col1, (c_dataset, buffer_sizes) in enumerate(datasets.items()):
                results = []
                ax = axes[col1] #attention
                for k in range(len(training_settings)):

                        dictlist, _ = get_data(path, training_settings[k], c_dataset, model, "/logs_NC.txt")
                        arguments_base, _, _ = get_arguments(dictlist[0])
                        #arguments_base.remove('seed')
                        tasks, within_var, _ = get_arrays(dictlist[0], 'within_var')
                        _, between_var, _ = get_arrays(dictlist[0], 'between_var')
                        between_var_overall, _, _ = get_arrays(dictlist[0], pre_string2='between_var_overall_task')
                        between_var_current, _, _ = get_arrays(dictlist[0], pre_string2='between_var_current_task')
                        between_var_previous, _, _ = get_arrays(dictlist[0], pre_string2='between_var_previous_task')
                        _, cos_movement, _ = get_arrays(dictlist[0], 'cos_movement')
                        df = get_dataframe(dictlist, arguments_base + ['seed'] + within_var + between_var + between_var_overall + between_var_current + between_var_previous + cos_movement)
                        df = df[df['result_type'].isin(list(['buffer', 'train_dataset', 'test_dataset']))]

                        for col2, buffer_size in enumerate(buffer_sizes):
                                current_df = df[df['buffer_size'] == buffer_size].copy()
                                
                                #Split buffer and train rows
                                buffer_df = current_df[current_df['result_type'] == 'buffer'].set_index('seed')
                                train_df = current_df[current_df['result_type'] == 'train_dataset'].set_index('seed')
                                test_df = current_df[current_df['result_type'] == 'test_dataset'].set_index('seed')

                                for i in range(0, 9):
                                        for j in range(1, 10+1-i):
                                                if training_settings[k] == 'class-il':
                                                        buffer_df[f'NC_{j}_task{j+i}'] = buffer_df[f'within_var_{j}_task{j+i}'] / buffer_df[f'between_var_overall_task{j+i}']
                                                        train_df[f'NC_{j}_task{j+i}'] = train_df[f'within_var_{j}_task{j+i}'] / train_df[f'between_var_overall_task{j+i}']
                                                else:
                                                        buffer_df[f'NC_{j}_task{j+i}'] = buffer_df[f'within_var_{j}_task{j+i}'] / buffer_df[f'between_var_{j}_task{j+i}']
                                                        train_df[f'NC_{j}_task{j+i}'] = train_df[f'within_var_{j}_task{j+i}'] / train_df[f'between_var_{j}_task{j+i}']

                                # Drop the 'result' column before subtraction
                                buffer_df = buffer_df.drop(columns=arguments_base)
                                train_df = train_df.drop(columns=arguments_base)
                                test_df = test_df.drop(columns=arguments_base)


                                current_df = train_df-buffer_df

                                for i in range(0, 9):
                                        to_average = []
                                        for j in range(1, 10+1-i):
                                                to_average.append(f'within_var_{j}_task{j+i}' )
                                        current_result = pd.DataFrame({'tasks_afterwards': i, 'var_mean': (current_df[to_average].mean(axis=0)), 'setting': training_settings[k]})
                                        results.append(current_result) 

                results = pd.concat(results)
                                
                # Remove the boxes around each plot
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['left'].set_visible(False)
                ax.spines['bottom'].set_visible(False)

                
                sns.lineplot(data=results, x='tasks_afterwards', y='var_mean', errorbar=('ci', 95), palette=palette, hue='setting', style='setting', ax=ax, marker='o', dashes=False, legend=True)
                ax.legend_.remove()
                
                ax.set_title(f'{plotting_name(c_dataset)}')
                ax.set_xlabel("Task Switches")
                ax.set_ylabel("")
                ax.xaxis.set_major_locator(ticker.MultipleLocator(2))
                #ax.yaxis.set_major_locator(ticker.MultipleLocator(5))
                ax.grid(visible=True, which='major', axis='x', linestyle=':', linewidth=0.5, color='lightgrey')  # Dashed grid for x-axis
                ax.grid(visible=True, which='major', axis='y', linestyle='--', linewidth=0.5, color='lightgrey')   

        fig.suptitle("Within Class Variance Train-Buffer", fontsize=10, y=.95)
        
        plt.tight_layout()           
        handles, _ = ax.get_legend_handles_labels()
        labels = ['Single-Head', 'Multi-Head']
        #fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(0.325, 0.78), ncol=1)
        fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(0.31, 0.4), ncol=1)
                
        fig.savefig(path + f"/figure7_1.pdf", dpi=800)
        fig.clf()  

def figure7_2():
        datasets = {'seq-cifar100': [500], 'seq-tinyimg': [1000], 'seq-cub200': [600]}
        model = 'er_buf'
        training_settings = ['class-il', 'task-il']

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
        # Create the dictionary with the updated assignments
        palette = sns.color_palette("Set1") 

        fig, axes = plt.subplots(1, 3, figsize=(8, 3), dpi=800, sharey=False)

        for col1, (c_dataset, buffer_sizes) in enumerate(datasets.items()):
                results = []
                results2 = []
                ax = axes[col1] #attention
                for k in range(len(training_settings)):

                        dictlist, _ = get_data(path, training_settings[k], c_dataset, model, "/logs_NC.txt")
                        arguments_base, _, _ = get_arguments(dictlist[0])
                        #arguments_base.remove('seed')
                        tasks, within_var, _ = get_arrays(dictlist[0], 'within_var')
                        _, between_var, _ = get_arrays(dictlist[0], 'between_var')
                        between_var_overall, _, _ = get_arrays(dictlist[0], pre_string2='between_var_overall_task')
                        between_var_current, _, _ = get_arrays(dictlist[0], pre_string2='between_var_current_task')
                        between_var_previous, _, _ = get_arrays(dictlist[0], pre_string2='between_var_previous_task')
                        _, cos_movement, _ = get_arrays(dictlist[0], 'cos_movement')
                        _, cos_mean_distance, _ = get_arrays(dictlist[0], 'cos_mean_distance')
                        df = get_dataframe(dictlist, arguments_base + ['seed'] + within_var + between_var + between_var_overall + between_var_current + between_var_previous + cos_movement + cos_mean_distance)
                        df = df[df['result_type'].isin(list(['buffer', 'train_dataset', 'test_dataset']))]

                        for col2, buffer_size in enumerate(buffer_sizes):
                                current_df = df[df['buffer_size'] == buffer_size].copy()
                                
                                #Split buffer and train rows
                                buffer_df = current_df[current_df['result_type'] == 'buffer'].set_index('seed')
                                train_df = current_df[current_df['result_type'] == 'train_dataset'].set_index('seed')
                                test_df = current_df[current_df['result_type'] == 'test_dataset'].set_index('seed')

                                # Drop the 'result' column before subtraction
                                buffer_df = buffer_df.drop(columns=arguments_base)
                                train_df = train_df.drop(columns=arguments_base)
                                test_df = test_df.drop(columns=arguments_base)

                                current_df = test_df

                                for i in range(0, 9):
                                        to_average = []
                                        to_average2 = []
                                        for j in range(1, 10+1-i):
                                                to_average.append(f'cos_movement_{j}_task{j+i}' )
                                                to_average2.append(f'cos_mean_distance_{j}_task{j+i}' )
                                        current_result = pd.DataFrame({'tasks_afterwards': i, 'var_mean': (current_df[to_average].mean(axis=0)), 'setting': training_settings[k]})
                                        current_result2 = pd.DataFrame({'tasks_afterwards': i, 'var_mean': (current_df[to_average2].mean(axis=0)), 'setting': training_settings[k]})

                                        results.append(current_result) 
                                        results2.append(current_result2)

                results = pd.concat(results)
                results2 = pd.concat(results2)
                                
                # Remove the boxes around each plot
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['left'].set_visible(False)
                ax.spines['bottom'].set_visible(False)

                
                sns.lineplot(data=results, x='tasks_afterwards', y='var_mean', errorbar=('ci', 95), palette=palette, hue='setting', style='setting', ax=ax, marker='o', dashes=False, legend=True)
                sns.lineplot(data=results2, x='tasks_afterwards', y='var_mean', errorbar=('ci', 95), palette=palette, hue='setting', style='setting', ax=ax, marker='', dashes=[(5, 2), (5, 2)], legend=False)
                ax.legend_.remove()
                
                ax.set_title(f'{plotting_name(c_dataset)}')
                ax.set_xlabel("Task Switches")
                ax.set_ylabel("")
                ax.xaxis.set_major_locator(ticker.MultipleLocator(2))
                #ax.yaxis.set_major_locator(ticker.MultipleLocator(5))
                ax.grid(visible=True, which='major', axis='x', linestyle=':', linewidth=0.5, color='lightgrey')  # Dashed grid for x-axis
                ax.grid(visible=True, which='major', axis='y', linestyle='--', linewidth=0.5, color='lightgrey')   

        fig.suptitle("Cosine Similarity class means", fontsize=10, y=.95)
        
        plt.tight_layout()           
        handles, _ = ax.get_legend_handles_labels()
        labels = ['Single-Head', 'Multi-Head']
        fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(0.97, 0.59), ncol=1)
                
        fig.savefig(path + f"/figure7_2.pdf", dpi=800)
        fig.clf()   

def figure7_extra():
        datasets = {'seq-cifar100-224': [2000], 'seq-tinyimg-224': [5000]}
        model = 'er_buf'
        training_settings = ['class-il', 'task-il']

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
        # Create the dictionary with the updated assignments
        palette = sns.color_palette("Set1") 

        fig, axes = plt.subplots(1, 3, figsize=(8, 3), dpi=800, sharey=False)

        for col1, (c_dataset, buffer_sizes) in enumerate(datasets.items()):
                results = []
                ax = axes[col1] #attention
                for k in range(len(training_settings)):
                        dictlist, _ = get_data(path, training_settings[k], c_dataset, model)
                        arguments_base, _, _ = get_arguments(dictlist[0])
                        accmean_columns, acctask_columns, result_accuracy = get_arrays(dictlist[0])

                        df = get_dataframe(dictlist, arguments_base + accmean_columns + acctask_columns)

                        df = df[df['result_type'].isin(list(['features_cil']))]

                        for col2, buffer_size in enumerate(buffer_sizes):
                                current_df = df[df['buffer_size'] == buffer_size].copy()

                                for i in range(0, 10):
                                        current_result = pd.DataFrame({'task': i+1, 'accuracy': current_df[accmean_columns[i]], 'setting': training_settings[k]})
                                        results.append(current_result) 

                results = pd.concat(results)
                                
                # Remove the boxes around each plot
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['left'].set_visible(False)
                ax.spines['bottom'].set_visible(False)

                
                sns.lineplot(data=results, x='task', y='accuracy', errorbar=('ci', 95), palette=palette, hue='setting', style='setting', ax=ax, marker='o', dashes=False, legend=True)

                ax.legend_.remove()        
                ax.set_title(f'{plotting_name(c_dataset)}')
                ax.set_xlabel("Training Task")
                if col1 == 0:
                        ax.set_ylabel("Test Accuracy [%]")
                else:
                        ax.set_ylabel("")
                ax.xaxis.set_major_locator(ticker.MultipleLocator(2))
                #ax.yaxis.set_major_locator(ticker.MultipleLocator(5))
                ax.grid(visible=True, which='major', axis='x', linestyle=':', linewidth=0.5, color='lightgrey')  # Dashed grid for x-axis
                ax.grid(visible=True, which='major', axis='y', linestyle='--', linewidth=0.5, color='lightgrey')   

        fig.suptitle("Fitting a separate Linear Classifier for each task", fontsize=10, y=.95)
        
        plt.tight_layout()           
        handles, _ = ax.get_legend_handles_labels()
        labels = ['Single-Head', 'Multi-Head']
        fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(0.33, 0.38), ncol=1)
                
        fig.savefig(path + f"/figure7_extra.pdf", dpi=800)
        fig.clf()  

figure7_extra()
def figure8():

        # Define the directory path
        directory_path = './data/dataframes/bias'

        # Initialize an empty list to store dataframes
        dataframes = []

        # Iterate over the files in the directory
        for filename in os.listdir(directory_path):
                if filename.endswith(".csv"):  # Assuming the files are in CSV format
                        # Extract dataset, buffersize, and seed from the filename
                        parts = filename.split('_')
                        dataset = parts[0]
                        buffersize = parts[1]
                        seed = parts[2].split('.')[0]  # Remove the file extension

                        # Load the dataframe
                        df = pd.read_csv(os.path.join(directory_path, filename))

                        # Add the new columns
                        df['dataset'] = dataset
                        df['buffersize'] = buffersize
                        df['seed'] = seed

                        # Append the dataframe to the list
                        dataframes.append(df)

        # Concatenate all dataframes into a single dataframe
        final_df = pd.concat(dataframes, ignore_index=True)

        # Display the final dataframe
        print(final_df)

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
        # Create the dictionary with the updated assignments
        palette = sns.color_palette("spring", 3)  # Requesting 6 colors to access cyan

        # Get unique datasets and buffer sizes
        datasets = final_df['dataset'].unique()
        #scratch = datasets[0]
        #datasets[0] = datasets[1]
        #datasets[1] = datasets[2]
        #datasets[2] = scratch
        buffersizes = final_df['buffersize'].unique()

        # Create subplots
        fig, axes = plt.subplots(1,len(datasets), figsize=(9, 2.25), sharey=True, dpi=800)

        # Plot data
        for i, dataset in enumerate(datasets):
                ax = axes[i]
                subset = final_df[(final_df['dataset'] == dataset)]
                subset['result_type'] = pd.Categorical(subset['result_type'], categories=['output', 'buffer', 'features'], ordered=True)
                
                sns.barplot(x='task', y='probability', data=subset, ax=ax, hue='result_type', palette=palette, dodge=False)


                ax.set_title(f'{plotting_name(dataset)}')
                ax.set_xlabel("Training Task")
                
                if i==0:
                        ax.set_ylabel("Probability Mass")
                else:
                        ax.set_ylabel("")

                # Add dotted grid lines
                ax.grid(True, linestyle='--', linewidth=1.0, axis='y', zorder=2)

                # Remove the boxes around each plot
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['left'].set_visible(False)
                ax.spines['bottom'].set_visible(False)

                ax.legend_.remove()

        #fig.suptitle("Probability mass on the current task", fontsize=10)
        # labels = ["1", "3", "5", "7", "9"]
        # legend_handles = [mpatches.Patch(color=palette[i]) for i in range(5)]
        #fig.legend(legend_handles, labels, loc='center', bbox_to_anchor=(0.5, 0.05), ncol=6)
        handles, labels = axes[0].get_legend_handles_labels()
        labels = ['ER','ER-BUF','ER-OR']
        fig.legend(handles, labels, bbox_to_anchor=(0.98, 0.835), ncol=1)
        fig.tight_layout()
        fig.savefig(path + f"/figure8.pdf", dpi=800)
        fig.clf() 

def figure8_NC():
        
        # Define the directory path
        directory_path = './data/dataframes/nc'

        # Initialize an empty list to store dataframes
        dataframes = []

        # Iterate over the files in the directory
        for filename in os.listdir(directory_path):
                if filename.endswith(".csv"):  # Assuming the files are in CSV format
                        # Extract dataset, buffersize, and seed from the filename
                        parts = filename.split('_')
                        datase = parts[0]
                        buffersize = parts[1]
                        seed = parts[2].split('.')[0]  # Remove the file extension

                        # Load the dataframe
                        df = pd.read_csv(os.path.join(directory_path, filename))

                        # Add the new columns
                        df['dataset'] = datase
                        df['buffersize'] = buffersize
                        df['seed'] = seed

                        if(buffersize == '0'):
                                dataframes.append(df)
                        

        # Concatenate all dataframes into a single dataframe
        final_df = pd.concat(dataframes, ignore_index=True)
        records = []
        for i in [2, 5, 8]:  # 2, 4, 6, 8, 10
                within_col = f"within_var_{i}"
                between_col = f"between_var_{i}"
                ratio = final_df[within_col] / final_df[between_col]

                records.append(pd.DataFrame({
                        "dataset": final_df["dataset"],
                        "sgd": final_df["buffersize"],
                        "epoch": final_df["epoch"],
                        "Task": final_df["task"],
                        "ratio": ratio,
                        "var_index": i
                        }))

        plot_df = pd.concat(records, ignore_index=True)

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
        # Create the dictionary with the updated assignments
        palette = sns.color_palette("rocket", 4)[1:]  # Requesting 6 colors to access cyan
        datasets = final_df['dataset'].unique()
        #scratch = datasets[0]
        #datasets[0] = datasets[1]
        #datasets[1] = datasets[2]
        #datasets[2] = scratch
        fig, axes = plt.subplots(1,len(datasets), figsize=(6, 3), dpi=800)

        for i, dataset in enumerate(datasets):
                ax = axes[i]
                subset = plot_df[(plot_df['dataset'] == dataset)]

                sns.lineplot(data=subset, x='epoch', y='ratio', hue='var_index', ax=ax, palette=palette, dashes=False, marker='')
                
                ax.set_title(f'{plotting_name(dataset)} ({plotting_name("task-il")})')
                ax.set_xlabel("Training Task")
                if i==0:
                        ax.set_ylabel(f"Neural Collapse")
                else:
                        ax.set_ylabel("")

                # Add dotted grid lines
                ax.grid(visible=True, which='major', axis='x', linestyle=':', linewidth=0.5, color='lightgrey')  # Dashed grid for x-axis
                ax.grid(visible=True, which='major', axis='y', linestyle='--', linewidth=0.5, color='lightgrey')   

                # Remove the boxes around each plot
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['left'].set_visible(False)
                ax.spines['bottom'].set_visible(False)

                if dataset=="seq-cifar100":
                        ax.set_xticks([i * 50 for i in range(0, 10, 1)])
                        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x//50)+1}"))
                        ax.set_ylim(ax.get_ylim()[0], 6)
                elif dataset=="seq-tinyimg":
                        ax.set_xticks([i * 100 for i in range(0, 10, 1)])
                        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x//100)+1}"))
                        ax.set_ylim(ax.get_ylim()[0], 6)
                elif dataset=="seq-cub200":
                        ax.set_xticks([i * 30 for i in range(0, 10, 1)])
                        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x//30)+1}"))

                ax.legend_.remove()

        handles, labels = axes[0].get_legend_handles_labels()
        labels=['Task 2', 'Task 5', 'Task 8']
        fig.legend(handles, labels, bbox_to_anchor=(0.45, 0.89), ncol=1)
        fig.tight_layout()
        fig.savefig(path + f"/figure8_NC.pdf", dpi=800)
        fig.clf() 

def table1():
        dataset='seq-cub200'
        model = 'er_buf'
        training_setting = 'task-il'

        dictlist, filepath = get_data(path, training_setting, dataset, model)
        arguments_base, arguments_keep, arguments_combine = get_arguments(dictlist[0])
        accmean_columns, acctask_columns, result_accuracy = get_arrays(dictlist[0])
        df = get_dataframe(dictlist, arguments_base + accmean_columns + acctask_columns)
        df = df[df['result_type'] == 'features_til']
        df = df[df['buffer_size'] == 200]

        #print(df[['lr', 'accmean_task10', 'buffer_size']])#.mean(axis=0))
        print(f'previous_task_pm: {df[acctask_columns[-len(accmean_columns):-1]].mean(axis=1).mean(axis=0)}')
        print(f'previous_task: {(max(df[acctask_columns[-len(accmean_columns):-1]].mean(axis=1)) - min(df[acctask_columns[-len(accmean_columns):-1]].mean(axis=1)) ) / 2}')

        print(f'current_task: {df[acctask_columns[-1]].mean(axis=0)}')
        print(f'current_task_pm: {(max(df[acctask_columns[-1]]) - min(df[acctask_columns[-1]]) ) / 2}')

        print(f'overall_pm: {df[accmean_columns[-1]].mean(axis=0)}')
        print(f'overall: {(max(df[accmean_columns[-1]]) - min(df[accmean_columns[-1]]) ) / 2}')

def figure9():
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
        # Create the dictionary with the updated assignments
        palette = sns.color_palette("tab10", 3)  # Requesting 6 colors to access cyan

        datasets = ['seq-cifar100','seq-tinyimg', 'seq-cub200']
        model = 'er_buf'
        training_settings = ['task-il', 'class-il']
        fig, axes = plt.subplots(1, 6, figsize=(15, 3), dpi=800)

        total = 0
        
        for row, c_dataset in enumerate(datasets):
                for col  in range(len(training_settings)):               
                        dictlist, _ = get_data(path, training_settings[col], c_dataset, model, "/logs_NC.txt")
                        arguments_base, _, _ = get_arguments(dictlist[0])
                        tasks, within_var, _ = get_arrays(dictlist[0], 'within_var')
                        _, between_var, _ = get_arrays(dictlist[0], 'between_var')
                        _, features_norm, _ = get_arrays(dictlist[0], 'features_norm')
                        between_var_overall, _, _ = get_arrays(dictlist[0], pre_string2='between_var_overall_task')
                        between_var_current, _, _ = get_arrays(dictlist[0], pre_string2='between_var_current_task')
                        between_var_previous, _, _ = get_arrays(dictlist[0], pre_string2='between_var_previous_task')
                        _, cos_movement, _ = get_arrays(dictlist[0], 'cos_movement')
                        _, cos_mean_distance, _ = get_arrays(dictlist[0], 'cos_mean_distance')
                        df_nc = get_dataframe(dictlist, arguments_base + ['seed'] + within_var + between_var + between_var_overall + between_var_current + between_var_previous + cos_movement + features_norm + cos_mean_distance)
                        df_nc = df_nc[df_nc['result_type'].isin(list(['train_dataset']))]
                        
                        if c_dataset == 'seq-cifar100':
                                df_nc = df_nc[df_nc['buffer_size']<=2000]
                        elif c_dataset == 'seq-tinyimg':
                                df_nc = df_nc[df_nc['buffer_size']<=5000]
                        else:
                                df_nc = df_nc[df_nc['buffer_size']<=800]
                        
                        dictlist, _ = get_data(path, training_settings[col], c_dataset, model, "/logs.txt")
                        arguments_base, arguments_keep, arguments_combine = get_arguments(dictlist[0])
                        accmean_columns, acctask_columns, result_accuracy = get_arrays(dictlist[0])

                        df_acc = get_dataframe(dictlist, arguments_base + [result_accuracy] + ["seed"] + acctask_columns)
                        df_acc = df_acc[df_acc['result_type'].isin(list(['features_til']))]
                        
                        if c_dataset == 'seq-cifar100':
                                df_nc = df_nc[df_nc['buffer_size']<=2000]
                        elif c_dataset == 'seq-tinyimg':
                                df_nc = df_nc[df_nc['buffer_size']<=5000]
                        else:
                                    df_nc = df_nc[df_nc['buffer_size']<=800]
                        
                        for i in range(1, 11):
                                for j in range(1, i):
                                        df_acc[f'forgetting_{j}_task{i}'] = df_acc[f'accuracy_{j}_task{j}'] - df_acc[f'accuracy_{j}_task{i}']
                        
                        
                        ax = axes[total] #attention

                        results = []
                        
                        df_merged = pd.merge(df_nc, df_acc, on=['seed', 'buffer_size', 'model', 'lr', 'dataset'], how='inner')

                        ax = axes[total] #attention

                        results = []
                        for i in range(1, 11):
                                for j in range(1, i):
                                        current_result = pd.DataFrame({
                                                'training_task': i, 'evaluating_task': j, 'task_delta': i-j,
                                                'buffer_size': df_merged['buffer_size'], 
                                                'NC': (df_merged[f'within_var_{j}_task{i}']/df_merged[f'between_var_{j}_task{i}']) , 
                                                'NC_delta': (df_merged[f'within_var_{j}_task{i}']/df_merged[f'between_var_{j}_task{i}']- df_merged[f'within_var_{j}_task{j}']/df_merged[f'between_var_{j}_task{j}']) ,
                                                'avg_accuracy': df_merged[result_accuracy]/100, 
                                                'forgetting': df_merged[f'forgetting_{j}_task{i}'],#/df_merged[result_accuracy], 
                                                'seed': df_merged['seed']})
                                        results.append(current_result) 


                        results = pd.concat(results)
                        # sns.lineplot(data=results, x='training_task', y='accuracy', hue='evaluating_task', style='result_type', marker='o', legend=True, ax=ax, palette=palette)
                        sns.regplot(data=results, x='forgetting', y='NC_delta', ax=ax, scatter=False, color='red', line_kws={"linestyle": "--"})
                        sns.scatterplot(data=results, x='forgetting', y='NC_delta', ax=ax,  zorder=0, marker='o', edgecolor='white', s=50, color=palette[row], alpha=0.5)
                        
                        ax.set_title(f'{plotting_name(c_dataset)} ({plotting_name(training_settings[col])})')
                        ax.set_xlabel("Feature Forgetting [%]")
                        if total==0:
                                ax.set_ylabel(f"Neural Inflation")
                        else:
                                ax.set_ylabel("")

                        # Add dotted grid lines
                        ax.grid(True, linestyle='--', linewidth=1.0, axis='y', zorder=2)

                        # Remove the boxes around each plot
                        ax.spines['top'].set_visible(False)
                        ax.spines['right'].set_visible(False)
                        ax.spines['left'].set_visible(False)
                        ax.spines['bottom'].set_visible(False)

                        # ax.legend_.remove()
                        total += 1
                        
        # labels = ["1", "3", "5", "7", "9"]
        # legend_handles = [mpatches.Patch(color=palette[i]) for i in range(5)]
        #fig.legend(legend_handles, labels, loc='center', bbox_to_anchor=(0.5, 0.05), ncol=6)
        handles, labels = axes[0].get_legend_handles_labels()
        labels = ['Task 1','Previous Tasks Average', 'Current Task']
        # fig.legend(handles, labels, bbox_to_anchor=(0.9, 0.1), ncol=3)
        fig.tight_layout()
        fig.savefig(path + f"/figure9.pdf", dpi=800)
        fig.clf() 

def table2():
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
        # Create the dictionary with the updated assignments
        palette = sns.color_palette("coolwarm", 2)  # Requesting 6 colors to access cyan

        datasets = {'seq-cifar100': [500], 'seq-tinyimg': [1000], 'seq-cub200': [200]}
        model = 'er_bounds'
        training_settings = ['task-il', 'class-il']
        fig, axes = plt.subplots(1, 6, figsize=(12, 3), dpi=800)
        #plt.rcParams['text.usetex'] = True # TeX rendering

        total = 0
        
        for row, (c_dataset, buffer_sizes) in enumerate(datasets.items()):
                for col, current_setting  in enumerate(training_settings):
                        dictlist, _ = get_data(path, training_settings[col], c_dataset, model, "/logs.txt")
                        arguments_base, _, _ = get_arguments(dictlist[0])

                        accmean, acctask, _ = get_arrays(dictlist[0])
                        df = get_dataframe(dictlist, arguments_base + ['seed'] + accmean + acctask)
                        df = df[df['result_type'].isin(list(['features_til']))]

                        
                        buffer_size = buffer_sizes[0]
                        ax = axes[total] #attention
                        current_df = df[df['buffer_size'] == buffer_size]

                        results = []
                        for i in range(1, len(accmean) + 1):
                                for j in range(1, i+1):
                                        if current_setting == "class-il":
                                                current_result = pd.DataFrame({'training_task': i, 'evaluating_task': j, 
                                                                'accuracy': (current_df[f'accuracy_{j}_task{i}']), 'result_type': current_df['result_type'], 'task_type':j==i})
                                        else: 
                                                current_result = pd.DataFrame({'training_task': i, 'evaluating_task': j, 
                                                                'accuracy': (current_df[f'accuracy_{j}_task{i}']), 'result_type': current_df['result_type'], 'task_type':j==i})
                                        results.append(current_result) 

                        results = pd.concat(results)
                        # sns.lineplot(data=results, x='training_task', y='accuracy', hue='evaluating_task', style='result_type', marker='o', legend=True, ax=ax, palette=palette)
                        sns.barplot(data=results, x='training_task', y='accuracy', hue='task_type', ax=ax, errorbar=('ci', 95), errwidth=2, palette=palette, zorder=0, edgecolor='black', dodge=False)
                        
                        bars = ax.patches
                        # Group bars by x position
                        bars_by_x = {}
                        for bar in bars:
                                x_pos = round(bar.get_x(), 5)  # Round to avoid floating point errors
                                if x_pos not in bars_by_x:
                                        bars_by_x[x_pos] = []
                                bars_by_x[x_pos].append(bar)

                        # Sort and adjust z-order within each x-group
                        for x_pos, bars_group in bars_by_x.items():
                                bars_group.sort(key=lambda bar: bar.get_height())  # Sort by height
                                for i, bar in enumerate(bars_group):
                                        bar.set_zorder(-i)  # Lower bars get lower z-order

                        ax.set_title(f'{plotting_name(c_dataset)} ({plotting_name(training_settings[col])})')
                        ax.set_xlabel("Training Task")
                        if total==0:
                                ax.set_ylabel(f"Test Accuracy [%]")
                        else:
                                ax.set_ylabel("")

                        ax.set_ylim(50, 100)
                        # Add dotted grid lines
                        ax.grid(True, linestyle='--', linewidth=1.0, axis='y', zorder=1)

                        # Remove the boxes around each plot
                        ax.spines['top'].set_visible(False)
                        ax.spines['right'].set_visible(False)
                        ax.spines['left'].set_visible(False)
                        ax.spines['bottom'].set_visible(False)
                        
                        ax.legend_.remove()
                        total += 1
                        
        # labels = ["1", "3", "5", "7", "9"]
        # legend_handles = [mpatches.Patch(color=palette[i]) for i in range(5)]
        #fig.legend(legend_handles, labels, loc='center', bbox_to_anchor=(0.5, 0.05), ncol=6)
        handles, _ = axes[0].get_legend_handles_labels()
        labels = ['Previous Tasks','Current Task']
        fig.legend(handles, labels, bbox_to_anchor=(0.65, 0.11), ncol=2)
        fig.tight_layout(rect=[0, 0.05, 1, 1])
        fig.savefig(path + f"/table2.pdf", dpi=800)
        fig.clf() 

def figure13():
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
        # Create the dictionary with the updated assignments
        palette = sns.color_palette("coolwarm", 2)  # Requesting 6 colors to access cyan
        #palette = ['palegreen', 'palegreen']

        datasets = {'seq-cifar100': [1000], 'seq-tinyimg': [2000]}#, 'seq-cub200': [200]}
        model = 'er_buf'
        training_settings = ['task-il', 'class-il']
        fig, axes = plt.subplots(1, 4, figsize=(8, 3), dpi=800)


        total = 0
        
        for row, (c_dataset, buffer_sizes) in enumerate(datasets.items()):
                for col, current_setting  in enumerate(training_settings):
                        dictlist, _ = get_data(path, training_settings[col], c_dataset, model, "/logs_NC.txt")
                        arguments_base, _, _ = get_arguments(dictlist[0])

                        tasks, within_var, _ = get_arrays(dictlist[0], 'within_var')
                        _, between_var, _ = get_arrays(dictlist[0], 'between_var')
                        _, features_norm, _ = get_arrays(dictlist[0], 'features_norm')
                        between_var_overall, _, _ = get_arrays(dictlist[0], pre_string2='between_var_overall_task')
                        between_var_current, _, _ = get_arrays(dictlist[0], pre_string2='between_var_current_task')
                        between_var_previous, _, _ = get_arrays(dictlist[0], pre_string2='between_var_previous_task')
                        _, cos_movement, _ = get_arrays(dictlist[0], 'cos_movement')
                        _, cos_mean_distance, _ = get_arrays(dictlist[0], 'cos_mean_distance')
                        _, acc, _ = get_arrays(dictlist[0])
                        df = get_dataframe(dictlist, acc + arguments_base + ['seed'] + within_var + between_var + between_var_overall + between_var_current + between_var_previous + cos_movement + features_norm + cos_mean_distance)
                        df = df[df['result_type'].isin(list(['buffer']))]

                        
                        buffer_size = buffer_sizes[0]
                        ax = axes[total] #attention
                        current_df = df[df['buffer_size'] == buffer_size]

                        results = []
                        for i in range(1, len(tasks) + 1):
                                for j in range(1, i+1): #attention
                                        if current_setting == "class-il":
                                                current_result = pd.DataFrame({'training_task': i, 'evaluating_task': j, 
                                                                'accuracy': current_df[f'between_var_{j}_task{i}'], 'result_type': current_df['result_type'], 'task_type':j==i})
                                        else: 
                                                current_result = pd.DataFrame({'training_task': i, 'evaluating_task': j, 
                                                                'accuracy': current_df[f'between_var_{j}_task{i}'], 'result_type': current_df['result_type'], 'task_type':j==i})
                                        results.append(current_result) 

                        results = pd.concat(results)
                        # sns.lineplot(data=results, x='training_task', y='accuracy', hue='evaluating_task', style='result_type', marker='o', legend=True, ax=ax, palette=palette)
                        sns.barplot(data=results, x='training_task', y='accuracy', hue='task_type', ax=ax, errorbar=('ci', 95), errwidth=2, palette=palette, zorder=0, edgecolor='black', dodge=False, legend=True)
                        
                        bars = ax.patches
                        # Group bars by x position
                        bars_by_x = {}
                        for bar in bars:
                                x_pos = round(bar.get_x(), 5)  # Round to avoid floating point errors
                                if x_pos not in bars_by_x:
                                        bars_by_x[x_pos] = []
                                bars_by_x[x_pos].append(bar)

                        # Sort and adjust z-order within each x-group
                        for x_pos, bars_group in bars_by_x.items():
                                bars_group.sort(key=lambda bar: bar.get_height())  # Sort by height
                                for i, bar in enumerate(bars_group):
                                        bar.set_zorder(-i)  # Lower bars get lower z-order

                        ax.set_title(f'{plotting_name(c_dataset)} ({plotting_name(training_settings[col])})')
                        ax.set_xlabel("Training Task")
                        if total==0:
                                ax.set_ylabel(f"Test Accuracy [%]")
                        else:
                                ax.set_ylabel("")

                        # Add dotted grid lines
                        ax.grid(True, linestyle='--', linewidth=1.0, axis='y', zorder=2)

                        # Remove the boxes around each plot
                        ax.spines['top'].set_visible(False)
                        ax.spines['right'].set_visible(False)
                        ax.spines['left'].set_visible(False)
                        ax.spines['bottom'].set_visible(False)

                        ax.legend_.remove()
                        total += 1

        for i in [0, 2]:#, 4]:
                y_min = min(axes[i].get_ylim()[0], axes[i+1].get_ylim()[0])
                y_max = max(axes[i].get_ylim()[1], axes[i+1].get_ylim()[1])
                axes[i].set_ylim(y_min, y_max)
                axes[i+1].set_ylim(y_min, y_max)           

        handles, _ = axes[0].get_legend_handles_labels()
        labels = ['Previous Tasks','Current Task']
        fig.legend(handles, labels, bbox_to_anchor=(0.7, 0.11), ncol=2)
        fig.tight_layout(rect=[0, 0.05, 1, 1])
        fig.savefig(path + f"/figure13.pdf", dpi=800)
        fig.clf() 

figure13()
def output_vs_feature(): #figure 2
        #sns.set_theme(style="whitegrid")
        colors = sns.color_palette("deep")  # Requesting 6 colors to access cyan

        # Create the dictionary with the updated assignments
        color_dict = {
        "ER": colors[0],  # blue
        "ER-OR": colors[9],  # cyan
        "DER": colors[2],  # green
        "FDR": colors[4],  # violet
        "iCaRL": colors[1],  # orange
        "JOINT": (0, 0, 0)  # black
        }
        datasets = ['seq-cub200']
        training_setting = ['class-il', 'task-il']
        models = ['er_buf']

        fig, axes = plt.subplots(1, len(datasets) * len(training_setting), figsize=(14, 3.5), dpi=800)
        for i, setting in enumerate(list(itertools.product(datasets, training_setting))):
                data_buffer = []
                data_nobuffer = []

                if setting[0] == 'seq-cifar10':
                        infinite_size = 2500
                        gap = [2200, 2300]
                elif setting[0] == 'seq-cifar100':
                        infinite_size = 6000
                        gap = [5400, 5600]
                elif setting[0] == 'seq-tinyimg':
                        infinite_size = 12000
                        gap = [10800, 11200]
                elif setting[0] == 'seq-cub200':
                        infinite_size = 2500

                for model in models:
                        dictlist, filepath = get_data(path, setting[1], setting[0], model)
                        arguments_base, arguments_keep, arguments_combine = get_arguments(dictlist[0])
                        accmean_columns, acctask_columns, result_accuracy = get_arrays(dictlist[0])

                        df = get_dataframe(dictlist, arguments_base + [result_accuracy])
                        if model in ['er_buf']:
                                if setting[1] == "class-il":
                                        features_version = "features_cil"
                                else:
                                        features_version = "features_til"
                                df = df[df['result_type'].isin(list(['output', features_version]))]
                                df.loc[df['result_type'] == features_version, 'model'] = 'ER-OR'
                                buffer_sizes = df['buffer_size'].sort_values().tolist()
                                df["buffer_size"] = df["buffer_size"].replace(buffer_sizes[-1], infinite_size)
                        elif model in ['der', 'fdr']:
                                df = df[df['result_type'].isin(list(['output']))]
                                buffer_sizes = df['buffer_size'].sort_values().tolist()
                                df["buffer_size"] = df["buffer_size"].replace(buffer_sizes[-1], infinite_size)
                        else:
                                df = df[df['result_type'].isin(list(['output']))]

                        df['model'] = df.apply(lambda row: plotting_name(row['model']), axis=1)
                        print(df)
                        if 'buffer_size' in arguments_base:
                                df = df[df['buffer_size'] < 50000]

                                data_buffer.append(df)
                        else:
                                data_nobuffer.append(df)   
                
                data_buffer = pd.concat(data_buffer)

                #small values
                sns.lineplot(data=data_buffer, x='buffer_size', y=result_accuracy, hue='model',  marker='o', style='result_type', legend=False, palette=color_dict, ax=axes[i])

                # Customize the grid  
                axes[i].grid(visible=True, which='major', axis='x', linestyle=':', linewidth=0.5, color='grey')  # Dashed grid for x-axis
                axes[i].grid(visible=True, which='major', axis='y', linestyle='--', linewidth=0.5, color='grey')     

                axes[i].set_title(f'{plotting_name(setting[0])}, {plotting_name(setting[1])}-head')
                axes[i].set_xlabel("Buffer Size")


                if i==0:
                        axes[i].set_ylabel("Test Accuracy [%]")
                else:
                        axes[i].set_ylabel("")
        
        legend_handles = []
        for label, color in color_dict.items():
                if label == "JOINT":
                        legend_handles.append(mlines.Line2D([0], [0], color=color, label=label))
                elif label == "ER-OR":
                        legend_handles.append(mlines.Line2D([0], [0], color=color, marker='o', linestyle=':', markersize=4, label=label))
                else:
                        legend_handles.append(mlines.Line2D([0], [0], color=color, marker='o', markersize=4, label=label))
        labels = color_dict.keys()

        fig.legend(legend_handles, labels, loc='center', bbox_to_anchor=(0.5, 0.05), ncol=6)
        fig.tight_layout(rect=[0, 0.05, 1, 1])
        plt.savefig(path + f"/Output_vs_Features.pdf", dpi=800)
        plt.clf() 

def output_vs_features2():
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
        # Create the dictionary with the updated assignments
        colors = sns.color_palette("viridis", 6)  # Requesting 6 colors to access cyan
        #colors = sns.color_palette("mako", 4)
        color_dict = {
        #"ER": colors[1],  # blue
        #"ER-OR": "black",  # cyan
        #"ER-BUF": colors[3],  # orange
        #"DER": colors[2],  # green
        #"DER-OR": "black",  # green
        #"FDR": colors[3],  # violet
        #"FDR-OR": "black",  # violet
        "iCaRL": colors[4],  # orange
        "iCaRL-OR": "black",  # orange
        "JOINT": "silver"  # black
        }
        datasets = ['seq-cifar100', 'seq-tinyimg']
        training_setting = ['task-il','class-il']
        models = ['sgd', 'icarl']#, 'der', 'fdr', 'icarl']

        fig, axes = plt.subplots(1, len(datasets) * len(training_setting), figsize=(8, 3), dpi=800)
        for i, setting in enumerate(list(itertools.product(training_setting, datasets))):
                data_buffer = []
                data_nobuffer = []

                if setting[1] == 'seq-cifar10':
                        infinite_size = 2500
                        gap = [2200, 2300]
                elif setting[1] == 'seq-cifar100':
                        infinite_size = 6000
                        gap = [5400, 5600]
                elif setting[1] == 'seq-tinyimg':
                        infinite_size = 12000
                        gap = [10800, 11200]

                for model in models:
                        dictlist, filepath = get_data(path, setting[0], setting[1], model)
                        arguments_base, arguments_keep, arguments_combine = get_arguments(dictlist[0])
                        accmean_columns, acctask_columns, result_accuracy = get_arrays(dictlist[0])

                        df = get_dataframe(dictlist, arguments_base + [result_accuracy])
                        if model in ['er_bounds']:
                                df = df[df['result_type'].isin(list(['output', 'features', 'buffer']))]
                                df.loc[df['result_type'] == 'features', 'model'] = 'ER-OR'
                                df.loc[df['result_type'] == 'buffer', 'model'] = 'ER-BUF'
                                buffer_sizes = df['buffer_size'].sort_values().tolist()
                                df["buffer_size"] = df["buffer_size"].replace(buffer_sizes[-1], infinite_size)
                        elif model in ['der', 'fdr', 'icarl']:
                                df = df[df['result_type'].isin(list(['output', 'features']))]
                                df.loc[df['result_type'] == 'features', 'model'] = 'iCaRL-OR'
                                buffer_sizes = df['buffer_size'].sort_values().tolist()
                                #df["buffer_size"] = df["buffer_size"].replace(buffer_sizes[-1], infinite_size)
                                df = df[df['buffer_size'] > 0]
                        else:
                                df = df[df['result_type'].isin(list(['output']))]

                        df['model'] = df.apply(lambda row: plotting_name(row['model']), axis=1)
                        if 'buffer_size' in arguments_base:
                                df = df[df['buffer_size'] < 50000]

                                data_buffer.append(df)
                        else:
                                data_nobuffer.append(df)   
                
                data_buffer = pd.concat(data_buffer)
                data_nobuffer = pd.concat(data_nobuffer)

                all_buffer_sizes = data_buffer['buffer_size'].unique()
                df_buffer_sizes = pd.DataFrame({'buffer_size': all_buffer_sizes})
                data_nobuffer = data_nobuffer.merge(df_buffer_sizes, how='cross')

                data_buffer_rest = data_buffer
                #data_buffer_OR = data_buffer[data_buffer['model'] == 'ER-OR']
                #data_buffer_rest = data_buffer[data_buffer['model'] != 'ER-OR']



                sns.lineplot(data=data_buffer_rest, x='buffer_size', y=result_accuracy, hue='model',  marker='o',  markersize=5, linestyle='--', legend=False, palette=color_dict, ax=axes[i])
                sns.lineplot(data=data_nobuffer, x='buffer_size', y=result_accuracy, hue='model', marker='',  linestyle='-', legend=False, palette=color_dict, ax=axes[i], linewidth=0.5)
                #sns.lineplot(data=data_buffer_OR, x='buffer_size', y=result_accuracy, hue='model',  marker='D',  markersize=5, linestyle=':', legend=False, palette=color_dict, ax=axes[i])



                # Remove the boxes around each plot
                axes[i].spines['top'].set_visible(False)
                axes[i].spines['right'].set_visible(False)
                axes[i].spines['left'].set_visible(False)
                axes[i].spines['bottom'].set_visible(False)


                y_min = min(data_buffer[result_accuracy].min(), data_nobuffer[result_accuracy].min(), data_buffer_rest[result_accuracy].min())
                y_max = max(data_buffer[result_accuracy].max(), data_nobuffer[result_accuracy].max(), data_buffer_rest[result_accuracy].max())
                axes[i].set_ylim(y_min - 5, y_max)

                current_yticks = axes[i].get_yticks()
                #y_min, y_max = axes[i].get_ylim()  # Store exact limits
                axes[i].set_yticks(current_yticks)  # Freeze ticks
                #axes[i].set_ylim((y_min, y_max))  # Reset exact limits

                if setting[1] == 'seq-cifar10':
                        xticks = [0, 200, 500, 1000, 1500, 2000, 2500]
                elif setting[1] == 'seq-cifar100':
                        xticks = [0, 1000, 2000, 3000, 4000, 5000, 6000]
                elif setting[1] == 'seq-tinyimg':
                        xticks = [0, 2000, 4000, 6000, 8000, 10000, 12000]
                
                axes[i].set_xticks(xticks)  # Ensure tick positions remain the same
                xticklabels = [str(int(tick)) if tick != max(xticks) else 'inf' for tick in xticks]  # Replace max tick with 'inf'
                axes[i].set_xticklabels(xticklabels)  # Set the new labels

                #Make the fontsize of the axis smaller
                for tick in axes[i].get_xticklabels():
                        tick.set_rotation(45)
                axes[i].tick_params(axis='both', which='major', labelsize=8)
                
                # # Optional: Add a small gap visual indicator
                # #axes[i].axvline(x=gap[0], color='grey', linestyle='--', lw=1, alpha=0.7)
                # #axes[i].axvline(x=gap[1], color='grey', linestyle='--', lw=1, alpha=0.7)
                # #y_pos = axes[i].get_ylim()[0] - (axes[i].get_ylim()[1] - axes[i].get_ylim()[0]) * 0.05  # Small offset below x-axis
                # axes[i].text((gap[0] + gap[1])//2, y_min, "...", fontsize=14, ha='center', va='center', color='black', zorder=4)
                # axes[i].text((gap[0] + gap[1])//2, y_max, "", fontsize=14, ha='center', va='center', color='black', zorder=4)

                # axes[i].fill_betweenx((y_min+1, y_max-1), gap[0], gap[1], color='white', zorder=3)
                
                # Customize the grid  
                axes[i].grid(visible=True, which='major', axis='x', linestyle=':', linewidth=0.5, color='lightgrey')  # Dashed grid for x-axis
                axes[i].grid(visible=True, which='major', axis='y', linestyle='--', linewidth=0.5, color='lightgrey')     

                axes[i].set_title(f'{plotting_name(setting[1])} ({plotting_name(setting[0])})', fontsize=10)
                axes[i].set_xlabel("Buffer Size")



                if i==0:
                        axes[i].set_ylabel("Test Accuracy [%]", fontsize=8)
                else:
                        axes[i].set_ylabel("")

                
                

                
        legend_handles = []
        for label, color in color_dict.items():
                if label == "JOINT":
                        legend_handles.append(mlines.Line2D([0], [0], color=color, label=label))
                #elif label == "ER-OR":
                #        legend_handles.append(mlines.Line2D([0], [0], color=color, marker='o', linestyle=':', markersize=4, label=label))
                else:
                        legend_handles.append(mlines.Line2D([0], [0], color=color, marker='o', markersize=4, label=label))
        labels = color_dict.keys()
        # Move ER-OR to the first position in the legend
        legend_handles =  [legend_handles[1]] + legend_handles[:1] + legend_handles[2:]
        labels =  ['iCaRL-OR'] + [label for label in labels if label != 'iCaRL-OR']
        fig.legend(legend_handles, labels, loc='center', bbox_to_anchor=(0.2, 0.45), ncol=1, title_fontsize=8, fontsize=8)
        fig.tight_layout()
        plt.savefig(path + f"/Output_vs_Features.pdf", dpi=800)
        plt.clf() 

def buffersize_vs_nc():
        sns.set_theme(style="whitegrid")
        #color_mapping = {"accuracy": palette[2], "train_dataset": palette[1], "buffer": palette[0]}
        line_mapping = {'Current Task': (1, 0), 'Previous Tasks': (5, 2)}
        datasets = ['seq-cifar10', 'seq-tinyimg']
        training_setting = ['class-il', 'task-il']
        model = 'er_bounds'
        fig, axes = plt.subplots(1, len(datasets)*len(training_setting), figsize=(16, 4))

        for i, (ax, setting) in enumerate(zip(axes, list(itertools.product(datasets, training_setting)))):
                dictlist, filepath = get_data(path, setting[1], setting[0], model, '/logs.txt')
                arguments_base, arguments_keep, arguments_combine = get_arguments(dictlist[0])
                accmean_columns, acctask_columns, result_accuracy = get_arrays(dictlist[0])

                df_accuracy = get_dataframe(dictlist, arguments_base + acctask_columns[-len(accmean_columns):])
                df_accuracy = df_accuracy[df_accuracy['buffer_size'] < 50000]
                df_accuracy = df_accuracy[df_accuracy['result_type'].isin(['buffer'])]
                df_accuracy['Current Task'] = df_accuracy[acctask_columns[-1]]
                df_accuracy['Previous Tasks'] = df_accuracy[acctask_columns[-3]]
                df_accuracy = pd.melt(df_accuracy, id_vars=arguments_base, value_vars=['Current Task', 'Previous Tasks'], var_name='task', value_name='accuracy')

                dictlist, filepath = get_data(path, setting[1], setting[0], model, '/logs_NC.txt')
                _, within_var, _ = get_arrays(dictlist[0], 'within_var')
                _, cos_distance, _ = get_arrays(dictlist[0], 'cos_distance')

                df_nc = get_dataframe(dictlist, arguments_base + within_var[-len(accmean_columns):] + within_var[-len(accmean_columns):])
                df_nc = df_nc[df_nc['buffer_size'] < 50000]
                df_nc['Current Task'] = df_nc[within_var[-1]]
                df_nc['Previous Tasks'] = df_nc[within_var[-4]]
                df_nc = pd.melt(df_nc, id_vars=arguments_base, value_vars=['Current Task', 'Previous Tasks'], var_name='task', value_name='NC')
                

                sns.lineplot(data=df_nc, x='buffer_size', y='NC', hue='result_type', style='task', marker='o', dashes=line_mapping, legend=(i==0))#, ax=ax, palette=color_mapping)

                ax2 = ax.twinx()
                sns.lineplot(data=df_accuracy, x='buffer_size', y='accuracy', style='task',  marker='o', dashes=line_mapping, legend=(i==0), ax=ax2)#, color=color_mapping['accuracy'])   
                
                #if i==0:
                #        handles, labels = ax.get_legend_handles_labels()
                #        ax.legend_.remove()

                ax.set_title(plotting_name(setting[0]) + ' (' + plotting_name(setting[1]) + ')')
                ax.set_xlabel("Buffer Size")
                #if i==0:
                        #ax.set_ylabel("NC")
                        #ax2.set_ylabel('Test Accuracy', color=color_mapping['accuracy'])
                        #ax2.tick_params(axis='y', labelcolor=color_mapping['accuracy'])
                #else:
                        #ax.set_ylabel("")
                        #ax2.set_ylabel('')
        
        #fig.legend(handles, labels, loc='center', bbox_to_anchor=(0.5, 0.05), ncol=5)
        fig.tight_layout(rect=[0, 0.05, 1, 1])
        fig.savefig(path + f"/BufferSize_vs_NC_TIL.png")
        fig.clf() 

def NC_sgd(): #figure 3
        palette = sns.color_palette("mako_r")
        datasets = {'seq-cifar100': [2000]}#, 'seq-tinyimg': [5000]}
        model = 'er_bounds'
        training_settings = ['class-il']
        fig, axes = plt.subplots(1, 2, figsize=(16, 4.2), dpi=800)
        #plt.rcParams['text.usetex'] = True # TeX rendering

        for k in range(len(training_settings)):
                for row, (c_dataset, buffer_sizes) in enumerate(datasets.items()):
                        dictlist, _ = get_data(path, training_settings[k], c_dataset, model, "/logs_NC.txt")
                        arguments_base, _, _ = get_arguments(dictlist[0])
                        tasks, within_var, _ = get_arrays(dictlist[0], 'within_var')
                        _, between_var, _ = get_arrays(dictlist[0], 'between_var')
                        _, cos_movement, _ = get_arrays(dictlist[0], 'cos_movement')
                        _, cos_mean_distance, _ = get_arrays(dictlist[0], 'cos_mean_distance')
                        df = get_dataframe(dictlist, arguments_base + ['seed'] + within_var + between_var + cos_movement + cos_mean_distance)
                        df = df[df['result_type'].isin(list(['train_dataset']))]

                        for col, buffer_size in enumerate(buffer_sizes):
                                ax = axes[row + col] #attention
                                current_df = df[df['buffer_size'] == buffer_size]

                                results = []
                                for i in range(1, len(tasks) + 1):
                                        for j in range(1, i+1):
                                                current_result = pd.DataFrame({'training_task': i, 'evaluating_task': j, 
                                                                        'accuracy': (current_df[f'cos_mean_distance_{j}_task{i}']), 'result_type': current_df['result_type']})
                                                results.append(current_result) 

                                results = pd.concat(results)
                                sns.lineplot(data=results, x='training_task', y='accuracy', hue='evaluating_task', style='result_type', marker='o', legend=True, ax=ax, palette=palette)
                                #sns.barplot(data=results, x='training_task', y='accuracy', hue='evaluating_task', ax=ax, errorbar=('ci', 95), err_kws={'linewidth': 2}, palette=palette, zorder=2, legend=False)
                                
                                ax.set_title(f'{plotting_name(c_dataset)}, {plotting_name(training_settings[k])}-head')
                                ax.set_xlabel("Training Task")
                                if col==0:
                                        ax.set_ylabel(f"NC")
                                else:
                                        ax.set_ylabel("")

                                # Add dotted grid lines
                                ax.grid(True, linestyle='--', linewidth=0.5, axis='y', zorder=0)
                        
        labels = ["1", "3", "5", "7", "9"]
        legend_handles = [mpatches.Patch(color=palette[i]) for i in range(5)]
        #fig.legend(legend_handles, labels, loc='center', bbox_to_anchor=(0.5, 0.05), ncol=6)
        fig.legend(legend_handles, labels, loc='center', bbox_to_anchor=(0.5, 0.05), ncol=5)
        fig.tight_layout(rect=[0, 0.05, 1, 1])
        fig.savefig(path + f"/NC_sgd.pdf", dpi=800)
        fig.clf() 

def NC_fixed_buffer(): #figure 4
        datasets = {'seq-cub200': [1000]}#, 'seq-tinyimg': [5000]}
        model = 'er_bounds'
        training_settings = ['class-il', 'task-il']
        fig, axes = plt.subplots(1, 4, figsize=(14, 3.5), dpi=800)

        
        for col1, (c_dataset, buffer_sizes) in enumerate(datasets.items()):
                for k in range(len(training_settings)):
                        dictlist, _ = get_data(path, training_settings[k], c_dataset, model, "/logs_NC.txt")
                        arguments_base, _, _ = get_arguments(dictlist[0])
                        tasks, within_var, _ = get_arrays(dictlist[0], 'within_var')
                        _, between_var, _ = get_arrays(dictlist[0], 'between_var')
                        _, l2_distance, _ = get_arrays(dictlist[0], 'l2_distance')
                        _, cos_distance, _ = get_arrays(dictlist[0], 'cos_distance')
                        df = get_dataframe(dictlist, arguments_base + ['seed'] + within_var + between_var + l2_distance + cos_distance)
                        df = df[df['result_type'].isin(list(['buffer', 'train_dataset']))]

                        for col2, buffer_size in enumerate(buffer_sizes):
                                ax = axes[k+ 2*col1 + col2] #attention
                                current_df = df[df['buffer_size'] == buffer_size].copy()
                                
                                '''
                                for i in range(1, len(tasks) + 1):
                                        for j in range(1, i + 1):
                                                current_df[f'within_var_{j}_task{i}'] = current_df[f'within_var_{j}_task{i}'] / current_df[f'between_var_{j}_task{i}']
                                '''
                                #Split buffer and train rows
                                buffer_df = current_df[current_df['result_type'] == 'buffer'].set_index('seed')
                                train_df = current_df[current_df['result_type'] == 'train_dataset'].set_index('seed')

                                # Drop the 'result' column before subtraction
                                buffer_df = buffer_df.drop(columns=arguments_base)
                                train_df = train_df.drop(columns=arguments_base)

                                current_df = (train_df - buffer_df)

                                results = []
                                for i in range(0, 5):
                                        to_average = []
                                        for j in range(1, 10+1-i):
                                                print(f'{j}_{j+i}')
                                                to_average.append(f'within_var_{j}_task{j+i}')
                                        current_result = pd.DataFrame({'tasks_afterwards': i, 'var_mean': (current_df[to_average].mean(axis=0))})
                                        results.append(current_result) 

                                print('yaaaaaaaaaaaaaaaaa')
                                results = pd.concat(results)
                                sns.barplot(data=results, x='tasks_afterwards', y='var_mean', ax=ax, errorbar=('ci', 95), err_kws={'linewidth': 2, 'color': 'black'}, zorder=2, color=sns.color_palette("deep")[0])

                                current_yticks = ax.get_yticks()[1:][::2]
                                y_min, y_max = ax.get_ylim()  # Store exact limits

                                ax.set_yticks(current_yticks)  # Freeze ticks
                                ax.set_ylim((y_min, y_max))  # Reset exact limits
                                
                                ax.set_title(f'{plotting_name(c_dataset)}, {plotting_name(training_settings[k])}-head')
                                ax.set_xlabel("Tasks passed")
                                if k+col1+col2==0:
                                        ax.set_ylabel("Within Class Variance")
                                else:
                                        ax.set_ylabel("")
                                ax.grid(True, linestyle='--', linewidth=0.5, axis='y', zorder=0)

        plt.tight_layout()             
        fig.savefig(path + f"/NC_fixed_buffer.pdf", dpi=800)
        fig.clf() 

def NC_fixed_buffer2():
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
        # Create the dictionary with the updated assignments
        palette = sns.color_palette("coolwarm", 2)  # Requesting 6 colors to access cyan

        datasets = {'seq-cifar100': [2000], 'seq-tinyimg': [5000], 'seq-cub200': [500]}
        model = 'er_bounds'
        training_settings = ['task-il', 'class-il']
        fig, axes = plt.subplots(1, 6, figsize=(12, 3), dpi=800)
        #plt.rcParams['text.usetex'] = True # TeX rendering

        total = 0
        for col  in range(len(training_settings)):
                for row, (c_dataset, buffer_sizes) in enumerate(datasets.items()):
                        if c_dataset == 'seq-cub200': 
                                path = base_path() + "results_paper2/"
                        else: 
                                path = base_path() + "results_paper2/"

                        dictlist, _ = get_data(path, training_settings[col], c_dataset, model, "/logs_NC.txt")
                        arguments_base, _, _ = get_arguments(dictlist[0])
                        tasks, within_var, _ = get_arrays(dictlist[0], 'within_var')
                        _, between_var, _ = get_arrays(dictlist[0], 'between_var')
                        _, l2_distance, _ = get_arrays(dictlist[0], 'l2_distance')
                        _, cos_distance, _ = get_arrays(dictlist[0], 'cos_distance')
                        df = get_dataframe(dictlist, arguments_base + ['seed', 'permute_classes'] + within_var + between_var + l2_distance + cos_distance)
                        df = df[df['result_type'].isin(list(['train_dataset']))]
                        df = df[df['permute_classes']==0]


                        buffer_size = buffer_sizes[0]
                        ax = axes[total] #attention
                        current_df = df[df['buffer_size'] == buffer_size]

                        results = []
                        for i in range(1, len(tasks) + 1):
                                for j in range(1, i+1):
                                        current_result = pd.DataFrame({'training_task': i, 'evaluating_task': j, 
                                                                'accuracy': (current_df[f'within_var_{j}_task{i}'] / current_df[f'between_var_{j}_task{i}']), 'result_type': current_df['result_type'], 'task_type':j==i})
                                        results.append(current_result) 

                        results = pd.concat(results)
                        # sns.lineplot(data=results, x='training_task', y='accuracy', hue='evaluating_task', style='result_type', marker='o', legend=True, ax=ax, palette=palette)
                        sns.barplot(data=results, x='training_task', y='accuracy', hue='task_type', ax=ax, errorbar=('ci', 95), errwidth=2, palette=palette, zorder=0, edgecolor='black', dodge=False)
                        
                        ax.set_title(f'{plotting_name(c_dataset)} ({plotting_name(training_settings[col])}-head)')
                        ax.set_xlabel("Training Task")
                        if total==0:
                                ax.set_ylabel(f"Within Variance")
                        else:
                                ax.set_ylabel("")

                        # Add dotted grid lines
                        ax.grid(True, linestyle='--', linewidth=1.0, axis='y', zorder=2)

                        # Remove the boxes around each plot
                        ax.spines['top'].set_visible(False)
                        ax.spines['right'].set_visible(False)
                        ax.spines['left'].set_visible(False)
                        ax.spines['bottom'].set_visible(False)

                        ax.legend_.remove()
                        total += 1
                        
        # labels = ["1", "3", "5", "7", "9"]
        # legend_handles = [mpatches.Patch(color=palette[i]) for i in range(5)]
        #fig.legend(legend_handles, labels, loc='center', bbox_to_anchor=(0.5, 0.05), ncol=6)
        handles, _ = axes[0].get_legend_handles_labels()
        labels = ['Previous Tasks','Current Task']
        fig.legend(handles, labels, bbox_to_anchor=(0.65, 0.11), ncol=2)
        fig.tight_layout(rect=[0, 0.05, 1, 1])
        fig.savefig(path + f"/NC_fixed_buffer.pdf", dpi=800)
        fig.clf() 

def NC_tasks():
        sns.set_theme(style="whitegrid")
        palette = sns.color_palette("mako_r")
        datasets = {'seq-cifar100': [2000], 'seq-tinyimg': [5000]}
        model = 'er_bounds'
        training_settings = ['class-il', 'task-il']
        fig, axes = plt.subplots(1, 4, figsize=(16, 8))

        for k in range(len(training_settings)):
                for row, (c_dataset, buffer_sizes) in enumerate(datasets.items()):
                        dictlist, _ = get_data(path, training_settings[k], c_dataset, model, "/logs_NC.txt")
                        arguments_base, _, _ = get_arguments(dictlist[0])
                        tasks, accuracy, _ = get_arrays(dictlist[0])
                        tasks, within_var, _ = get_arrays(dictlist[0], 'within_var')
                        _, between_var, _ = get_arrays(dictlist[0], 'between_var')
                        _, l2_distance, _ = get_arrays(dictlist[0], 'l2_distance')
                        _, cos_distance, _ = get_arrays(dictlist[0], 'cos_distance')
                        df = get_dataframe(dictlist, arguments_base + ['seed', 'permute_classes'] + accuracy + within_var + between_var + l2_distance + cos_distance)
                        df = df[df['result_type'].isin(list(['train_dataset']))]
                        df = df[df['permute_classes'] == 1]

                        for col, buffer_size in enumerate(buffer_sizes):
                                ax = axes[2*k +row + col] #attention
                                current_df = df[df['buffer_size'] == buffer_size]

                                results = []
                                for i in range(1, len(tasks) + 1):
                                        for j in range(1, i + 1):
                                                if(j%2==0 and c_dataset=='seq-tinyimg'):
                                                        continue
                                                current_result = pd.DataFrame({'training_task': i, 'evaluating_task': j, 
                                                                        'accuracy': (current_df[f'within_var_{j}_task{i}'] / current_df[f'between_var_{j}_task{i}']), 'result_type': current_df['result_type']})
                                                results.append(current_result) 

                                results = pd.concat(results)
                                sns.lineplot(data=results, x='training_task', y='accuracy', hue='evaluating_task', style='result_type', marker='o', legend=True, ax=ax, palette=palette)
                                #if row+col==0:
                                #        handles, labels = ax.get_legend_handles_labels()
                                #        ax.legend_.remove()
                                
                                ax.set_title(f'{plotting_name(c_dataset)} {plotting_name(training_settings[k])}')
                                ax.set_xlabel("Training Task")
                                if col==0:
                                        ax.set_ylabel("within var")
                                else:
                                        ax.set_ylabel("")
                        
        #fig.legend(handles, labels, loc='center', bbox_to_anchor=(0.5, 0.05), ncol=5)
        #fig.tight_layout(rect=[0, 0.05, 1, 1])
        fig.savefig(path + f"/NC_tasks.png")
        fig.clf() 

def NC_tasks2():
        sns.set_theme(style="whitegrid")
        palette = sns.color_palette("mako_r")
        datasets = {'seq-cifar100': [2000], 'seq-tinyimg': [5000]}
        model = 'er_bounds'
        training_settings = ['class-il', 'task-il']
        fig, axes = plt.subplots(2, 2, figsize=(8, 8), dpi=1000)

        for k in range(len(training_settings)):
                for col1, (c_dataset, buffer_sizes) in enumerate(datasets.items()):
                        dictlist, _ = get_data(path, training_settings[k], c_dataset, model, "/logs_NC.txt")
                        arguments_base, _, _ = get_arguments(dictlist[0])
                        tasks, within_var, _ = get_arrays(dictlist[0], 'within_var')
                        _, between_var, _ = get_arrays(dictlist[0], 'between_var')
                        _, l2_distance, _ = get_arrays(dictlist[0], 'l2_distance')
                        _, cos_distance, _ = get_arrays(dictlist[0], 'cos_distance')
                        df = get_dataframe(dictlist, arguments_base + ['seed'] + within_var + between_var + l2_distance + cos_distance)
                        df = df[df['result_type'].isin(list(['buffer', 'train_dataset']))]

                        for col2, buffer_size in enumerate(buffer_sizes):
                                ax = axes[k, col1 + col2] #attention
                                current_df = df[df['buffer_size'] == buffer_size].copy()
                                
                                """
                                for i in range(1, len(tasks) + 1):
                                        for j in range(1, i + 1):
                                                current_df[f'within_var_{j}_task{i}'] = current_df[f'within_var_{j}_task{i}'] / current_df[f'between_var_{j}_task{i}']
                                """
                                #Split buffer and train rows
                                buffer_df = current_df[current_df['result_type'] == 'buffer'].set_index('seed')
                                train_df = current_df[current_df['result_type'] == 'train_dataset'].set_index('seed')

                                # Drop the 'result' column before subtraction
                                buffer_df = buffer_df.drop(columns=arguments_base)
                                train_df = train_df.drop(columns=arguments_base)

                                current_df = (buffer_df - train_df)

                                results = []
                                for i in range(1, len(tasks) + 1):
                                        for j in range(1, i + 1):
                                                if(j%2==0 and c_dataset=='seq-tinyimg'):
                                                        continue
                                                current_result = pd.DataFrame({'training_task': i, 'evaluating_task': j, 
                                                                        'accuracy': (current_df[f'within_var_{j}_task{i}'])})
                                                results.append(current_result) 

                                results = pd.concat(results)
                                sns.lineplot(data=results, x='training_task', y='accuracy', hue='evaluating_task', marker='o', legend=True, ax=ax, palette=palette)
                                
                                ax.set_title(f'{plotting_name(c_dataset)}, {plotting_name(training_settings[k])}')
                                ax.set_xlabel("Training Task")
                                if col1+col2==0:
                                        ax.set_ylabel("Buffer - Train, within var")
                                else:
                                        ax.set_ylabel("")
                                ax.get_legend().set_title(None)

        plt.tight_layout()             
        fig.savefig(path + f"/NC2_tasks.png", dpi=1000)
        fig.clf() 

def read_dataframe():
        dataset='seq-tinyimg'
        model = 'er_bounds'
        training_setting = 'task-il'

        dictlist, filepath = get_data(path, training_setting, dataset, model)
        arguments_base, arguments_keep, arguments_combine = get_arguments(dictlist[0])
        accmean_columns, acctask_columns, result_accuracy = get_arrays(dictlist[0])
        df = get_dataframe(dictlist, ['permute_classes'] + arguments_base + accmean_columns + acctask_columns)
        df = df[df['result_type'] == 'features']
        df = df[df['buffer_size'] == 000]
        df = df[df['permute_classes'] == 1]
        #print(df[['lr', 'accmean_task10', 'buffer_size']])#.mean(axis=0))
        print((max(df[acctask_columns[-1]]) - min(df[acctask_columns[-1]]) ) / 2)
        print(df[acctask_columns[-1]].mean(axis=0))

        print(df[acctask_columns[-len(accmean_columns):-1]].mean(axis=1).mean(axis=0))
        print( (max(df[acctask_columns[-len(accmean_columns):-1]].mean(axis=1)) - min(df[acctask_columns[-len(accmean_columns):-1]].mean(axis=1)) ) / 2)

        print((max(df[accmean_columns[-1]]) - min(df[accmean_columns[-1]]) ) / 2)
        print(df[accmean_columns[-1]].mean(axis=0))

#old code
def sample_portion():
        sns.set_theme(style="whitegrid")
        color_mapping = {"accuracy": palette[2], "train_dataset": palette[1], "buffer": palette[0]}
        line_mapping = {'Current Task': (1, 0), 'Previous Tasks': (5, 2)}
        datasets = ['seq-cifar100']
        training_setting = 'task-il'
        model = 'er_portion'
        fig, axes = plt.subplots(1, len(datasets), figsize=(8, 4))

        for i, (ax, dataset) in enumerate(zip([axes], datasets)):
                dictlist, filepath = get_data(path, training_setting, dataset, model, '/logs.txt')
                arguments_base, arguments_keep, arguments_combine = get_arguments(dictlist[0])
                accmean_columns, acctask_columns, result_accuracy = get_arrays(dictlist[0])

                df_accuracy = get_dataframe(dictlist, arguments_base + acctask_columns[-len(accmean_columns):])
                df_accuracy = df_accuracy[df_accuracy['buffer_size'] == 2000]
                df_accuracy = df_accuracy[df_accuracy['result_type'].isin(['buffer'])]
                df_accuracy['Current Task'] = df_accuracy[acctask_columns[-1]]
                df_accuracy['Previous Tasks'] = df_accuracy[acctask_columns[-len(accmean_columns):-1]].mean(axis=1)
                df_accuracy = pd.melt(df_accuracy, id_vars=arguments_base, value_vars=['Current Task', 'Previous Tasks'], var_name='task', value_name='accuracy')
         
                dictlist, filepath = get_data(path, training_setting, dataset, model, '/logs_NC.txt')
                accmean_columns, within_var, result_accuracy = get_arrays(dictlist[0], 'within_var')
                _, between_var, result_accuracy = get_arrays(dictlist[0], 'between_var')
                df_nc = get_dataframe(dictlist, arguments_base + within_var[-len(accmean_columns):] + between_var[-len(accmean_columns):])
                df_nc = df_nc[df_nc['result_type'].isin(['buffer', 'train_dataset'])]
                df_nc['Current Task'] = df_nc[within_var[-1]]
                df_nc['Previous Tasks'] = df_nc[within_var[-len(accmean_columns):-1]].mean(axis=1)
                df_nc = pd.melt(df_nc, id_vars=arguments_base, value_vars=['Current Task', 'Previous Tasks'], var_name='task', value_name='NC')
                
                sns.lineplot(data=df_nc, x='portion', y='NC', hue='result_type', style='task', marker='o', dashes=line_mapping, legend=True, ax=ax, palette=color_mapping)
                
                ax2 = ax.twinx()
                sns.lineplot(data=df_accuracy, x='portion', y='accuracy', style='task',  marker='o', dashes=line_mapping, legend=False, ax=ax2, color=color_mapping['accuracy'])   
                
                
                #if i==0:
                #        handles, labels = ax.get_legend_handles_labels()
                #        ax.legend_.remove()
                
                ax.set_title(plotting_name(dataset) + ' (' + plotting_name(training_setting) + ')')
                ax.set_xlabel("Ratio used for replay")
                if i==0:
                        ax.set_ylabel("within var")
                        ax2.set_ylabel('Test Accuracy')
                        ax2.tick_params(axis='y', labelcolor=color_mapping['accuracy'])
                else:
                        ax.set_ylabel("")
                        ax.set_ylabel('')
        
        #fig.legend(handles, labels, loc='center', bbox_to_anchor=(0.5, 0.05), ncol=5)
        #fig.tight_layout(rect=[0, 0.05, 1, 1])
        fig.savefig(path + f"/SamplePortion.png")
        fig.clf() 

def bias_plot():
        sns.set_theme(style="whitegrid")
        color_mapping = {"ER offline": '#F28E2B', "DER online": '#4E79A7', "DER offline": '#59A14F'}
        datasets = {'seq-cifar10': [0, 500],}
        model = 'er_bounds2'
        training_setting = 'class-il'
        fig, axes = plt.subplots(len(datasets), 2, figsize=(8, 4))

        for row, (c_dataset, buffer_sizes) in enumerate(datasets.items()):
                dictlist, filepath = get_data(path, training_setting, c_dataset, model)
                arguments_base, arguments_keep, arguments_combine = get_arguments(dictlist[0])
                accmean_columns, acctask_columns, result_accuracy = get_arrays(dictlist[0])
                df = get_dataframe(dictlist, arguments_base + acctask_columns + [result_accuracy])
                replace(df, "model", plotting_name(model))

                df = group_df(df, {'result_type': 'output'}, arguments_keep, arguments_combine, result_accuracy)
                for col, buffer_size in enumerate(buffer_sizes):
                        ax = axes[row*len(datasets) + col]
                        current_df = df[df['buffer_size'] == buffer_size]

                        results = []
                        for i, _ in enumerate(accmean_columns):
                                if i==0:
                                        continue
                                task_number = i+1
                                current_column = f"accuracy_{task_number}_task{task_number}"
                                previous_columns = [f"accuracy_{1}_task{task_number}"]
                                for j in range(2, task_number):
                                        previous_columns.append(f"accuracy_{j}_task{task_number}")
                                current_result = pd.DataFrame({'task': task_number, 'current_task': current_df[current_column],
                                                        'mean_previous_tasks': current_df[previous_columns].mean(axis=1), 'result_type': current_df['result_type']})       
                                current_result['bias'] = current_result['mean_previous_tasks'] / current_result['current_task']
                                results.append(current_result)
                        results = pd.concat(results)

                        sns.lineplot(data=results, x='task', y='bias', hue='result_type', marker='o', legend=(col+row==0), ax=ax)#, palette=color_mapping)
                        if row+col==0:
                                handles, labels = ax.get_legend_handles_labels()
                                ax.legend_.remove()
                        
                        ax.set_title(f'Buffer Size = {buffer_size}')
                        ax.set_xlabel("Task")
                        if col==0:
                                ax.set_ylabel("Bias")
                        else:
                                ax.set_ylabel("")
                        
        fig.legend(handles, labels, loc='center', bbox_to_anchor=(0.5, 0.05), ncol=5)
        fig.tight_layout(rect=[0, 0.05, 1, 1])
        fig.savefig(path + f"/bias.png")
        fig.clf() 

def task_plot2():
        sns.set_theme(style="whitegrid")
        color_mapping = {"ER offline": '#F28E2B', "DER online": '#4E79A7', "DER offline": '#59A14F'}
        datasets = {'seq-cifar10': [200, 2000], 'seq-tinyimg': [1000, 5000]}
        model = 'er_bounds'
        training_setting = 'class-il'
        result_type = {'solid_line': 'output','dashed_line': 'features', 'dotted_line': 'buffer'}
        version = 'accuracy'
        fig, axes = plt.subplots(len(datasets), 2, figsize=(8, 8))

        for row, (c_dataset, buffer_sizes) in enumerate(datasets.items()):
                dictlist, filepath = get_data(path, training_setting, c_dataset, model)
                arguments_base, arguments_keep, arguments_combine = get_arguments(dictlist[0])
                accmean_columns, acctask_columns, result_accuracy = get_arrays(dictlist[0], version)
                df = get_dataframe(dictlist, arguments_base + acctask_columns + [result_accuracy])
                replace(df, "model", plotting_name(model))
                df = df[df['result_type'].isin(list(result_type.values()))]

                df = group_df(df, {'result_type': 'output'}, arguments_keep, arguments_combine, result_accuracy)
                for col, buffer_size in enumerate(buffer_sizes):
                        ax = axes[row, col] #attention
                        current_df = df[df['buffer_size'] == buffer_size]

                        results = []
                        for i, _ in enumerate(accmean_columns, start=1):
                                for j in range(1, i+1):
                                        current_result = pd.DataFrame({'training_task': i, 'evaluating_task': j, 
                                                                       'accuracy': current_df[version+f"_{j}_task{i}"], 'result_type': current_df['result_type']})
                                        results.append(current_result) 

                        results = pd.concat(results)
                        sns.lineplot(data=results, x='training_task', y='accuracy', hue='evaluating_task', style='result_type', markers=True, dashes=True, legend=(col+row==0), ax=ax)#, palette=color_mapping)
                        if row+col==0:
                                handles, labels = ax.get_legend_handles_labels()
                                ax.legend_.remove()
                        
                        ax.set_title(f'Buffer Size = {buffer_size}')
                        ax.set_xlabel("Training Task")
                        if col==0:
                                ax.set_ylabel("Accuracy")
                        else:
                                ax.set_ylabel("")
                        
        fig.legend(handles, labels, loc='center', bbox_to_anchor=(0.5, 0.05), ncol=5)
        fig.tight_layout(rect=[0, 0.05, 1, 1])
        fig.savefig(path + f"/task.png")
        fig.clf() 

def NC_plot():
        sns.set_theme(style="whitegrid")
        color_mapping = {"ER offline": '#F28E2B', "DER online": '#4E79A7', "DER offline": '#59A14F'}
        datasets = {'seq-cifar10': [200, 1500]}
        model = 'er_bounds'
        training_setting = 'class-il'
        result_type = {'solid_line': 'buffer','dashed_line': 'test_dataset', 'dotted_line': 'extra_buffer'}
        version = 'intra_class_var'
        fig, axes = plt.subplots(len(datasets), 2, figsize=(8, 4))

        for row, (c_dataset, buffer_sizes) in enumerate(datasets.items()):
                dictlist, filepath = get_data(path, training_setting, c_dataset, model, "/logs_NC.txt")
                arguments_base, arguments_keep, arguments_combine = get_arguments(dictlist[0])
                accmean_columns, acctask_columns, result_accuracy = get_arrays(dictlist[0], version, 'inter_class_var')
                df = get_dataframe(dictlist, arguments_base + accmean_columns + acctask_columns + [result_accuracy])
                replace(df, "model", plotting_name(model))
                df = df[df['result_type'].isin(list(result_type.values()))]

                #df = group_df(df, None, arguments_keep, arguments_combine, result_accuracy)
                for col, buffer_size in enumerate(buffer_sizes):
                        ax = axes[row + col] #attention
                        current_df = df[df['buffer_size'] == buffer_size]

                        results = []
                        for i, _ in enumerate(accmean_columns, start=1):
                                for j in range(1, i+1):
                                        if(j%2==1 and c_dataset=='seq-tinyimg'):
                                                continue
                                        current_result = pd.DataFrame({'training_task': i, 'evaluating_task': j, 
                                                                       'accuracy': (current_df[version +f"_{j}_task{i}"]  /  1.0), 'result_type': current_df['result_type']})
                                        results.append(current_result) 

                        results = pd.concat(results)
                        sns.lineplot(data=results, x='training_task', y='accuracy', hue='evaluating_task', style='result_type', markers=True, dashes=True, legend=(col+row==0), ax=ax)#, palette=color_mapping)
                        if row+col==0:
                                handles, labels = ax.get_legend_handles_labels()
                                ax.legend_.remove()
                        
                        ax.set_title(f'Buffer Size = {buffer_size}')
                        ax.set_xlabel("Training Task")
                        if col==0:
                                ax.set_ylabel("Intra Class Var")
                        else:
                                ax.set_ylabel("")
                        
        fig.legend(handles, labels, loc='center', bbox_to_anchor=(0.5, 0.05), ncol=5)
        fig.tight_layout(rect=[0, 0.05, 1, 1])
        fig.savefig(path + f"/NC.png")
        fig.clf() 

def chunking_plot():
        sns.set_theme(style="whitegrid")
        datasets = ['chu-cifar10']
        
        for i, dataset in enumerate(datasets, start=1): 
                plt.subplot(1, len(datasets), i)

                dictlist, filepath = get_data(path, 'class-il', dataset, 'sgd')
                arguments_base, arguments_keep, arguments_combine = get_arguments(dictlist[0])

                for current_dictionary in dictlist:
                        current_number = current_dictionary['chunking']
                        current_dictionary['result'] = current_dictionary[f'accuracy_1_task{current_number}']

                df = get_dataframe(dictlist, arguments_base + ['result'])
                results = group_df(df, {'type': 'output'}, arguments_keep, arguments_combine, 'result')

                #print(data_buffer)
                sns.lineplot(data=results, x='chunking', y='result', hue='type', marker='o', legend=(i==1))

                plt.title(plotting_name(dataset))
                plt.xlabel('Chunks')
                if(i == 1):
                        plt.ylabel('Accuracy')
                else:
                        plt.ylabel('')

                plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=5)

        plt.tight_layout()
        plt.savefig(path + f"/chunking.png")
        plt.close()
        plt.clf() 
