import sys
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import ast
import re
from functools import reduce

from conf import base_path

path = base_path() + "results/"
datasets = sys.argv[1]
datasets = datasets.split(',')
scenario = sys.argv[2]

def preprocess_line(line):
       # Remove the 'device' entry
        line = re.sub(r"'device': device\(.*?\),?", "", line)
        return line

def get_data(path, scenario, dataset, model):
        filepath = path + scenario + "/" + dataset + "/" + model
        with open(filepath + "/logs.txt", "r") as file:
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
        if "type" in dictionary:
                base.append("type")
        if "chunking" in dictionary:
                base.append("chunking")
                keep.append("chunking")
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

def get_arrays(dataset):
        accmean_columns = list()
        acctask_columns = list()
        amount_task = 0
        if(any(substring in dataset for substring in ["seq-tinyimg"])):
                amount_task = 10
        elif(any(substring in dataset for substring in ["rot-mnist", "perm-mnist"])):
                amount_task = 20
        elif(any(substring in dataset for substring in ["seq-mnist", "seq-cifar10"])):
                amount_task = 5
        for i in range(1, amount_task+1):
                accmean_columns.append(f"accmean_task{i}")
                if(scenario == "domain-il"):
                        continue
                for j in range(1, i+1):
                        acctask_columns.append(f"accuracy_{j}_task{i}")
        result_accuracy = accmean_columns[-1]

        return accmean_columns, acctask_columns, result_accuracy

def plotting_name(logging_name):
        if logging_name == 'der_tempbounds':
                return 'DER offline'
        
        elif logging_name == 'seq-cifar10':
                return 'Seq-Cifar10'
        elif logging_name == 'chu-cifar10':
                return 'Chunking-Cifar10'
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
        print(grouped)
        #Calculate max of mean for arguments we should keep for plotting
        best = grouped.groupby(keep_arguments, as_index=False).agg({result_accuracy: 'max'})
        best = grouped.merge(best[result_accuracy], on=[result_accuracy], how='inner')
        print(best)
        #Filter dataframe according to best arguments
        best = best.drop(result_accuracy, axis=1)
        result = df.merge(best, on=(keep_arguments + group_arguments), how='inner')
        print(result)
        return result


def output_plot():
        sns.set_theme(style="whitegrid")
        color_mapping = {"ER offline": '#F28E2B', "DER online": '#4E79A7', "DER offline": '#59A14F'}
        datasets = ['seq-cifar10']
        models = ['er_bounds', 'er_balanced', 'er_balanced_fixed']
        version = 'output'

        for i, dataset in enumerate(datasets, start=1): 
                plt.subplot(1, len(datasets), i)

                data_buffer = []
                data_nobuffer = []
                for model in models:
                        dictlist, filepath = get_data(path, 'class-il', dataset, model)
                        arguments_base, arguments_keep, arguments_combine = get_arguments(dictlist[0])

                        accmean_columns, acctask_columns, result_accuracy = get_arrays(dataset)

                        df = get_dataframe(dictlist, arguments_base + [result_accuracy])
                        df = df[df['type'] == version]
                        df.drop('type', axis=1)
                        replace(df, "model", plotting_name(model))

                        result = group_df(df, None, arguments_keep, arguments_combine, result_accuracy)
                        if 'buffer_size' in arguments_base:
                                data_buffer.append(result)
                        else:
                                data_nobuffer.append(result)   

                data_buffer = pd.concat(data_buffer)
                #data_nobuffer = pd.concat(data_nobuffer)

                #all_buffer_sizes = data_buffer['buffer_size'].unique()
                #df_buffer_sizes = pd.DataFrame({'buffer_size': all_buffer_sizes})
                #data_nobuffer = data_nobuffer.merge(df_buffer_sizes, how='cross')
                
                sns.lineplot(data=data_buffer, x='buffer_size', y=result_accuracy, hue='model', marker='o')#, palette=color_mapping, legend=False)
                #sns.lineplot(data=data_buffer, x='buffer_size', y=result_accuracy, hue='model', marker='', palette=color_mapping, legend=False)

                plt.title(plotting_name(dataset) + ' (CIL)')
                plt.xlabel('Buffer Size')
                if(i == 1):
                        plt.ylabel('Accuracy')
                else:
                        plt.ylabel('')

        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=5)

        plt.tight_layout()
        plt.savefig(path + f"/output.png")
        plt.close()
        plt.clf() 


def bias_plot():
        sns.set_theme(style="whitegrid")
        color_mapping = {"ER offline": '#F28E2B', "DER online": '#4E79A7', "DER offline": '#59A14F'}
        datasets = ['seq-cifar10']
        models = ['der_tempbounds']

        for i, dataset in enumerate(datasets, start=1): 
                plt.subplot(1, len(datasets), i)

                data_buffer = []
                for model in models:
                        dictlist, filepath = get_data(path, 'class-il', dataset, model)
                        arguments_base, arguments_keep, arguments_combine = get_arguments(dictlist[0])

                        accmean_columns, acctask_columns, result_accuracy = get_arrays(dataset)

                        df = get_dataframe(dictlist, arguments_base + acctask_columns[-len(accmean_columns):] + [result_accuracy])
                        replace(df, "model", plotting_name(model))

                        result = group_df(df, {'type': 'output'}, arguments_keep, arguments_combine, result_accuracy)
                        result['current_task'] = result[acctask_columns[-1]]
                        result['mean_previous_task'] = result[acctask_columns[-len(accmean_columns):-1]].mean(axis=1)
                        result['bias'] = result['mean_previous_task'] / result['current_task']
                        result.drop(acctask_columns[-len(accmean_columns):-1], axis=1)
                        print(result)
                        data_buffer.append(result)
  
                data_buffer = pd.concat(data_buffer)
                #print(data_buffer)
                
                sns.lineplot(data=data_buffer, x='buffer_size', y='current_task', hue='type', marker='o', legend=(i==1))#, palette=color_mapping)

                plt.title(plotting_name(dataset) + ' (CIL)')
                plt.xlabel('Buffer Size')
                if(i == 1):
                        plt.ylabel('Bias')
                else:
                        plt.ylabel('')

                plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=5)

        plt.tight_layout()
        plt.savefig(path + f"/bias.png")
        plt.close()
        plt.clf() 

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

chunking_plot()