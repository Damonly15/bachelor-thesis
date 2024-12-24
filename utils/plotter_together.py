import sys
import os

mammoth_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import ast
import re

dataset = sys.argv[1]
scenario = sys.argv[2]
algorithm = sys.argv[3]
dataset_feature = sys.argv[4]

def preprocess_line(line):
       # Remove the 'device' entry
        line = re.sub(r"'device': device\(.*?\),?", "", line)
        return line

def get_data(base, type, dataset, model):
        filepath = base + "/data/results_thesis/" + type + "/" + dataset + "/" + model
        with open(filepath + "/logs.txt", "r") as file:
                dictlist = list()
                for i, line in enumerate(file, start=1):
                        line = preprocess_line(line)
                        dictlist.append(ast.literal_eval(line))
                return dictlist, filepath

def get_arguments(dictionary):
        base = ["dataset", "model"]
        if "buffer_size" in dictionary:
                base.append("buffer_size")
        if "lr" in dictionary:
                base.append("lr")
        if "alpha" in dictionary:
                base.append("alpha")
        if "beta" in dictionary:
                base.append("beta")
        if "temperature" in dictionary:
                base.append("temperature")
        return base

def get_dataframe(dictlist, keep_arguments):
        df = pd.DataFrame(dictlist)
        df = df.drop(columns=[col for col in df.columns if col not in (keep_arguments)])

        #if the temperature is above 100, we used the mse approximation
        if "temperature" in keep_arguments:
                index = df['temperature'] > 100
                df["temperature"] = df["temperature"].astype("str")
                df.loc[index, 'temperature'] = 'mse'
        return df

def replace(dataframe, key, new_value):
        dataframe[key] = new_value

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

def dump_df(df, group_arguments, result_accuracy, filepath, temperature):
        df_group = df.groupby(group_arguments).mean()
        df_group["runs"] = df.groupby(group_arguments).size()
        df_group["range"] =  (df.groupby(group_arguments)[result_accuracy].max() - df.groupby(group_arguments)[result_accuracy].min()) / 2
        df_group = df_group.reset_index()

        if temperature != 0:
                if not os.path.exists(filepath + f"/temperature{temperature}"):
                        os.makedirs(filepath + f"/temperature{temperature}")
                with open(filepath + f"/temperature{temperature}/summary_temp{temperature}.txt", "w") as file:
                        file.write(df_group[group_arguments + ["runs", result_accuracy, "range"]].to_string())
        else: 
                with open(filepath + "/summary.txt", "w") as file:
                        file.write(df_group[group_arguments + ["runs", result_accuracy, "range"]].to_string())

def group_df(df, group_arguments, result_accuracy, filepath, temperature):
        result = pd.DataFrame()
        for current_buffersize in df["buffer_size"].unique():
                c_df = df[df["buffer_size"] == current_buffersize]
                average_result = c_df[group_arguments + [result_accuracy]]
                average_result = average_result.groupby(group_arguments).mean().reset_index()

                if result_accuracy == "forgetting":
                        max_average_res = average_result.loc[average_result["accmean_task5"].idxmax()]
                else:
                        max_average_res = average_result.loc[average_result[result_accuracy].idxmax()]
                for argument in group_arguments:
                        c_df = c_df[(c_df[argument] == max_average_res[argument])]
                result = pd.concat([result, c_df])

        """
        if temperature != 0:
                if not os.path.exists(filepath + f"/temperature{temperature}"):
                        os.makedirs(filepath + f"/temperature{temperature}")
                with open(filepath + f"/temperature{temperature}/best_temp{temperature}.txt", "w") as file:
                        result_grouped = result.groupby(group_arguments).mean().reset_index()
                        file.write(result_grouped[group_arguments + [result_accuracy]].to_string())
        else:
                with open(filepath + "/best.txt", "w") as file:
                        result_grouped = result.groupby(group_arguments).mean().reset_index()
                        file.write(result_grouped[group_arguments + [result_accuracy]].to_string())
        """
        return result

def group_dfscaling(df, group_arguments, result_accuracy, filepath, temperature):
        result = pd.DataFrame()
        for current_buffersize in df["buffer_size"].unique():
                for backbone in df["backbone"].unique():
                        c_df = df[(df["buffer_size"] == current_buffersize) & (df["backbone"] == backbone)]
                        average_result = c_df[group_arguments + [result_accuracy]]
                        average_result = average_result.groupby(group_arguments).mean().reset_index()

                        if result_accuracy == "forgetting":
                                max_average_res = average_result.loc[average_result["accmean_task5"].idxmax()]
                        else:
                                max_average_res = average_result.loc[average_result[result_accuracy].idxmax()]
                        for argument in group_arguments:
                                c_df = c_df[(c_df[argument] == max_average_res[argument])]
                        result = pd.concat([result, c_df])

        return result

def deroffline_deronline_eroffline():
        dictlist_der, filepath_der = get_data(mammoth_path, scenario, dataset, "der")
        dictlist_dertempbounds, filepath_dertempbounds = get_data(mammoth_path, scenario, dataset, "der_tempbounds")
        dictlist_er, filepath_er = get_data(mammoth_path, scenario, dataset, "er_bounds")

        arguments_der = get_arguments(dictlist_der[0])
        arguments_der_temperature= get_arguments(dictlist_dertempbounds[0])
        arguments_er = get_arguments(dictlist_er[0])

        accmean_columns, acctask_columns, result_accuracy = get_arrays(dataset)

        df_der = get_dataframe(dictlist_der, arguments_der + [result_accuracy])
        replace(df_der, "model", "DER online")
        df_dertempbounds = get_dataframe(dictlist_dertempbounds, arguments_der_temperature + [result_accuracy])
        replace(df_dertempbounds, "model", "DER offline")
        df_er = get_dataframe(dictlist_er, arguments_er + [result_accuracy])
        replace(df_er, "model", "ER offline")

        for current_temperature in df_dertempbounds["temperature"].unique():
                c_df_dertempbounds = df_dertempbounds[df_dertempbounds["temperature"] == current_temperature]

                dump_df(df_der, arguments_der, result_accuracy, filepath_der, 0)
                dump_df(c_df_dertempbounds, arguments_der_temperature, result_accuracy, filepath_dertempbounds, current_temperature)
                dump_df(df_er, arguments_er, result_accuracy, filepath_er, 0)

                result = group_df(df_der, arguments_der, result_accuracy, filepath_der, 0)
                current_result = group_df(c_df_dertempbounds, arguments_der_temperature, result_accuracy, filepath_dertempbounds, current_temperature)
                result = pd.concat([result, current_result])
                current_result = group_df(df_er, arguments_er, result_accuracy, filepath_er, 0)
                result = pd.concat([result, current_result])

                #create the plot
                sns.set_theme(style="whitegrid")
                color_mapping = {"ER offline": '#F28E2B', "DER online": '#4E79A7', "DER offline": '#59A14F'}
                sns.lineplot(data=result, x='buffer_size', y=result_accuracy, hue='model', marker='o', palette=color_mapping, legend=True)
                #plt.title(f'Accuracy vs Buffer Size by Model for Temperature = {current_temperature}')
                plt.title(f'Seq-Cifar10 (CIL)')
                plt.xlabel('Buffer Size')
                plt.ylabel('')
                plt.legend(title='')
                #plt.legend(loc='lower right')

                #change
                plt.savefig(filepath_er + f"/summary_plot_temp{current_temperature}.png")
                plt.close()
                plt.clf() 

def deroffline_derpretrained_derpretrained2():
        dictlist_dertempbounds, filepath_dertempbounds = get_data(mammoth_path, scenario, dataset, "der_tempbounds")
        dictlist_derpretrained, filepath_derpretrained = get_data(mammoth_path, scenario, dataset, "der_pretrained")
        dictlist_derpretrained2, filepath_derpretrained2 = get_data(mammoth_path, scenario, dataset, "der_pretrained2")
        dictlist_er, filepath_er = get_data(mammoth_path, scenario, dataset, "er_bounds")

        arguments_der_temperature= get_arguments(dictlist_dertempbounds[0])

        accmean_columns, acctask_columns, result_accuracy = get_arrays(dataset)

        df_dertempbounds = get_dataframe(dictlist_dertempbounds, arguments_der_temperature + [result_accuracy])
        replace(df_dertempbounds, "model", "DER offline")
        df_derpretrained = get_dataframe(dictlist_derpretrained, arguments_der_temperature + [result_accuracy])
        replace(df_derpretrained, "model", "DER JTTeacher")
        df_derpretrained2 = get_dataframe(dictlist_derpretrained2, arguments_der_temperature + [result_accuracy])
        replace(df_derpretrained2, "model", "DER JTTeacher + masking")


        for current_temperature in df_dertempbounds["temperature"].unique():
                c_df_dertempbounds = df_dertempbounds[df_dertempbounds["temperature"] == current_temperature]
                c_df_derpretrained = df_derpretrained[df_derpretrained["temperature"] == current_temperature]
                c_df_derpretrained2 = df_derpretrained2[df_derpretrained2["temperature"] == current_temperature]

                dump_df(c_df_dertempbounds, arguments_der_temperature, result_accuracy, filepath_dertempbounds, current_temperature)
                dump_df(c_df_derpretrained, arguments_der_temperature, result_accuracy, filepath_derpretrained, current_temperature)
                dump_df(c_df_derpretrained2, arguments_der_temperature, result_accuracy, filepath_derpretrained2, current_temperature)

                result = group_df(c_df_dertempbounds, arguments_der_temperature, result_accuracy, filepath_dertempbounds, current_temperature)
                current_result = group_df(c_df_derpretrained, arguments_der_temperature, result_accuracy, filepath_derpretrained, current_temperature)
                result = pd.concat([result, current_result])
                current_result = group_df(c_df_derpretrained2, arguments_der_temperature, result_accuracy, filepath_derpretrained2, current_temperature)
                result = pd.concat([result, current_result])
        
                #create the plot
                sns.set_theme(style="whitegrid")
                color_mapping = {"DER offline": '#59A14F', "DER JTTeacher + masking": '#F28E2B', "DER JTTeacher": '#17becf'}
                sns.lineplot(data=result, x='buffer_size', y=result_accuracy, hue='model', marker='o', palette=color_mapping, legend=False)
                #plt.title(f'Accuracy vs Buffer Size by Model for Temperature = {current_temperature}')
                plt.title(f'Seq-MNIST (CIL)')
                plt.xlabel('Buffer Size')
                plt.ylabel('Accuracy')
                #plt.legend(title='Model')
                #plt.legend(loc='lower right')

                #change
                plt.savefig(filepath_er + f"/summary_plot_temp{current_temperature}.png")
                plt.close()
                plt.clf() 

def derpretrained_derpermuted_erlabelsmoothing():
        dictlist_dertempbounds, filepath_dertempbounds = get_data(mammoth_path, scenario, dataset, "der_tempbounds")
        dictlist_derpretrained, filepath_derpretrained = get_data(mammoth_path, scenario, dataset, "der_pretrained")
        dictlist_derpermuted, filepath_derpermuted = get_data(mammoth_path, scenario, dataset, "der_permuted")
        dictlist_erlabelsmoothing, filepath_erlabelsmoothing = get_data(mammoth_path, scenario, dataset, "er_labelsmoothing")
        dictlist_er, filepath_er = get_data(mammoth_path, scenario, dataset, "er_bounds")

        arguments_der_temperature= get_arguments(dictlist_derpretrained[0])
        arguments_er = get_arguments(dictlist_erlabelsmoothing[0])

        accmean_columns, acctask_columns, result_accuracy = get_arrays(dataset)

        df_dertempbounds = get_dataframe(dictlist_dertempbounds, arguments_der_temperature + [result_accuracy])
        replace(df_dertempbounds, "model", "DER offline")
        df_derpretrained = get_dataframe(dictlist_derpretrained, arguments_der_temperature + [result_accuracy])
        replace(df_derpretrained, "model", "DER JTTeacher")      
        df_derpermuted = get_dataframe(dictlist_derpermuted, arguments_der_temperature + [result_accuracy])
        replace(df_derpermuted, "model", "DER JTTeacher + permuting")
        df_erlabelsmoothing = get_dataframe(dictlist_erlabelsmoothing, arguments_er + [result_accuracy])
        replace(df_erlabelsmoothing, "model", "Label Smoothing")

        for current_temperature in df_derpretrained["temperature"].unique():
                c_df_dertempbounds = df_dertempbounds[df_dertempbounds["temperature"] == current_temperature]
                c_df_derpretrained = df_derpretrained[df_derpretrained["temperature"] == current_temperature]
                c_df_derpermuted = df_derpermuted[df_derpermuted["temperature"] == current_temperature]

                dump_df(c_df_dertempbounds, arguments_der_temperature, result_accuracy, filepath_dertempbounds, current_temperature)
                dump_df(c_df_derpretrained, arguments_der_temperature, result_accuracy, filepath_derpretrained, current_temperature)
                dump_df(c_df_derpermuted, arguments_der_temperature, result_accuracy, filepath_derpermuted, current_temperature)
                dump_df(df_erlabelsmoothing, arguments_er, result_accuracy, filepath_erlabelsmoothing, 0)

                result = group_df(c_df_derpermuted, arguments_der_temperature, result_accuracy, filepath_derpermuted, current_temperature)
                current_result = group_df(c_df_derpretrained, arguments_der_temperature, result_accuracy, filepath_derpretrained, current_temperature)
                result = pd.concat([result, current_result])
                current_result = group_df(c_df_dertempbounds, arguments_der_temperature, result_accuracy, filepath_dertempbounds, current_temperature)
                result = pd.concat([result, current_result])
                current_result = group_df(df_erlabelsmoothing, arguments_er, result_accuracy, filepath_erlabelsmoothing, 0)
                result = pd.concat([result, current_result])
                #create the plot
                sns.set_theme(style="whitegrid")
                color_mapping = {"Label Smoothing": '#F28E2B', "DER JTTeacher + permuting": '#4E79A7', "DER JTTeacher": '#17becf', "DER offline": '#59A14F'}
                sns.lineplot(data=result, x='buffer_size', y=result_accuracy, hue='model', marker='o', palette=color_mapping, legend=False)
                #plt.title(f'Accuracy vs Buffer Size by Model for Temperature = {current_temperature}')
                plt.title(f'Seq-MNIST (CIL)')
                plt.xlabel('Buffer Size')
                plt.ylabel('Accuracy')
                #plt.legend(title='Model')
                #plt.legend(loc='lower right')

                #change
                plt.savefig(filepath_er + f"/summary_plot_temp{current_temperature}.png")
                plt.close()
                plt.clf() 

def temperature():
        dictlist_dertempbounds, filepath_dertempbounds = get_data(mammoth_path, scenario, dataset, "der_tempbounds")
        arguments_der_temperature= get_arguments(dictlist_dertempbounds[0])

        accmean_columns, acctask_columns, result_accuracy = get_arrays(dataset)

        df_dertempbounds = get_dataframe(dictlist_dertempbounds, arguments_der_temperature + [result_accuracy])
        replace(df_dertempbounds, "model", "DER offline")

        for current_temperature in df_dertempbounds["temperature"].unique():
                c_df_dertempbounds = df_dertempbounds[df_dertempbounds["temperature"] == current_temperature]
                dump_df(c_df_dertempbounds, arguments_der_temperature, result_accuracy, filepath_dertempbounds, current_temperature)

        df_dertempbounds = df_dertempbounds[df_dertempbounds['temperature'].isin(['1.0', '5.0', '20.0', 'mse'])]
        #create the plot
        sns.set_theme(style="whitegrid")
        color_mapping = {"1.0": '#F28E2B', "20.0": '#4E79A7', "5.0": '#17becf', "mse": '#59A14F'}
        sns.lineplot(data=df_dertempbounds, x='buffer_size', y=result_accuracy, hue='temperature', marker='o', legend=True, palette=color_mapping)
        #plt.title(f'Accuracy vs Buffer Size by Model for Temperature = {current_temperature}')
        plt.title(f'Seq-Cifar10 (CIL)')
        plt.xlabel('Buffer Size')
        plt.ylabel('Accuracy')
        plt.legend(title='Temperature')
        plt.legend(loc='lower right')

        #change
        plt.savefig(filepath_dertempbounds + f"/summary_plot.png")
        plt.close()
        plt.clf() 

def buffer_accuracy():
        path = mammoth_path + "/plot_dataframes/buffer_accuracy/" + dataset

        # List all files in the folder
        files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]

        # Initialize a list to hold the extracted numbers
        buffer_sizes = set()
        # Define the regex pattern to extract the number after two underscores
        pattern = r'^[^_]+_[^_]+_(\d+)_[^_]+\.txt$'

        # Loop through each file
        for file in files:
                # Match the pattern
                match = re.match(pattern, file)
                if match:
                        # Extract the number and convert it to an integer
                        number = int(match.group(1))
                        # Append the number to the list
                        buffer_sizes.add(number)
        buffer_sizes = list(buffer_sizes)
        buffer_sizes.sort()

        df_smaller_buffer = []
        df_bigger_buffer = []
        for file in files:
                file_path = os.path.join(path, file)
                #Get buffer size
                match = re.match(pattern, file)
                if match:
                        number = int(match.group(1))
                
                df = pd.read_csv(file_path, sep='\t')
                df['model'] = df['model'].replace('der_tempbounds', 'DER offline')
                df['model'] = df['model'].replace('der', 'DER online')
                if number == buffer_sizes[0]:
                        df_smaller_buffer.append(df)
                else:
                        df_bigger_buffer.append(df)

        df_smaller_buffer = pd.concat(df_smaller_buffer, ignore_index=True)
        df_smaller_buffer = df_smaller_buffer[df_smaller_buffer['model'].isin(['DER offline', 'DER online'])]
        df_bigger_buffer = pd.concat(df_bigger_buffer, ignore_index=True)
        df_bigger_buffer = df_bigger_buffer[df_bigger_buffer['model'].isin(['DER offline', 'DER online'])]
        #create the plot
        sns.set_theme(style="whitegrid")
        for buffer_size, current_df in zip(buffer_sizes, [df_smaller_buffer, df_bigger_buffer]):
                for version in ["buf_logits_accuracy", "buf_train_accuracy"]:
                        color_mapping = {"DER online": '#4E79A7', "DER offline": '#59A14F'}
                        sns.lineplot(data=current_df, x='epoch', y=version, hue='model', legend=True, marker=None, errorbar=None, palette=color_mapping)
                        plt.title('Seq-Cifar10 (CIL)')
                        plt.xlabel('Epoch')
                        plt.ylabel('')
                        plt.legend(title='')

                        #change
                        plt.savefig(path + "/buffersize_" + str(buffer_size) + "_" + version)
                        plt.close()
                        plt.clf() 

def buffer_bias():
        path = mammoth_path + "/plot_dataframes/bias/" + dataset

        # List all files in the folder
        files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]

        # Initialize a list to hold the extracted numbers
        buffer_sizes = set()
        # Define the regex pattern to extract the number after two underscores
        pattern = r'^[^_]+_[^_]+_(\d+)_[^_]+\.txt$'

        # Loop through each file
        for file in files:
                # Match the pattern
                match = re.match(pattern, file)
                if match:
                        # Extract the number and convert it to an integer
                        number = int(match.group(1))
                        # Append the number to the list
                        buffer_sizes.add(number)
        buffer_sizes = list(buffer_sizes)
        buffer_sizes.sort()

        df_smaller_buffer = []
        df_bigger_buffer = []
        for file in files:
                file_path = os.path.join(path, file)
                #Get buffer size
                match = re.match(pattern, file)
                if match:
                        number = int(match.group(1))
                
                df = pd.read_csv(file_path, sep='\t')
                df['model'] = df['model'].replace('der_tempbounds', 'DER offline')
                df['model'] = df['model'].replace('der_pretrained', 'DER JTTeacher')
                df['model'] = df['model'].replace('der_permuted', 'DER JTTeacher + permuting')
                df['model'] = df['model'].replace('der_pretrained2', 'DER JTTeacher + masking')
                df['model'] = df['model'].replace('er_labelsmoothing', 'Label Smoothing')
                if number == buffer_sizes[0]:
                        df_smaller_buffer.append(df)
                else:
                        df_bigger_buffer.append(df)

        df_smaller_buffer = pd.concat(df_smaller_buffer, ignore_index=True)
        df_smaller_buffer = df_smaller_buffer[df_smaller_buffer['model'].isin(['DER offline', 'DER JTTeacher', 'DER JTTeacher + masking'])]
        df_bigger_buffer = pd.concat(df_bigger_buffer, ignore_index=True)
        df_bigger_buffer = df_bigger_buffer[df_bigger_buffer['model'].isin(['DER offline', 'DER JTTeacher', 'DER JTTeacher + masking'])]
        #create the plot
        sns.set_theme(style="whitegrid")
        for buffer_size, current_df in zip(buffer_sizes, [df_smaller_buffer, df_bigger_buffer]):
                color_mapping = {"DER offline": '#59A14F', "DER JTTeacher + masking": '#F28E2B', "DER JTTeacher": '#17becf'}
                plt.figure(figsize=(8, 6.3))
                sns.catplot(data=current_df, x='task', y='current_task_probability', hue='model', kind='bar', palette=color_mapping, legend=True)
                if(buffer_size == buffer_sizes[0]):
                        plt.title('Seq-Cifar10 (CIL), small buffer', y=0.98)
                else:
                        plt.title('Seq-Cifar10 (CIL), large buffer', y=0.98)
                plt.xlabel('Task')
                plt.ylabel('Probability')
                plt.legend(title='')

                #change
                plt.savefig(path + "/buffersize_" + str(buffer_size))
                plt.close()
                plt.clf() 

def BN_training():
        path = mammoth_path + "/plot_dataframes/BN/" + dataset

        # List all files in the folder
        files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]

        df = []
        for file in files:
                file_path = os.path.join(path, file)               
                c_df = pd.read_csv(file_path, sep='\t')
                df.append(c_df)

        df = pd.concat(df, ignore_index=True)
        df['model'] = df['model'].replace('der', 'DER online')
        df['model'] = df['model'].replace('der_tempbounds_eval', 'DER offline + global moments')
        df['model'] = df['model'].replace('der_tempbounds_train', 'DER offline + mini batch moments')

        #create the plot
        sns.set_theme(style="whitegrid")
        plt.figure(figsize=(8, 6))
        color_mapping = {"DER online": '#4E79A7', 'DER offline + global moments': '#59A14F', 'DER offline + mini batch moments': '#8C8C8C'}
        ax = sns.lineplot(data=df, x='epoch', y="train_accuracy", hue='model', marker=None, errorbar=None, palette=color_mapping)
        sns.move_legend(
                ax, "lower center",
                bbox_to_anchor=(.5, 1), ncol=3, title=None, frameon=False,)
        
        # Create a dictionary to map epoch to task
        epoch_to_task = df.drop_duplicates('epoch').set_index('epoch')['task'].to_dict()

        #adjust x axis naming
        # Generate x-tick labels: show task only once when it changes
        xticks_labels = []
        last_task = None
        for epoch in df['epoch'].unique():
                current_task = epoch_to_task[epoch]
                # Only add the task number if it is different from the last task
                if current_task != last_task:
                        xticks_labels.append(str(current_task))  # Display task number
                        last_task = current_task
                else:
                        xticks_labels.append('')  # Keep empty to avoid repeating tasks

        # Replace x-tick labels with corresponding task values (displaying only once)
        plt.xticks(ticks=df['epoch'].unique(), labels=xticks_labels)

        # Disable default vertical grid lines
        plt.grid(visible=False, axis='x')

        # Manually add vertical grid lines for each unique task
        unique_tasks = df.drop_duplicates('task')['epoch'].values
        for task_epoch in unique_tasks:
                plt.axvline(x=task_epoch, color='gray', linestyle='-', linewidth=0.8)
        
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')

        #change
        plt.savefig(path + "/summary")
        plt.close()
        plt.clf() 

def tracking():
        path = mammoth_path + "/plot_dataframes/BN/" + dataset

        # List all files in the folder
        files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]

        df = []
        for file in files:
                file_path = os.path.join(path, file)               
                c_df = pd.read_csv(file_path, sep='\t')
                df.append(c_df)

        df = pd.concat(df, ignore_index=True)
        df['model'] = df['model'].replace('der', 'DER online')
        df['model'] = df['model'].replace('der_tempbounds_eval', 'DER offline + global moments')
        df['model'] = df['model'].replace('der_tempbounds_train', 'DER offline + mini-batch moments')

        #create the plot
        sns.set_theme(style="whitegrid")
        plt.figure(figsize=(8, 6))
        color_mapping = {"DER online": '#4E79A7', 'DER offline + global moments': '#59A14F', 'DER offline + mini-batch moments': '#8C8C8C'}
        ax = sns.lineplot(data=df, x='epoch', y="train_accuracy", hue='model', marker=None, errorbar=None, palette=color_mapping)
        sns.move_legend(
                ax, "lower center",
                bbox_to_anchor=(.5, 1), ncol=3, title=None, frameon=False,)
        
        # Create a dictionary to map epoch to task
        epoch_to_task = df.drop_duplicates('epoch').set_index('epoch')['task'].to_dict()

        #adjust x axis naming
        # Generate x-tick labels: show task only once when it changes
        xticks_labels = []
        last_task = None
        for epoch in df['epoch'].unique():
                current_task = epoch_to_task[epoch]
                # Only add the task number if it is different from the last task
                if current_task != last_task:
                        xticks_labels.append(str(current_task))  # Display task number
                        last_task = current_task
                else:
                        xticks_labels.append('')  # Keep empty to avoid repeating tasks

        # Replace x-tick labels with corresponding task values (displaying only once)
        plt.xticks(ticks=df['epoch'].unique(), labels=xticks_labels)

        # Disable default vertical grid lines
        plt.grid(visible=False, axis='x')

        # Manually add vertical grid lines for each unique task
        unique_tasks = df.drop_duplicates('task')['epoch'].values
        for task_epoch in unique_tasks:
                plt.axvline(x=task_epoch, color='gray', linestyle='-', linewidth=0.8)
        
        plt.xlabel('Task')
        plt.ylabel('Accuracy')

        #change
        plt.savefig(path + "/summary")
        plt.close()
        plt.clf() 

def feature_forgetting_current():
        dictlist_dertempbounds, filepath_dertempbounds = get_data(mammoth_path, scenario, dataset, algorithm)
        dictlist_dertempbounds_feature, filepath_dertempbounds_feature = get_data(mammoth_path, scenario, dataset_feature, algorithm)

        arguments_der_temperature= get_arguments(dictlist_dertempbounds_feature[0])

        accmean_columns, acctask_columns, result_accuracy = get_arrays(dataset)

        df_dertempbounds = get_dataframe(dictlist_dertempbounds, arguments_der_temperature + acctask_columns + [result_accuracy])
        replace(df_dertempbounds, "model", "Network head")
        df_dertempbounds_feature = get_dataframe(dictlist_dertempbounds_feature, arguments_der_temperature + acctask_columns + [result_accuracy])
        replace(df_dertempbounds_feature, "model", "Fitted head")

        for current_temperature in df_dertempbounds_feature["temperature"].unique():
                c_df_dertempbounds = df_dertempbounds[df_dertempbounds["temperature"] == current_temperature]
                c_df_dertempbounds = group_df(c_df_dertempbounds, arguments_der_temperature, result_accuracy, filepath_dertempbounds, current_temperature)
                c_df_dertempbounds_feature = df_dertempbounds_feature[df_dertempbounds_feature["temperature"] == current_temperature]
                c_df_dertempbounds_feature = group_df(c_df_dertempbounds_feature, arguments_der_temperature, result_accuracy, filepath_dertempbounds_feature, current_temperature)

                c_df = pd.concat([c_df_dertempbounds, c_df_dertempbounds_feature], ignore_index=True)
                c_df = c_df[c_df["buffer_size"] == 200]

                results = pd.DataFrame(columns=['model', 'task', 'current_task', 'mean_previous_tasks'])
                for i,_ in enumerate(accmean_columns, start=1):
                        print(i)
                        current_column = f"accuracy_{i}_task{i}"
                        previous_columns = [f"accuracy_{1}_task{i}"]
                        for j in range(2, i):
                                previous_columns.append(f"accuracy_{j}_task{i}")
                        current = pd.DataFrame({'model': c_df['model'], 'task': i, 'current_task': c_df[current_column], 'mean_previous_tasks': c_df[previous_columns].mean(axis=1)})       
                        results =  pd.concat([results, current], ignore_index=True)
                
                
                # Set the aesthetic style of the plots
                sns.set(style="whitegrid")

                color_mapping = {'Network head': '#59A14F', 'Fitted head': '#8C8C8C'}
                plot = sns.lineplot(data=results, x='task', y='mean_previous_tasks', hue='model', marker='o', palette=color_mapping, legend=False)
                lines = plot.get_lines()
                for i, line in enumerate(lines):
                        line.set_linestyle('--')

                sns.lineplot(data=results, x='task', y='current_task', hue='model', marker='o', palette=color_mapping, legend=True)

                # Customize plot
                plt.xlabel('Task')
                plt.ylabel('')
                plt.title('Seq-Cifar10 (CIL)')
                plt.legend(title="")
                plt.grid(True)

                #change
                plt.savefig(filepath_dertempbounds + f"/summary_plot_temp{current_temperature}.png")
                plt.close()
                plt.clf() 
                
def feature_forgetting_task():
        dictlist_dertempbounds_feature, filepath_dertempbounds_feature = get_data(mammoth_path, scenario, dataset_feature, algorithm)
        arguments_der_temperature= get_arguments(dictlist_dertempbounds_feature[0])
        accmean_columns, acctask_columns, result_accuracy = get_arrays(dataset_feature)
        df_dertempbounds_feature = get_dataframe(dictlist_dertempbounds_feature, arguments_der_temperature + acctask_columns + [result_accuracy])

        for current_temperature in df_dertempbounds_feature["temperature"].unique():
                c_df_dertempbounds_feature = df_dertempbounds_feature[df_dertempbounds_feature["temperature"] == current_temperature]
                c_df = group_df(c_df_dertempbounds_feature, arguments_der_temperature, result_accuracy, filepath_dertempbounds_feature, current_temperature)

                results = pd.DataFrame(columns=['task', 'current_task', 'mean_previous_tasks', 'buffer_size'])
                for i,_ in enumerate(accmean_columns, start=1):
                        print(i)
                        current_column = f"accuracy_{i}_task{i}"
                        previous_columns = [f"accuracy_{1}_task{i}"]
                        for j in range(2, i):
                                previous_columns.append(f"accuracy_{j}_task{i}")
                        current = pd.DataFrame({'task': i, 'current_task': c_df[current_column], 'mean_previous_tasks': c_df[previous_columns].mean(axis=1), 'buffer_size': c_df['buffer_size']})       
                        results =  pd.concat([results, current], ignore_index=True)
                
                
                # Set the aesthetic style of the plots
                sns.set(style="whitegrid")
                plot = sns.lineplot(data=results, x='task', y='mean_previous_tasks', hue='buffer_size', marker='o', palette=(sns.color_palette("mako", n_colors=4))[::-1], legend=False)
                lines = plot.get_lines()
                for i, line in enumerate(lines):
                        line.set_linestyle('--')

                sns.lineplot(data=results, x='task', y='current_task', hue='buffer_size', marker='o', palette=(sns.color_palette("mako", n_colors=4))[::-1], legend=True)

                handles, labels = plot.get_legend_handles_labels()
                print(labels)
                new_labels = ['Zero', 'Small', 'Large', 'Infinity']
                plt.legend(handles=handles, labels=new_labels, title="")

                # Customize plot
                plt.xlabel('Task')
                plt.ylabel('')
                plt.title('Seq-Cifar10 (CIL)')
                plt.grid(True)

                #change
                plt.savefig(filepath_dertempbounds_feature + f"/summary_plot_temp{current_temperature}.png")
                plt.close()
                plt.clf() 

def infinite_buffer():
        accmean_columns, acctask_columns, result_accuracy = get_arrays(dataset)
        filepath = mammoth_path + "/data/results/" + scenario + "/" + dataset + "/" + 'infinite_buffer'

        with open(filepath + "/logs.txt", "r") as file:
                dictlist_der = list()
                dictlist_er = list()

                for i, line in enumerate(file, start=1):
                        line = preprocess_line(line)
                        line = ast.literal_eval(line)
                        if 'temperature' in line:
                                if line['model'] == 'der_boundsbalanced':
                                        continue
                                line['model'] = "DER offline"
                                dictlist_der.append(line)
                        elif 'buffer_size' in line:
                                if line['model'] == 'er_bounds':
                                        line['model'] = 'ER offline'
                                        #workaround for grouping
                                        line['buffer_size'] = 111000
                                else: 
                                        line['model'] = "ER balanced"
                                dictlist_er.append(line)
                        else:           
                                line['model'] = "Joint"
                                line[result_accuracy] = line.pop('accmean_task1')
                                line['buffer_size'] = 0
                                dictlist_er.append(line)

        arguments_der = get_arguments(dictlist_der[0])
        arguments_er = get_arguments(dictlist_er[0])

        df_der = get_dataframe(dictlist_der, arguments_der + [result_accuracy])
        df_er = get_dataframe(dictlist_er, arguments_er + [result_accuracy])

        df_der = group_df(df_der, arguments_der, result_accuracy, filepath, 1)
        df_er = group_df(df_er, arguments_er, result_accuracy, filepath, 2)

        df = pd.concat([df_der, df_er])

        #create the plot
        sns.set_theme(style="whitegrid")
        color_mapping = {"ER offline": '#F28E2B', "Joint": '#17becf', "DER offline": '#59A14F', 'ER balanced': '#8C8C8C'}
        hue_order = ['DER offline', 'ER offline', 'ER balanced', 'Joint']
        print(df)
        sns.barplot(data=df, y=result_accuracy, hue="model", legend=True, palette = color_mapping, hue_order=hue_order)
        plt.ylim(70, 100)
        plt.title(f'Seq-Cifar10 (CIL)')
        plt.xlabel('Model')
        plt.ylabel('')
        plt.legend(title='')
        #plt.legend(loc='lower right')

        #change
        plt.savefig(filepath + f"/summary.png")
        plt.close()
        plt.clf() 

def infinite_buffer_tasks():
        _, _, result_accuracy = get_arrays(dataset)
        filepath = mammoth_path + "/data/results/" + scenario + "/" + dataset + "/" + 'infinite_buffer'

        with open(filepath + "/logs.txt", "r") as file:
                dictlist_der = list()
                dictlist_er = list()

                for i, line in enumerate(file, start=1):
                        line = preprocess_line(line)
                        line = ast.literal_eval(line)
                        if 'temperature' in line:
                                if line['model'] == 'der_boundsbalanced':
                                        continue
                                line['model'] = "DER offline"
                                dictlist_der.append(line)
                        elif 'buffer_size' in line:
                                if line['model'] == 'er_bounds':
                                        line['model'] = 'ER offline'
                                        #workaround for grouping
                                        line['buffer_size'] = 111000
                                else: 
                                        line['model'] = "ER balanced"
                                dictlist_er.append(line)

        acctask_columns = []
        if(any(substring in dataset for substring in ["seq-tinyimg"])):
                amount_task = 10
        elif(any(substring in dataset for substring in ["rot-mnist", "perm-mnist"])):
                amount_task = 20
        elif(any(substring in dataset for substring in ["seq-mnist", "seq-cifar10"])):
                amount_task = 5
        
        for j in range(1, amount_task+1):
                acctask_columns.append(f"accuracy_{j}_task{amount_task}")

        print(acctask_columns)
        arguments_der = get_arguments(dictlist_der[0])
        arguments_er = get_arguments(dictlist_er[0])

        df_der = get_dataframe(dictlist_der, arguments_der + acctask_columns + [result_accuracy])
        df_er = get_dataframe(dictlist_er, arguments_er + acctask_columns + [result_accuracy])

        df_der = group_df(df_der, arguments_der, result_accuracy, filepath, 1)
        df_er = group_df(df_er, arguments_er, result_accuracy, filepath, 2)

        df = pd.concat([df_der, df_er])

        df['Mean previous tasks'] = df[acctask_columns[:-1]].mean(axis=1)  
        df['Current task'] = df[acctask_columns[-1]]
        print(df)

        melted_df = pd.melt(df, id_vars='model', value_vars=['Mean previous tasks', 'Current task'],
                    var_name='result_type', value_name='value')

        #create the plot
        sns.set_theme(style="whitegrid")
        print(melted_df)
        #hue_order = ['Current task', 'Mean previous tasks']
        sns.barplot(data=melted_df, y='value', x='model', hue='result_type', legend=False)
        plt.ylim(20, 70)
        plt.title(f'Seq-TinyIMG (CIL)')
        plt.xlabel('Method')
        plt.ylabel('')
        #plt.legend(title='')
        #plt.legend(loc='lower right')

        #change
        plt.savefig(filepath + f"/summary.png")
        plt.close()
        plt.clf() 

def scaling():
        dictlist_der, filepath_der = get_data(mammoth_path, scenario, dataset, algorithm)
        arguments_der = get_arguments(dictlist_der[0]) + ["backbone"]
        accmean_columns, acctask_columns, result_accuracy = get_arrays(dataset)
        df_der = get_dataframe(dictlist_der, arguments_der + [result_accuracy, "forgetting"])
        #df_der['backbone'] = df_der['backbone'].astype(str)

        result = group_dfscaling(df_der, arguments_der, result_accuracy, filepath_der, 0)
        print(result)
        
        sns.set_theme(style="whitegrid")
        plot = sns.barplot(data=result, x='buffer_size', y=result_accuracy, hue="backbone", legend=True, palette=sns.color_palette("crest"))
        plt.ylim(30, 80)

        #handles, labels = plot.get_legend_handles_labels()
        #print(labels)
        #new_labels = ['100K', '400K', '1.6M', '4.8M', '11.7M', '21.8M']
        #plt.legend(handles=handles, labels=new_labels, title="")

        plt.title(f'Seq-TinyIMG (CIL)')
        plt.xlabel('Buffer Size')
        plt.ylabel('')
        #plt.legend(title='Backbone')
        #plt.legend(loc='lower right')

        #change
        plt.savefig(filepath_der + f"/summary_accuracy.png")
        plt.close()
        plt.clf() 

        #forgetting
        sns.set_theme(style="whitegrid")
        sns.barplot(data=result, x='buffer_size', y="forgetting", hue="backbone", legend=True, palette=sns.color_palette("crest"))
        plt.ylim(30, 80)

        handles, labels = plot.get_legend_handles_labels()
        print(labels)
        new_labels = ['100K', '400K', '1.6M', '4.8M', '11.7M', '21.8M']
        plt.legend(handles=handles, labels=new_labels, title="")

        plt.title(f'Seq-TinyIMG (CIL)')
        plt.xlabel('Buffer Size')
        plt.ylabel('Forgetting')
        #plt.legend(title='Backbone')
        #plt.legend(loc='lower right')

        #change
        plt.savefig(filepath_der + f"/summary_forgetting.png")
        plt.close()
        plt.clf() 

def scaling_previous():
        dictlist_der_buffer, filepath_der_buffer = get_data(mammoth_path, scenario, dataset, "der_scaling_buffer")
        dictlist_der_normal, filepath_der_normal = get_data(mammoth_path, scenario, dataset_feature, "der_scaling")
        dictlist_er_buffer, filepath_er_buffer = get_data(mammoth_path, scenario, dataset, "er_scaling_buffer")
        dictlist_er_normal, filepath_er_normal = get_data(mammoth_path, scenario, dataset_feature, "er_scaling")
        arguments_der = get_arguments(dictlist_der_normal[0]) + ["backbone"]
        arguments_er = get_arguments(dictlist_er_normal[0]) + ["backbone"]

        acctask_columns = []
        if(any(substring in dataset for substring in ["seq-tinyimg"])):
                amount_task = 10
        elif(any(substring in dataset for substring in ["rot-mnist", "perm-mnist"])):
                amount_task = 20
        elif(any(substring in dataset for substring in ["seq-mnist", "seq-cifar10"])):
                amount_task = 5
        
        for j in range(1, amount_task):
                acctask_columns.append(f"accuracy_{j}_task{amount_task}")

        df_der_buffer = get_dataframe(dictlist_der_buffer, arguments_der + acctask_columns)
        replace(df_der_buffer, "model", "DER buffer")
        df_der_buffer['accuracy'] = df_der_buffer[acctask_columns].mean(axis=1)       

        df_der_normal = get_dataframe(dictlist_der_normal, arguments_der + acctask_columns)
        replace(df_der_normal, "model", "DER offline")
        df_der_normal['accuracy'] = df_der_normal[acctask_columns].mean(axis=1)

        df_er_buffer = get_dataframe(dictlist_er_buffer, arguments_er + acctask_columns)
        replace(df_er_buffer, "model", "ER buffer")
        df_er_buffer['accuracy'] = df_er_buffer[acctask_columns].mean(axis=1)       

        df_er_normal = get_dataframe(dictlist_er_normal, arguments_er + acctask_columns)
        replace(df_er_normal, "model", "ER offline")
        df_er_normal['accuracy'] = df_er_normal[acctask_columns].mean(axis=1)

        for current_buffersize in df_der_normal["buffer_size"].unique():
                c_df_der_buffer = df_der_buffer[df_der_buffer["buffer_size"] == current_buffersize]
                c_df_der_normal = df_der_normal[df_der_normal["buffer_size"] == current_buffersize]
                c_df_er_buffer = df_er_buffer[df_er_buffer["buffer_size"] == current_buffersize]
                c_df_er_normal = df_er_normal[df_er_normal["buffer_size"] == current_buffersize]


                c_df_der_buffer = group_dfscaling(c_df_der_buffer, arguments_der, "accuracy", filepath_der_buffer, 0)
                c_df_der_normal = group_dfscaling(c_df_der_normal, arguments_der, "accuracy", filepath_der_normal, 0)
                c_df_er_buffer = group_dfscaling(c_df_er_buffer, arguments_er, "accuracy", filepath_er_buffer, 0)
                c_df_er_normal = group_dfscaling(c_df_er_normal, arguments_er, "accuracy", filepath_er_normal, 0)

                result = pd.concat([c_df_der_buffer, c_df_der_normal, c_df_er_buffer, c_df_er_normal])

                sns.set_theme(style="whitegrid")
                order = [1, 2, 3, 4, 5, 6]
                hue_order = ["DER buffer", "DER offline", "ER buffer", "ER offline"]
                color_mapping = {"ER offline": '#F28E2B', "DER buffer": '#4E79A7', "DER offline": '#59A14F', 'ER buffer': '#C73A2D'}
                sns.barplot(data=result, x='backbone', y="accuracy", hue="model", legend=False, order=order, palette=color_mapping, hue_order=hue_order)

                custom_labels = ['100K', '400K', '1.6M', '4.8M', '11.7M', '21.8M']
                plt.xticks(ticks=range(len(order)), labels=custom_labels)

                #plt.ylim(60, 90)
                plt.title(f'Seq-TinyIMG (CIL)')
                plt.xlabel('Network Size')
                plt.ylabel('Accuracy')
                #plt.legend(title='')
                #plt.legend(loc='lower right')

                #change
                plt.savefig(filepath_der_buffer + f"/summary_accuracy_buffersize{current_buffersize}.png")
                plt.close()
                plt.clf()       

def BN_separating():
        dictlist_der_together, filepath_der_together = get_data(mammoth_path, scenario, dataset, "der_together")
        dictlist_der, filepath_der = get_data(mammoth_path, scenario, dataset, "der_separated_notalign")
        dictlist_der_separated, filepath_der_separated = get_data(mammoth_path, scenario, dataset, "der_separated")

        ditclist_er_separated, filepath_er_separated = get_data(mammoth_path, scenario, dataset, "er_separated")
        ditclist_er_together, filepath_er_together = get_data(mammoth_path, scenario, dataset, "er_together")

        arguments_der = get_arguments(dictlist_der[0])
        arguments_er = get_arguments(ditclist_er_together[0])
        _, _, result_accuracy = get_arrays(dataset)

        df_der_together = get_dataframe(dictlist_der_together, arguments_der + [result_accuracy])
        df_der_together["method"] = "DER online"
        df_der_together["version"] = 2
        df_der = get_dataframe(dictlist_der, arguments_der + [result_accuracy])
        df_der["method"] = "DER online"
        df_der["version"] = 0
        df_der_separated = get_dataframe(dictlist_der_separated, arguments_der + [result_accuracy])
        df_der_separated["method"] = "DER online"
        df_der_separated["version"] = 1

        df_er_separated = get_dataframe(ditclist_er_separated, arguments_er  + ["align_bn", result_accuracy])
        df_er_separated["method"] = "ER online"
        df_er_separated["version"] = df_er_separated["align_bn"]

        df_er_together = get_dataframe(ditclist_er_together, arguments_er + [result_accuracy])
        df_er_together["method"] = "ER online"
        df_er_together["version"] = 2

        df = pd.concat([df_der_together, df_der, df_der_separated, df_er_separated, df_er_together], ignore_index=True)

        for current_buffersize in df["buffer_size"].unique():
                c_df = df[df["buffer_size"] == current_buffersize]

                #printing
                sns.set_theme(style="whitegrid")
                order = [2, 0, 1]
                color_mapping = {"ER online": '#F28E2B', "DER online": '#4E79A7'}
                plot = sns.barplot(data=c_df, x='version', y=result_accuracy, hue="method", legend=False, palette=color_mapping, order=order)
                #plt.ylim(50, 90)

                custom_labels = ['Together', 'Separated', 'Separated + BN']
                plt.xticks(ticks=range(len(order)), labels=custom_labels)

                #hatch bars
                bars_to_hatch = [(2, 'ER online'), (0, 'DER online')]

                for i, bar in enumerate(plot.patches):
                        
                        # Check if this bar is in the bars_to_hatch list
                        if(i==1 or i==3):
                                bar.set_hatch('//')  # Choose any hatch pattern, e.g., '/', '\\', 'x'

                plt.title(f'Seq-TinyIMG (CIL)')
                plt.xlabel('Forward Pass')
                plt.ylabel('')
                #plt.legend(title='')
                #plt.legend(loc='lower right')

                #change
                plt.savefig(filepath_er_separated + f"/summary_accuracy_buffersize{current_buffersize}.png")
                plt.close()
                plt.clf() 

def LN_separating():
        dictlist_der_together, filepath_der_together = get_data(mammoth_path, scenario, dataset, "der_together")
        dictlist_der, filepath_der = get_data(mammoth_path, scenario, dataset, "der")

        ditclist_er_separated, filepath_er_separated = get_data(mammoth_path, scenario, dataset, "er_separated")
        ditclist_er, filepath_er_ = get_data(mammoth_path, scenario, dataset, "er")

        arguments_der = get_arguments(dictlist_der[0])
        arguments_er = get_arguments(ditclist_er[0])
        _, _, result_accuracy = get_arrays(dataset)

        df_der_together = get_dataframe(dictlist_der_together, arguments_der + [result_accuracy])
        df_der_together =  group_df(df_der_together, arguments_der, result_accuracy, filepath_der_together, 0)
        df_der_together["method"] = "DER online"
        df_der_together["version"] = 2
        df_der = get_dataframe(dictlist_der, arguments_der + [result_accuracy])
        df_der = group_df(df_der, arguments_der, result_accuracy, filepath_der, 0)
        df_der["method"] = "DER online"
        df_der["version"] = 0

        df_er_separated = get_dataframe(ditclist_er_separated, arguments_er  + [result_accuracy])
        df_er_separated["method"] = "ER online"
        df_er_separated["version"] = 0

        df_er_together = get_dataframe(ditclist_er, arguments_er + [result_accuracy])
        df_er_together["method"] = "ER online"
        df_er_together["version"] = 2

        df = pd.concat([df_der_together, df_der, df_der, df_er_separated, df_er_together], ignore_index=True)

        for current_buffersize in df_der_together["buffer_size"].unique():
                c_df = df[df["buffer_size"] == current_buffersize]

                #printing
                sns.set_theme(style="whitegrid")
                order = [2, 0]
                color_mapping = {"ER online": '#F28E2B', "DER online": '#4E79A7'}
                plot = sns.barplot(data=c_df, x='version', y=result_accuracy, hue="method", legend=False, palette=color_mapping, order=order)
                plt.ylim(4, 12)

                custom_labels = ['Together', 'Separated']
                plt.xticks(ticks=range(len(order)), labels=custom_labels)

                for i, bar in enumerate(plot.patches):
                        # Check if this bar is in the bars_to_hatch list
                        if(i==1 or i==2):
                                bar.set_hatch('//')  # Choose any hatch pattern, e.g., '/', '\\', 'x'

                plt.title(f'Seq-TinyIMG (CIL)')
                plt.xlabel('Forward Pass')
                plt.ylabel('Accuracy')
                #plt.legend(title='')
                #plt.legend(loc='lower right')

                #change
                plt.savefig(filepath_er_separated + f"/summary_accuracy_buffersize{current_buffersize}.png")
                plt.close()
                plt.clf() 

scaling_previous()
'''
dictlist_dertempbounds, filepath_dertempbounds = get_data(mammoth_path, scenario, dataset_feature, "der_tempbounds")
dictlist_derpretrained, filepath_derpretrained = get_data(mammoth_path, scenario, dataset_feature, "der_pretrained")
dictlist_derpermuted, filepath_derpermuted = get_data(mammoth_path, scenario, dataset_feature, "der_permuted")
dictlist_erlabelsmoothing, filepath_erlabelsmoothing = get_data(mammoth_path, scenario, dataset_feature, "er_labelsmoothing")
dictlist_er, filepath_er = get_data(mammoth_path, scenario, dataset_feature, "er_bounds")

arguments_der_temperature= get_arguments(dictlist_derpretrained[0])
arguments_er = get_arguments(dictlist_erlabelsmoothing[0])

accmean_columns, acctask_columns, result_accuracy = get_arrays(dataset)

df_dertempbounds = get_dataframe(dictlist_dertempbounds, arguments_der_temperature + [result_accuracy])
replace(df_dertempbounds, "model", "DER offline")
df_derpretrained = get_dataframe(dictlist_derpretrained, arguments_der_temperature + [result_accuracy])
replace(df_derpretrained, "model", "DER JTTeacher")      
df_derpermuted = get_dataframe(dictlist_derpermuted, arguments_der_temperature + [result_accuracy])
replace(df_derpermuted, "model", "DER JTTeacher + permuting")
df_erlabelsmoothing = get_dataframe(dictlist_erlabelsmoothing, arguments_er + [result_accuracy])
replace(df_erlabelsmoothing, "model", "Label Smoothing")

current_temperature = df_dertempbounds["temperature"].unique()[0]

c_df_dertempbounds = df_dertempbounds[df_dertempbounds["temperature"] == current_temperature]
c_df_derpretrained = df_derpretrained[df_derpretrained["temperature"] == current_temperature]
c_df_derpermuted = df_derpermuted[df_derpermuted["temperature"] == current_temperature]

dump_df(c_df_dertempbounds, arguments_der_temperature, result_accuracy, filepath_dertempbounds, current_temperature)
dump_df(c_df_derpretrained, arguments_der_temperature, result_accuracy, filepath_derpretrained, current_temperature)
dump_df(c_df_derpermuted, arguments_der_temperature, result_accuracy, filepath_derpermuted, current_temperature)
dump_df(df_erlabelsmoothing, arguments_er, result_accuracy, filepath_erlabelsmoothing, 0)

result = group_df(c_df_derpermuted, arguments_der_temperature, result_accuracy, filepath_derpermuted, current_temperature)
current_result = group_df(c_df_derpretrained, arguments_der_temperature, result_accuracy, filepath_derpretrained, current_temperature)
result = pd.concat([result, current_result])
current_result = group_df(c_df_dertempbounds, arguments_der_temperature, result_accuracy, filepath_dertempbounds, current_temperature)
result = pd.concat([result, current_result])
current_result = group_df(df_erlabelsmoothing, arguments_er, result_accuracy, filepath_erlabelsmoothing, 0)
result = pd.concat([result, current_result])

#create the plot
sns.set_theme(style="whitegrid")
color_mapping = {"Label Smoothing": '#F28E2B', "DER JTTeacher + permuting": '#4E79A7', "DER JTTeacher": '#17becf', "DER offline": '#59A14F'}
plot = sns.lineplot(data=result, x='buffer_size', y=result_accuracy, hue='model', marker='o', palette=color_mapping, legend=False)

lines = plot.get_lines()
for i, line in enumerate(lines):
    line.set_linestyle("--")

dictlist_der, filepath_der = get_data(mammoth_path, scenario, dataset, "der")
dictlist_dertempbounds, filepath_dertempbounds = get_data(mammoth_path, scenario, dataset, "der_tempbounds")
dictlist_er, filepath_er = get_data(mammoth_path, scenario, dataset, "er_bounds")

dictlist_dertempbounds, filepath_dertempbounds = get_data(mammoth_path, scenario, dataset, "der_tempbounds")
dictlist_derpretrained, filepath_derpretrained = get_data(mammoth_path, scenario, dataset, "der_pretrained")
dictlist_derpermuted, filepath_derpermuted = get_data(mammoth_path, scenario, dataset, "der_permuted")
dictlist_erlabelsmoothing, filepath_erlabelsmoothing = get_data(mammoth_path, scenario, dataset, "er_labelsmoothing")
dictlist_er, filepath_er = get_data(mammoth_path, scenario, dataset, "er_bounds")

arguments_der_temperature= get_arguments(dictlist_derpretrained[0])
arguments_er = get_arguments(dictlist_erlabelsmoothing[0])

accmean_columns, acctask_columns, result_accuracy = get_arrays(dataset)

df_dertempbounds = get_dataframe(dictlist_dertempbounds, arguments_der_temperature + [result_accuracy])
replace(df_dertempbounds, "model", "DER offline")
df_derpretrained = get_dataframe(dictlist_derpretrained, arguments_der_temperature + [result_accuracy])
replace(df_derpretrained, "model", "DER JTTeacher")      
df_derpermuted = get_dataframe(dictlist_derpermuted, arguments_der_temperature + [result_accuracy])
replace(df_derpermuted, "model", "DER JTTeacher + permuting")
df_erlabelsmoothing = get_dataframe(dictlist_erlabelsmoothing, arguments_er + [result_accuracy])
replace(df_erlabelsmoothing, "model", "Label Smoothing")

c_df_dertempbounds = df_dertempbounds[df_dertempbounds["temperature"] == current_temperature]
c_df_derpretrained = df_derpretrained[df_derpretrained["temperature"] == current_temperature]
c_df_derpermuted = df_derpermuted[df_derpermuted["temperature"] == current_temperature]

dump_df(c_df_dertempbounds, arguments_der_temperature, result_accuracy, filepath_dertempbounds, current_temperature)
dump_df(c_df_derpretrained, arguments_der_temperature, result_accuracy, filepath_derpretrained, current_temperature)
dump_df(c_df_derpermuted, arguments_der_temperature, result_accuracy, filepath_derpermuted, current_temperature)
dump_df(df_erlabelsmoothing, arguments_er, result_accuracy, filepath_erlabelsmoothing, 0)

result = group_df(c_df_derpermuted, arguments_der_temperature, result_accuracy, filepath_derpermuted, current_temperature)
current_result = group_df(c_df_derpretrained, arguments_der_temperature, result_accuracy, filepath_derpretrained, current_temperature)
result = pd.concat([result, current_result])
current_result = group_df(c_df_dertempbounds, arguments_der_temperature, result_accuracy, filepath_dertempbounds, current_temperature)
result = pd.concat([result, current_result])
current_result = group_df(df_erlabelsmoothing, arguments_er, result_accuracy, filepath_erlabelsmoothing, 0)
result = pd.concat([result, current_result])

#create the plot
sns.set_theme(style="whitegrid")
color_mapping = {"Label Smoothing": '#F28E2B', "DER JTTeacher + permuting": '#4E79A7', "DER JTTeacher": '#17becf', "DER offline": '#59A14F'}
sns.lineplot(data=result, x='buffer_size', y=result_accuracy, hue='model', marker='o', palette=color_mapping, legend=True)
#plt.title(f'Accuracy vs Buffer Size by Model for Temperature = {current_temperature}')
plt.title(f'Seq-Cifar10 (CIL)')
plt.xlabel('Buffer Size')
plt.ylabel('')
plt.legend(title='')
#plt.legend(loc='lower right')

#change
plt.savefig(filepath_er + f"/summary_plot_temp{current_temperature}.png")
plt.close()
plt.clf() 
'''


"""
dictlist_der, filepath_der = get_data(mammoth_path, scenario, dataset, "der")
dictlist_dertempbounds, filepath_dertempbounds = get_data(mammoth_path, scenario, dataset, "der_tempbounds")
dictlist_derpretrained, filepath_derpretrained = get_data(mammoth_path, scenario, dataset, "der_pretrained")
dictlist_derpretrained2, filepath_derpretrained2 = get_data(mammoth_path, scenario, dataset, "der_pretrained2")
dictlist_derpermuted, filepath_derpermuted = get_data(mammoth_path, scenario, dataset, "der_permuted")
dictlist_er, filepath_er = get_data(mammoth_path, scenario, dataset, "er_bounds")
dictlist_erlabelsmoothing, filepath_erlabelsmoothing = get_data(mammoth_path, scenario, dataset, "er_labelsmoothing")

arguments_der = get_arguments(dictlist_der[0])
arguments_der_temperature= get_arguments(dictlist_dertempbounds[0])
arguments_er = get_arguments(dictlist_er[0])

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

df_der = get_dataframe(dictlist_der, arguments_der + [result_accuracy])
replace(df_der, "model", "DER-online")
df_dertempbounds = get_dataframe(dictlist_dertempbounds, arguments_der_temperature + [result_accuracy])
replace(df_dertempbounds, "model", "DER-offline")
df_derpretrained = get_dataframe(dictlist_derpretrained, arguments_der_temperature + [result_accuracy])
replace(df_derpretrained, "model", "DER-pretrained")
df_derpretrained2 = get_dataframe(dictlist_derpretrained2, arguments_der_temperature + [result_accuracy])
replace(df_derpretrained2, "model", "DER-pretrained + masking")
df_derpermuted = get_dataframe(dictlist_derpermuted, arguments_der_temperature + [result_accuracy])
replace(df_derpermuted, "model", "DER-pretrained + permuting")
df_er = get_dataframe(dictlist_er, arguments_er + [result_accuracy])
replace(df_er, "model", "ER-offline")
df_erlabelsmoothing = get_dataframe(dictlist_erlabelsmoothing, arguments_er + [result_accuracy])
replace(df_erlabelsmoothing, "model", "ER-offline + labelsmoothing")

for current_temperature in df_dertempbounds["temperature"].unique():
        c_df_dertempbounds = df_dertempbounds[df_dertempbounds["temperature"] == current_temperature]
        c_df_derpretrained = df_derpretrained[df_derpretrained["temperature"] == current_temperature]
        c_df_derpretrained2 = df_derpretrained2[df_derpretrained2["temperature"] == current_temperature]
        c_df_derpermuted = df_derpermuted[df_derpermuted["temperature"] == current_temperature]

        dump_df(df_der, arguments_der, result_accuracy, filepath_der, 0)
        dump_df(c_df_dertempbounds, arguments_der_temperature, result_accuracy, filepath_dertempbounds, current_temperature)
        dump_df(c_df_derpretrained, arguments_der_temperature, result_accuracy, filepath_derpretrained, current_temperature)
        dump_df(c_df_derpretrained2, arguments_der_temperature, result_accuracy, filepath_derpretrained2, current_temperature)
        dump_df(c_df_derpermuted, arguments_der_temperature, result_accuracy, filepath_derpermuted, current_temperature)
        dump_df(df_er, arguments_er, result_accuracy, filepath_er, 0)
        dump_df(df_erlabelsmoothing, arguments_er, result_accuracy, filepath_erlabelsmoothing, 0)

        result = group_df(df_der, arguments_der, result_accuracy, filepath_der, 0)
        current_result = group_df(c_df_dertempbounds, arguments_der_temperature, result_accuracy, filepath_dertempbounds, current_temperature)
        result = pd.concat([result, current_result])
        current_result = group_df(c_df_derpretrained, arguments_der_temperature, result_accuracy, filepath_derpretrained, current_temperature)
        result = pd.concat([result, current_result])
        current_result = group_df(c_df_derpretrained2, arguments_der_temperature, result_accuracy, filepath_derpretrained2, current_temperature)
        result = pd.concat([result, current_result])
        current_result = group_df(c_df_derpermuted, arguments_der_temperature, result_accuracy, filepath_derpermuted, current_temperature)
        result = pd.concat([result, current_result])
        current_result = group_df(df_er, arguments_er, result_accuracy, filepath_er, 0)
        result = pd.concat([result, current_result])
        current_result = group_df(df_erlabelsmoothing, arguments_er, result_accuracy, filepath_erlabelsmoothing, 0)
        result = pd.concat([result, current_result])

        #create the plot
        sns.set_theme(style="whitegrid")
        sns.lineplot(data=result, x='buffer_size', y=result_accuracy, hue='model', marker='o', legend=True)
        #plt.title(f'Accuracy vs Buffer Size by Model for Temperature = {current_temperature}')
        plt.title(f'b) Seq-Cifar10')
        plt.xlabel('Buffer Size')
        plt.ylabel('')
        plt.legend(title='Model')

        #change
        plt.savefig(filepath_er + f"/summary_plot_temp{current_temperature}.png")
        plt.close()
        plt.clf() 
"""