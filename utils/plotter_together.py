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

def preprocess_line(line):
       # Remove the 'device' entry
        line = re.sub(r"'device': device\(.*?\),?", "", line)
        return line

def get_data(base, type, dataset, model):
        filepath = base + "/data/results/" + type + "/" + dataset + "/" + model
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
                average_result= c_df.groupby(group_arguments).mean().reset_index()
                max_average_res = average_result.loc[average_result[result_accuracy].idxmax()]
                for argument in group_arguments:
                        c_df = c_df[(c_df[argument] == max_average_res[argument])]
                result = pd.concat([result, c_df])

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
        return result


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
replace(df_der, "model", "der-online")
df_dertempbounds = get_dataframe(dictlist_dertempbounds, arguments_der_temperature + [result_accuracy])
replace(df_dertempbounds, "model", "der-offline")
df_derpretrained = get_dataframe(dictlist_derpretrained, arguments_der_temperature + [result_accuracy])
replace(df_derpretrained, "model", "der-pretrained")
df_derpretrained2 = get_dataframe(dictlist_derpretrained2, arguments_der_temperature + [result_accuracy])
replace(df_derpretrained2, "model", "der-pretrained + masking")
df_derpermuted = get_dataframe(dictlist_derpermuted, arguments_der_temperature + [result_accuracy])
replace(df_derpermuted, "model", "der-pretrained + permuting")
df_er = get_dataframe(dictlist_er, arguments_er + [result_accuracy])
replace(df_er, "model", "er-offline")
df_erlabelsmoothing = get_dataframe(dictlist_erlabelsmoothing, arguments_er + [result_accuracy])
replace(df_erlabelsmoothing, "model", "er-offline + labelsmoothing")

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

        #result = group_df(df_der, arguments_der, result_accuracy, filepath_der, 0)
        #current_result = group_df(c_df_dertempbounds, arguments_der_temperature, result_accuracy, filepath_dertempbounds, current_temperature)
        #result = pd.concat([result, current_result])
        #current_result = group_df(c_df_derpretrained, arguments_der_temperature, result_accuracy, filepath_derpretrained, current_temperature)
        #result = pd.concat([result, current_result])
        #current_result = group_df(c_df_derpretrained2, arguments_der_temperature, result_accuracy, filepath_derpretrained2, current_temperature)
        #result = pd.concat([result, current_result])
        #current_result = group_df(c_df_derpermuted, arguments_der_temperature, result_accuracy, filepath_derpermuted, current_temperature)
        #result = pd.concat([result, current_result])
        #current_result = group_df(df_er, arguments_er, result_accuracy, filepath_er, 0)
        #result = pd.concat([result, current_result])
        #current_result = group_df(df_erlabelsmoothing, arguments_er, result_accuracy, filepath_erlabelsmoothing, 0)
        #result = pd.concat([result, current_result])

        result = group_df(c_df_derpermuted, arguments_der_temperature, result_accuracy, filepath_derpermuted, current_temperature)
        current_result = group_df(c_df_derpretrained, arguments_der_temperature, result_accuracy, filepath_derpretrained, current_temperature)
        result = pd.concat([result, current_result])
        current_result = group_df(df_erlabelsmoothing, arguments_er, result_accuracy, filepath_erlabelsmoothing, 0)
        result = pd.concat([result, current_result])

        #create the plot
        sns.set_theme(style="whitegrid")
        sns.lineplot(data=result, x='buffer_size', y=result_accuracy, hue='model', marker='o')
        #plt.title(f'Accuracy vs Buffer Size by Model for Temperature = {current_temperature}')
        plt.title(f'c) Seq-TinyIMG')
        plt.xlabel('Buffer Size')
        plt.ylabel('Mean Accuracy')
        plt.legend(title='Model')

        #change
        plt.savefig(filepath_er + f"/summary_plot_temp{current_temperature}.png")
        plt.close()
        plt.clf() 
