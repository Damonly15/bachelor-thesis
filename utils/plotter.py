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
model = sys.argv[3]

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

def group_df(df, group_arguments, result_accuracy, filepath, temperature):
        result = pd.DataFrame()
        for current_buffersize in df["buffer_size"].unique():
                c_df = df[df["buffer_size"] == current_buffersize]
                average_result= c_df.groupby(group_arguments).mean().reset_index()
                max_average_res = average_result.loc[average_result[result_accuracy].idxmax()]
                for argument in group_arguments:
                        c_df = c_df[(c_df[argument] == max_average_res[argument])]
                result = pd.concat([result, c_df])

        if not os.path.exists(filepath + f"/temperature{temperature}"):
            os.makedirs(filepath + f"/temperature{temperature}")
        with open(filepath + f"/temperature{temperature}/best_temp{temperature}.txt", "w") as file:
            result_grouped = result.groupby(group_arguments).mean().reset_index()
            file.write(result_grouped[group_arguments + [result_accuracy]].to_string())
        return result

def dump_df(df, group_arguments, result_accuracy, filepath, temperature):
        df_group = df.groupby(group_arguments).mean()
        df_group["runs"] = df.groupby(group_arguments).size()
        df_group["range"] =  (df.groupby(group_arguments)[result_accuracy].max() - df.groupby(group_arguments)[result_accuracy].min()) / 2
        df_group = df_group.reset_index()
        if not os.path.exists(filepath + f"/temperature{temperature}"):
            os.makedirs(filepath + f"/temperature{temperature}")
        with open(filepath + f"/temperature{temperature}/summary_temp{temperature}.txt", "w") as file:
            file.write(df_group[group_arguments + ["runs", result_accuracy, "range"]].to_string())

dictlist, filepath = get_data(mammoth_path, scenario, dataset, model)
arguments = get_arguments(dictlist[0])

accmean_columns = list()
acctask_columns = list()
amount_task = 0
if(dataset == "seq-tinyimg"):
        amount_task = 10
elif(dataset in ["rot-mnist", "perm-mnist"]):
        amount_task = 20
elif(dataset in["seq-mnist", "seq-cifar10"]):
        amount_task = 5
for i in range(1, amount_task+1):
        accmean_columns.append(f"accmean_task{i}")
        if(scenario == "domain-il"):
                continue
        for j in range(1, i+1):
                acctask_columns.append(f"accuracy_{j}_task{i}")
result_accuracy = accmean_columns[-1]

df = get_dataframe(dictlist, arguments + [result_accuracy])

if "temperature" in arguments:
        for current_temperature in df["temperature"].unique():
                c_df = df[df["temperature"] == current_temperature]

                dump_df(c_df, arguments, result_accuracy, filepath, current_temperature)
else:
       dump_df(df, arguments, result_accuracy, filepath, 0)