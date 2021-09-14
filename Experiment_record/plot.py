import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas import DataFrame
import seaborn as sns
import os
import sys


df = pd.read_csv("/home/gamma/wb_alchemy/sub_project/Experiment_record/multi_steps_0913_01.csv")
df = df.drop(df[df["episode"]>100].index)
# df = df.drop(df[1500<df["episode"]].index)
sns.set(rc = {'figure.figsize':(10,5)})


def plot_original():
    # data = df.query("experiment != 'Reptile'")
    # data["smooth_average_distance"] = data["distance"].ewm(span=80).mean()
    plt.title(" different meta algorithms in RL")
    sns.lineplot(data=df, x='episode', y='distance', hue='experiment')
    # sns.lineplot(data=df, x='episode', y='smooth_average_distance', hue="experiment")

    # data2 = df.query("experiment == 'Medium_See_goal'")
    # data2["smooth_average_distance"] = data2["distance"].ewm(span=80).mean()
    # sns.lineplot(data=data2, x='episode', y='distance', palette=['gray'])
    # sns.lineplot(data=data2, x='episode', y='smooth_average_distance', palette=['green'],hue="experiment")

    # data3 = df.query("experiment == 'Original_See_goal'")
    # data3["smooth_average_distance"] = data3["distance"].ewm(span=80).mean()
    # sns.lineplot(data=data3, x='episode', y='distance', palette=['gray'])
    # sns.lineplot(data=data3, x='episode', y='smooth_average_distance', palette=['red'],hue="experiment")
    plt.savefig("/home/gamma/wb_alchemy/sub_project/Experiment_record/{}".format("multi_steps_0913_03.png"))
    print("done")


plot_original()


