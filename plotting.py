import numpy as np
import os
import json

from Helper import smooth
from Helper import LearningCurvePlot


# func to plot

#TODO add option for boolean values
def plot_hyperparameter(data, names, title, param_name, param_key, filename):
    plot = LearningCurvePlot(title)
    for name in names:
        experiment = data[name]
        result = np.array(experiment['results']).T
        x = result[0]
        y = result[1]
        param = experiment[param_key]
        label = param_name + ": " + str(param)
        plot.add_curve(x, y, label)
    plot.save(filename)

def plot_compare_tn_rb(data, names, title, filename):
    plot = LearningCurvePlot(title)
    for name in names:
        experiment = data[name]
        result = np.array(experiment['results']).T
        x = result[0]
        y = result[1]
        label = "DQN"
        if "tn" in name:
            label += " with Target Network on"
        if "rb" in name:
            label += " with Replay Buffer on"
        plot.add_curve(x, y, label)
    plot.save(filename)


# code to plot experiments
def plot_experiments():
    data = load_data("data.json")
    experiments = extract(data)
    
    # plot 1 Learning rate
    names = ['DQN', 'DQN_rb_1000']
    title, param_key, param_name = "Learning rate compared of DQN agents", "learning_rate", "Learning rate"
    plot_hyperparameter(data, names, title, param_name, param_key, "lr.png")

    # plot 2
    names = ['DQN', 'DQN_rb_1000', 'DQN_tn_100', 'DQN_rb_1000_tn_100']
    title = "Target Network and Replay Buffer compared"
    plot_compare_tn_rb(data, names, title, "TN RB compared.png")

    print()

def extract(data):
    experiments = []
    for key in data.keys():
        experiment = data[key]
        experiment['name'] = key
        experiments.append(experiment)
    return experiments

def load_data(data_path):
    if os.path.exists(data_path):
        # Load the JSON file
        with open(data_path, 'r') as file:
            data = json.load(file)
        print("JSON file loaded successfully.")
    else:
        data = {}
        print("JSON file does not exist.")
    k = data.keys()
    print(k)
    return data

if __name__ == '__main__':
    plot_experiments()