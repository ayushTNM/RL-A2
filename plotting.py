import numpy as np
import os
import json

from Helper import smooth
from Helper import LearningCurvePlot


# func to plot
def plot(data, names, title, param_name):
    plot = LearningCurvePlot(title)
    for name in names:
        experiment = data[name]
        result = np.array(experiment['results']).T
        x = result[0]
        y = result[1]
        param = experiment[param_name]
        label = param_name + ": " + param
        plot.add_curve(x, y, label)


# code to plot experiments
def plot_experiments():
    data_path = "data.json"
    if os.path.exists(data_path):
        # Load the JSON file
        with open(data_path, 'r') as file:
            data = json.load(file)
        print("JSON file loaded successfully.")
    else:
        data = {}
        print("JSON file does not exist.")
    print(data['DQN'])
    k = data.keys()
    print(k)
    experiments = extract(data)

    # plot 1 Learning rate
    names = ['DQN', 'DQN_rb_1000']
    title, param_name = "Learning rate compared of DQN agents", "Learning rate"
    plot(data, names, title, param_name)

    # plot 2
    names = ['DQN', 'DQN_rb_1000']
    title, param_name = " ε compared of DQN agents", "ε"
    plot(data, names, title, param_name)


    print()
    #TODO settings to show which hyper parameter

def extract(data):
    experiments = []
    for key in data.keys():
        experiment = data[key]
        experiment['name'] = key
        experiments.append(experiment)
    return experiments

if __name__ == '__main__':
    plot_experiments()