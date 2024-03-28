import numpy as np
import os
import json

from Helper import smooth
from Helper import LearningCurvePlot


# func to plot

#TODO experiment code is not made to do these experiments
def plot_hyperparameter(data, names, title, filename, param_name, param_key):
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
        if "tn" in name and "rb" in name: #TODO make dependant on actual settings, not on name
            label = "DQN with target network and replay buffer"
        elif "tn" in name: #TODO make this prettier
            label += "DQN with target network"
        elif "rb" in name:
            label += "DQN with replay buffer"
        else:
            label = "Default DQN"

        plot.add_curve(x, y, label)
    plot.save(filename)

def plot_action_select(data, names, title, filename):
    plot = LearningCurvePlot(title)
    for name in names:
        experiment = data[name]
        result = np.array(experiment['results']).T
        x = result[0]
        y = result[1]
        
        action_selection_kwargs = experiment["action_selection_kwargs"]
        policy = action_selection_kwargs["policy"]
        if policy == "egreedy":
            label = "ε-greedy"
        elif policy == "ann_egreedy":
            label = "Annealing ε-greedy"
        elif policy == "softmax":
            label = "Softmax"
        elif policy == "ann_softmax":
            label = "Annealing softmax" 
        else:
            label = ""
        plot.add_curve(x, y, label)
    plot.save(filename)


# code to plot experiments
def plot_experiments():
    data = load_data("data.json")
    experiments = extract(data)
    
    # plot 1 Learning rate
    names = ['DQN', 'DQN_lr_0.005', 'DQN_lr_0.01']
    title = "Learning rate compared of DQN agents"
    param_key = "learning_rate"
    param_name_show = "Learning rate"
    plot_hyperparameter(data, names, title, "lr.png", param_name_show, param_key)

    # plot 2
    names = ['DQN', 'DQN_rb', 'DQN_tn', 'DQN_rb_tn']
    title = "Target Network and Replay Buffer compared"
    plot_compare_tn_rb(data, names, title, "tn_rb_compared.png")

    #plot 3 
    names = ['DQN', 'DQN_as_softmax', 'DQN_as_egreedy', 'DQN_as_annsoftmax']
    title = "Different action selection methods compared"
    plot_action_select(data, names, title, "Action_select_compared")



    print()

def extract(data):
    experiments = []
    for key in data.keys():
        experiment = data[key]
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