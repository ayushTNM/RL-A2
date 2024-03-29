
# -*- coding: utf-8 -*-
"""
Practical for course 'Reinforcement Learning',
Leiden University, The Netherlands
By Thomas Moerland
"""

import numpy as np
import time
import os
import json

from DQN import DQN
from Helper import smooth


def average_over_repetitions(n_repetitions, n_timesteps, learning_rate, gamma, action_selection_kwargs, use_replay_buffer=True, replay_buffer_size=1000, use_target_net=True, target_net_delay=100, smoothing_window=None, eval_interval=500, name=None):

    returns_over_repetitions = []
    now = time.time()
    # progress_bar_desc = f"{f'{epsilon=}' if policy ==
    #                        'egreedy' else f'{temp=}'}|lr={learning_rate}"
    for rep in range(n_repetitions):  # Loop over repetitions
        print(f"Repetition {rep+1}/{n_repetitions}:")
        returns, timesteps = DQN(n_timesteps, learning_rate, gamma, action_selection_kwargs, use_replay_buffer=use_replay_buffer,
                                 replay_buffer_size=replay_buffer_size, use_target_net=use_target_net, target_net_delay=target_net_delay, eval_interval=eval_interval)

        returns_over_repetitions.append(returns)

    print('Running one setting takes {} minutes'.format((time.time()-now)/60))
    # average over repetitions
    learning_curve = np.mean(np.array(returns_over_repetitions), axis=0)
    if smoothing_window is not None:
        # additional smoothing
        learning_curve = smooth(learning_curve, smoothing_window)
    return learning_curve, timesteps


    # Settings
    # Experiment
    n_repetitions = 5
    smoothing_window = 9  # Must be an odd number. Use 'None' to switch smoothing off!
    
    # Exploration
    action_selection_kwargs = dict(policy='ann_egreedy', epsilon_start=0.1, epsilon_decay=0.995, epsilon_min=0.01)

    hp = dict(n_timesteps = 25001, eval_interval = 500, learning_rate = 0.001, gamma = 1.0, action_selection_kwargs=action_selection_kwargs)
    
    # parameters of different runs to save for plotting
    runs_kwargs = [dict(use_replay_buffer=False, use_target_net=False),\
                dict(use_replay_buffer=True, replay_buffer_size=1000, use_target_net=False),\
                dict(use_replay_buffer=False, use_target_net=True, target_net_delay=100),
                dict(use_replay_buffer=True, replay_buffer_size=1000, use_target_net=True, target_net_delay=100)]
    
    # Define the path to the JSON file
    data_path = "data.json"

    # Check if the file exists
    if os.path.exists(data_path):
        # Load the JSON file
        with open(data_path, 'r') as file:
            data = json.load(file)
        print("JSON file loaded successfully.")
    else:
        data = {}
        print("JSON file does not exist.")
    
    for params in runs_kwargs:
        
        config = "DQN" if experiment_comment is None else "DQN_" + experiment_comment
        if "use_replay_buffer" in params and params["use_replay_buffer"]:
            config += f"_rb_{params['replay_buffer_size']}"
        if "use_target_net" in params and params["use_target_net"]:
            config += f"_tn_{params['target_net_delay']}"

        if config in data and not overwrite:
            print(f'Configuration {config} already found, skipping..')
            continue
        
        learning_curve, timesteps = average_over_repetitions(n_repetitions, **hp, smoothing_window=smoothing_window)
        data.update({config:{**hp,"results":np.column_stack([timesteps, learning_curve]).tolist()}})
        
        with open(data_path, "w") as file:
            json.dump(data, file, indent=2)

def experiment(experiment_comment=None, overwrite=False):
    # Settings
    # Experiment
    n_repetitions = 5
    smoothing_window = 9  # Must be an odd number. Use 'None' to switch smoothing off!
    
    # Exploration, standard action selection kwargs
    action_selection_kwargs = dict(policy='ann_egreedy', epsilon_start=0.1, epsilon_decay=0.995, epsilon_min=0.01)

    # standard hyperparameters
    hp = dict(n_timesteps = 25001, eval_interval = 500, learning_rate = 0.001, gamma = 1.0, action_selection_kwargs=action_selection_kwargs, use_replay_buffer=False, replay_buffer_size=1000, use_target_net=False, target_net_delay=100)
    
    #for the action runs different act sel kwargs
    egreedy_as_kwargs = dict(policy='egreedy', epsilon=0.1)
    softmax_as_kwargs = dict(policy='softmax', temp=1)
    ann_softmax_as_kwargs = dict(policy='ann_softmax', temp_start=1, temp_decay=0.998, temp_min=0.1)    

 

    # For every run set the hps that must overwrite the standards hps, also set a name for the run
    runs_kwargs = [
                #runs to compare with/without rb and tn
                dict(name="DQN", use_replay_buffer=False, use_target_net=False),#standard run, used in all experiments 
                dict(name="DQN_rb", use_replay_buffer=True, replay_buffer_size=1000, use_target_net=False),\
                dict(name="DQN_tn", use_replay_buffer=False, use_target_net=True, target_net_delay=100),
                dict(name="DQN_rb_tn", use_replay_buffer=True, replay_buffer_size=1000, use_target_net=True, target_net_delay=100,),
                #runs to compare learning rate
                dict(name="DQN_lr_0.005", learning_rate=0.005),
                dict(name="DQN_lr_0.01", learning_rate=0.01),
                # runs to compare action selection
                dict(name="DQN_as_softmax", action_selection_kwargs=softmax_as_kwargs),
                dict(name="DQN_as_egreedy", action_selection_kwargs=egreedy_as_kwargs),
                dict(name="DQN_as_annsoftmax", action_selection_kwargs=ann_softmax_as_kwargs)
]
    
    # Define the path to the JSON file
    data_path = "data.json"

    # Check if the file exists
    if os.path.exists(data_path):
        # Load the JSON file
        with open(data_path, 'r') as file:
            data = json.load(file)
        print("JSON file loaded successfully.")
    else:
        data = {}
        print("JSON file does not exist.")
    
    for params in runs_kwargs:
        
        name = params["name"]
        print(name)

        if name in data and not overwrite:
            print(f'Configuration {name} already found, skipping..')
            continue

        new_params = {**hp, **params}
        
        learning_curve, timesteps = average_over_repetitions(n_repetitions, **new_params, smoothing_window=smoothing_window)
        data.update({name:{**new_params,"results":np.column_stack([timesteps, learning_curve]).tolist()}})
        
        with open(data_path, "w") as file:
            json.dump(data, file, indent=2)


if __name__ == '__main__':
    experiment(overwrite=False)