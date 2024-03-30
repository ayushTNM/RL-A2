#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Practical for master course 'Reinforcement Learning',
Leiden University, The Netherlands
By Thomas Moerland
"""

import numpy as np
import torch

class BaseNNAgent:

    def __init__(self, Qnet, n_actions):
        self.Qnet = Qnet
        self.n_actions = n_actions
        
    def select_action(self, s, episode = None, **kwargs):
          
        with torch.no_grad():
            q_values = self.Qnet(s)

        if 'greedy' in kwargs['policy']:
            # Greedy action
            a = torch.argmax(q_values).item()
            
            # In case of annealing
            if kwargs['policy'].startswith("ann"):
                if episode == None:
                    raise ValueError("Provide episode")
                kwargs["epsilon"] = max(kwargs['epsilon_start'] * kwargs['epsilon_decay'] ** episode, kwargs['epsilon_min'])

            if 'egreedy' in kwargs['policy']:
                if 'epsilon' not in kwargs or kwargs['epsilon'] is None:
                    raise KeyError("Provide an epsilon")
                if np.random.rand() < kwargs['epsilon']:
                    # Explore: choose a random action with probability epsilon
                    a = np.random.choice(np.delete(np.arange(self.n_actions), a))
                        
        elif 'softmax' in kwargs['policy']:
            # In case of annealing
            if kwargs['policy'].startswith("ann"):
                if episode == None:
                    raise ValueError("Provide episode")
                kwargs["temp"] = max(kwargs['temp_start'] * kwargs['temp_decay'] ** episode, kwargs['temp_min'])

            if kwargs["temp"] is None:
                raise KeyError("Provide a temperature")

            a = np.random.choice(self.n_actions,p=torch.softmax(q_values.cpu()/kwargs["temp"], dim=0).numpy())

        return a

    def update(self):
        raise NotImplementedError('For each agent you need to implement its specific back-up method') # Leave this and overwrite in subclasses in other files


    def evaluate(self,eval_env, dev, n_eval_episodes=10):
        returns = []  # list to store the reward per episode
        for _ in range(n_eval_episodes):
            s = torch.tensor(eval_env.reset()[0], dtype=torch.float32, device=dev)
            R_ep = 0
            term,trunc = False, False
            while not term and not trunc:
                a = self.select_action(s, **dict(policy='greedy'))
                s_prime, r, term, trunc, _ = eval_env.step(a)
                R_ep += r
                
                s = torch.tensor(s_prime, dtype=torch.float32, device=dev)
            returns.append(R_ep)
        mean_return = np.mean(returns)
        return mean_return
