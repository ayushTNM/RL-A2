#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Practical for master course 'Reinforcement Learning',
Leiden University, The Netherlands
By Thomas Moerland
"""

import numpy as np
import torch

def softmax(x, temp):
    ''' Computes the softmax of vector x with temperature parameter 'temp' '''
    x = x / temp # scale by temperature
    z = x - max(x) # substract max to prevent overflow of softmax 
    return np.exp(z)/np.sum(np.exp(z)) # compute softmax

class BaseNNAgent:

    def __init__(self, Qnet, n_actions):
        self.Qnet = Qnet
        self.n_actions = n_actions
        
    def select_action(self, s, policy='egreedy', epsilon=None, temp=None):
          
        with torch.no_grad():
            q_values = self.Qnet(s)

        if policy in ['egreedy','greedy']:
            # Greedy action
            a = torch.argmax(q_values).item()

            if policy == 'egreedy':
                if epsilon is None:
                    raise KeyError("Provide an epsilon")
                if np.random.rand() < epsilon:
                    # Explore: choose a random action with probability epsilon
                    a = np.random.choice(np.delete(np.arange(self.n_actions), a))
                        
        elif policy == 'softmax':
            if temp is None:
                raise KeyError("Provide a temperature")

            a = np.random.choice(self.n_actions,p=softmax(q_values,temp))
            
        return a
        
    def update(self):
        raise NotImplementedError('For each agent you need to implement its specific back-up method') # Leave this and overwrite in subclasses in other files


    def evaluate(self,eval_env,n_eval_episodes=30, max_episode_length=100):
        returns = []  # list to store the reward per episode
        for i in range(n_eval_episodes):
            s = eval_env.reset()
            R_ep = 0
            for t in range(max_episode_length):
                a = self.select_action(s, 'greedy')
                s_prime, r, done = eval_env.step(a)
                R_ep += r
                if done:
                    break
                else:
                    s = s_prime
            returns.append(R_ep)
        mean_return = np.mean(returns)
        return mean_return
