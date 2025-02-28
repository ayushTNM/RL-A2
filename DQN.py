import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import copy
from tqdm import tqdm
import random as rn
import argparse

from Agent import BaseNNAgent

dev = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
print(dev)

# Define the neural network architecture
class QNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, 32)
        self.linear2 = nn.Linear(32, 32)
        self.output_layer = nn.Linear(32, output_dim)

        # Initialization using Xavier
        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.xavier_uniform_(self.linear2.weight)
        nn.init.xavier_uniform_(self.output_layer.weight)

    def forward(self, inputs):
        # Forward pass through the layers with sigmoid activation
        x = torch.relu(self.linear1(inputs))
        x = torch.relu(self.linear2(x))
        x = self.output_layer(x)
        return x
    

# Define the DQN agent
class DQNAgent(BaseNNAgent):
    def __init__(self, n_states, n_actions, lr, gamma, replay_buffer_size=None, target_net_delay=None):
        super().__init__(QNetwork(n_states,n_actions).to(dev), n_actions)
        self.target_net_delay = target_net_delay
        self.target_net = copy.deepcopy(self.Qnet) if target_net_delay is not None else None
        self.optimizer = optim.Adam(self.Qnet.parameters(), lr=lr)
        self.train_step = 0
        self.loss_fn = nn.MSELoss()
        self.gamma = gamma
        self.replay_buffer = ReplayBuffer(replay_buffer_size) if replay_buffer_size else None

    def update(self, state, action, next_state, reward, done):
        self.train_step += 1
        if self.target_net and self.train_step % self.target_net_delay == 0:
            self.target_net = copy.deepcopy(self.Qnet)
        
        if self.replay_buffer:
            self.replay_buffer.add((state, action, next_state, reward, done))
            return self.optimize_step(*self.replay_buffer.sample())
        return self.optimize_step(state, action, next_state, reward, done)

    def optimize_step(self, state, action, next_state, reward, done):
        q_values = self.Qnet(state)
        
        next_q_values = self.target_net(next_state) if self.target_net else self.Qnet(next_state)
            
        target_q = q_values.clone().detach()
        target_q[action] = reward + self.gamma * torch.max(next_q_values).item() * (1 - done)

        self.optimizer.zero_grad()
        loss = self.loss_fn(q_values, target_q)
        loss.backward()
        self.optimizer.step()
        return loss.item()

class ReplayBuffer(list):
    def __init__(self, max_size, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_size = max_size
    
    def add(self, e):
        if len(self) == self.max_size:
            self[rn.randrange(len(self))] = e
        else:
            self.append(e)
    
    def sample(self):
        return rn.choice(self)

def DQN(n_timesteps, learning_rate, gamma, action_selection_kwargs, replay_buffer_size=None, target_net_delay=None, eval_interval=2500, play_best_net=False, verbose=False):
    
    env = gym.make("CartPole-v1")
    eval_env = gym.make("CartPole-v1")
    agent = DQNAgent(env.observation_space.shape[0], env.action_space.n, learning_rate, gamma, \
        replay_buffer_size=replay_buffer_size, \
        target_net_delay=target_net_delay)
    
    if play_best_net:
        best_net = None
        best_net_res = 0
    
    state = torch.tensor(env.reset()[0], dtype=torch.float32, device=dev)
    term, trunc = False, False
    total_episode_loss = 0
    total_episode_reward = 0
    episode_start_step = 0
    episode = 0
    
    progress_bar = tqdm(range(n_timesteps), desc =  f"Ep: {episode + 1}, Avg. Loss: None, Tot. Rew.: None", \
        bar_format='{l_bar}{bar}| Ts.: {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
    eval_timesteps, eval_returns = [], []
    best_total_reward = 0

    for ts in progress_bar:
        if ts % eval_interval == 0:
            eval_returns.append(agent.evaluate(eval_env,dev))
            eval_timesteps.append(ts)
            
            if play_best_net and eval_returns[-1] > best_net_res:
                best_net = copy.deepcopy(agent.Qnet)
                best_net_res = eval_returns[-1]
        
        action = agent.select_action(state, episode = episode,**action_selection_kwargs)
        next_state, reward, term, trunc, _ = env.step(action)

        next_state = torch.tensor(next_state, dtype=torch.float32, device=dev)
        reward = torch.tensor(reward, dtype=torch.float32, device=dev)
        done_flag = torch.tensor(term or trunc, dtype=torch.float32, device=dev)

        loss = agent.update(state, action, next_state, reward, done_flag)

        total_episode_loss += loss
        state = next_state
        total_episode_reward += reward.item()
        
        if term or trunc:
            avg_loss = total_episode_loss / (ts - episode_start_step + 1)
            if total_episode_reward > best_total_reward: best_total_reward = total_episode_reward
            progress_bar.desc =  f"Ep: {episode + 1}, Avg. Loss: {avg_loss:3f}, Tot. Rew.: {total_episode_reward}, Best Tot. Rew.: {best_total_reward}"
            
            state = torch.tensor(env.reset()[0], dtype=torch.float32, device=dev)
            total_episode_loss = 0
            total_episode_reward = 0
            episode_start_step = ts
            episode += 1
    
    if play_best_net:
        play(agent, best_net)
    
    if verbose:
        print('Final evaluation result:', eval_returns[-1])
    
    return eval_returns, eval_timesteps

def play(agent, net):
    env = gym.make("CartPole-v1", render_mode="human")
    agent.Qnet = net
    state = torch.tensor(env.reset()[0], dtype=torch.float32, device=dev)
    term, trunc = False, False
    total_reward = 0

    while not term and not trunc:
        action = agent.select_action(state.unsqueeze(0), policy='greedy')
        next_state, reward, term, trunc, _ = env.step(action)
        next_state = torch.tensor(next_state, dtype=torch.float32, device=dev)
        total_reward += reward
        state = next_state
    print("Total reward of live episode:", total_reward)
    env.close()

def test(args):
    # Exploration, standard action selection kwargs
    action_selection_kwargs = {
        'ann_egreedy': dict(policy='ann_egreedy', epsilon_start=0.1, epsilon_decay=0.995, epsilon_min=0.01),
        'egreedy': dict(policy='egreedy', epsilon=0.1),
        'softmax': dict(policy='softmax', temp=1),
        'ann_softmax': dict(policy='ann_softmax', temp_start=1, temp_decay=0.998, temp_min=0.1)
        }

    # standard hyperparameters
    hp = dict(n_timesteps = args.n, eval_interval = 500, learning_rate = 0.001, gamma = 1.0, action_selection_kwargs=action_selection_kwargs[args.as_policy], replay_buffer_size=(1000 if args.replay_buffer else None), target_net_delay=(100 if args.target_net else None))
    
    DQN(**hp, play_best_net=True, verbose=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    to_bool = lambda s: s == 'True'
    parser.add_argument('--replay_buffer', type=to_bool, default=True, help='whether to use a replay buffer')
    parser.add_argument('--target_net', type=to_bool, default=True, help='whether to use a target network')
    parser.add_argument('--as_policy', default='ann_egreedy', choices=['ann_egreedy', 'egreedy', 'softmax', 'ann_softmax'], help='type of action selection (exploration) policy')
    parser.add_argument('--n', type=int, default=25001, help='number of train steps')
    args = parser.parse_args()
    test(args)
