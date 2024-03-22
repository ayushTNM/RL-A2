import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import copy
# import matplotlib.pyplot as plt
from tqdm import tqdm
import random as rn

from Agent import BaseNNAgent

if torch.cuda.is_available():  
  dev = torch.device("cuda:0")
else:  
  dev = torch.device("cpu")
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
        # Forward pass through the layers
        x = torch.selu(self.linear1(inputs))
        x = torch.selu(self.linear2(x))
        x = self.output_layer(x)
        return x

# Define the DQN agent
class DQNAgent(BaseNNAgent):
    def __init__(self, n_states, n_actions, lr, gamma, use_replay_buffer=True, replay_buffer_size=1000, use_target_net=True, target_net_delay=1000):
        super().__init__(QNetwork(n_states, n_actions).to(dev), n_actions)
        self.use_target_net = use_target_net
        self.target_net_delay = target_net_delay
        self.target_net = QNetwork(n_states, n_actions).to(dev) if use_target_net else None
        self.optimizer = optim.Adam(self.Qnet.parameters(), lr=lr)
        self.train_step = 0
        self.loss_fn = nn.MSELoss()
        self.gamma = gamma
        self.use_replay_buffer = use_replay_buffer
        self.replay_buffer = ReplayBuffer(replay_buffer_size) if use_replay_buffer else None

    def update(self, state, action, next_state, reward, done):
        self.train_step += 1
        if self.use_target_net and self.train_step % self.target_net_delay == 0:
            self.target_net = copy.deepcopy(self.Qnet)
        
        if self.use_replay_buffer:
            self.replay_buffer.add((state, action, next_state, reward, done))
            return self.optimize_step(*self.replay_buffer.sample())
        return self.optimize_step(state, action, next_state, reward, done)

    def optimize_step(self, state, action, next_state, reward, done):
        q_values = self.Qnet(state)
        if self.use_target_net:
            next_q_values = self.target_net(next_state)
        else:
            next_q_values = self.Qnet(next_state)

        target = reward + self.gamma * torch.max(next_q_values).item() * (1 - done)
        target_q = q_values.clone().detach()
        target_q[0][action] = target

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

def DQN(n_timesteps, learning_rate, gamma, use_replay_buffer, replay_buffer_size, use_target_net, target_net_delay, eval_interval, **action_selection_kwargs):
    
    env = gym.make("CartPole-v1")
    eval_env = gym.make("CartPole-v1")
    agent = DQNAgent(env.observation_space.shape[0], env.action_space.n, learning_rate, gamma, \
        use_replay_buffer=use_replay_buffer, replay_buffer_size=replay_buffer_size, \
        use_target_net=use_target_net, target_net_delay=target_net_delay)
    
    state = torch.tensor(env.reset()[0], dtype=torch.float32, device=dev)
    term, trunc = False, False
    total_episode_loss = 0
    total_episode_reward = 0
    episode_start_step = 0
    episode_iteration = 0
    
    progress_bar = tqdm(range(1, n_timesteps)) # range from 1 as we add 1 to n_timesteps in experimentation
    eval_timesteps, eval_returns = [], []

    for ts in progress_bar:
        # _train_step = ts
        
        # If needed for annealing
        if action_selection_kwargs['policy'].startswith('ann'):
            action_selection_kwargs.update(dict(episode_iteration=episode_iteration))
        
        action = agent.select_action(state.unsqueeze(0), **action_selection_kwargs)
        next_state, reward, term, trunc, _ = env.step(action)

        next_state = torch.tensor(next_state, dtype=torch.float32, device=dev)
        reward = torch.tensor(reward, dtype=torch.float32, device=dev)
        done_flag = torch.tensor(term or trunc, dtype=torch.float32, device=dev)

        loss = agent.update(state.unsqueeze(0), action, next_state.unsqueeze(0), reward, done_flag)

        total_episode_loss += loss
        state = next_state
        total_episode_reward += reward.item()
        
        if ts % eval_interval == 0:
            eval_returns.append(agent.evaluate(eval_env,dev))
            eval_timesteps.append(ts)
            print(eval_returns,eval_timesteps)

        if term or trunc:
            avg_loss = total_episode_loss / (ts - episode_start_step + 1)
            progress_bar.desc =  f"Ep it.: {episode_iteration + 1}, Avg. Loss: {avg_loss:3f}, Tot. R.: {total_episode_reward}"
            # if not best_net or avg_loss < min(losses) or total_episode_reward > list(best_net.keys())[0]:
            #     if not best_net or total_episode_reward >= list(best_net.keys())[0]: # In case avg_loss < min(lossses)
            #         best_net = {total_episode_reward:copy.deepcopy(agent.Qnet)}
            # losses.append(avg_loss)
            
            state = torch.tensor(env.reset()[0], dtype=torch.float32, device=dev)
            total_episode_loss = 0
            total_episode_reward = 0
            episode_start_step = ts
            episode_iteration += 1
            
    return eval_returns, eval_timesteps

def test():
    n_timesteps = 25000
    gamma =0.99
    lr = 0.001
    use_replay_buffer = True
    replay_buffer_size = 1000
    use_target_net = True
    target_net_delay = 1000
    action_selection_kwargs = dict(policy='ann_egreedy', epsilon_start=0.1, epsilon_decay=0.995, epsilon_min=0.01)
    eval_interval = 2500
    
    DQN(n_timesteps, lr, gamma, use_replay_buffer, replay_buffer_size, use_target_net, target_net_delay, eval_interval, **action_selection_kwargs)

if __name__ == '__main__':
    test()
