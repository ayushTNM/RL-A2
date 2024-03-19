import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import copy
# import matplotlib.pyplot as plt
from tqdm import tqdm
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
    def __init__(self, n_states, n_actions, lr, gamma):
        super().__init__(QNetwork(n_states, n_actions).to(dev), n_actions)
        self.optimizer = optim.Adam(self.Qnet.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
        self.gamma = gamma

    def update(self, state, action, next_state, reward, done):
        q_values = self.Qnet(state)
        next_q_values = self.Qnet(next_state)

        target = reward + self.gamma * torch.max(next_q_values).item() * (1 - done)
        target_q = q_values.clone().detach()
        target_q[0][action] = target

        self.optimizer.zero_grad()
        loss = self.loss_fn(q_values, target_q)
        loss.backward()
        self.optimizer.step()
        return loss.item()

# Hyperparameters
lr = 0.001
gamma = 0.99
policy='egreedy'
# epsilon = 0.01
epsilon_start = 0.1
epsilon_decay = 0.995
epsilon_min = 0.01
temp = 0.1
num_episodes = 500

losses = []
best_net = {}

# Initialize environment and agent
env = gym.make("CartPole-v1")
agent = DQNAgent(env.observation_space.shape[0], env.action_space.n, lr, gamma)
progress_bar = tqdm(range(num_episodes))

# training loop
for episode in progress_bar:
    state = torch.tensor(env.reset()[0], dtype=torch.float32, device=dev)
    term, trunc = False, False
    total_loss = 0
    total_reward = 0
    iteration = 0
    while not term and not trunc:
        iteration += 1
        epsilon = max(epsilon_start * epsilon_decay ** episode, epsilon_min)
        action = agent.select_action(state.unsqueeze(0), policy=policy, epsilon=epsilon, temp=temp)
        next_state, reward, term, trunc, _ = env.step(action)

        next_state = torch.tensor(next_state, dtype=torch.float32, device=dev)
        reward = torch.tensor(reward, dtype=torch.float32, device=dev)
        done_flag = torch.tensor(term or trunc, dtype=torch.float32, device=dev)

        loss = agent.update(state.unsqueeze(0), action, next_state.unsqueeze(0), reward, done_flag)
        total_loss += loss
        state = next_state
        total_reward += reward.item()

    avg_loss = total_loss/iteration
    progress_bar.desc =  f"Ep. {episode + 1}, Avg. Loss {avg_loss:3f}, Tot. R.: {total_reward}"
    if not best_net or avg_loss < min(losses) or total_reward > list(best_net.keys())[0]:
        if total_reward >= list(best_net.keys())[0]: # In case avg_loss < min(lossses)
            best_net = {total_reward:copy.deepcopy(agent.Qnet)}

    losses.append(avg_loss)
# plt.plot(losses)
# plt.xlabel('Episode')
# plt.ylabel('Loss')
# plt.title('Loss Across Episodes')
# plt.show()

# Show best run
env = gym.make("CartPole-v1", render_mode="human")
agent.Qnet = list(best_net.values())[0]
state = torch.tensor(env.reset()[0], dtype=torch.float32, device=dev)
term, trunc = False, False
total_reward = 0

while not term and not trunc:
    action = agent.select_action(state.unsqueeze(0), 'greedy')
    next_state, reward, term, trunc, _ = env.step(action)
    next_state = torch.tensor(next_state, dtype=torch.float32, device=dev)
    total_reward += reward.item()
    state = next_state
print("Total reward:",total_reward)
env.close()
