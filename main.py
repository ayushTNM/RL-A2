import gymnasium as gym
import random as rn
import torch

from model import model

# env = gym.make("CartPole-v1", render_mode='human')

# observation, info = env.reset(seed=42)
# for _ in range(1000):
#    action = env.action_space.sample()
#    observation, reward, terminated, truncated, info = env.step(action)

#    if terminated or truncated:
#        observation, info = env.reset()
#env.close()

def train_dqn(model, N, M, sample_size=100, gamma=0.9, learning_rate=1e-3, render_mode=None):
    env = gym.make("CartPole-v1", render_mode=render_mode)

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    loss_fn = torch.nn.MSELoss()
    
    to_torch = lambda ss: torch.tensor(ss, dtype=torch.float32)

    D = []
    for ep in range(1, M + 1):
        s, _ = env.reset()
        for t in range(1, 501):
            q_values = model(to_torch([s]))[0]
            a = q_values.argmax().item()
            s_next, r, term, trunc, _ = env.step(a)
            if len(D) < N:
                D.append((s, a, r, s_next, term or trunc))
            else:
                D[rn.randrange(len(D))] = (s, a, r, s_next, term or trunc)
            
            if len(D) > sample_size:
                samples = rn.sample(D, sample_size)
            else:
                samples = D
            
            s = s_next
            
            ys = [r if term else r + gamma * max(model(to_torch([s_next]))[0]) for s, a, r, s_next, term in samples]
            ss, _as, rs, _, terms = zip(*samples)
            ss = [to_torch(s) for s in ss]
            
            # model.optimize([(y - model(s)[a]) ** 2 for (s, a, _, _, _), y in zip(samples, ys))])
            
            
            # print(torch.stack(ss).shape)
            loss = loss_fn(model(torch.stack(ss))[:, _as], to_torch(ys))
            # Backpropagation
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            if term or trunc:
                break
    env.close()

train_dqn(model, 100, 100)
