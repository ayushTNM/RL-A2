import gymnasium as gym
import random as rn
env = gym.make("CartPole-v1", render_mode='human')

observation, info = env.reset(seed=42)
for _ in range(1000):
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        observation, info = env.reset()
env.close()

def train_dqn(model, N, M, sample_size=100, gamma=0.9):
    D = []
    for ep in range(1, M + 1):
        s, _ = env.reset()
        for t in range(1, 501):
            q_values = model(s)
            a = selection_policy(q_values)
            s_next, r, term, trunc, _ = env.step(a)
            if len(D) < N:
                D.append((s, a, r, s_next, term or trunc))
            else:
                D[rn.randrange(len(D))] = (s, a, r, s_next, term or trunc)
            
            if len(D) > sample_size
                samples = rn.sample(D, sample_size)
            else:
                samples = D
            
            ys = [r if term else r + gamma * max(model(s)) for s, a, r, s_next, term in samples]
            
            model.optimize([(y - model(s)[a]) ** 2 for (s, a, _, _, _), y in zip(samples, ys))])
