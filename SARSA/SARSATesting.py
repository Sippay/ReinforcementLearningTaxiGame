import gymnasium as gym
import numpy as np
import random
import torch

Q_table = np.load('q_table_sarsa.npy')

# Set a master seed for reproducibility
master_seed = 42
random.seed(master_seed)
np.random.seed(master_seed)
torch.manual_seed(master_seed)

# Generate a list of random seeds for each episode
num_episodes = 1000
episode_seeds = [random.randint(0, 10000) for _ in range(num_episodes)]

env = gym.make('Taxi-v3', render_mode='human')
total_moves, total_penalties, total_reward = 0, 0, 0
episodes = 1000

for i in range(episodes):
    current_seed = episode_seeds[i]
    state = env.reset(seed=current_seed)[0]
    moves, penalties, rewards = 0, 0, 0

    done = False
    while not done:
        # print(f"q_table: {q_table}")
        action = np.argmax(Q_table[state])
        state, reward, done, truncated, _ = env.step(action)

        if reward == -10:
            penalties += 1

        rewards += reward
        moves += 1
        if truncated:
            break

    total_reward += rewards

    # if i % 10 == 0:
    #     print(f"Episode {i} finished after {moves} moves with {penalties} penalties and {rewards} reward.")

    total_penalties += penalties
    total_moves += moves

# print()
# print(f"Results after {episodes} episodes:")
# print(f"Average steps per episode: {total_moves / episodes}")
# print(f"Average reward per episode: {total_reward/ episodes}")