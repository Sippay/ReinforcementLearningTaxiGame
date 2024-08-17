import gym
import numpy as np
import torch
import random

# Set a master seed for reproducibility
master_seed = 42
random.seed(master_seed)
np.random.seed(master_seed)
torch.manual_seed(master_seed)

# Generate a list of random seeds for each episode
num_episodes = 1000
episode_seeds = [random.randint(0, 10000) for _ in range(num_episodes)]

q_table = np.load('Q-Learning/Q_table/Q_Table_Q_Learning.npy')

# Evaluation of the agent
env = gym.make('Taxi-v3', render_mode='human')
total_moves, total_penalties, total_rewards = 0, 0, 0
episodes = 100

for i in range(episodes):
    current_seed = episode_seeds[i]
    state = env.reset(seed=current_seed)[0]
    moves, penalties, rewards = 0, 0, 0
    
    done = False
    while not done:
        # print(f"q_table: {q_table}")
        action = np.argmax(q_table[state])
        state, reward, done, truncated, _ = env.step(action)
        
        if reward == -10:
            penalties += 1
        
        rewards += reward
        moves += 1
        
        if done or truncated:
            break
    
    total_rewards += rewards
    
    if i % 10 == 0:
        print(f"Episode {i} finished after {moves} moves with {penalties} penalties and {rewards} reward.")

    total_penalties += penalties
    total_moves += moves

print()
print(f"Results after {episodes} episodes:")
print(f"Average rewards per episode: {total_rewards / episodes}")
print(f"Average moves per episode: {total_moves / episodes}")
print(f"Average penalties per episode: {total_penalties / episodes}")

env.close()