import random
import torch
import numpy as np
import gym

# Set a master seed for reproducibility
master_seed = 42
random.seed(master_seed)
np.random.seed(master_seed)
torch.manual_seed(master_seed)

# Generate a list of random seeds for each episode
num_episodes = 1000
episode_seeds = [random.randint(0, 10000) for _ in range(num_episodes)]

# Evaluation of the agent
env = gym.make('Taxi-v3', render_mode='human')
n_actions = env.action_space.n
total_moves, total_penalties = 0, 0
episodes = 100
totalReward = 0
max_moves = 1000

for i in range(episodes):
    current_seed = episode_seeds[i]
    state = env.reset(seed=current_seed)[0]
    moves, penalties, rewards = 0, 0, 0
    
    done = False
    while not done and moves < max_moves:
        moves += 1
        action = np.random.randint(0, n_actions - 1)
        next_state, reward, done, truncated, _ = env.step(action)
        
        rewards += reward
        
        if reward == -10:
            penalties += 1
        
        rewards += reward
        moves += 1
    
    totalReward += rewards
    
    if i % 10 == 0:
        print(f"Episode {i} finished after {moves} moves with {penalties} penalties and {rewards} reward.")

    total_penalties += penalties
    total_moves += moves


print(f"\nAverage steps per episode: {total_moves / episodes}")
print(f"Average reward per episode: {totalReward / episodes}")
print()