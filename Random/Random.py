import gym
import numpy as np
import matplotlib.pyplot as plt
import time

env = gym.make('Taxi-v3')
env.reset()

n_states = env.observation_space.n
n_actions = env.action_space.n
q_table = np.zeros([env.observation_space.n, env.action_space.n])

# Set the number of episodes
num_episodes = 8000

all_rewards = []
all_moves = []
all_penalties = []
start_time = time.time()
max_moves = 1000

# Training the agent
for i in range(num_episodes):
    state = env.reset()[0]
    truncated = False
    done = False
    moves, rewards, penalties = 0, 0, 0
    
    while not done and moves < max_moves:
        moves += 1
        # Choose a random action
        action = np.random.randint(0, n_actions - 1)
        
        next_state, reward, done, truncated, _ = env.step(action)
        
        if reward == -10:
            penalties += 1
            
        rewards += reward
        moves += 1
        
        state = next_state
    
    all_rewards.append(rewards)
    all_moves.append(moves)
    all_penalties.append(penalties)
    
    if i == 1:
        print ("Episode: ", i, "Rewards: ", rewards, "Moves: ", moves, "Penalties: ", penalties)
        
    if i == 6999:
        print ("Episode: ", i, "Rewards: ", rewards, "Moves: ", moves, "Penalties: ", penalties)
        
end_time = time.time()

print(f"Training finished.\n, Training time: {end_time - start_time:.2f}")   

# Plotting metrics
fig, axs = plt.subplots(3, figsize=(6, 10))

# Plot the number of moves per episode
axs[0].plot(all_moves)
axs[0].set_title('Number of moves per episode')
axs[0].set_xlabel('Episode')
axs[0].set_ylabel('Moves')

# Plot the number of penalties per episode
axs[1].plot(all_penalties)
axs[1].set_title('Number of penalties per episode')
axs[1].set_xlabel('Episode')
axs[1].set_ylabel('Penalties')

# Plot the reward per episode
axs[2].plot(all_rewards)
axs[2].set_title('Random Training Progress')
axs[2].set_xlabel('Episode')
axs[2].set_ylabel('Total Reward')

plt.tight_layout()
plt.show()

import random
import torch

# Set a master seed for reproducibility
master_seed = 42
random.seed(master_seed)
np.random.seed(master_seed)
torch.manual_seed(master_seed)

# Generate a list of random seeds for each episode
num_episodes = 1000
episode_seeds = [random.randint(0, 10000) for _ in range(num_episodes)]

# Evaluation of the agent
env = gym.make('Taxi-v3')
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