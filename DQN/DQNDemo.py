import gym
import numpy as np
import torch
from DQNAgent import DQNAgent
import random

# Set a master seed for reproducibility
master_seed = 42
random.seed(master_seed)
np.random.seed(master_seed)
torch.manual_seed(master_seed)

# Generate a list of random seeds for each episode
num_episodes = 1000
episode_seeds = [random.randint(0, 10000) for _ in range(num_episodes)]

# Initialize the environment and get state and action sizes
env = gym.make('Taxi-v3', render_mode='human')
stateSize = env.observation_space.n
actionSize = env.action_space.n

# Initialize the agent and load the weights
agent = DQNAgent(stateSize, actionSize)
# Load the entire saved state
saved_state = torch.load('DQN/weights/taxi_dqn_weightsv11.pth')

# Extract the train network state dict
train_network_state_dict = saved_state['train_network_state_dict']

# Load the state dict into the agent's train network
agent.train_network.load_state_dict(train_network_state_dict)

agent.epsilon = 0.01  # Small epsilon for some exploration during testing

# Test the agent
numEpisodes = 1000
totalSteps = 0
successfulEpisodes = 0
totalReward = 0

for episode in range(numEpisodes):
    current_seed = episode_seeds[episode]
    state, _ = env.reset(seed=current_seed)
    state = (state,)  # Convert to tuple
    RewardPerEp = 0
    steps = 0
    done = False
    truncated = False

    while not (done or truncated):
        action = agent.act(state)
        stepResult = env.step(action)
        
        if len(stepResult) == 5:
            nextState, reward, done, truncated, _ = stepResult
        else:
            nextState, reward, done, _ = stepResult
            truncated = False
            
        nextState = (nextState,)  # Convert to tuple
        RewardPerEp += reward
        state = nextState
        steps += 1
    
    totalSteps += steps
    totalReward += RewardPerEp
    if RewardPerEp > 0:
        successfulEpisodes += 1

print(f"\nAverage steps per episode: {totalSteps / numEpisodes}")
print(f"Average reward per episode: {totalReward / numEpisodes}")
print()

env.close()