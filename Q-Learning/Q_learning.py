import gym
import numpy as np
import matplotlib.pyplot as plt

# Taxi-v3
env = gym.make('Taxi-v3')

# Q-Table
n_states = env.observation_space.n
n_actions = env.action_space.n
q_table = np.zeros([env.observation_space.n, env.action_space.n])

# Hyperparameters
alpha = 0.1
gamma = 0.9
epsilon = 1
epsilon_min = 0.1
epsilon_decay = 0.99
num_episodes = 8000

all_moves = []
all_penalties = []
all_rewards = []

# Training the agent
for i in range(num_episodes):
    state = env.reset()[0]
        
    moves, penalties, rewards, = 0, 0, 0
    truncated = False
    done = False
    
    while not done:
        if np.random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(q_table[state])
        
        next_state, reward, done, truncated, _ = env.step(action)

        # Update Q-Table
        old_value = q_table[state, action]
        next_max = np.max(q_table[next_state])
        new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
        q_table[state, action] = new_value
        
        state = next_state
        moves += 1

        if reward == -10:
            penalties += 1
        
        rewards += reward
        
        if done or truncated:
            break
            
    # Epsilon decay
    if epsilon > epsilon_min:
        epsilon = epsilon * epsilon_decay
    
    all_moves.append(moves)
    all_penalties.append(penalties)
    all_rewards.append(rewards)
    
    if i == 20:
        print("Q table at episode 50")
        print(q_table)
        print("Moves: ", moves)
        print("Penalties: ", penalties)
        print("Rewards: ", rewards)
        print()
    
    if i == 3500:
        print("Q table at episode 3500")
        print(q_table)
        print("Moves: ", moves)
        print("Penalties: ", penalties)
        print("Rewards: ", rewards)
        print()
    
        
print("Training finished.\n")
np.save('Q-Learning/Q_table/Q_Table_Q_Learning.npy', q_table)

# Plotting metrics
fig, axs = plt.subplots(3, figsize=(10, 10))

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
axs[2].set_title('Q Learning Training Progress')
axs[2].set_xlabel('Episode')
axs[2].set_ylabel('Rewards')

plt.tight_layout()
plt.show()