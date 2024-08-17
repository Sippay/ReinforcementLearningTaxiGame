import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import time

# create environment
env = gym.make('Taxi-v3')

start = time.time()
#define parameters
epsilon = 1
epsilon_decay = 0.999
min_epsilon = 0.001
total_eps = 8000
discount = 0.999
max_moves = 10000
alpha = 0.1

#define Q table
Q_table = np.zeros((env.observation_space.n, env.action_space.n))
all_moves = []
all_penalties = []
all_rewards = []

#choice of action
def choose_action(state):
    """
    Uses a balance of exploration and exploitation by utilising epsilon. As training goes on, epsilon gets lower and lower,
    allowing the algorithm to switch to exploitation. At the very start, it is 100% exploration

    state: the state for which we need to find an action for
    """
    action = 0
    if np.random.uniform(0,1) < epsilon:
        action = env.action_space.sample()
    else:
        action = np.argmax(Q_table[state])
    return action

#updating Q-table
def update(state, action, reward, future_state, future_action, moves):
    """
    takes an estimate of the future reward and updates the Q table using that. This update function uses SARSA which means
    it takes the current Q value multiplied by the discount factor, adds it to the reward.

    The estimate is updated per move in an episode and it slowly is trained. It starts off being wrong at the start
    """
    target = reward + discount * Q_table[future_state, future_action]
    Q_table[state, action] = (1-alpha)*Q_table[state, action] + alpha * target

# for each episode, get a starting state, and choose an action.
# While the episode is not finished, keep on choosing actions and updating Q_table.
# At the end, the Q_table is returned and tested against.
# Note that at the end of an episode, epsilon's value is reduced
# so that the model will naturally over time prefer exploiting rather than exploring
for episode in range(1, total_eps+1):
    t = 0
    state = env.reset()
    state = state[0]
    action = choose_action(state)
    moves, penalties, total_reward = 0, 0, 0
    done = False

    while not done:
        future_state, reward, done, truncated, _ = env.step(action)
        future_action = choose_action(future_state)
        moves += 1
        update(state, action, reward, future_state, future_action, moves)
        state = future_state
        action = future_action

        if reward <= -10:
            penalties += 1

        total_reward += reward

        if moves == max_moves:
            break
    
    if epsilon > min_epsilon:
        epsilon *= epsilon_decay

    # store values
    all_rewards.append(total_reward)
    all_moves.append(moves)
    all_penalties.append(penalties)
    if episode % 50 == 0:
        print(f"episode: {episode}, Reward = {total_reward}, epsilon: {epsilon}, moves: {moves}")

end = time.time()

print(f"Training time: {end-start}")
np.save('SARSA/Q_table/q_table_sarsaV2.npy', Q_table)

# Plotting metrics
fig, axs = plt.subplots(3, figsize=(7, 10))

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
axs[2].set_title('SARSA Training Progress')
axs[2].set_xlabel('Episode')
axs[2].set_ylabel('Total Reward')

plt.tight_layout()
plt.show()