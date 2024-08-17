import gym
import matplotlib.pyplot as plt
from DQNAgent import DQNAgent

#Environment
env = gym.make("Taxi-v3")

# Training function
def train_dqn(episodes, batch_size):
    print(f"Training - State size: {env.observation_space.n}, Action size: {env.action_space.n}")
    agent = DQNAgent(env.observation_space.n, env.action_space.n)
    rewards_list = []

    for episode in range(episodes):
        state, _ = env.reset()
        state = (state,) # Convert to tuple
        total_reward = 0
        done = False
        truncated = False

        while not (done or truncated):
            action = agent.act(state)
            step_result = env.step(action)
            
            if len(step_result) == 5:
                next_state, reward, done, truncated, _ = step_result
            else:
                next_state, reward, done, _ = step_result
                truncated = False
            
            next_state = (next_state,)  # Convert to tuple
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

            if len(agent.memory) > batch_size:
                agent.replay(batch_size)
                
        if agent.epsilon > agent.epsilon_min:
            agent.epsilon *= agent.epsilon_decay

        if episode % 50 == 0:
            agent.update_target_network()

        rewards_list.append(total_reward)
        
        if episode % 50 == 0:
            print(f"Episode: {episode}, Total Reward: {total_reward}, Epsilon: {agent.epsilon:.2f}")

    return agent, rewards_list

# Train the agent
episodes = 8000
batch_size = 64
trained_agent, rewards_history = train_dqn(episodes, batch_size)

# Save the trained model
trained_agent.save_weights('DQN/weights/DQN_weight.pth')

# Plot the results
plt.figure(figsize=(10, 5))
plt.plot(rewards_history)
plt.title('DQN Training Progress')
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.show()