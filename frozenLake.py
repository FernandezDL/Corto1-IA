import numpy as np
import gym
from gym.envs.toy_text.frozen_lake import generate_random_map

env = gym.make('FrozenLake-v1', is_slippery=True, desc=generate_random_map(size=4), render_mode="human")

# Valores de inicio aleatorios
action_space_size = env.action_space.n
state_space_size = env.observation_space.n
q_table = np.random.rand(state_space_size, action_space_size)

num_episodes = 300
max_steps_per_episode = 100

learning_rate = 0.1
discount_rate = 0.99

exploration_rate = 1
max_exploration_rate = 1
min_exploration_rate = 0.01
exploration_decay_rate = 0.001

rewards_all_episodes = []

for episode in range(num_episodes):
    state = env.reset()[0]
    done = False
    rewards_current_episode = 0

    for step in range(max_steps_per_episode):
        exploration_rate_threshold = np.random.uniform(0, 1)
        if exploration_rate_threshold > exploration_rate:
            action = np.argmax(q_table[state, :])
        else:
            action = env.action_space.sample()

        new_state, reward, done, _, _ = env.step(action)

        q_table[state, action] = q_table[state, action] * (1 - learning_rate) + \
                                 learning_rate * (reward + discount_rate * np.max(q_table[new_state, :]))

        state = new_state
        rewards_current_episode += reward

        if done:
            print(f"Episode {episode + 1} finished after {step + 1} steps with {'Success' if reward == 1 else 'Failure'}")

            break

    exploration_rate = min_exploration_rate + \
                        (max_exploration_rate - min_exploration_rate) * np.exp(-exploration_decay_rate * episode)
    rewards_all_episodes.append(rewards_current_episode)

num_test_episodes = 100
total_test_reward = 0

for episode in range(num_test_episodes):
    state = env.reset()[0]
    done = False
    episode_reward = 0

    while not done:
        action = np.argmax(q_table[state, :])
        new_state, reward, done, _, _ = env.step(action)
        state = new_state
        episode_reward += reward

    total_test_reward += episode_reward
    print(f"Test Episode {episode + 1}: Reward = {episode_reward}")

print("=========================================================================")
print("Promedio de exitos: ", total_test_reward/100)

env.close()

