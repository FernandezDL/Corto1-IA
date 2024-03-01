import numpy as np
import gym
from gym.envs.toy_text.frozen_lake import generate_random_map

env = gym.make('FrozenLake-v1', is_slippery=True, desc=generate_random_map(size=4), render_mode="human")

# Valores de inicio aleatorios
action_space_size = env.action_space.n
state_space_size = env.observation_space.n
q_table = np.random.rand(state_space_size, action_space_size)

num_episodes = 100
max_steps_per_episode = 100

learning_rate = 0.1
discount_rate = 0.99

exploration_rate = 1
max_exploration_rate = 1
min_exploration_rate = 0.01
exploration_decay_rate = 0.001

rewards_all_episodes = []

for episode in range(num_episodes):
    print(episode)
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
            break

    exploration_rate = min_exploration_rate + \
                        (max_exploration_rate - min_exploration_rate) * np.exp(-exploration_decay_rate * episode)
    rewards_all_episodes.append(rewards_current_episode)

env.close()