import math
import random
import numpy as np

import gym

class RandomAgent():
    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, observation, reward, done):
        return self.action_space.sample()

class SimpleQLearningAgent():
    def __init__(self, action_space, observation_space, observation_bin_sizes):
        self.action_space = action_space
        self.observation_space = observation_space
        self.observation_bin_sizes = observation_bin_sizes

        self.observation_space.low[1] = -0.8
        self.observation_space.high[1] = 0.8
        self.observation_space.low[3] = -math.radians(40)
        self.observation_space.high[3] = math.radians(40)

        self.observation_bins = tuple([np.linspace(observation_space.low[i], observation_space.high[i], size-1) for i, size in enumerate(observation_bin_sizes)])

        self.q_value_table = np.zeros(observation_bin_sizes + (2, ))

    def act(self, observation, episode):
        exploration_rate = max(0.01, min(1.0, 1.0 - math.log10((episode + 1) / 20.0)))

        observation_bins = [int(np.digitize(observation_item, self.observation_bins[i])) if len(self.observation_bins[i]) > 0 else 0 for i, observation_item in enumerate(observation)]

        return self.action_space.sample() if random.random() < exploration_rate else np.argmax(self.q_value_table[tuple(observation_bins)])

    def train(self, old_observation, action, new_observation, reward, done, episode):
        learning_rate = max(0.1, min(0.5, 1.0 - math.log10((episode + 1) / 20.0)))

        old_observation_bins = [int(np.digitize(observation_item, self.observation_bins[i])) if len(self.observation_bins[i]) > 0 else 0 for i, observation_item in enumerate(old_observation)]
        new_observation_bins = [int(np.digitize(observation_item, self.observation_bins[i])) if len(self.observation_bins[i]) > 0 else 0 for i, observation_item in enumerate(new_observation)]

        old_q_value = self.q_value_table[tuple(old_observation_bins)][action]

        sample = reward + 1.0 * (np.max(self.q_value_table[tuple(new_observation_bins)]) if not done else 0)

        new_q_value = (1 - learning_rate) * old_q_value + learning_rate * sample

        self.q_value_table[tuple(old_observation_bins)][action] = new_q_value

if __name__ == '__main__':
    env = gym.make('CartPole-v0')

    randomAgent = RandomAgent(env.action_space)

    episodes = 10
    total_reward = 0
    done = False

    for i in range(episodes):
        total_reward = 0

        observation = env.reset()
        env.render()

        while True:
            action = randomAgent.act(observation, total_reward, done)

            observation, reward, done, _ = env.step(action)
            env.render()

            total_reward += reward

            if done:
                print 'Episode {}: {}'.format(str(i+1), total_reward)
                break

    random.seed()

    simpleQLearningAgent = SimpleQLearningAgent(env.action_space, env.observation_space, (2, 2, 8, 4))

    episodes = 1800
    total_reward = 0
    done = False

    solved = 0

    for i in range(episodes):
        total_reward = 0

        old_observation = env.reset()

        while True:
            action = simpleQLearningAgent.act(old_observation, i)

            new_observation, reward, done, _ = env.step(action)

            simpleQLearningAgent.train(old_observation, action, new_observation, reward, done, i)

            old_observation = new_observation

            total_reward += reward

            if done:
                print 'Episode {}: {}'.format(i+1, total_reward)

                if total_reward >= 195:
                    solved += 1

                break

    print 'Solved {} times'.format(solved)
