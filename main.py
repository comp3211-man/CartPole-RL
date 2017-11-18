import argparse

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
    def __init__(self, action_space, observation_space, observation_bin_sizes, parameters):
        self.action_space = action_space
        self.observation_space = observation_space
        self.observation_bin_sizes = observation_bin_sizes

        self.parameters = {
            'learning_rate_max': parameters.learning_rate_max,
            'learning_rate_min': parameters.learning_rate_min,
            'learning_rate_decrease_rate': parameters.learning_rate_decrease_rate,
            'discount': parameters.discount,
            'exploration_rate_max': parameters.exploration_rate_max,
            'exploration_rate_min': parameters.exploration_rate_min,
            'exploration_rate_decrease_rate': parameters.exploration_rate_decrease_rate
        }

        self.observation_space.low[1] = parameters.x_dot_min
        self.observation_space.high[1] = parameters.x_dot_max
        self.observation_space.low[3] = math.radians(parameters.theta_dot_min)
        self.observation_space.high[3] = math.radians(parameters.theta_dot_max)

        self.observation_bins = tuple([np.linspace(observation_space.low[i] + ((observation_space.high[i] - observation_space.low[i]) / size), observation_space.high[i], size-1, endpoint=False) for i, size in enumerate(observation_bin_sizes)])

        self.q_value_table = np.zeros(observation_bin_sizes + (2, ))

    def act(self, observation, episode):
        exploration_rate = max(self.parameters['exploration_rate_min'], min(self.parameters['exploration_rate_max'], 1.0 - math.log10((episode + 1) / self.parameters['exploration_rate_decrease_rate'])))

        observation_bins = [int(np.digitize(observation_item, self.observation_bins[i])) if len(self.observation_bins[i]) > 0 else 0 for i, observation_item in enumerate(observation)]

        return self.action_space.sample() if random.random() < exploration_rate else np.argmax(self.q_value_table[tuple(observation_bins)])

    def train(self, old_observation, action, new_observation, reward, done, episode):
        learning_rate = max(self.parameters['learning_rate_min'], min(self.parameters['learning_rate_max'], 1.0 - math.log10((episode + 1) / self.parameters['learning_rate_decrease_rate'])))

        old_observation_bins = [int(np.digitize(observation_item, self.observation_bins[i])) if len(self.observation_bins[i]) > 0 else 0 for i, observation_item in enumerate(old_observation)]
        new_observation_bins = [int(np.digitize(observation_item, self.observation_bins[i])) if len(self.observation_bins[i]) > 0 else 0 for i, observation_item in enumerate(new_observation)]

        old_q_value = self.q_value_table[tuple(old_observation_bins)][action]

        sample = reward + self.parameters['discount'] * (np.max(self.q_value_table[tuple(new_observation_bins)]) if not done else 0)

        new_q_value = (1 - learning_rate) * old_q_value + learning_rate * sample

        self.q_value_table[tuple(old_observation_bins)][action] = new_q_value

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Q-learning agent for CartPole problem of OpenAI Gym')

    parser.add_argument('--learning-rate-max', default=0.5, type=float)
    parser.add_argument('--learning-rate-min', default=0.1, type=float)
    parser.add_argument('--learning-rate-decrease-rate', default=20.0, type=float)

    parser.add_argument('--discount', default=1.0, type=float)

    parser.add_argument('--exploration-rate-max', default=1.0, type=float)
    parser.add_argument('--exploration-rate-min', default=0.01, type=float)
    parser.add_argument('--exploration-rate-decrease-rate', default=20.0, type=float)

    parser.add_argument('--x-bins', default=2, type=int)
    parser.add_argument('--x-dot-bins', default=2, type=int)
    parser.add_argument('--theta-bins', default=8, type=int)
    parser.add_argument('--theta-dot-bins', default=4, type=int)

    parser.add_argument('--x-dot-min', default=-0.8, type=float)
    parser.add_argument('--x-dot-max', default=0.8, type=float)
    parser.add_argument('--theta-dot-min', default=-40, type=float)
    parser.add_argument('--theta-dot-max', default=40, type=float)

    parser.add_argument('--episodes', default=1800, type=int)

    parser.add_argument('--ui', action='store_true')

    parser.add_argument('--random-agent', action='store_true')
    parser.add_argument('--random-agent-episodes', default=10, type=int)

    args = parser.parse_args()

    env = gym.make('CartPole-v0')

    if args.random_agent:
        randomAgent = RandomAgent(env.action_space)

        episodes = args.random_agent_episodes
        total_reward = 0
        done = False

        for i in range(episodes):
            total_reward = 0

            observation = env.reset()
            if args.ui:
                env.render()

            while True:
                action = randomAgent.act(observation, total_reward, done)

                observation, reward, done, _ = env.step(action)
                if args.ui:
                    env.render()

                total_reward += reward

                if done:
                    print 'Episode {}: {}'.format(str(i+1), total_reward)
                    break

    random.seed()

    simpleQLearningAgent = SimpleQLearningAgent(env.action_space, env.observation_space, (args.x_bins, args.x_dot_bins, args.theta_bins, args.theta_dot_bins), args)

    episodes = args.episodes
    total_reward = 0
    done = False

    solved = 0
    solved_consecutive = 0

    for i in range(episodes):
        total_reward = 0

        old_observation = env.reset()
        if args.ui:
            env.render()

        while True:
            action = simpleQLearningAgent.act(old_observation, i)

            new_observation, reward, done, _ = env.step(action)
            if args.ui:
                env.render()

            simpleQLearningAgent.train(old_observation, action, new_observation, reward, done, i)

            old_observation = new_observation

            total_reward += reward

            if done:
                print 'Episode {}: {}'.format(i+1, total_reward)

                solved = solved + 1 if total_reward >= 195 else solved
                solved_consecutive = solved_consecutive + 1 if total_reward >= 195 else 0

                break

        if solved_consecutive >= 100:
            break

    print 'Solved {} times, {} times consecutively'.format(solved, solved_consecutive)
    if solved_consecutive >= 100:
        print 'Problem solved'
