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

    def rate_cal(self, episode, rate_decay, rate_max, rate_min):
         rate = 1.0 - math.log10((episode + 1.0) / rate_decay)
         return max(rate_min, min(rate_max, rate))

    def act(self, observation, episode):
        rate_min = self.parameters['exploration_rate_min']
        rate_max = self.parameters['exploration_rate_max']
        rate_decay = self.parameters['exploration_rate_decrease_rate']
        self.exploration_rate=self.rate_cal(episode, rate_decay, rate_max, rate_min)

        observation_bins = [int(np.digitize(observation_item, self.observation_bins[i])) if len(self.observation_bins[i]) > 0 else 0 for i, observation_item in enumerate(observation)]

        return self.action_space.sample() if random.random() < self.exploration_rate else np.argmax(self.q_value_table[tuple(observation_bins)])

    def train(self, old_observation, action, new_observation, reward, done, episode):
        rate_min = self.parameters['learning_rate_min']
        rate_max = self.parameters['learning_rate_max']
        rate_decay = self.parameters['learning_rate_decrease_rate']
        self.learning_rate=self.rate_cal(episode, rate_decay, rate_max, rate_min)

        old_observation_bins = [int(np.digitize(observation_item, self.observation_bins[i])) if len(self.observation_bins[i]) > 0 else 0 for i, observation_item in enumerate(old_observation)]
        new_observation_bins = [int(np.digitize(observation_item, self.observation_bins[i])) if len(self.observation_bins[i]) > 0 else 0 for i, observation_item in enumerate(new_observation)]

        old_q_value = self.q_value_table[tuple(old_observation_bins)][action]

        sample = reward + self.parameters['discount'] * (np.max(self.q_value_table[tuple(new_observation_bins)]) if not done else 0)

        new_q_value = (1 - self.learning_rate) * old_q_value + self.learning_rate * sample

        self.q_value_table[tuple(old_observation_bins)][action] = new_q_value

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Q-learning agent for CartPole problem of OpenAI Gym')

    parser.add_argument('--learning-rate-max', default=0.5, type=float, help='maximum learning rate, 0 <= learning_rate_max <= 1')
    parser.add_argument('--learning-rate-min', default=0.1, type=float, help='minimum learning rate, 0 <= learning_rate_min <= 1')
    parser.add_argument('--learning-rate-decrease-rate', default=20.0, type=float, help='learning rate decrease rate, learning_rate_decrease_rate > 0, higher decrease rate means slower decreasing learning rate function')
    parser.add_argument('--discount', default=1.0, type=float, help='discount, 0 <= discount <= 1')
    parser.add_argument('--exploration-rate-max', default=1.0, type=float, help='maximum exploration rate, 0 <= exploration_rate_max <= 1')
    parser.add_argument('--exploration-rate-min', default=0.01, type=float, help='minimum exploration rate, 0 <= exploration_rate_min <= 1')
    parser.add_argument('--exploration-rate-decrease-rate', default=20.0, type=float, help='exploration rate decrease rate, exploration_rate_decrease_rate > 0, higher decrease rate means slower decreasing exploration rate function')
    parser.add_argument('--x-bins', default=2, type=int, help='number of bins for cart position')
    parser.add_argument('--x-dot-bins', default=2, type=int, help='number of bins for cart velocity')
    parser.add_argument('--theta-bins', default=8, type=int, help='number of bins for pole angle')
    parser.add_argument('--theta-dot-bins', default=4, type=int, help='number of bins for pole angular velocity')
    parser.add_argument('--x-dot-min', default=-0.8, type=float, help='minimum of cart velocity domain')
    parser.add_argument('--x-dot-max', default=0.8, type=float, help='maximum of cart velocity domain')
    parser.add_argument('--theta-dot-min', default=-40, type=float, help='minimum of pole angular velocity domain')
    parser.add_argument('--theta-dot-max', default=40, type=float, help='maximum of pole angular velocity domain')
    parser.add_argument('--episodes', default=1800, type=int, help='number of episodes')
    parser.add_argument('--ui', default=0, type=int, help='render the UI starting from the uith episode')
    parser.add_argument('--random', action = 'store_true')
    parser.add_argument('--output-print', action = 'store_true')

    args = parser.parse_args()
    if args.random:
        args.exploration_rate_decrease_rate=random.randrange(1,150)*1.0
        args.learning_rate_decrease_rate=random.randrange(1,150)*1.0
        args.learning_rate_max=random.randrange(500,1000)/1000.0
        args.learning_rate_min = random.randrange(1, 500) / 1000.0
        args.exploration_rate_max = random.randrange(500, 1000) / 1000.0
        args.exploration_rate_min = random.randrange(1, 500) / 1000.0
    param={'learning':[args.learning_rate_max,args.learning_rate_min,args.learning_rate_decrease_rate],'exploration':[args.exploration_rate_max,args.exploration_rate_min,args.exploration_rate_decrease_rate]}
    if args.random:
        print('Param {}'.format(param))
    env = gym.make('CartPole-v0')

    random.seed()

    simpleQLearningAgent = SimpleQLearningAgent(env.action_space, env.observation_space, (args.x_bins, args.x_dot_bins, args.theta_bins, args.theta_dot_bins), args)

    episodes = args.episodes
    total_reward = 0
    done = False

    solved = 0
    solved_consecutive = 0
    solved_max_consecutive=0
    for i in range(episodes):
        total_reward = 0

        old_observation = env.reset()
        if args.ui > 0 and i >= args.ui - 1:
            env.render()

        while True:
            action = simpleQLearningAgent.act(old_observation, i)

            new_observation, reward, done, _ = env.step(action)
            if args.ui > 0 and i >= args.ui - 1:
                env.render()

            simpleQLearningAgent.train(old_observation, action, new_observation, reward, done, i)

            old_observation = new_observation

            total_reward += reward

            if done:
                if (args.output_print):
                    print(
                    'Episode {}: {}, ;Learn {}, Exp {}'.format(i + 1, total_reward, simpleQLearningAgent.learning_rate,
                                                               simpleQLearningAgent.exploration_rate))
                solved = solved + 1 if total_reward >= 195 else solved
                solved_consecutive = solved_consecutive + 1 if total_reward >= 195 else 0
                solved_max_consecutive = max(solved_max_consecutive, solved_consecutive)
                break

        if solved_consecutive >= 100:
                break
    if (args.output_print):
        print("Solved: {}({})".format(solved, solved_max_consecutive))
        if solved_consecutive >= 100:
            print('Problem solved')
    else:
        if solved_consecutive >= 100:
            file = open("sucess.txt", 'a')
            output = ("{}({}), {}\n".format(solved, solved_max_consecutive, param))
            print(output)
            file.write(output)
            file.close()
        else:
            file = open("fail.txt", 'a')
            output = ("{}({}), {}\n".format(solved, solved_max_consecutive, param))
            print(output)
            file.write(output)
            file.close()