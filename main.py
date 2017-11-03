import gym

class RandomAgent():
    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, observation, reward, done):
        return self.action_space.sample()

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
