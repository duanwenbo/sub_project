"""
# time: 07/09/2021
# update: /
# author: Bobby
Main function to launch training in a single reinforcement environment
"""

import gym
from Algorithms.PPO import PPO
from Entities.agent import Agent
from Networks.Discrete import Actor_net, Critic_net
from Environment.navigation import Navigation

def train():
    # env = gym.make("CartPole-v0")
    env = Navigation(goal=(4.3,2.1))
    action_space, observation_space = env.action_space.n, env.observation_space.shape[0]
    critic_net = Critic_net(input=observation_space,
                            hidden=64,
                            output=1)
    actor_net = Actor_net(input=observation_space,
                          hidden=64,
                          output=action_space)
    agent = Agent(critic_net = critic_net,
                  actor_net = actor_net,
                  env = env,
                  device = "cpu",
                  algo = PPO,
                  LEARNING_RATE = 0.0005)
    for i in range(5000):
        trajectory, ep_rewards, distance = agent.sample_trajectory()
        agent.learn(trajectory)
        print("episode: {}   distance: {}".format(i, distance))

if __name__ == "__main__":
    train()
        