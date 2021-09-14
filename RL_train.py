"""
# time: 07/09/2021
# update: 09/09/2021
# author: Bobby
Main function to launch training in a single reinforcement environment
"""

import torch
from Algorithms.PPO import PPO
from Algorithms.VPG import VPG
from Entities.agent import RLAgent
from Networks.Discrete import Actor_net, Critic_net
from Entities.environment_sampler import Environment_sampler
import csv
import gym

def train():
    #env = gym.make("CartPole-v0")
    env = Environment_sampler(see_goal=True, obstacles="Easy").single_env(goal=(0.1,0.1))
    action_space, observation_space = env.action_space.n, env.observation_space.shape[0]
    critic_net = Critic_net(input=observation_space,
                            hidden=64,
                            output=1)
    actor_net = Actor_net(input=observation_space,
                          hidden=64,
                          output=action_space)
    
    # actor_net = torch.load("maml.pkl")

     
    agent = RLAgent(critic_net = critic_net,
                  actor_net = actor_net,
                  env = env,
                  device = "cpu",
                  algo = VPG,
                  LEARNING_RATE = 0.001)
    for i in range(5):
        trajectory, ep_rewards, distance = agent.sample_trajectory()
        agent.learn(trajectory)
        print("episode: {}   distance: {}".format(i,distance))
        with open("/home/gamma/wb_alchemy/sub_project/Chongkai/difficulty_test.csv", "a+") as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow(["lbd","VPG",i,distance])

if __name__ == "__main__":
    train()
        