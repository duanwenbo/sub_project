"""
# time: 09/09/2021
# update: /
# author: Bobby
an implementation of MAML Algorithm
"""
import time

from Entities.agent import MetaAgent
import torch
from Networks.Discrete import Actor_net, Critic_net
from Entities.environment_sampler import Environment_sampler
from Entities.clone_module import clone_module
import yaml
from copy import deepcopy
import wandb
import csv
from Algorithms.PPO import PPO
from Algorithms.VPG import VPG

EPISODE_LENGTH = 3000
INNER_LEARNING_RATE = 0.0003
OUTER_LEARNING_RATE = 0.0005
STEP_NUM = 1

def train(n):
    # initialize test environments
    environments = Environment_sampler(see_goal=True, 
                                    obstacles="Medium").multi_env(env_num=8,
                                                                    goal_distribution="average")
    # discrete action space env 
    action_space, observation_space = environments[0].action_space.n, environments[0].observation_space.shape[0]
    # create policy net and baseline net
    policy_net = Actor_net(input=observation_space,
                        hidden=128,
                        output=action_space)
    baseline_net = Critic_net(input=observation_space,
                            hidden=128,
                            output=1)
    policy_optimizer = torch.optim.Adam(policy_net.parameters(), 
                                        lr=OUTER_LEARNING_RATE)# essential parameters


    # main training loop
    for i in range(EPISODE_LENGTH):
        cumulative_loss = 0.  # used for optimizing the meta_learner
        cumulative_distance = 0.
        # cumulative_rewards = 0.  # used for evaluating the mete_learner
        for j, environment in enumerate(environments):
            meta_leaner = MetaAgent(critic_net=deepcopy(baseline_net),
                                actor_net=clone_module(policy_net), 
                                env=environment,
                                device="cpu",
                                LEARNING_RATE=INNER_LEARNING_RATE,
                                algo=VPG)

            # expeiment 1: one step opt ---> multiple step opt
            for _ in range(STEP_NUM):
                trajectory,_,_ = meta_leaner.sample_trajectory()
                meta_leaner.learn(trajectory)
            new_trajectory, ep_reward, distance = meta_leaner.sample_trajectory()
            one_step_opt_loss = meta_leaner.policy_loss(new_trajectory)
            cumulative_loss += one_step_opt_loss
            cumulative_distance += distance

            print("task:{}, distance:{}".format(j, round(distance,2)))
        
        policy_optimizer.zero_grad()
        cumulative_loss.backward()
        policy_optimizer.step()

        ep_distance = round(cumulative_distance/len(environments),2)
        print("##############################")
        print("episode:{}  distance:{}".format(i, ep_distance))
        print("##############################")

        with open("/home/gamma/wb_alchemy/sub_project/Chongkai/maml_02.csv", "a+") as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow(["maml_wide_distribution", "VPG",  i, ep_distance,n])
    torch.save(policy_net, "maml.pkl")

if __name__ == "__main__":
    for i in range(1):
        t1 = time.time()
        train(i)
        t2 = time.time()
        td = t2-t1
        with open("/home/gamma/wb_alchemy/sub_project/Chongkai/maml_02.csv", "a+") as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow(["time", td])
