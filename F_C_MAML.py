"""
# time: 09/09/2021
# update: /
# author: Bobby
an implementation of MAML Algorithm
"""

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


# initialize test environments
environments = Environment_sampler(see_goal=True, 
                                  obstacles="Medium").multi_env(env_num=12,
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

def MAML(tasks,n):
    # main training loop
    for i in range(EPISODE_LENGTH):
        cumulative_loss = 0.  # used for optimizing the meta_learner
        cumulative_distance = 0.
        # cumulative_rewards = 0.  # used for evaluating the mete_learner
        for j, environment in enumerate(tasks):
            meta_leaner = MetaAgent(critic_net=deepcopy(baseline_net),
                                actor_net=clone_module(policy_net),
                                env=environment,
                                device="cpu",
                                LEARNING_RATE=INNER_LEARNING_RATE,
                                algo=VPG)
        
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

        with open("/home/gamma/wb_alchemy/sub_project/Chongkai/difficulty_classify.csv", "a+") as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow(["curriculum", "VPG",  i, ep_distance,n])

def difficult_classifier(total_tasks):
    """experimental trial"""
    task_scheduler = [[],[],[]]  # [[easy],[medium], [hard]]
    for task in total_tasks:
        if 5< task.goal[0]<10 and 5<task.goal[1]<10:
            task_scheduler[0].append(task)
        elif 0<task.goal[0]<5 and 5<task.goal[1]<10:
            task_scheduler[1].append(task)
        elif 5<task.goal[0]<10 and 0<task.goal[1]<5:
            task_scheduler[1].append(task)
        elif 0<task.goal[0]<5 and 0<task.goal[1]<5:
            task_scheduler[2].append(task)
        else:
            raise AttributeError("Check your goal") 
    return task_scheduler


def train(n):
    privious_net = policy_net.state_dict()
    tasks = difficult_classifier(environments)
    for levels in tasks:
        policy_net.load_state_dict(privious_net)
        MAML(levels,n)
        privious_net = policy_net.state_dict()
    
    torch.save(policy_net, "difficulty_classifier_{}.pkl".format(n))
     
if __name__ =="__main__":
    for i in range(3):
        train(i)
    
    


