"""
# time: 09/09/2021
# update: /
# author: Bobby
Creating tasks for both RL and Meta_RL according to different requirements
"""
from Environment.navigationPro import NavigationPro
import gym

class Environment_sampler:
    def __init__(self, see_goal=False, obstacles="None") -> None:
        self.see_goal = see_goal
        self.obstacles = obstacles
        self.env = NavigationPro

    def single_env(self, goal=(0.1,0.1)):
        """for a single env test in reinforcement learning"""
        env = self.env(goal, self.see_goal, self.obstacles)
        return env
    
    def single_env_gym(self, env="CartPole-v1"):
        """for validation purpose"""
        return gym.make(env)
        
    def multi_env(self, env_num=5, goal_distribution="average"):
        pass