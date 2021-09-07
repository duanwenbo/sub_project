from gym import Env
from gym.spaces import Discrete, Box
import numpy as np


class Navigation(Env):
    metadata = {"render.modes":["console"]}
    def __init__(self, goal) -> None:
        super(Navigation, self).__init__()
        self.action_space = Discrete(8)
        self.observation_space = Box(low=np.array([0,0]), high=np.array([10,10]))
        self.state = np.array([9.,9.])
        self.episode_length = 100
        self.action_dict = {"0":(0,0.1),"1":(0.1,0.1),"2":(0.1,0),"3":(0.1,-0.1),"4":(0,-0.1),"5":(-0.1,-0.1),"6":(-0.1,0),"7":(-0.1,0.1)}
        # self.goal = (random.uniform(0,10), random.uniform(0,10))
        self.goal = goal # for model test

    def step(self, action):
        self.episode_length -= 1
        x_increment, y_increment = self.action_dict[str(action)][0], self.action_dict[str(action)][1]
        # x_increment, y_increment = 0.3 * math.cos(math.radians(action)), 0.3 * math.sin(math.radians(action))
        self.state = (self.state[0]+ x_increment, self.state[1]+y_increment)
        distance = ((self.state[0] - self.goal[0])**2 + (self.state[1] - self.goal[1])**2)
        # amplify_factor= max([0.008, self.episode_length / 100]) # normalized amplified factor
        reward = (-distance) 
        if self.episode_length <=0:
            done = True
        elif distance <= 0.05:
            done = True
        else:
            done = False
        return np.array(self.state), reward, done, distance

    def reset(self):
        self.state =  np.array([9.,9.])
        self.episode_length = 100
        return self.state

    def render(self):
        pass



if __name__ == "__main__":
    a = np.array([1,2])
    b = np.array([4,4,a[0]])
    print(b)