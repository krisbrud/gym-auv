import sys, os
import gym
from stable_baselines import PPO2, DDPG, TD3
from gym.utils import seeding
import numpy as np
import numpy.linalg as linalg
from scipy.ndimage.filters import gaussian_filter1d, gaussian_filter
import shapely.geometry

import gym_auv.utils.geomutils as geom
from gym_auv.objects.auv import AUV2D
from gym_auv.objects.path import RandomCurveThroughOrigin
from gym_auv.objects.obstacles import StaticObstacle
from gym_auv.environment import BaseShipScenario

from matplotlib import pyplot as plt

class PathColavControlEnv(BaseShipScenario):
    def __init__(self, env_config, test_mode, pilot):
        self.pilot = PPO2.load(pilot)
        self.log_lambda = 0
        self.cur_lambda = 0
        self.reward_shift = 100
        self.progression_filter = 0
        self.pathfollow_improvement_filter = 0
        self.filter_alpha = 0.99
        super().__init__(env_config, test_mode)
        self.action_space = gym.spaces.Box(
            low=np.array([-0.1]),
            high=np.array([0.01]),
            dtype=np.float32
        )
        
        low_obs = [-1]*self.n_observations
        high_obs = [1]*self.n_observations
        low_obs[self.nstates - 1] = -10000
        high_obs[self.nstates - 1] = 10000
        self.observation_space = gym.spaces.Box(
            low=np.array(low_obs),
            high=np.array(high_obs),
            dtype=np.float32
        )

    def apply_action(self, action, obs):
        self.log_lambda = np.clip(self.log_lambda + action[0], -7, 0)
        #self.cur_lambda = np.clip(self.cur_lambda, 0.01, 1.1)

        x = obs.copy()
        x[6] = np.power(10, self.log_lambda) # action[0] # lambda
        #x[4] = x[4] + action[1] # heading error
        self.config["reward_lambda"] = x[6]
        control_signal, _states = self.pilot.predict(x, deterministic=True)
        #print('Applying action: ', action, 'control signal', control_signal)
        self.vessel.step(control_signal)

    def step_reward(self, action, obs):
        done = False
        step_reward = 0
        info = {"collision": False, 'path_reward': None, 'closeness_reward': None}

        for obst_dist, _, obst in self.nearby_obstacles:
            if obst_dist <= 0:
                if (not obst.collided):
                    obst.collided = True
                    self.collisions += 1
                info["collision"] = True
                step_reward = self.config["min_reward"]
                done = True
                break

        if (not done):
            path_reward = self.get_path_reward()
            info['path_reward'] = path_reward
            closeness_reward = self.get_closeness_reward()
            info['closeness_reward'] = closeness_reward

            progression = (self.path_prog[-1] - self.path_prog[max(-len(self.path_prog), -2)])/self.vessel.max_speed
            pathfollow_improvement = self.past_path_rewards[-1] - self.past_path_rewards[max(-len(self.past_path_rewards), -2)] 
            
            self.progression_filter = self.filter_alpha*self.progression_filter + (1-self.filter_alpha)*progression
            self.pathfollow_improvement_filter = self.filter_alpha*self.pathfollow_improvement_filter + (1-self.filter_alpha)*pathfollow_improvement
            step_reward = self.progression_filter + 100*self.pathfollow_improvement_filter - 0.5

        step_reward = max(self.config["min_reward"] - self.cumulative_reward, step_reward)

        return done, step_reward, info