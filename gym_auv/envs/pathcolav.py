"""
This module implements the AUV gym environment through the AUVenv class.
"""

import sys, os
import gym
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

class PathColavEnv(BaseShipScenario):
    """
    Creates an environment with a vessel and a path.
    Attributes
    ----------
    config : dict
        The configuration disctionary specifying rewards,
        look ahead distance, simulation timestep and desired cruise
        speed.
    nstates : int
        The number of state variables passed to the agent.
    vessel : gym_auv.objects.auv.AUV2D
        The AUV that is controlled by the agent.
    path : gym_auv.objects.path.RandomCurveThroughOrigin
        The path to be followed.
    np_random : np.random.RandomState
        Random number generator.
    reward : float
        The accumulated reward
    path_prog : float
        Progression along the path in terms of arc length covered.
    past_actions : np.array
        All actions that have been perfomed.
    action_space : gym.spaces.Box
        The action space. Consists of two floats that must take on
        values between -1 and 1.
    observation_space : gym.spaces.Box
        The observation space. Consists of
        self.nstates + self.nsectors floats that must be between
        0 and 1.
    """

    def __init__(self, env_config, test_mode):
        super().__init__(env_config, test_mode)
        self.action_space = gym.spaces.Box(
            low=np.array([-1, -1]),
            high=np.array([1, 1]),
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
        action[0] = (action[0] + 1)/2 # For algorithms that require symmetric action spaces
        self.vessel.step(action)

    def step_reward(self, action, obs):
        """
        Calculates the step_reward and decides whether the episode
        should be ended.

        Returns
        -------
        done : bool
            If True the episode is ended.
        step_reward : double
            The reward for performing action at his timestep.
        """
        done = False
        step_reward = 0
        info = {"collision": False, 'path_reward': None, 'closeness_reward': None}

        for obst_dist, _, obst in self.nearby_obstacles:
            if obst_dist <= 0:
                if (not obst.collided):
                    obst.collided = True
                    self.collisions += 1
                info["collision"] = True
                if self.config["end_on_collision"]:
                    step_reward = self.config["min_reward"]*(1-self.config["reward_lambda"])
                    done = True
                    break
                elif self.config["teleport_on_collision"]:
                    step_reward = self.config["min_reward"]*(1-self.config["reward_lambda"])
                    self.vessel.teleport_back(200)
                    break

        if (not done):
            path_reward = self.get_path_reward()
            info['path_reward'] = path_reward

            closeness_reward = self.get_closeness_reward(collision=info["collision"])
            info['closeness_reward'] = closeness_reward

            #print(('{:.2f}, '*len(self.sensor_obst_closenesses)).format(*self.sensor_obst_closenesses))

            step_reward = self.config["reward_lambda"]*path_reward + \
                (1-self.config["reward_lambda"])*closeness_reward - \
                self.living_penalty + \
                self.config["reward_speed"]*self.vessel.speed/self.vessel.max_speed - \
                self.config["penalty_rudder_angle_change"]*self.vessel.smoothed_rudder_change - \
                self.config["penalty_rudder_angle"]*self.vessel.smoothed_rudder

            # print(('{:.2f}, '*6).format(
            #     self.heading_progress, path_reward, closeness_reward, self.living_penalty, 
            #     self.vessel.smoothed_rudder_change, self.vessel.smoothed_rudder
            # ))
            
            #step_reward = max(-self.config["penalty_collision"], step_reward)

        step_reward = max(self.config["min_reward"] - self.cumulative_reward, step_reward)

        return done, step_reward, info