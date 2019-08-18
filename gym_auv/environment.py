import os
import gym
from gym import spaces
import numpy as np

from gym.utils import seeding, EzPickle
from gym_auv.rendering import render_env, init_env_viewer, FPS

from numpy.random import random

class BaseShipScenario(gym.Env, EzPickle):
    """Creates an environment with a vessel and a path.
    
    Attributes:
        config : dict
            The configuration disctionary specifying rewards,
            look ahead distance, simulation timestep and desired cruise
            speed.
        nsectors : int
            The number of obstacle sectors.
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
    
    Raises:
        NotImplementedError: Method is not implemented.
    """

    metadata = {
        'render.modes': ['human', 'rgb_array', 'state_pixels'],
        'video.frames_per_second': FPS
    }

    def __init__(self, env_config):
        """
        The __init__ method declares all class atributes and calls
        the self.reset() to intialize them properly.

        Parameters
        ----------
        env_config : dict
            Configuration parameters for the environment.
            Must have the following members:
            reward_ds
                The reward for progressing ds along the path in
                one timestep. reward += reward_ds*ds.
            reward_speed_error
                reward += reward_speed_error*speed_error where the
                speed error is abs(speed-cruise_speed)/max_speed.
            reward_cross_track_error
                reward += reward_cross_track_error*cross_track_error
            la_dist
                The look ahead distance.
            t_step_size
                The timestep
            cruise_speed
                The desired cruising speed.
        """
        self.config = env_config
        self.nstates = 6
        self.vessel = None
        self.path = None
        self.obstacles = None

        self.np_random = None

        self.cumulative_reward = 0
        self.past_rewards = None
        self.path_prog = None
        self.past_actions = None
        self.past_obs = None
        self.past_errors = None
        self.t_step = None
        self.episode = 0

        init_env_viewer(self)

        self.reset()

        self.action_space = gym.spaces.Box(
            low=np.array([0, -1]),
            high=np.array([1, 1]),
            dtype=np.float32
        )
        nobservations = self.nstates + self.nsectors
        low_obs = [-1]*nobservations
        high_obs = [1]*nobservations
        low_obs[self.nstates - 1] = -10000
        high_obs[self.nstates - 1] = 10000
        self.observation_space = gym.spaces.Box(
            low=np.array(low_obs),
            high=np.array(high_obs),
            dtype=np.float32
        )
        

    def step(self, action):
        """
        Simulates the environment for one timestep when action
        is performed

        Parameters
        ----------
        action : np.array
            [propeller_input, rudder_position].
        Returns
        -------
        obs : np.array
            Observation of the environment after action is performed.
        step_reward : double
            The reward for performing action at his timestep.
        done : bool
            If True the episode is ended.
        info : dict
            Empty, is included because it is required of the
            OpenAI Gym frameowrk.
        """
        self.past_actions = np.vstack([self.past_actions, action])
        self.vessel.step(action)

        if (self.path is not None):
            prog = self.path.get_closest_arclength(self.vessel.position)
            self.path_prog = np.append(self.path_prog, prog)

        obs = self.observe()
        self.past_obs = np.vstack([self.past_obs, obs])
        done, step_reward, info = self.step_reward()
        self.past_rewards = np.append(self.past_rewards, step_reward)
        self.cumulative_reward += step_reward

        self.t_step += 1

        return obs, step_reward, done, info

    def reset(self):
        """
        Resets the environment by reseeding and calling self.generate.

        Returns
        -------
        obs : np.array
            The initial observation of the environment.
        """
        self.vessel = None
        self.path = None
        self.cumulative_reward = 0
        self.path_prog = None
        self.past_actions = np.array([[0, 0]])
        self.past_rewards = np.array([])
        self.past_errors = {
            'speed': np.array([]),
            'cross_track': np.array([]),
        }
        self.obstacles = []

        if self.np_random is None:
            self.seed()

        self.generate()
        obs = self.observe()
        self.past_obs = np.array([obs])
        self.t_step = 0
        self.episode += 1
        return obs

    def close(self):
        self.viewer.close()

    def generate(self):
        raise NotImplementedError()

    def observe(self):
        raise NotImplementedError()

    def step_reward(self):
        raise NotImplementedError()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def render(self, mode='human'):
        image_arr = render_env(self, mode)
        return image_arr


