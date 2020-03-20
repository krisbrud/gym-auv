import gym
import numpy as np
from gym.utils import seeding

from gym_auv.objects.vessel import Vessel
from gym_auv.objects.rewarder import *

import gym_auv.rendering.render2d as render2d
import gym_auv.rendering.render3d as render3d

class ASV_Scenario(gym.Env):
    """
    Creates an environment with a vessel and a path.
    """

    metadata = {
        'render.modes': ['human', 'rgb_array', 'state_pixels'],
        'video.frames_per_second': render2d.FPS
    }

    def __init__(self, env_config, test_mode=False, render_mode='2d', verbose=False):
        """
        The __init__ method declares all class atributes and calls
        the self.reset() to intialize them properly.

        Parameters
        ----------
            env_config : dict
                Configuration parameters for the environment. 
                The default values are set in __init__.py
            test_mode : bool
                If test_mode is True, the environment will not be autonatically reset 
                due to too low cumulative reward or too large distance from the path. 
            render_mode : {'2d', '3d', 'both'}
                Whether to use 2d or 3d rendering. 'both' is currently broken.
            verbose
                Whether to print debugging information.
        """
        
        self.test_mode = test_mode
        self.render_mode = render_mode
        self.verbose = verbose
        self.config = env_config
        
        # Setting dimension of observation vector
        self.n_observations = len(Vessel.NAVIGATION_STATES) + 3*self.config["n_sectors"]

        self.episode = 0
        self.total_t_steps = 0
        self.t_step = 0
        self.history = []
        self.rewarder = ColavRewarder()

        # Declaring attributes
        self.obstacles = None
        self.vessel = None
        self.path = None
        
        self.reached_goal = None
        self.collision = None
        self.progress = None
        self.cumulative_reward = None
        self.last_reward = None
        self.last_episode = None
        self.rng = None
        self._tmp_storage = None

        self.action_space = gym.spaces.Box(
            low=np.array([-1, -1]),
            high=np.array([1, 1]),
            dtype=np.float32
        )
        self.observation_space = gym.spaces.Box(
            low=np.array([-1]*self.n_observations),
            high=np.array([1]*self.n_observations),
            dtype=np.float32
        )

        # Initializing rendering
        self.viewer2d = None
        self.viewer3d = None
        if self.render_mode == '2d' or self.render_mode == 'both':
            render2d.init_env_viewer(self)
        if self.render_mode == '3d' or self.render_mode == 'both':
            render3d.init_env_viewer(self, autocamera=self.config["autocamera3d"])

        self.reset()

    def reset(self):
        """
        Resets the environment by reseeding and calling self._generate.

        Returns
        -------
        obs : np.array
            The initial observation of the environment.
        """

        # Seeding
        if self.rng is None:
            self.seed()

        # Saving information about episode
        if self.t_step:
           self._save_latest_episode()

        # Incrementing counters
        self.episode += 1
        self.total_t_steps += self.t_step

        # Resetting all internal variables
        self.cumulative_reward = 0
        self.t_step = 0
        self.last_reward = 0
        self.reached_goal = False
        self.collision = False
        self.progress = 0

        # Generating a new environment
        if self.verbose:    print('Generating scenario...')
        self._generate()
        if self.verbose:    print('Generated scenario')

        # Resetting rewarder instance
        self.rewarder.reset(self.vessel)

        # Initializing 3d viewer
        if self.render_mode == '3d':
            render3d.init_boat_model(self)
            self.viewer3d.create_path(self.path)

        # Getting initial observation vector
        obs = self.observe()[0]
        if self.verbose:    print('Calculated initial observation')

        # Resetting temporary data storage
        self._tmp_storage = {
            'cross_track_error': [],
        }

        return obs

    def _generate(self):
        raise NotImplementedError

    def observe(self):
        navigation_states, reached_goal, progress = self.vessel.navigate(self.path)
        sector_closenesses, sector_velocities, collision = self.vessel.perceive(self.obstacles)

        obs = np.concatenate([navigation_states, sector_closenesses, sector_velocities])
        return (obs, collision, reached_goal, progress) 

    def step(self, action):
        """
        Simulates the environment for one timestep when action
        is performed.

        Parameters
        ----------
        action : np.array
            [thrust_input, torque_input].
        Returns
        -------
        obs : np.array
            Observation of the environment after action is performed.
        reward : double
            The reward for performing action at his timestep.
        done : bool
            If True the episode is ended, due to either a collision or having reached the goal position.
        info : dict
            Dictionary with data used for reporting or debugging
        """

        action[0] = (action[0] + 1)/2 # Done to be compatible with RL algorithms that require symmetric action spaces
        if np.isnan(action).any(): action = np.zeros(action.shape)

        # If the environment is dynamic, calling self.update will change it.
        self._update()

        # Updating vessel state from its dynamics model
        self.vessel.step(action)

        # Getting observation vector
        obs, collision, reached_goal, progress = self.observe()
        self.collision = collision
        self.reached_goal = reached_goal
        self.progress = progress

        # Receiving agent's reward
        reward = self.rewarder.calculate()
        self.last_reward = reward
        self.cumulative_reward += reward

        info = {}
        info['collision'] = collision
        info['reached_goal'] = reached_goal
        info['progress'] = progress

        # Testing criteria for ending the episode
        done = any([
            collision,
            reached_goal,
            self.t_step > self.config["max_timesteps"],
            self.cumulative_reward < self.config["min_cumulative_reward"]
        ])

        self._save_latest_step()

        return obs, reward, done, info

    def _update(self):
        dt = self.config["t_step_size"]
        [obst.update(dt) for obst in self.obstacles if not obst.static]

    def close(self):
        if self.viewer2d is not None:
            self.viewer2d.close()
        if self.viewer3d is not None:
            self.viewer3d.close()

    def render(self, mode='human'):
        """
        Render the environment.
        """

        if self.render_mode == '2d' or self.render_mode == 'both':
            image_arr = render2d.render_env(self, mode)
        if self.render_mode == '3d' or self.render_mode == 'both':
            image_arr = render3d.render_env(self, mode, self.config["t_step_size"])
        return image_arr

    def seed(self, seed=None):
        self.rng, seed = seeding.np_random(seed)
        return [seed]

    def _save_latest_step(self):
        self._tmp_storage['cross_track_error'].append(abs(self.vessel.last_navi_state_dict['cross_track_error']))

    def _save_latest_episode(self):
        self.last_episode = {
            'path': self.path(np.linspace(0, self.path.length, 1000)) if self.path is not None else None,
            'path_taken': self.vessel.path_taken,
            'obstacles': []
        }
        self.history.append({
            'cross_track_error': np.array(self._tmp_storage['cross_track_error']).mean(),
            'reached_goal': int(self.reached_goal),
            'collision': int(self.collision),
            'reward': self.cumulative_reward,
            'timesteps': self.t_step,
            'progress': self.progress
        })