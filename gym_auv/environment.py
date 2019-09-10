import os
import gym
from gym import spaces
from numpy.random import random
import numpy as np
import numpy.linalg as linalg
import gym_auv.utils.geomutils as geom

from gym.utils import seeding, EzPickle
from gym_auv.rendering import render_env, init_env_viewer, FPS

class BaseShipScenario(gym.Env):
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
            self.nstates + self.nsectors*2 floats that must be between
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
        self.nstates = 7
        self.nsectors = self.config["n_sectors"]
        self.nsensors = self.config["n_sensors_per_sector"]*self.config["n_sectors"]
        self.nrings = self.config["n_rings"]
        sum_radius = 30
        self.ring_depths = [sum_radius]
        self.rings = [sum_radius/2]
        if (self.config["rear_detection"]):
            self.ring_sectors = [7+4*i for i in range(self.nrings)]
        else:
            self.ring_sectors = [12 for i in range(self.nrings)]
        self.n_detection_grid_sections = sum(self.ring_sectors)
        for j in range(self.nrings-1):
            if (self.config["rear_detection"]):
                radius = 2*np.pi*sum_radius/(self.ring_sectors[j+1] - 2*np.pi)
            else:
                radius = np.pi*sum_radius/(self.ring_sectors[j+1] - np.pi)
            self.ring_depths.append(radius)
            self.rings.append(sum_radius + radius/2)
            sum_radius += radius

        self.sensor_angle = (4*np.pi/3)/(self.nsensors + 1)
        self.detection_images = [np.zeros((self.ring_sectors[i]),) for i in range(self.nrings)]
        self.feasibility_images = [np.zeros((self.ring_sectors[i]),) for i in range(self.nrings)]
        self.sensor_angles = [-2*np.pi/3 + (i + 1)*self.sensor_angle for i in range(self.nsensors)]
        if (self.config["rear_detection"]):
            self.sector_angles = [
                [-np.pi + (isector + 0.5)/(self.ring_sectors[iring])*2*np.pi for isector in range(self.ring_sectors[iring])] for iring in range(self.nrings)
            ]
        else:
            self.sector_angles = [
                [-2*np.pi/3 + (isector + 0.5)/(self.ring_sectors[iring])*4*np.pi/3 for isector in range(self.ring_sectors[iring])] for iring in range(self.nrings)
            ]
        
        self.n_observations = self.nstates
        if (self.config["detection_grid"]):
            self.n_observations += self.n_detection_grid_sections
        if (self.config["lidars"]):
            self.lidar_obs_index = self.n_observations
            self.n_observations += self.nsectors

        self.sensor_obst_intercepts = [None for isensor in range(self.nsensors)]
        self.sector_active = [0 for isector in range(self.nsectors)]
        self.sensor_obst_measurements = np.zeros((self.nsensors, ))
        self.sensor_path_arclengths = np.zeros((self.nsensors, ))
        self.sensor_path_index = None
        self.sensor_order = range(self.nsensors)
        self.vessel = None
        self.path = None
        self.obstacles = None
        self.nearby_obstacles = None
        self.look_ahead_point = None
        self.look_ahead_arclength = None
        self.reached_goal = None

        self.np_random = None

        self.cumulative_reward = 0
        self.past_rewards = None
        self.max_path_prog = None
        self.target_arclength = None
        self.path_prog = None
        self.past_actions = None
        self.past_obs = None
        self.past_errors = None
        self.t_step = 0
        self.total_t_steps = 0
        self.episode = 0
        self.last_episode = []
        self.history = []

        self.collisions = None
        self.sensor_updates = None

        init_env_viewer(self)

        self.reset()

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
        action[0] = (action[0] + 1)/2
        self.past_actions = np.vstack([self.past_actions, action])
        self.vessel.step(action)

        la_heading = self.path.get_direction(self.target_arclength)
        self.la_heading_error = float(geom.princip(la_heading - self.vessel.heading))
        path_position = self.path(self.target_arclength) - self.vessel.position
        target_heading = np.arctan2(path_position[1], path_position[0])
        self.heading_error = float(geom.princip(target_heading - self.vessel.heading))
        self.la_distance = linalg.norm(path_position)
        path_direction = self.path.get_direction(self.max_path_prog)
        track_errors = geom.Rzyx(0, 0, -path_direction).dot(
            np.hstack([self.path(self.max_path_prog) - self.vessel.position, 0])
        )
        self.along_track_error = track_errors[0]
        self.cross_track_error = track_errors[1]
        self.cross_track_errors = np.append(self.cross_track_errors, abs(self.cross_track_error))
        self.speed_error = (linalg.norm(self.vessel.velocity) - self.config["cruise_speed"])/self.vessel.max_speed

        self.past_errors['speed'] = np.append(self.past_errors['speed'], self.speed_error)
        self.past_errors['cross_track'] = np.append(self.past_errors['cross_track'], self.cross_track_error)
        #self.past_errors['d_cross_track'] = np.append(self.past_errors['d_cross_track'], d_cross_track_error)
        self.past_errors['la_heading'] = np.append(self.past_errors['la_heading'], self.la_heading_error)
        self.past_errors['heading'] = np.append(self.past_errors['heading'], self.heading_error)

        #closest_point_distance, _, closest_arclength = self.path.get_closest_point_distance(self.vessel.position)
        #closest_point_heading_error = geom.princip(self.path.get_direction(closest_arclength) - self.vessel.course)
        course_path_angle = geom.princip(self.path.get_direction(self.max_path_prog) - self.vessel.course)
        dprog = np.cos(course_path_angle)*self.vessel.speed*self.config["t_step_size"] - self.along_track_error*0.05
        prog = min(max(0, self.max_path_prog + dprog), self.path.length)

        if prog > self.max_path_prog:
            self.max_path_prog = prog

        self.path_prog = np.append(self.path_prog, prog)

        if (self.look_ahead_arclength is None):
            target_arclength_candidate = self.max_path_prog + self.config["min_la_dist"]
        else:
            target_arclength_candidate = max(self.look_ahead_arclength, self.max_path_prog + self.config["min_la_dist"])
        
        if (target_arclength_candidate > self.target_arclength):
            self.target_arclength = min(target_arclength_candidate, self.path.length)

        obs = self.observe()
        assert not np.isnan(obs).any(), 'Observation vector "{}" contains nan values.'.format(str(obs))
        self.past_obs = np.vstack([self.past_obs, obs])
        done, step_reward, info = self.step_reward()
        info['progress'] = prog/self.path.length
        self.past_rewards = np.append(self.past_rewards, step_reward)
        self.cumulative_reward += step_reward

        self.t_step += 1
        self.total_t_steps += 1
        if (self.t_step > self.config["max_timestemps"]):
            done = True

        return obs, step_reward, done, info

    def reset(self):
        """
        Resets the environment by reseeding and calling self.generate.

        Returns
        -------
        obs : np.array
            The initial observation of the environment.
        """

        if (self.t_step > 0):
            self.last_episode = {
                'path': self.path(np.linspace(0, self.path.s_max, 1000)),
                'path_taken': self.vessel.path_taken,
            }
            self.history.append({
                'collisions': self.collisions,
                'cross_track_error': self.cross_track_errors.mean(),
                'collision_baselines': self.get_collision_baseline(),
                'progress': self.max_path_prog/self.path.length,
                'reached_goal': int(self.reached_goal),
                'reward': self.cumulative_reward,
                'timesteps': self.t_step
            })

        self.vessel = None
        self.path = None
        self.cumulative_reward = 0
        self.max_path_prog = 0
        self.target_arclength = 0
        self.path_prog = None
        self.past_obs = None
        self.past_actions = np.array([[0, 0]])
        self.past_rewards = np.array([])
        self.past_errors = {
            'speed': np.array([]),
            'cross_track': np.array([]),
            'heading': np.array([]),
            'la_heading': np.array([]),
            'd_cross_track': np.array([]),
        }
        
        self.obstacles = []
        self.nearby_obstacles = []
        self.t_step = 0
        self.look_ahead_point = None
        self.look_ahead_arclength = None
        self.reached_goal = False

        self.heading_error_la = 0
        self.heading_error = 0
        self.la_distance = 0
        self.along_track_error = 0
        self.cross_track_error = 0

        self.collisions = 0
        self.cross_track_errors = np.array([])
        self.sensor_updates = 0

        if self.np_random is None:
            self.seed()

        self.generate()
        obs = self.observe()
        assert not np.isnan(obs).any(), 'Observation vector "{}" contains nan values.'.format(str(obs))
        self.past_obs = np.array([obs])
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
    
    def get_collision_baseline(self):
        baseline = 0
        for obst in self.obstacles:
            closest_point_distance, _, _ = self.path.get_closest_point_distance(obst.position)
            if (closest_point_distance <= obst.radius):
                baseline += 1
        return baseline