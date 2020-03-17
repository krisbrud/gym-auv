import os
import gym
import json
import pickle
import time
from gym import spaces
from numpy.random import random
import numpy as np
import numpy.linalg as linalg
import gym_auv.utils.geomutils as geom

from scipy.ndimage.filters import gaussian_filter1d, gaussian_filter
import shapely.geometry, shapely.errors
from gym_auv.objects.vessel import Vessel
from gym_auv.objects.path import RandomCurveThroughOrigin, Path
from gym_auv.objects.obstacles import CircularObstacle, PolygonObstacle, VesselObstacle

from gym.utils import seeding, EzPickle
import gym_auv.rendering.render2d as render2d
import gym_auv.rendering.render3d as render3d


class Environment(gym.Env):
    """Creates an environment with a vessel and a path.
    """

    metadata = {
        'render.modes': ['human', 'rgb_array', 'state_pixels'],
        'video.frames_per_second': render2d.FPS
    }

    # Number of observation features unrelated to rangefinder sensors
    N_STATES = 11

    def __init__(self, env_config, test_mode=False, render_mode='2d', detect_moving=False, verbose=False):
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
            detect_moving : bool
                Whether to feed obstacle velocities to the agent or not.
            verbose
                Whether to print debugging information.
        """
        
        self.input_config = env_config
        self.test_mode = test_mode
        self.render_mode = render_mode
        self.detect_movement = detect_moving
        self.verbose = verbose
        self.viewer2d = None
        self.viewer3d = None
        self.config = dict(env_config).copy()
        self.n_states = Environment.N_STATES
        self.n_sectors = self.config["n_sectors"]
        self.n_sensors = self.config["n_sensors_per_sector"]*self.config["n_sectors"]
        self.sensor_angle = 2*np.pi/(self.n_sensors)
        self.sensor_angles = [-np.pi + (i + 1)*self.sensor_angle for i in range(self.n_sensors)]
        self.n_sensors_per_sector = [0]*self.config["n_sectors"]
        self.sector_start_indeces = [0]*self.config["n_sectors"]
        last_isector = -1
        for isensor in range(self.n_sensors):
            isector = self.config["sector_partition_fun"](self, isensor)
            if isector != last_isector:
                last_isector = isector
                self.sector_start_indeces[isector] = isensor
            self.n_sensors_per_sector[isector] += 1

        # Setting dimension of observation vector
        self.n_observations = self.n_states
        self.n_observations += self.n_sectors
        if self.detect_movement:
            self.n_observations += 2*self.n_sectors # x and y columns

        # Declaring internal variables
        self.sensor_obst_intercepts_hist = []
        self.sensor_obst_intercepts_transformed_hist = None
        self.sensor_obst_intercepts = [None for isensor in range(self.n_sensors)]
        self.sector_active = [0 for isector in range(self.n_sectors)]
        self.sector_empty = [0 for isector in range(self.n_sectors)]
        self.sector_clear = [0 for isector in range(self.n_sectors)]
        self.sector_closeness = [0.0 for isector in range(self.n_sectors)]
        self.sector_last_heartbeat = [0 for isector in range(self.n_sectors)]
        self.sensor_obst_closenesses = np.zeros((self.n_sensors, ))
        self.sensor_obst_reldx = np.zeros((self.n_sensors, ))
        self.sensor_obst_reldy = np.zeros((self.n_sensors, ))
        self.sensor_obst_distances = np.ones((self.n_sensors, ))*self.config["sensor_range"]
        self.sensor_path_arclengths = np.zeros((self.n_sensors, ))
        self.sensor_path_index = None
        self.sensor_order = range(self.n_sensors)
        self.vessel = None
        self.path = None
        self.vessel_obstacles = None
        self.obstacles = None
        self.nearby_obstacles = None
        self.reached_goal = None
        self.last_checkpoint_time = None
        self.np_random = None
        self.critical_angle = None
        self.cumulative_reward = 0
        self.past_rewards = None
        self.past_path_rewards = None
        self.past_closeness_rewards = None
        self.max_path_prog = None
        self.target_arclength = None
        self.path_prog_hist = None
        self.past_actions = None
        self.past_obs = None
        self.past_errors = None
        self.t_step = 0
        self.total_t_steps = 0
        self.episode = 0
        self.last_episode = None
        self.history = []
        self.measurement_history = None
        self.collisions = None
        self.sensor_updates = None
        self.action_space = gym.spaces.Box(
            low=np.array([-1, -1]),
            high=np.array([1, 1]),
            dtype=np.float32
        )
        low_obs = [-1]*self.n_observations
        high_obs = [1]*self.n_observations
        low_obs[self.n_states - 1] = -10000
        high_obs[self.n_states - 1] = 10000
        self.observation_space = gym.spaces.Box(
            low=np.array(low_obs),
            high=np.array(high_obs),
            dtype=np.float32
        )

        # Initializing rendering
        if self.render_mode == '2d' or self.render_mode == 'both':
            render2d.init_env_viewer(self)
        if self.render_mode == '3d' or self.render_mode == 'both':
            render3d.init_env_viewer(self, autocamera=self.config["autocamera3d"])

        self.reset()

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
        if np.isnan(action).any():
            if self.verbose:
                print('Warning: action is NaN - replacing with zero')
            action = np.zeros(action.shape)

        # Updating path information
        if self.path is not None and self.t_step % self.config["update_interval_path"] == 0:
            la_heading = self.path.get_direction(self.target_arclength) # Tangential path direction at look-ahead point
            self.heading_error_la = float(geom.princip(la_heading - self.vessel.heading))
            path_position = self.path(self.target_arclength) - self.vessel.position # Vector difference between look-ahead point and vessel position
            self.la_distance = linalg.norm(path_position)
            self.target_heading = np.arctan2(path_position[1], path_position[0])
            self.heading_error = float(geom.princip(self.target_heading - self.vessel.course))
            goal_position = self.path(self.path.length) - self.vessel.position # Vector difference between goal position and vessel position
            self.goal_heading = np.arctan2(goal_position[1], goal_position[0])
            self.goal_heading_error = float(geom.princip(self.goal_heading - self.vessel.course))
            path_direction = self.path.get_direction(self.max_path_prog)  # Tangential path direction at reference point
            track_errors = geom.Rzyx(0, 0, -path_direction).dot(
                np.hstack([self.path(self.max_path_prog) - self.vessel.position, 0])
            )
            self.along_track_error = track_errors[0]
            self.cross_track_error = track_errors[1]
            self.cross_track_errors = np.append(self.cross_track_errors, abs(self.cross_track_error))
            self.course_path_angle = geom.princip(path_direction - self.vessel.course)
            self.heading_progress = np.cos(self.course_path_angle)*self.vessel.speed # Decomposed progress of vessel in the path direction
            self.speed_error = self.vessel.speed - self.config["cruise_speed"]

            self.past_errors['speed'] = np.append(self.past_errors['speed'], self.speed_error)
            self.past_errors['cross_track'] = np.append(self.past_errors['cross_track'], self.cross_track_error)
            self.past_errors['along_track'] = np.append(self.past_errors['along_track'], self.along_track_error)
            self.past_errors['la_heading'] = np.append(self.past_errors['la_heading'], self.heading_error_la)
            self.past_errors['heading'] = np.append(self.past_errors['heading'], self.heading_error)
            self.past_errors['torque_change'] = np.append(self.past_errors['torque_change'], self.vessel.smoothed_torque_change)
            
            # Updating path reference point if vessel is going forward.
            closest_arclength = self.path.get_closest_arclength(self.vessel.position)
            prog = closest_arclength
            if prog > self.max_path_prog:
                self.max_path_prog = prog
                self.last_checkpoint_time = self.t_step
        
            self.path_prog_hist = np.append(self.path_prog_hist, prog)

            # Updating look-ahead reference point / target point
            self.target_arclength = min(self.max_path_prog + self.config["look_ahead_distance"], self.path.length)

        # If the environment is dynamic, calling self.update will change it.
        self.update()

        # Obtaining the observation vector which is fed to the agent
        obs, collision = self.observe()
        assert not np.isnan(obs).any(), 'Observation vector "{}" contains nan values.'.format(str(obs))
        self.past_obs = np.vstack([self.past_obs, obs])

        # Updating vessel state from its dynamics model
        self.vessel.step(action)
        if self.past_actions is None:
            self.past_actions = np.array([action])
        else:
            self.past_actions = np.vstack([self.past_actions, action])

        # Receiving agent's reward as well as whether the episode is done or not
        done, reward, info = self._get_reward(collision)
        info['collision'] = collision
        self.past_rewards = np.append(self.past_rewards, reward)
        self.past_path_rewards = np.append(self.past_path_rewards, info['path_reward'])
        self.past_closeness_rewards = np.append(self.past_closeness_rewards, info['closeness_reward'])
        self.cumulative_reward += reward
        
        # Testing criteria for ending episode because vessel is too far from the path
        if self.path is not None and self.t_step % self.config["update_interval_path"] == 0:
            info['progress'] = prog/self.path.length
            info['collisions'] = self.collisions
            info['cross_track_error'] = self.cross_track_error
            info['heading_error'] = self.heading_error
            closest_point_distance, _, _ = self.path.get_closest_point_distance(self.vessel.position)
            if closest_point_distance > self.config["max_distance"] and not self.test_mode:
                done = True

        # Incrementing counters
        self.t_step += 1
        self.total_t_steps += 1

        # Testing criteria for ending the episode
        if self.t_step > self.config["max_timestemps"]:
            done = True
        if self.cumulative_reward <= self.config["min_cumulative_reward"] and not self.test_mode:
            done = True
        if collision:
            if self.config["end_on_collision"]:
                # Ending episode
                reward = self.config["min_cumulative_reward"]*(1-self.config["reward_lambda"])
                done = True
            elif self.config["teleport_on_collision"]:
                # Teleporting vesesl back in time
                reward = self.config["min_cumulative_reward"]*(1-self.config["reward_lambda"])
                self.vessel.teleport_back(200)

        dist_to_endpoint = linalg.norm(self.vessel.position - self.path.end)
        if (abs(self.path_prog_hist[-1] - self.path.length) < self.config["min_goal_progress"] or dist_to_endpoint < self.config["min_goal_distance"]):
            done = True
            self.reached_goal = True

        return obs, reward, done, info
        
    def _get_reward(self, collision):
        """
        Calculates the step reward and decides whether the episode
        should be ended.

        Returns
        -------
        done : bool
            If True the episode is ended.
        reward : float
            The reward for performing action at his timestep.
        info : dict
            Dictionary with extra information.
        """
        done = False
        reward = 0
        info = {'path_reward': None, 'closeness_reward': None}

        # Calculating reward
        if not done:
            path_reward = self.get_path_reward()
            info['path_reward'] = path_reward

            closeness_reward = self.get_closeness_reward(collision=collision)
            info['closeness_reward'] = closeness_reward

            # Calculating total reward
            reward = self.config["reward_lambda"]*path_reward + \
                (1-self.config["reward_lambda"])*closeness_reward - \
                self.living_penalty + \
                self.config["reward_eta"]*self.vessel.speed/self.vessel.max_speed - \
                self.config["penalty_yawrate"]*abs(self.vessel.yawrate) - \
                self.config["penalty_torque_change"]*abs(self.vessel.smoothed_torque_change)

        # Capping reward so that it will not lead to a cumulative reward less than the minimum
        reward = max(self.config["min_cumulative_reward"] - self.cumulative_reward, reward)

        # A trick to improve performance?
        if reward < 0:
            reward *= 2 

        return done, reward, info

    def get_path_reward(self):
        cross_track_performance = np.exp(-self.config["reward_gamma_y_e"]*np.abs(self.cross_track_error))
        path_reward = (1 + np.cos(self.heading_error)*self.vessel.speed/self.vessel.max_speed)*(1 + cross_track_performance) - 1
        return path_reward

    def get_closeness_reward(self, collision=False):
        if collision:
            return -1/self.config["reward_gamma_x"]
        closeness_reward_num = 0
        closeness_reward_den = 0
        if self.n_sensors > 0:
            for isensor in range(self.n_sensors):
                sensor_angle = self.sensor_angles[isensor]
                sensor_distance = self.sensor_obst_distances[isensor]
                sensor_weight = 1 / (1 + np.abs(self.config["reward_gamma_theta"]*sensor_angle))
                sensor_raw_penalty = 1/(self.config["reward_gamma_x"]*(max(sensor_distance, 1))**2)
                sensor_reward = sensor_weight*sensor_raw_penalty
                closeness_reward_num += sensor_reward
                closeness_reward_den += sensor_weight
            closeness_reward = -closeness_reward_num/closeness_reward_den
        else:
            closeness_reward = 0
        return closeness_reward


    def reset(self):
        """
        Resets the environment by reseeding and calling self.generate.

        Returns
        -------
        obs : np.array
            The initial observation of the environment.
        """

        # Saving information about episode
        if (self.t_step > 0):
            self.last_episode = {
                'path': self.path(np.linspace(0, self.path.length, 1000)) if self.path is not None else None,
                'path_taken': self.vessel.path_taken,
                'obstacles': []
            }
            for obst in self.obstacles:
                if isinstance(obst, CircularObstacle):
                    self.last_episode['obstacles'].append(('circle', (obst.position, obst.radius)))
                elif isinstance(obst, PolygonObstacle):
                    self.last_episode['obstacles'].append(('polygon', (obst.points)))
            self.history.append({
                'collisions': self.collisions,
                'cross_track_error': self.cross_track_errors.mean(),
                'surge': self.vessel.prev_inputs[:, 0].mean(),
                'steer': 180/np.pi*self.vessel.prev_inputs[:, 1].mean(),
                'progress': (1.0 if self.reached_goal else self.max_path_prog/self.path.length) if self.path is not None else 0,
                'reached_goal': int(self.reached_goal),
                'reward': self.cumulative_reward,
                'timesteps': self.t_step,
                'timesteps_baseline': self.path.length/(self.config["cruise_speed"]*self.config["t_step_size"]) if self.path is not None else 0
            })

        # Resampling stochastic config parameters
        for param in self.config["stochastic_params"]:
            try:
                self.config[param] = self.input_config[param]()
            except TypeError as e:
                raise e
                pass

        # Resetting all internal variables
        self.vessel = None
        self.path = None
        self.cumulative_reward = 0
        self.max_path_prog = 0
        self.target_arclength = 0
        self.last_checkpoint_time = 0
        self.path_prog_hist = None
        self.past_obs = None
        self.past_actions = None
        self.past_rewards = np.array([0])
        self.past_path_rewards = np.array([0])
        self.past_closeness_rewards = np.array([0])
        self.past_errors = {
            'speed': np.array([]),
            'cross_track': np.array([]),
            'along_track': np.array([]),
            'heading': np.array([]),
            'la_heading': np.array([]),
            'd_cross_track': np.array([]),
            'torque_change': np.array([])
        }
        self.vessel_obstacles = []
        self.obstacles = []
        self.nearby_obstacles = []
        self.t_step = 0
        self.look_ahead_point = None
        self.look_ahead_arclength = None
        self.reached_goal = False
        living_penalty_alpha = 0.1
        self.living_penalty = self.config["reward_lambda"]*(2*living_penalty_alpha+1) + self.config["reward_eta"]*living_penalty_alpha
        self.heading_error_la = 0
        self.heading_error = 0
        self.goal_heading_error = 0
        self.target_heading = 0
        self.la_distance = 0
        self.along_track_error = 0
        self.cross_track_error = 0
        self.collisions = 0
        self.cross_track_errors = np.array([])
        self.sensor_updates = 0
        self.critical_angle = 0
        self.sensor_obst_intercepts_hist = []
        self.sensor_obst_intercepts_transformed_hist = []
        self.sensor_obst_intercepts = [None for isensor in range(self.n_sensors)]
        self.sector_active = [0 for isector in range(self.n_sectors)]
        self.sector_empty = [0 for isector in range(self.n_sectors)]
        self.sector_clear = [0 for isector in range(self.n_sectors)]
        self.sector_closeness = [0.0 for isector in range(self.n_sectors)]
        self.sector_last_heartbeat = [0 for isector in range(self.n_sectors)]
        self.sensor_obst_closenesses = np.zeros((self.n_sensors, ))
        self.sensor_obst_reldx = np.zeros((self.n_sensors, ))
        self.sensor_obst_reldy = np.zeros((self.n_sensors, ))
        self.sensor_obst_distances = np.ones((self.n_sensors, ))*self.config["sensor_range"]
        self.sensor_path_arclengths = np.zeros((self.n_sensors, ))
        if self.np_random is None:
            self.seed()

        # Generating a new environment
        self.generate()

        # Initializing 3d viewer
        if self.render_mode == '3d':
            render3d.init_boat(self)
            self.viewer3d.create_path(self.path)

        # Getting initial observation vector
        obs, _ = self.observe()
        assert not np.isnan(obs).any(), 'Observation vector "{}" contains nan values.'.format(str(obs))

        self.past_obs = np.array([obs])
        self.episode += 1

        return obs

    def update(self):
        for obstacle in self.obstacles:
            if isinstance(obstacle, VesselObstacle):
                obstacle.update(self.config["t_step_size"])

    def close(self):
        if self.viewer2d is not None:
            self.viewer2d.close()
        if self.viewer3d is not None:
            self.viewer3d.close()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def render(self, mode='human'):
        if self.render_mode == '2d' or self.render_mode == 'both':
            image_arr = render2d.render_env(self, mode)
        if self.render_mode == '3d' or self.render_mode == 'both':
            image_arr = render3d.render_env(self, mode, self.config["t_step_size"])
        return image_arr
    
    def _generate_obstacle(self, displacement_dist_std=150, obst_radius_distr=np.random.poisson, obst_radius_mean=30):
        min_distance = 0
        while min_distance <= 0:
            obst_displacement_dist = np.random.normal(0, displacement_dist_std)
            obst_arclength = (0.1 + 0.8*self.np_random.rand())*self.path.length
            obst_position = self.path(obst_arclength)
            obst_displacement_angle = geom.princip(self.path.get_direction(obst_arclength) - np.pi/2)
            obst_position += obst_displacement_dist*np.array([
                np.cos(obst_displacement_angle), 
                np.sin(obst_displacement_angle)
            ])
            obst_radius = max(1, obst_radius_distr(obst_radius_mean))

            vessel_distance_vec = geom.Rzyx(0, 0, -self.vessel.heading).dot(
                np.hstack([obst_position - self.vessel.position, 0])
            )
            vessel_distance = linalg.norm(vessel_distance_vec) - self.vessel.width - obst_radius
            goal_distance = linalg.norm(obst_position - self.path(self.path.length)) - obst_radius
            min_distance = min(vessel_distance, goal_distance)

        return (obst_position, obst_radius)

    def generate(self):
        """
        Sets up a default environemnt with path and static obstacles.
        Typically overridden by specific scenarios.
        """
        nwaypoints = int(np.floor(4*self.np_random.rand() + 2))
        self.path = RandomCurveThroughOrigin(self.np_random, nwaypoints, length=800)

        init_pos = self.path(0)
        init_angle = self.path.get_direction(0)

        init_pos[0] += 50*(self.np_random.rand()-0.5)
        init_pos[1] += 50*(self.np_random.rand()-0.5)
        init_angle = geom.princip(init_angle + 2*np.pi*(self.np_random.rand()-0.5))
        self.vessel = Vessel(self.config["t_step_size"], np.hstack([init_pos, init_angle]), width=4)
        prog = 0
        self.path_prog_hist = np.array([prog])
        self.max_path_prog = prog

        for _ in range(20):
            obstacle = CircularObstacle(*self._generate_obstacle())
            self.obstacles.append(obstacle)

    def observe(self):
        """
        Generates the observation of the environment.
        
        Returns
        -------
        obs : np.array
        """

        # Initializing the observation vector
        obs = np.zeros((self.n_observations,))

        # Setting path-related observations based on the variables that were
        # updated in the self.step method
        obs[0] = np.clip(self.vessel.velocity[0]/self.vessel.max_speed, -1, 1)
        obs[1] = np.clip(self.vessel.velocity[1]/0.26, -1, 1)
        obs[2] = np.clip(self.vessel.yawrate/0.55, -1, 1)
        obs[3] = np.clip(self.heading_error_la/np.pi, -1, 1)
        obs[4] = np.clip(self.heading_error/np.pi, -1, 1)
        obs[5] = np.clip(self.cross_track_error*20/self.path.length, -1, 1)
        obs[6] = np.clip(self.goal_heading_error/np.pi, -1, 1)
        obs[7] = np.clip(self.vessel.heading/np.pi, -1, 1)
        obs[8] = np.clip(self.vessel.crab_angle/np.pi, -1, 1)
        obs[9] = np.log10(self.config["reward_lambda"])
        obs[10] = self.config["reward_eta"]

        # Initializing obstacle detection-related values to the previous ones
        # in case they are not intended to be recalculated (for performance reasons)
        if (self.past_obs is not None):
            obs[self.n_states:] = self.past_obs[-1, self.n_states:]

        # Loading nearby obstacles so that the rest can be ignored
        # when calculating the sensor interception points (for performance reasons)
        if (self.t_step % self.config["sensor_interval_load_obstacles"] == 0):
            if self.verbose:
                print('Loading nearby obstacles...')
            vessel_center = shapely.geometry.Point(
                self.vessel.position[0], 
                self.vessel.position[1],
            )
            self.nearby_obstacles = []
            for obst in self.obstacles:
                obst_dist = float(vessel_center.distance(obst.boundary)) - self.vessel.width
                if (obst_dist < self.config["sensor_range"]):
                    self.nearby_obstacles.append((obst_dist, obst))    
                self.nearby_obstacles = sorted(self.nearby_obstacles, key=lambda x: x[0])
            if self.verbose:
                print('Loaded nearby obstacles ({} / {})...'.format(len(self.nearby_obstacles), len(self.obstacles)))

        collision = False

        # Updating sensor readings
        if (self.t_step % self.config["sensor_interval_obstacles"] == 0):
            self.sensor_updates += 1
            self.sensor_obst_intercepts = [None for isensor in range(self.n_sensors)]

            vessel_center = shapely.geometry.Point(
                self.vessel.position[0], 
                self.vessel.position[1],
            )
            
            # Testing if vessel has collided so that all observations can be set accordingly
            for _, obst in self.nearby_obstacles:
                obst_dist = float(vessel_center.distance(obst.boundary)) - self.vessel.width
                if obst_dist <= 0:
                    collision = True
            if collision:
                for isector in range(self.n_sectors):
                    obs[self.n_states + isector] = 1

            else:
                sector_lines = [None for isensor in range(self.n_sensors)]
                sector_processed = [False for isector in range(self.n_sectors)]
                sector_measurements = [np.zeros((self.n_sensors_per_sector[isector],)) for isector in range(self.n_sectors)]
                self.sector_active = [0 for isector in range(self.n_sectors)]

                for obst_dist, obst in self.nearby_obstacles:
                    if self.config["observe_obstacle_fun"](self.sensor_updates,  obst.last_obs_distance):
                        obst.last_obs_distance = self.config["sensor_range"]
                        obst.last_obs_linestring = []

                # Iterating over all sensors
                for isensor in range(self.n_sensors):
                    isector = self.config["sector_partition_fun"](self, isensor)
                    isensor_internal = isensor - self.sector_start_indeces[isector]
                    if self.config["sensor_rotation"] and (self.sensor_updates + 1) % (int(self.n_sectors/2) + 1) != abs(int(self.n_sectors/2)-isector):
                        continue
                    self.sector_active[isector] = 1
                    if not sector_processed[isector]:
                        self.sector_empty[isector] = 1

                    sensor_angle = self.sensor_angles[isensor]
                    global_sensor_angle = geom.princip(sensor_angle+self.vessel.heading)
                    self.sensor_obst_intercepts[isensor] = (
                        self.vessel.position[0] + np.cos(global_sensor_angle)*self.config["sensor_range"],
                        self.vessel.position[1] + np.sin(global_sensor_angle)*self.config["sensor_range"],
                    )
                    sector_lines[isensor] = shapely.geometry.LineString([(
                            self.vessel.position[0], 
                            self.vessel.position[1],
                        ),(
                            self.vessel.position[0] + np.cos(global_sensor_angle)*self.config["sensor_range"],
                            self.vessel.position[1] + np.sin(global_sensor_angle)*self.config["sensor_range"],
                        )
                    ])
                    self.sensor_obst_closenesses[isensor] = 0                
                    self.sensor_obst_reldx[isensor] = 0
                    self.sensor_obst_reldy[isensor] = 0
                    self.sensor_obst_distances[isensor] = self.config["sensor_range"]

                    # Iterating over all nearby obstacles
                    for obst_dist, obst in self.nearby_obstacles:
                        if not obst.valid:
                            continue
                        should_observe = self.config["observe_obstacle_fun"](self.sensor_updates,  obst.last_obs_distance)
                        try:
                            if should_observe:
                                # Calculating real intersection point between sensor ray and obstacle
                                obst_intersect = obst.boundary.intersection(sector_lines[isensor])
                            else:
                                # Calculating virtual intersection point between sensor ray and obstacle (for performance reasons)
                                obst_intersect = obst.last_obs_linestring.intersection(sector_lines[isensor])
                        except shapely.errors.TopologicalError as e:
                            # Obstacle geometry is invalid - ignoring it for the future
                            obst.valid = False
                            print('Encountered TopologicalError with obstacle - ignoring error')
                            continue
                        except AttributeError as e:
                            continue

                        if not obst_intersect.is_empty:
                            self.sector_empty[isector] = 0
                            try:
                                # Retrieving a list of intersection points
                                obst_intersections = [obst_intersect] if type(obst_intersect) in (shapely.geometry.Point, shapely.geometry.LineString) else list(obst_intersect.geoms)
                            except AttributeError as e:
                                continue

                            # Iterating over intersection points
                            for obst_intersection in obst_intersections:
                                # Converting the intersection object to a point if it is a string
                                # (happens if the sensor ray is parallell to the obstacle boundary)
                                if (type(obst_intersection) == shapely.geometry.LineString):
                                    obst_intersection = shapely.geometry.Point(obst_intersection.coords[0])

                                if should_observe:
                                    obst.last_obs_linestring.append(obst_intersection)

                                # Calculating distance from the vessel to the obstacle
                                distance = max(0, float(vessel_center.distance(obst_intersection)) - self.vessel.width)
                                if distance < obst.last_obs_distance and should_observe:
                                    obst.last_obst_distance = distance

                                # Calculating closeless (scaled inverse distance)
                                closeness = 1 - np.clip(distance/self.config["sensor_range"], 0, 1)

                                # Updating sensor reading if the obstacle reading is closer than the existing one
                                if (closeness > self.sensor_obst_closenesses[isensor]):
                                    self.sensor_obst_closenesses[isensor] = closeness
                                    self.sensor_obst_distances[isensor] = distance
                                    self.sensor_obst_intercepts[isensor] = (obst_intersection.x, obst_intersection.y)

                                    # Updating decomposed obstacle velocity 
                                    if not obst.static:
                                        obst_speed_homogenous = geom.to_homogeneous([obst.dx, obst.dy])
                                        obst_speed_rel_homogenous = geom.Rz(-global_sensor_angle - np.pi/2).dot(obst_speed_homogenous)
                                        obst_speed_rel = geom.to_cartesian(obst_speed_rel_homogenous)
                                        self.sensor_obst_reldx[isensor] = obst_speed_rel[0]
                                        self.sensor_obst_reldy[isensor] = obst_speed_rel[1]

                                    else:
                                        self.sensor_obst_reldx[isensor] = 0
                                        self.sensor_obst_reldy[isensor] = 0

                    
                    # Saving the measurement to the data structure containing measurements for each sensor sector
                    sector_measurements[isector][isensor_internal] = self.sensor_obst_distances[isensor]

                # Updating virtual obstacles
                for obst_dist, obst in self.nearby_obstacles:
                    if self.config["observe_obstacle_fun"](self.sensor_updates, obst.last_obs_distance):
                        if len(obst.last_obs_linestring) >= 2:
                            obst.last_obs_linestring = shapely.geometry.LineString(obst.last_obs_linestring)

                self.sensor_obst_intercepts_hist.append(self.sensor_obst_intercepts.copy())

                # Iterating over all sectors to finalize observation vector
                for isector in range(self.n_sectors):

                    # Testing conditions to ignore sector (and keep old value)
                    if self.config["sensor_rotation"] and (self.sensor_updates + 1) % (int(self.n_sectors/2) + 1) != abs(int(self.n_sectors/2)-isector):
                        continue

                    measurements = sector_measurements[isector]

                    # Calculating maximum feasible distance according to Feasibility Pooling algorithm 
                    feasible_distance, critical_sensor_index = geom.feasibility_pooling(
                        x=measurements, 
                        W=self.vessel.width, 
                        theta=self.sensor_angle, 
                        N_sensors=self.n_sensors_per_sector[isector]
                    )

                    # Calculating feasible closeness
                    if self.config["sensor_log_transform"]:
                        feasible_closeness = 1 - np.clip(np.log(1 + feasible_distance)/np.log(1 + self.config["sensor_range"]), 0, 1)
                    else:
                        feasible_closeness = 1 - np.clip(feasible_distance/self.config["sensor_range"], 0, 1)

                    # Setting observation vector value
                    obs[self.n_states + isector] = feasible_closeness
                    self.sector_closeness[isector] = feasible_closeness

                    if self.detect_movement:
                        if critical_sensor_index is None:
                            critical_sensor_index = 0
                        obs[self.n_states + self.n_sectors + isector] = self.sensor_obst_reldx[self.sector_start_indeces[isector] + critical_sensor_index]
                        obs[self.n_states + self.n_sectors*2 + isector] = self.sensor_obst_reldy[self.sector_start_indeces[isector] + critical_sensor_index]
                    
                    self.sector_last_heartbeat[isector] = self.t_step

        return (obs, collision)
    
    def save(self, filepath):
        self.vessel._state = np.array(self.vessel._state)
        self.path.waypoints = np.array(self.path.waypoints)
        environment_data = {
            'vessel_state': self.vessel._state.tolist(),
            'path': self.path.waypoints.tolist(),
            'obstacles': []
        }
        for obst in self.obstacles:
            if obst.static:
                obstacle_data = {
                    'position': obst.position.tolist(),
                    'radius': obst.radius
                }
                environment_data['obstacles'].append(obstacle_data)
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(environment_data, f)
        except OSError as e:
            print("Couldn't save enviromentment_data ({})".format(str(e)))

    def load(self, filepath):
        with open(filepath, 'rb') as f:
            environment_data = pickle.load(f)
        self.vessel.reset(environment_data['vessel_state'][:3])
        self.path = Path(np.array(environment_data['path']))
        self.obstacles = []
        for obst_data in environment_data['obstacles']:
            obst = CircularObstacle(np.array(obst_data['position']), obst_data['radius'])
            self.obstacles.append(obst)
