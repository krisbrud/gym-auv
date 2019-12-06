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
import shapely.geometry
from gym_auv.objects.auv import AUV2D
from gym_auv.objects.path import RandomCurveThroughOrigin, ParamCurve
from gym_auv.objects.obstacles import StaticObstacle

from gym.utils import seeding, EzPickle
from gym_auv.rendering import render_env, init_env_viewer, FPS

def feasibility_pooling(x, W, theta, N_sensors):
    sort_idx = np.argsort(x, axis=None)
    for idx in sort_idx:
        surviving = x > x[idx] + W
        d = x[idx]*theta
        opening_width = 0
        opening_span = 0
        opening_start = -theta*(N_sensors-1)/2
        found_opening = False
        for isensor, lidar_surviving in enumerate(surviving):
            if (lidar_surviving):
                opening_width += d
                opening_span += theta
                if (opening_width > W):
                    opening_center = opening_start + opening_span/2
                    if (abs(opening_center) < theta*(N_sensors-1)/4):
                        found_opening = True
            else:
                opening_width += 0.5*d
                opening_span += 0.5*theta
                if (opening_width > W):
                    opening_center = opening_start + opening_span/2
                    if (abs(opening_center) < theta*(N_sensors-1)/4):
                        found_opening = True
                opening_width = 0
                opening_span = 0
                opening_start = -theta*(N_sensors-1)/2 + isensor*theta

        if (not found_opening): 
            return x[idx]

    return np.max(x)

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

    def __init__(self, env_config, test_mode=False):
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
        
        self.env_config = env_config
        self.config = dict(env_config)
        self.test_mode = test_mode
        self.nstates = 8
        self.nsectors = self.config["n_sectors"]
        self.nsensors = self.config["n_sensors_per_sector"]*self.config["n_sectors"]
        self.nrings = self.config["n_rings"]
        sum_radius = 30 if self.nrings > 1 else self.config["lidar_range"]
        self.ring_depths = [sum_radius]
        self.rings = [sum_radius/2]
        self.ring_sectors = [self.nsectors for i in range(self.nrings)]
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
            if (self.config["sector_clear_column"]):
                self.n_observations += self.nsectors

        self.sensor_obst_intercepts = [None for isensor in range(self.nsensors)]
        self.sector_active = [0 for isector in range(self.nsectors)]
        self.sector_empty = [0 for isector in range(self.nsectors)]
        self.sector_clear = [0 for isector in range(self.nsectors)]
        self.sector_closeness = [0.0 for isector in range(self.nsectors)]
        self.sector_last_heartbeat = [0 for isector in range(self.nsectors)]
        self.sensor_obst_closenesses = np.zeros((self.nsensors, ))
        self.sensor_obst_distances = np.ones((self.nsensors, ))*self.config["lidar_range"]
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

        self.max_clearness = np.sum(np.clip(np.cos(self.sector_angles[0]), 0, 1))

        self.np_random = None
        self.critical_angle = None
        self.security_margin = None

        self.cumulative_reward = 0
        self.past_rewards = None
        self.past_path_rewards = None
        self.past_closeness_rewards = None
        self.max_path_prog = None
        self.target_arclength = None
        self.path_prog = None
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

        init_env_viewer(self)

        self.reset()

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

        if (self.path is not None):
            la_heading = self.path.get_direction(self.target_arclength)
            self.heading_error_la = float(geom.princip(la_heading - self.vessel.heading))
            path_position = self.path(self.target_arclength) - self.vessel.position
            self.target_heading = np.arctan2(path_position[1], path_position[0])
            self.heading_error = float(geom.princip(self.target_heading - self.vessel.course))
            goal_position = self.path(self.path.length) - self.vessel.position
            self.goal_heading = np.arctan2(goal_position[1], goal_position[0])
            self.goal_heading_error = float(geom.princip(self.goal_heading - self.vessel.course))
            self.la_distance = linalg.norm(path_position)
            path_direction = self.path.get_direction(self.max_path_prog)
            track_errors = geom.Rzyx(0, 0, -path_direction).dot(
                np.hstack([self.path(self.max_path_prog) - self.vessel.position, 0])
            )
            self.course_path_angle = geom.princip(self.path.get_direction(self.max_path_prog) - self.vessel.course)
            self.along_track_error = track_errors[0]
            self.d_cross_track_error = abs(track_errors[1]) - abs(self.cross_track_error)
            self.cross_track_error = track_errors[1]
            self.cross_track_errors = np.append(self.cross_track_errors, abs(self.cross_track_error))
            self.heading_progress = np.cos(self.course_path_angle)*self.vessel.speed
            self.speed_error = self.vessel.speed - self.config["cruise_speed"]

            self.past_errors['speed'] = np.append(self.past_errors['speed'], self.speed_error)
            self.past_errors['cross_track'] = np.append(self.past_errors['cross_track'], self.cross_track_error)
            self.past_errors['along_track'] = np.append(self.past_errors['along_track'], self.along_track_error)
            self.past_errors['d_cross_track'] = np.append(self.past_errors['d_cross_track'], self.d_cross_track_error)
            self.past_errors['la_heading'] = np.append(self.past_errors['la_heading'], self.heading_error_la)
            self.past_errors['heading'] = np.append(self.past_errors['heading'], self.heading_error)
            self.past_errors['rudder_change'] = np.append(self.past_errors['rudder_change'], self.vessel.smoothed_rudder_change)

            closest_point_distance, _, closest_arclength = self.path.get_closest_point_distance(self.vessel.position)# x0=self.max_path_prog)
            closest_point_heading_error = geom.princip(self.path.get_direction(closest_arclength) - self.vessel.course)
            
            if (
                closest_point_distance < self.config["max_closest_point_distance"] or
                abs(closest_point_heading_error < self.config["max_closest_point_heading_error"])):
                dprog = closest_arclength - self.max_path_prog
            else:
                dprog = self.heading_progress*self.config["t_step_size"] - self.along_track_error*0.05
            
            prog = closest_arclength# min(max(0, self.max_path_prog + dprog), self.path.length)

            if prog > self.max_path_prog:
                self.max_path_prog = prog

            self.path_prog = np.append(self.path_prog, prog)

            if (self.look_ahead_arclength is None):
                target_arclength_candidate = self.max_path_prog + self.config["min_la_dist"]
            else:
                target_arclength_candidate = max(self.look_ahead_arclength, self.max_path_prog + self.config["min_la_dist"])
            
            if (target_arclength_candidate > self.target_arclength):
                dtarget_arglength = (target_arclength_candidate - self.target_arclength)*1
                self.target_arclength += dtarget_arglength
                self.target_arclength = min(self.target_arclength, self.path.length)

        obs = self.observe()
        assert not np.isnan(obs).any(), 'Observation vector "{}" contains nan values.'.format(str(obs))
        self.past_obs = np.vstack([self.past_obs, obs])

        if (self.past_actions is None):
            self.past_actions = np.array([action])
        else:
            self.past_actions = np.vstack([self.past_actions, action])
        self.apply_action(action, obs)

        done, step_reward, info = self.step_reward(action, obs)
        self.past_rewards = np.append(self.past_rewards, step_reward)
        self.past_path_rewards = np.append(self.past_path_rewards, info['path_reward'])
        self.past_closeness_rewards = np.append(self.past_closeness_rewards, info['closeness_reward'])
        self.cumulative_reward += step_reward

        if (self.path is not None):
            info['progress'] = prog/self.path.length
            info['collisions'] = self.collisions
            info['cross_track_error'] = self.cross_track_error
            info['heading_error'] = self.heading_error
            if (closest_point_distance > self.config["max_distance"]):
                done = True

        self.t_step += 1
        self.total_t_steps += 1
        if (self.t_step > self.config["max_timestemps"]):
            done = True
        if (self.cumulative_reward <= self.config["min_reward"]):
            done = True
        dist_to_endpoint = linalg.norm(self.vessel.position - self.path.get_endpoint())
        if (abs(self.path_prog[-1] - self.path.length) < self.config["min_goal_progress"] or dist_to_endpoint < self.config["min_goal_closeness"]):
            done = True
            self.reached_goal = True

        # global_sector_angle = geom.princip(self.sector_angles[0]+self.vessel.heading)
        # sector_error = geom.princip(global_sector_angle - self.target_heading)
        # sector_target_heading_component = np.clip(np.cos(sector_error), 0, 1)
        # total_clearness = np.dot(self.sector_clear, sector_target_heading_component)/self.max_clearness
        # log_lambda = np.log10(self.config["reward_lambda"])
        # candidate_log_lambda = np.log10(np.clip(total_clearness**2, 10**(-15), 1))
        # alpha = 0.99 if candidate_log_lambda < log_lambda else 0.999
        # new_log_lambda = alpha*log_lambda + (1-alpha)*candidate_log_lambda
        # self.config["reward_lambda"] = np.power(10, new_log_lambda)

        #print(335, total_clearness, candidate_log_lambda)

        if (self.config["callable_update_interval"] is not None and self.t_step % self.config["callable_update_interval"] == 0):
            for k in self.env_config:
                if (callable(self.env_config[k])):
                    self.config[k] = self.env_config[k]()

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
                'path': self.path(np.linspace(0, self.path.s_max, 1000)) if self.path is not None else None,
                'path_taken': self.vessel.path_taken,
                'obstacles': [(obst.position, obst.radius) for obst in self.obstacles]
            }
            self.history.append({
                'collisions': self.collisions,
                'cross_track_error': self.cross_track_errors.mean(),
                'surge': self.vessel.prev_inputs[:, 0].mean(),
                'steer': 180/np.pi*self.vessel.prev_inputs[:, 1].mean(),
                'collision_baseline': self.get_collision_baseline() if self.path is not None else 0,
                'progress': (1.0 if self.reached_goal else self.max_path_prog/self.path.length) if self.path is not None else 0,
                'reached_goal': int(self.reached_goal),
                'reward': self.cumulative_reward,
                'timesteps': self.t_step,
                'timesteps_baseline': self.path.length/(self.config['cruise_speed']*self.config['t_step_size']) if self.path is not None else 0
            })

        self.vessel = None
        self.path = None
        self.cumulative_reward = 0
        self.max_path_prog = 0
        self.target_arclength = 0
        self.path_prog = None
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
            'rudder_change': np.array([])
        }
        
        for k in self.env_config:
            if (callable(self.env_config[k])):
                self.config[k] = self.env_config[k]()
        self.obstacles = []
        self.nearby_obstacles = []
        self.t_step = 0
        self.look_ahead_point = None
        self.look_ahead_arclength = None
        self.reached_goal = False
        living_penalty_alpha = 0.1
        self.living_penalty = self.config["reward_lambda"]*(2*living_penalty_alpha+1) + self.config["reward_speed"]*living_penalty_alpha

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
        self.security_margin = 0

        self.sensor_obst_intercepts = [None for isensor in range(self.nsensors)]
        self.sector_active = [0 for isector in range(self.nsectors)]
        self.sector_empty = [0 for isector in range(self.nsectors)]
        self.sector_clear = [0 for isector in range(self.nsectors)]
        self.sector_closeness = [0.0 for isector in range(self.nsectors)]
        self.sector_last_heartbeat = [0 for isector in range(self.nsectors)]
        self.sensor_obst_closenesses = np.zeros((self.nsensors, ))
        self.sensor_obst_distances = np.ones((self.nsensors, ))*self.config["lidar_range"]
        self.sensor_path_arclengths = np.zeros((self.nsensors, ))

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

    def apply_action(self, action):
        raise NotImplementedError()

    def step_reward(self, action):
        raise NotImplementedError()

    def get_path_reward(self):
        cross_track_performance = np.exp(-self.config["reward_gamma_y_e"]*np.abs(self.cross_track_error))
        path_reward = (1 + np.cos(self.heading_error)*self.vessel.speed/self.vessel.max_speed)*(1 + cross_track_performance) - 1
        return path_reward

    def get_closeness_reward(self, collision=False):
        if (collision):
            return -1/self.config["reward_gamma_x"]
        closeness_reward_num = 0
        closeness_reward_den = 0
        if (self.nsensors > 0):
            for isensor in range(self.nsensors):
                sensor_angle = self.sensor_angles[isensor]
                sensor_distance = self.sensor_obst_distances[isensor]
                sensor_weight = 1 / (1 + np.abs(self.config["reward_gamma_theta"]*sensor_angle))
                # if (sensor_distance <= 1 / np.sqrt(self.config["reward_gamma_x"]*self.config["penalty_collision"]) + self.config["security_margin"]):
                #     sensor_raw_penalty = self.config["penalty_collision"]
                # else:
                sensor_raw_penalty = 1/(self.config["reward_gamma_x"]*(max(sensor_distance, 1))**2)
                sensor_reward = sensor_weight*sensor_raw_penalty
                closeness_reward_num += sensor_reward
                closeness_reward_den += sensor_weight
            closeness_reward = -closeness_reward_num/closeness_reward_den
        else:
            closeness_reward = 0
        return closeness_reward

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

    def _create_obstacle(self):
        min_distance = 0
        while min_distance <= 0:
            obst_displacement_dist = np.random.normal(0, 150)
            obst_arclength = (0.1 + 0.8*self.np_random.rand())*self.path.s_max
            obst_position = self.path(obst_arclength)
            obst_displacement_angle = geom.princip(self.path.get_direction(obst_arclength) - np.pi/2)
            obst_position += obst_displacement_dist*np.array([
                np.cos(obst_displacement_angle), 
                np.sin(obst_displacement_angle)
            ])
            obst_radius = np.random.poisson(30)
            obstacle = StaticObstacle(obst_position, obst_radius)

            vessel_distance_vec = geom.Rzyx(0, 0, -self.vessel.heading).dot(
                np.hstack([obstacle.position - self.vessel.position, 0])
            )
            vessel_distance = linalg.norm(vessel_distance_vec) - self.vessel.width - obstacle.radius
            goal_distance = linalg.norm(obstacle.position - self.path(self.path.length)) - obstacle.radius
            min_distance = min(vessel_distance, goal_distance)

        return obstacle

    def generate(self):
        """
        Sets up the environment. Generates the path and
        initialises the AUV.
        """
        nwaypoints = int(np.floor(4*self.np_random.rand() + 2))
        self.path = RandomCurveThroughOrigin(self.np_random, nwaypoints, self.config["goal_dist"])

        init_pos = self.path(0)
        init_angle = self.path.get_direction(0)

        init_pos[0] += 50*(self.np_random.rand()-0.5)
        init_pos[1] += 50*(self.np_random.rand()-0.5)
        init_angle = geom.princip(init_angle + 2*np.pi*(self.np_random.rand()-0.5))
        self.vessel = AUV2D(self.config["t_step_size"], np.hstack([init_pos, init_angle]), width=4, adaptive_step_size=self.config["adaptive_step_size"])
        prog = 0
        self.path_prog = np.array([prog])
        self.max_path_prog = prog

        for _ in range(self.config["nobstacles"]):
            obstacle = self._create_obstacle()
            self.obstacles.append(obstacle)

    def save(self, filepath):
        self.vessel._state = np.array(self.vessel._state)
        self.path.init_waypoints = np.array(self.path.init_waypoints)
        environment_data = {
            'vessel_state': self.vessel._state.tolist(),
            'path': self.path.init_waypoints.tolist(),
            'obstacles': []
        }
        for obst in self.obstacles:
            obstacle_data = {
                'position': obst.position.tolist(),
                'radius': obst.radius
            }
            environment_data['obstacles'].append(obstacle_data)
        with open(filepath, 'wb') as f:
            pickle.dump(environment_data, f)

    def load(self, filepath):
        with open(filepath, 'rb') as f:
            environment_data = pickle.load(f)
        self.vessel.reset(environment_data['vessel_state'][:3])
        self.path = ParamCurve(np.array(environment_data['path']))
        self.obstacles = []
        for obst_data in environment_data['obstacles']:
            obst = StaticObstacle(np.array(obst_data['position']), obst_data['radius'])
            self.obstacles.append(obst)

    def observe(self):
        """
        Generates the observation of the environment.
        Parameters
        ----------
        action : np.array
            [propeller_input, rudder_position].

        Returns
        -------
        obs : np.array
            [
            surge velocity,
            sway velocity,
            yawrate,
            heading error,
            cross track error,
            propeller_input,
            rudder_position,
            ]
            All observations are between -1 and 1.
        """

        obs = np.zeros((self.n_observations,))

        obs[0] = np.clip(self.vessel.velocity[0]/self.vessel.max_speed, -1, 1)
        obs[1] = np.clip(self.vessel.velocity[1]/0.26, -1, 1)
        obs[2] = np.clip(self.vessel.yawrate/0.55, -1, 1)
        obs[3] = np.clip(self.heading_error_la/np.pi, -1, 1)
        obs[4] = np.clip(self.heading_error/np.pi, -1, 1)
        obs[5] = np.clip(self.cross_track_error/50, -1, 1)
        obs[6] = np.log10(self.config["reward_lambda"])
        obs[7] = np.clip(self.goal_heading_error/np.pi, -1, 1) #remove this
        #obs[6] = np.clip((self.la_distance - self.config["min_la_dist"])/(self.config["lidar_range"]-self.config["min_la_dist"]), -1, 1)

        if (self.past_obs is not None):
            obs[self.nstates:] = self.past_obs[-1, self.nstates:]

        if (self.t_step % self.config['sensor_interval_obstacles'] == 0):
            self.sensor_updates += 1
            vessel_center = shapely.geometry.Point(
                self.vessel.position[0], 
                self.vessel.position[1],
            )
            self.nearby_obstacles = []
            collision = False
            for obst in self.obstacles:
                distance_vec = geom.Rzyx(0, 0, -self.vessel.heading).dot(
                    np.hstack([obst.position - self.vessel.position, 0])
                )
                obst_dist = linalg.norm(distance_vec) - self.vessel.width - obst.radius
                if (obst_dist < self.config["lidar_range"]):
                    obst_ang = float(np.arctan2(distance_vec[1], distance_vec[0]))
                    self.nearby_obstacles.append((obst_dist, obst_ang, obst))
                if obst_dist + self.vessel.width <= 0:
                    collision = True
            self.nearby_obstacles = sorted(self.nearby_obstacles, key=lambda x: x[0])
            for obst_dist, obst_ang, obst in self.nearby_obstacles:
                obst.observed = False

            self.sensor_obst_intercepts = [None for isensor in range(self.nsensors)]

            if (self.t_step % self.config['sensor_interval_path'] == 0):
                last_look_ahead_arclength = self.look_ahead_arclength
                last_look_ahead_point = self.look_ahead_point
                self.look_ahead_arclength = None
                self.look_ahead_point = None
                max_look_ahead_arclength = 0

                if (self.sensor_path_index is None):
                    sort_key = lambda i: 0
                else:
                    sort_key = lambda i: self.sensor_path_arclengths[i] - abs(i - self.sensor_path_index)

                self.sensor_order = sorted(range(self.nsensors), key=sort_key, reverse=True)
                self.sensor_path_index = None
                stop_path_search = False

            if (self.config["detection_grid"] and self.t_step > 0):
                full_detection_image = np.array([])
                for iring in range(self.nrings):
                    radius = self.rings[iring]
                    old_detection_image = self.detection_images[iring].copy()

                    if (self.config["rear_detection"]):
                        image_rotation = min(1, abs(self.vessel.heading_change) / (2*np.pi/self.ring_sectors[iring]))
                    else:
                        image_rotation = min(1, abs(self.vessel.heading_change) / (np.pi/self.ring_sectors[iring]))
                    for isector in range(self.ring_sectors[iring]):
                        left_detection_value = old_detection_image[(isector - 1) % self.ring_sectors[iring]]
                        right_detection_value = old_detection_image[(isector + 1) % self.ring_sectors[iring]]
                        
                        if (self.vessel.heading_change > 0):
                            self.detection_images[iring][isector] = image_rotation*left_detection_value + (1-image_rotation)*self.detection_images[iring][isector]
                        if (self.vessel.heading_change > 0):
                            self.detection_images[iring][isector] = image_rotation*right_detection_value + (1-image_rotation)*self.detection_images[iring][isector]

                        sector_angle = self.sector_angles[iring][isector]
                        global_sector_angle = geom.princip(sector_angle+self.vessel.heading)
                        
                        center = np.array([
                            self.vessel.position[0] + np.cos(global_sector_angle)*radius,
                            self.vessel.position[1] + np.sin(global_sector_angle)*radius,
                        ])
                        intensity = 0
                        for obst_dist, obst_ang, obst in self.nearby_obstacles:
                            if (linalg.norm(obst.position - center) < obst.radius + self.ring_depths[iring]/2):
                                intensity = 1
                                break
                        
                        # 1 - linalg.norm(obst.position - center)/(obst.radius + self.ring_depths[iring]/2)
                        self.detection_images[iring][isector] = intensity

                        if (iring == 0):
                            self.feasibility_images[iring][isector] = 1 - self.detection_images[iring][isector]
                        else:
                            self.feasibility_images[iring][isector] = min(self.feasibility_images[iring-1][isector], 1 - self.detection_images[iring][isector])

                        full_detection_image = np.hstack((full_detection_image, self.detection_images[iring][isector]))

                    if (iring > 0):
                        for isector in range(self.ring_sectors[iring]):
                            if (self.detection_images[iring][isector] == 0):
                                neighbor = (isector + 1) % self.ring_sectors[iring]
                                self.feasibility_images[iring][neighbor] = max(self.feasibility_images[iring][isector], self.feasibility_images[iring-1][neighbor])
                        for isector in range(self.ring_sectors[iring]-1, -1, -1):
                            if (self.detection_images[iring][isector] == 0):
                                neighbor = (isector - 1) % self.ring_sectors[iring]
                                self.feasibility_images[iring][neighbor] = max(self.feasibility_images[iring][isector], self.feasibility_images[iring-1][neighbor])
                
                #self.detection_images = gaussian_filter(self.detection_images, 1)
                obs[self.nstates:self.nstates+self.n_detection_grid_sections] = full_detection_image

            if (self.config["lidars"]):            
                if (collision):
                    for isector in range(self.nsectors):
                        obs[self.lidar_obs_index + isector] = 1

                else:
                    sector_lines = [None for isensor in range(self.nsensors)]
                    sector_processed = [False for isector in range(self.nsectors)]
                    sector_measurements = [np.zeros((self.config["n_sensors_per_sector"],)) for isector in range(self.nsectors)]
                    skip_sectors = [self.t_step - self.sector_last_heartbeat[isector] < self.config["closeness_sector_delay"]*(1 - obs[self.lidar_obs_index + isector]) for isector in range(self.nsectors)]
                    self.sector_active = [0 for isector in range(self.nsectors)]
                    for isensor in self.sensor_order:
                        isensor_internal = isensor % self.config["n_sensors_per_sector"]
                        isector = isensor // self.config["n_sensors_per_sector"]
                        if (self.config["lidar_rotation"] and (self.sensor_updates + 1) % (int(self.nsectors/2) + 1) != abs(int(self.nsectors/2)-isector)):
                            continue
                        if (skip_sectors[isector]):
                            continue
                        self.sector_active[isector] = 1
                        if (not sector_processed[isector]):
                            self.sector_empty[isector] = 1
                        sensor_angle = self.sensor_angles[isensor]
                        global_sensor_angle = geom.princip(sensor_angle+self.vessel.heading)
                        self.sensor_obst_intercepts[isensor] = (
                            self.vessel.position[0] + np.cos(global_sensor_angle)*self.config["lidar_range"],
                            self.vessel.position[1] + np.sin(global_sensor_angle)*self.config["lidar_range"],
                        )
                        sector_lines[isensor] = shapely.geometry.LineString([(
                                self.vessel.position[0], 
                                self.vessel.position[1],
                            ),(
                                self.vessel.position[0] + np.cos(global_sensor_angle)*self.config["lidar_range"],
                                self.vessel.position[1] + np.sin(global_sensor_angle)*self.config["lidar_range"],
                            )
                        ])
                        self.sensor_obst_closenesses[isensor] = 0
                        self.sensor_obst_distances[isensor] = self.config["lidar_range"]

                        for obst_dist, obst_ang, obst in self.nearby_obstacles:
                            d_ang = geom.princip(obst_ang - sensor_angle)
                            if (d_ang > np.pi/2):
                                continue
                            obst_intersect = obst.circle.intersection(sector_lines[isensor])
                            
                            if (obst_intersect):
                                self.sector_empty[isector] = 0
                                obst.observed = True
                                try:
                                    obst_intersections = [obst_intersect] if type(obst_intersect) == shapely.geometry.Point else list(obst_intersect.geoms)
                                except AttributeError:
                                    continue
                                for obst_intersection in obst_intersections:
                                    distance = max(0, float(vessel_center.distance(obst_intersection)) - self.vessel.width)
                                    closeness = 1 - np.clip(distance/self.config["lidar_range"], 0, 1)
                                    if (closeness > self.sensor_obst_closenesses[isensor]):
                                        self.sensor_obst_closenesses[isensor] = closeness
                                        self.sensor_obst_distances[isensor] = distance
                                        self.sensor_obst_intercepts[isensor] = (obst_intersection.x, obst_intersection.y)
                        
                        sector_measurements[isector][isensor_internal] = self.sensor_obst_distances[isensor]
                        if (self.config["sensor_noise_std"] > 0):
                            sector_measurements[isector][isensor_internal] += np.random.normal(0, self.config["sensor_noise_std"])
                            sector_measurements[isector][isensor_internal] = np.clip(sector_measurements[isector][isensor_internal], 0, self.config["lidar_range"])

                        if (self.t_step % self.config['sensor_interval_path'] == 0):
                            self.sensor_path_arclengths[isensor] = 0
                            if (not stop_path_search):
                                path_intersect = sector_lines[isensor].intersection(self.path.line)
                                path_intersections = [path_intersect] if type(path_intersect) == shapely.geometry.Point else list(path_intersect.geoms)
                                for path_intersection in path_intersections:
                                    distance = float(vessel_center.distance(path_intersection))
                                    if (distance < self.sensor_obst_distances[isensor] and distance >= self.config["min_la_dist"]):
                                        if (last_look_ahead_arclength is not None and self.look_ahead_arclength is None):
                                            max_look_ahead_arclength = last_look_ahead_arclength
                                            self.look_ahead_point = last_look_ahead_point
                                            self.look_ahead_arclength = last_look_ahead_arclength
                                        if (type(path_intersection) == shapely.geometry.LineString):
                                            continue
                                        intersection_arclength = self.path.get_closest_arclength([path_intersection.x, path_intersection.y])
                                        if (intersection_arclength > max_look_ahead_arclength):
                                            if (last_look_ahead_arclength is None or intersection_arclength > last_look_ahead_arclength):
                                                max_look_ahead_arclength = intersection_arclength
                                                self.look_ahead_point = [path_intersection.x, path_intersection.y]
                                                self.look_ahead_arclength = intersection_arclength
                                                self.sensor_path_index = isensor
                                                if (distance > 0.9*self.config["lidar_range"]):
                                                    stop_path_search = True
                                                    break
                                        if (intersection_arclength > self.sensor_path_arclengths[isensor]):
                                            self.sensor_path_arclengths[isensor] = intersection_arclength

                    if (self.config["sensor_convolution_sigma"]):
                        self.sensor_obst_closenesses = gaussian_filter1d(self.sensor_obst_closenesses, sigma=self.config['sensor_convolution_sigma'])
                    
                    if (self.test_mode and self.config["security"] and self.nsensors > 0):
                        critical_sensor = np.argmax(self.sensor_obst_closenesses)
                        critical_closeness = self.sensor_obst_closenesses[critical_sensor]
                        if (critical_closeness > 0.5):
                            security_margin = min(self.config["security_margin"], 
                                self.config["lidar_range"]*(1 - critical_closeness)) - 1
                            alpha = self.config["security_smoothing_factor"]
                            self.security_margin = alpha*self.security_margin + (1-alpha)*security_margin
                            critical_angle = self.sensor_angles[critical_sensor]+self.vessel.heading
                            self.critical_angle = alpha*self.critical_angle + (1-alpha)*critical_angle
                            transformed_obst_measurements = self.sensor_obst_distances.copy()
                            transformed_lidar_surface = shapely.geometry.LineString([
                                (
                                    intercept[0] - self.security_margin*np.cos(self.critical_angle), 
                                    intercept[1] - self.security_margin*np.sin(self.critical_angle)
                                ) for intercept in self.sensor_obst_intercepts if intercept is not None
                            ])
                            sector_measurements = [np.zeros((self.config["n_sensors_per_sector"],)) for isector in range(self.nsectors)]
                            for isensor in range(self.nsensors):
                                isector = isensor // self.config["n_sensors_per_sector"]
                                isensor_internal = isensor % self.config["n_sensors_per_sector"]
                                if (sector_lines[isensor] is None):
                                    continue
                                surface_intersection = sector_lines[isensor].intersection(transformed_lidar_surface)
                                surface_intersection = [surface_intersection] if type(surface_intersection) == shapely.geometry.Point else list(surface_intersection.geoms)
                                if (surface_intersection):
                                    new_distance = float(vessel_center.distance(surface_intersection[0])) - self.vessel.width
                                    if (new_distance < self.sensor_obst_distances[isensor]):
                                        transformed_obst_measurements[isensor] = new_distance
                                        self.sensor_obst_intercepts[isensor] = (surface_intersection[0].x, surface_intersection[0].y)
                                sector_measurements[isector][isensor_internal] = transformed_obst_measurements[isensor]
                        else:
                            self.security_margin = 0
                            self.critical_angle = 0
                    for isector in range(self.nsectors):
                        if (self.config["lidar_rotation"] and (self.sensor_updates + 1) % (int(self.nsectors/2) + 1) != abs(int(self.nsectors/2)-isector)):
                            continue
                        if (skip_sectors[isector]):
                            continue
                        measurements = sector_measurements[isector]
                        if (self.config["save_measurements"]):
                            if (self.measurement_history is None):
                                self.measurement_history = np.array(measurements)
                            else:
                                self.measurement_history = np.vstack([self.measurement_history, measurements])
                        distance = feasibility_pooling(x=measurements, W=self.vessel.width, theta=self.sensor_angle, N_sensors=self.config["n_sensors_per_sector"])
                        if (distance == self.config["lidar_range"]):
                            closeness = 0
                            self.sector_clear[isector] = 1
                            if (self.config["sector_clear_column"]):
                                obs[self.lidar_obs_index + self.nsectors + isector] = 0
                        else:
                            # if (self.test_mode and self.config["security"]):
                            #     observed_distance = max(0, distance - self.config["obst_reward_range"])
                            #     max_distance = self.config["lidar_range"] - self.config["obst_reward_range"]
                            # else:
                            self.sector_clear[isector] = 0
                            observed_distance = max(0, distance)
                            max_distance = self.config["lidar_range"]
                            assert observed_distance >= 0, "Invalid observed_distance {}".format(observed_distance)
                            if (self.config["lidar_range_log_transform"]):
                                closeness = 1 - np.clip(np.log(1 + observed_distance)/np.log(1 + max_distance), 0, 1)
                            else:
                                closeness = 1 - np.clip(observed_distance/max_distance, 0, 1)
                            if (self.config["sector_clear_column"]):
                                obs[self.lidar_obs_index + self.nsectors + isector] = 1
                        obs[self.lidar_obs_index + isector] = closeness if not np.isnan(closeness) else 1.0
                        self.sector_closeness[isector] = obs[self.lidar_obs_index + isector]
                        self.sector_last_heartbeat[isector] = self.t_step
        
        return obs