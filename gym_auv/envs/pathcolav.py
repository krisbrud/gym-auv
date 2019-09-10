"""
This module implements the AUV gym environment through the AUVenv class.
"""

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

    def step_reward(self):
        """
        Calculates the step_reward and decides whether the episode
        should be ended.

        Parameters
        ----------
        obs : np.array
            The observation of the environment.
        delta_path_prog : double
            How much the vessel has progressed along the path
            the last timestep.
        Returns
        -------
        done : bool
            If True the episode is ended.
        step_reward : double
            The reward for performing action at his timestep.
        """
        done = False
        step_reward = 0
        info = {"collision": False}
        progress = self.path_prog[-1] - self.path_prog[-2]
        max_prog = self.config["cruise_speed"]*self.config["t_step_size"]

        ds = np.clip(progress/max_prog, -1, 1)
        step_reward += ds*self.config["reward_ds"]
        if (ds < 0):
            step_reward += ds*self.config["penalty_negative_ds"]
        
        step_reward += (np.exp(-self.cross_track_error**2/200)*self.config["reward_cross_track_error"])
        #step_reward += (self.d_cross_track_error*self.config["reward_d_cross_track_error"])
        step_reward += (max(self.speed_error, 0)*self.config["reward_speed_error"])
        step_reward += (abs(self.la_heading_error)*self.config["reward_la_heading_error"])
        step_reward += (abs(self.heading_error)*self.config["reward_heading_error"])
        step_reward += abs(self.vessel.rudder_change)*self.config["reward_rudderchange"]
        step_reward += self.config["living_penalty"]

        dist_to_endpoint = linalg.norm(self.vessel.position - self.path.get_endpoint())

        for obst_dist, _, obst in self.nearby_obstacles:
            step_reward += max(self.config["reward_collision"], np.exp(self.config["obst_reward_range"] - obst_dist)*self.config["reward_closeness"])
            if obst_dist <= 0:
                if (not obst.collided):
                    obst.collided = True
                    self.collisions += 1
                info["collision"] = True
                if self.config["end_on_collision"]:
                    done = True
                    break

        step_reward = max(self.config["min_reward"] - self.cumulative_reward, step_reward)
        if (self.cumulative_reward <= self.config["min_reward"]):
            done = True
        if (abs(self.path_prog[-1] - self.path.length) < 2 or dist_to_endpoint < 5):
            done = True
            self.reached_goal = True

        return done, step_reward, info

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
        self.vessel = AUV2D(self.config["t_step_size"], np.hstack([init_pos, init_angle]), width=4)
        prog = 0
        self.path_prog = np.array([prog])
        self.max_path_prog = prog

        for _ in range(self.config["nobstacles"]):
            obstacle = self._create_obstacle()
            self.obstacles.append(obstacle)

    def _get_sector_lidar_distance(self, measurements):
        sort_idx = np.argsort(measurements, axis=None)
        for idx in sort_idx:
            surviving = measurements > measurements[idx]
            d = measurements[idx]*self.sensor_angle
            opening_width = 0
            found_opening = False
            for lidar_surviving in surviving:
                if (lidar_surviving):
                    opening_width += d
                    if (opening_width > self.vessel.width):
                        found_opening = True
                        break
                else:
                    opening_width += 0.5*d
                    if (opening_width > self.vessel.width):
                        found_opening = True
                        break
                    opening_width = 0

            if (not found_opening):
                return measurements[idx]
    
        raise AssertionError('Should not get here.')

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
        obs[5] = np.clip(self.cross_track_error/200, -1, 1)
        obs[6] = np.clip((self.la_distance - self.config["min_la_dist"])/(self.config["lidar_range"]-self.config["min_la_dist"]), -1, 1)

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
                if obst_dist <= 0:
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
                self.sensor_path_arclengths = np.zeros((self.nsensors, ))
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

                    sector_measurements = [[] for isector in range(self.nsectors)]
                    self.sensor_obst_measurements = np.zeros((self.nsensors, ))
                    self.sector_active = [0 for isector in range(self.nsectors)]
                    for isensor in self.sensor_order:
                        isector = isensor // self.config["n_sensors_per_sector"]
                        if (self.config["lidar_rotation"] and (self.sensor_updates + 1) % self.nsectors != isector):
                            continue
                        self.sector_active[isector] = 1
                        sensor_angle = self.sensor_angles[isensor]
                        global_sensor_angle = geom.princip(sensor_angle+self.vessel.heading)
                        sector_line = shapely.geometry.LineString([(
                                self.vessel.position[0], 
                                self.vessel.position[1],
                            ),(
                                self.vessel.position[0] + np.cos(global_sensor_angle)*self.config["lidar_range"],
                                self.vessel.position[1] + np.sin(global_sensor_angle)*self.config["lidar_range"],
                            )
                        ])

                        closest_obstacle_distance = self.config["lidar_range"]
                        for obst_dist, obst_ang, obst in self.nearby_obstacles:
                            d_ang = geom.princip(obst_ang - sensor_angle)
                            if (d_ang > np.pi/2):
                                continue
                            obst_intersect = obst.circle.intersection(sector_line)
                            
                            if (obst_intersect):
                                obst.observed = True
                                obst_intersections = [obst_intersect] if type(obst_intersect) == shapely.geometry.Point else list(obst_intersect.geoms)
                                for obst_intersection in obst_intersections:
                                    distance = float(vessel_center.distance(obst_intersection)) - self.vessel.width
                                    closeness = 1 - np.clip(distance/self.config["lidar_range"], 0, 1)
                                    if (closeness > self.sensor_obst_measurements[isensor]):
                                        self.sensor_obst_measurements[isensor] = closeness
                                        self.sensor_obst_intercepts[isensor] = (obst_intersection.x, obst_intersection.y)
                                        closest_obstacle_distance = distance
                        sector_measurements[isector].append(closest_obstacle_distance)

                        if (self.t_step % self.config['sensor_interval_path'] == 0):
                            if (not stop_path_search):
                                path_intersect = sector_line.intersection(self.path.line)
                                path_intersections = [path_intersect] if type(path_intersect) == shapely.geometry.Point else list(path_intersect.geoms)
                                for path_intersection in path_intersections:
                                    distance = float(vessel_center.distance(path_intersection))
                                    if (distance < closest_obstacle_distance and distance >= self.config["min_la_dist"]):
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

                    self.sensor_obst_measurements = gaussian_filter1d(self.sensor_obst_measurements, sigma=self.config['sensor_convolution_sigma'])

                    for isector in range(self.nsectors):
                        if (self.config["lidar_rotation"] and (self.sensor_updates + 1) % self.nsectors != isector):
                            continue
                        measurements = np.array(sector_measurements[isector])
                        distance = self._get_sector_lidar_distance(measurements)
                        if (self.config["lidar_range_log_transform"]):
                            closeness = 1 - np.clip(np.log(1 + distance)/np.log(1+self.config["lidar_range"]), 0, 1)
                        else:
                            closeness = 1 - np.clip(distance/self.config["lidar_range"], 0, 1)
                        obs[self.lidar_obs_index + isector] = closeness

        return obs