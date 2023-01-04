"""
This module implements an AUV that is simulated in the horizontal plane.
"""
from re import I
from typing import Callable, List, Tuple
import numpy as np
import numpy.linalg as linalg
import shapely.geometry, shapely.errors, shapely.strtree, shapely.ops, shapely.prepared
import gym_auv

import gym_auv.utils.constants as const
import gym_auv.utils.geomutils as geom
from gym_auv.objects.obstacles import BaseObstacle, LineObstacle
from gym_auv.objects.path import Path
from gym_auv.objects.vessel.sensor import (
    get_relative_positions_of_lidar_measurements,
    make_occupancy_grid,
    find_rays_to_simulate_for_obstacles,
    simulate_sensor,
)
from gym_auv.objects.vessel.odesolver import meyer_odesolver45
from gym_auv.objects.vessel.otter import Otter3DoF

class Vessel:
    def __init__(
        self, config: gym_auv.Config, init_state: np.ndarray, width: float = 4
    ) -> None:
        """
        Initializes and resets the vessel.

        Parameters
        ----------
        config : dict
            Dictionary containing the configuration parameters for
            the vessel
        init_state : np.ndarray
            The initial attitude of the veHssel [x, y, psi], where
            psi is the initial heading of the AUV.
        width : float
            The distance from the center of the AUV to its edge
            in meters.
        """

        self.config = config

        # Initializing private attributes
        self._width = config.vessel.vessel_width

        # self._n_sectors = self.config.vessel.n_sectors
        # self._n_sensors = self.config.vessel.n_sensors_per_sector * self._n_sectors
        self._n_sensors = config.sensor.n_lidar_rays
        self._d_sensor_angle = (
            2 * np.pi / (self._n_sensors)
        )  # radians TODO: Move to sensor?
        self._sensor_angles = np.array(
            [-np.pi + (i + 1) * self._d_sensor_angle for i in range(self._n_sensors)]
        )
        self._max_speed = 2
        self.dynamics_model = Otter3DoF()

        # self._sensor_internal_indeces = []
        # self._sensor_interval = max(1, int(1 / self.config.simulation.sensor_frequency))

        # Calculating feasible closeness
        if self.config.sensor.apply_log_transform and self.config.sensor.apply_distance_normalization:
            raise ValueError("Log transform and distance normalization cannot be set to True at the same time!")
        if self.config.sensor.apply_log_transform:
            self._get_closeness = lambda x: 1 - np.clip(
                np.log(1 + x) / np.log(1 + self.config.sensor.range), 0, 1
            )
        elif self.config.sensor.apply_distance_normalization:
            self._get_closeness = lambda x: 1 - np.clip(
                x / self.config.sensor.range, 0, 1
            )
        else:
            self._get_closeness = lambda x: x  # Do nothing to the input

        # Initializing vessel to initial position
        self.reset(init_state)

    @property
    def n_sensors(self) -> int:
        """Number of sensors."""
        return self._n_sensors

    @property
    def width(self) -> float:
        """Width of vessel in meters."""
        return self._width

    @property
    def position(self) -> np.ndarray:
        """Returns an array holding the position of the AUV in cartesian
        coordinates."""
        return self._state[0:2]

    @property
    def path_taken(self) -> np.ndarray:
        """Returns an array holding the path of the AUV in cartesian
        coordinates."""
        return self._prev_states[:, 0:2]

    @property
    def heading_taken(self) -> np.ndarray:
        """Returns an array holding the heading of the AUV for all timesteps."""
        return self._prev_states[:, 5]

    @property
    def actions_taken(self) -> np.ndarray:
        """Returns the actions taken (surge, rudder) over the current episode. The actions are probably normalized."""
        return self._prev_inputs

    @property
    def heading(self) -> float:
        """Returns the heading of the AUV with respect to true north."""
        return self._state[5]

    @property
    def velocity(self) -> np.ndarray:
        """Returns the surge and sway velocity of the AUV."""
        return self._state[3:5]

    @property
    def velocity_ned(self) -> np.ndarray:
        """Returns the surge and sway velocity of the AUV in the NED frame."""
        return geom.transform_ned_to_body(self.velocity, np.zeros(2), self.heading)

    @property
    def speed(self) -> float:
        """Returns the speed of the AUV."""
        return linalg.norm(self.velocity)

    @property
    def yaw_rate(self) -> float:
        """Returns the rate of rotation about the z-axis."""
        return self._state[5]

    @property
    def yaw_rate_taken(self) -> np.ndarray:
        """Returns the history of yaw rates"""
        return self._prev_states[:, 5]

    @property
    def max_speed(self) -> float:
        """Returns the maximum speed of the AUV."""
        return self._max_speed

    @property
    def course(self) -> float:
        """Returns the course angle of the AUV with respect to true north."""
        crab_angle = np.arctan2(self.velocity[1], self.velocity[0])
        return self.heading + crab_angle

    @property
    def sensor_angles(self) -> np.ndarray:
        """Array containg the angles each sensor ray relative to the vessel heading."""
        return self._sensor_angles

    # @property
    # def sector_angles(self) -> np.ndarray:
    #     """Array containg the angles of the center line of each sensor sector relative to the vessel heading."""
    #     return self._sector_angles

    @property
    def progress(self) -> float:
        """Returns the progress along the path. Can take values between 0 and 1."""
        return self._progress

    @property
    def prev_timestep_progress(self) -> float:
        """Returns the progress during the previous timestep"""
        return self._prev_progress

    @property
    def max_progress(self) -> float:
        """Returns the maximum progress along the path in the current episode. Can take values between 0 and 1."""
        return self._max_progress

    def reset(self, init_state: np.ndarray) -> None:
        """
        Resets the vessel to the specified initial state.

        Parameters
        ----------
        init_state : np.ndarray
            The initial attitude of the veHssel [x, y, psi], where
            psi is the initial heading of the AUV.
        """
        init_speed = np.zeros(3, dtype=np.float64)
        init_propeller = np.zeros(2, dtype=np.float64)
        init_state = np.array(init_state, dtype=np.float64)
        # init_speed = np.array(init_speed, dtype=np.float64)
        self._state = np.hstack([init_state, init_speed])
        self._prev_states = np.vstack([self._state])
        self._input = [0, 0]
        self._prev_inputs = np.vstack([self._input])
        self._last_sensor_dist_measurements = (
            np.ones((self._n_sensors,)) * self.config.sensor.range
        )
        self._last_sensor_speed_measurements = np.zeros((2, self._n_sensors))
        self._last_navi_state_dict = dict(
            (state, 0) for state in self._get_navigation_observation_keys()
        )
        self._collision = False
        self._progress = 0
        self._max_progress = 0
        self._reached_goal = False

        self._step_counter = 0
        self._perceive_counter = 0
        self._nearby_obstacles = []

    def step(self, action: list) -> None:
        """
        Simulates the vessel one step forward after applying the given action.

        Parameters
        ----------
        action : np.ndarray[thrust_input, torque_input]
        """
        # self._input = np.array(
        #     [self._thrust_surge(action[0]), self._moment_steer(action[1])]
        # )
        self._input = np.array(action)
        w, q = meyer_odesolver45(
            self._state_dot, self._state, self.config.simulation.t_step_size
        )

        self._state = q
        # self._state[5] = geom.princip(self._state[5])

        self._prev_states = np.vstack([self._prev_states, self._state])
        self._prev_inputs = np.vstack([self._prev_inputs, self._input])

        self._step_counter += 1

    def perceive(self, obstacles: List[BaseObstacle]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulates the sensor suite and returns observation arrays of the environment.

        Returns
        -------
        if self.lidar_preprocessor is not None:
            sector_closenesses : np.ndarray
            sector_velocities : np.ndarray

        """
        # PSEUDOCODE:
        # Find nearby obstacles - done
        # Simulate sensors for nearby obstacles
        # Postprocess the observations

        # Initializing variables
        sensor_range = self.config.sensor.range

        # if (
        #     self._step_counter % self.config.simulation.sensor_interval_load_obstacles
        #     == 0
        # ):
        self._load_nearby_obstacles(obstacles)

        # Loading nearby obstacles, i.e. obstacles within the vessel's detection range
        if not self._nearby_obstacles:
            # Set feasible distances to sensor range, closeness and velocities to zero
            collision = False

            n_sensors = self.config.sensor.n_lidar_observations
            sensor_dist_measurements = np.ones((n_sensors,)) * sensor_range
            sensor_speed_measurements = np.zeros((2, n_sensors))
            sensor_blocked_arr = [False] * n_sensors
            output_closenesses = sensor_dist_measurements

        else:
            geom_targets = self._nearby_obstacles
            # Simulating all sensors using _simulate_sensor subroutine
            sensor_angles_ned = self._sensor_angles + self.heading

            (
                sensor_dist_measurements,
                _,  # sensor_speed_measurements,
                sensor_blocked_arr,
            ) = self._simulate_sensors_or_use_previous_measurement(
                sensor_angles_ned, self._p0_point, sensor_range, geom_targets
            )

            self._last_sensor_dist_measurements = sensor_dist_measurements
            # self._last_sensor_speed_measurements = sensor_speed_measurements
            distances = sensor_dist_measurements
            # output_velocities = sensor_speed_measurements

            # Calculating feasible closeness
            output_closenesses = self._get_closeness(distances)

            # Testing if vessel has collided
            collision = np.any(sensor_dist_measurements < self.width)

        self._collision = collision
        self._perceive_counter += 1

        if self.config.sensor.use_velocity_observations:
            raise NotImplementedError

        return output_closenesses, None

    def _load_nearby_obstacles(self, obstacles) -> None:
        self._nearby_obstacles = list(
            filter(
                lambda obst: float(self._p0_point.distance(obst.boundary)) - self._width
                < self.config.sensor.range,
                obstacles,
            )
        )

    @property
    def _p0_point(self) -> shapely.geometry.Point:
        """Position of vessel in North-East coords of NED as shapely point"""
        return shapely.geometry.Point(*self.position)

    def _simulate_sensors_or_use_previous_measurement(
        self,
        sensor_angles_ned: np.ndarray,
        p0_point: shapely.geometry.Point,
        sensor_range: float,
        geom_targets: List[BaseObstacle],
    ) -> Tuple[np.ndarray, np.ndarray, List[bool]]:
        """Simulates the sensors if it is time to do that, or uses the previous if not

        Returns:
            sensor_dist_measurements:   Distances for sensors
            sensor_speed_measurements:  Speed mesurements for sensors
            sensor_blocked_arr:         Whether the sensors are blocked
        """
        # activate_sensor = lambda i: (i % self._sensor_interval) == (
        #     self._perceive_counter % self._sensor_interval
        # )

        sensor_dist_measurements = []
        sensor_speed_measurements = []
        sensor_blocked_arr = []
        obstacles_to_simulate_per_ray = find_rays_to_simulate_for_obstacles(
            obstacles=geom_targets,
            p0_point=p0_point,
            heading=self.heading,
            angle_per_ray=self._d_sensor_angle,
            n_rays=self.n_sensors,
        )

        for i, ray_obstacles in enumerate(obstacles_to_simulate_per_ray):
            dist, speed, blocked = simulate_sensor(
                obstacles=ray_obstacles,
                sensor_angle=sensor_angles_ned[i],
                sensor_range=sensor_range,
                p0_point=p0_point,
            )

            sensor_dist_measurements.append(dist)
            sensor_speed_measurements.append(speed)
            sensor_blocked_arr.append(blocked)

        sensor_dist_measurements = np.array(sensor_dist_measurements)
        sensor_speed_measurements = np.array(sensor_speed_measurements).T

        return (sensor_dist_measurements, sensor_speed_measurements, sensor_blocked_arr)

    def navigate(self, path: Path) -> np.ndarray:
        """
        Calculates and returns navigation states representing the vessel's attitude
        with respect to the desired path.

        Returns
        -------
        navigation_states : np.ndarray
        """

        # Calculating path arclength at reference point, i.e. the point closest to the vessel
        vessel_arclength = path.get_closest_arclength(self.position)

        # Calculating tangential path direction at reference point
        path_direction = path.get_direction(vessel_arclength)

        closest_path_point_ned = path(vessel_arclength)
        relative_pos_nearest_path_point = geom.Rzyx(0, 0, -self.heading).dot(
            np.hstack([closest_path_point_ned - self.position, 0])
        )[:2]
        cross_track_error = geom.Rzyx(0, 0, -path_direction).dot(
            np.hstack([path(vessel_arclength) - self.position, 0])
        )[1]

        # Calculating tangential path direction at look-ahead point
        target_arclength = min(
            path.length, vessel_arclength + self.config.vessel.look_ahead_distance
        )
        look_ahead_path_direction = path.get_direction(target_arclength)
        look_ahead_heading_error = float(
            geom.princip(look_ahead_path_direction - self.heading)
        )

        # Calculating vector difference between look-ahead point and vessel position
        relative_pos_lookahead = path(target_arclength) - self.position

        # Calculating heading error
        target_heading = np.arctan2(
            relative_pos_lookahead[1], relative_pos_lookahead[0]
        )
        heading_error = float(geom.princip(target_heading - self.heading))

        # Calculating path progress
        self._prev_progress = self._progress
        progress = vessel_arclength / path.length
        self._progress = progress

        self._max_progress = max(progress, self._max_progress)
        new_progress = max(0, self.progress - self.prev_timestep_progress)

        # Deciding if vessel has reached the goal
        goal_distance = linalg.norm(path.end - self.position)
        reached_goal = (
            goal_distance <= self.config.episode.min_goal_distance
            or progress >= self.config.episode.min_path_progress
        )
        self._reached_goal = reached_goal
        path_error_x = relative_pos_nearest_path_point[0] / 100
        path_error_y = relative_pos_nearest_path_point[1] / 100
        lookahead_path_error_x = relative_pos_lookahead[0] / 100
        lookahead_path_error_y = relative_pos_lookahead[1] / 100
        
        # Calculate a unit vector pointing in the direction of the lookahead point
        lookahead_vector_normalized_ned = relative_pos_lookahead / linalg.norm(relative_pos_lookahead)

        self._last_navi_state_dict = {
            "surge_velocity": self.velocity[0],
            "sway_velocity": self.velocity[1],
            "yaw_rate": self.yaw_rate,
            "look_ahead_heading_error": look_ahead_heading_error,
            "heading_error": heading_error,
            "cross_track_error": cross_track_error / 100,
            "new_progress": new_progress,
            "target_heading": target_heading,
            "target_vector": relative_pos_lookahead,
            "look_ahead_path_direction": look_ahead_path_direction,
            "path_direction": path_direction,
            "vessel_arclength": vessel_arclength,
            "target_arclength": target_arclength,
            "goal_distance": goal_distance,
            "path_error_x": path_error_x,
            "path_error_y": path_error_y,
            "lookahead_path_error_x": lookahead_path_error_x,
            "lookahead_path_error_y": lookahead_path_error_y,
            "lookahead_vector_normalized_ned": lookahead_vector_normalized_ned,
            "velocity_ned": self.velocity_ned,
        }

        navigation_observation_keys = self._get_navigation_observation_keys()
        navigation_states = np.array(
            [self._last_navi_state_dict[key] for key in navigation_observation_keys]
        )

        return navigation_states

    def _get_navigation_observation_keys(self) -> List[str]:
        """Gets the navigation keys to include in the observation according to the settings in the config"""
        navigation_keys = []

        if self.config.sensor.observe_proprioceptive:
            navigation_keys.extend(["surge_velocity", "sway_velocity", "yaw_rate"])

        if self.config.sensor.observe_cross_track_error:
            navigation_keys.append("cross_track_error")

        if self.config.sensor.observe_heading_error:
            navigation_keys.append("heading_error")

        if self.config.sensor.observe_la_heading_error:
            navigation_keys.append("look_ahead_heading_error")

        if self.config.sensor.observe_new_progress:
            navigation_keys.append("new_progress")

        return navigation_keys

    def req_latest_data(self) -> dict:
        """Returns dictionary containing the most recent perception and navigation
        states."""
        latest_data = {
            "distance_measurements": self._last_sensor_dist_measurements,
            "speed_measurements": self._last_sensor_speed_measurements,
            "navigation": self._last_navi_state_dict,
            "collision": self._collision,
            "progress": self._progress,
            "reached_goal": self._reached_goal,
            # "max_progress":
        }

        # if self.config.sensor.use_feasibility_pooling:
        #     latest_data["feasible_distances"] = self._last_sector_feasible_dists

        return latest_data

    # def _state_dot_meyer(self, state):
    #     psi = state[2]
    #     nu = state[3:]

    #     tau = np.array([self._input[0], 0, self._input[1]])
    #         # TODO: Look over rudder action

    #     eta_dot = geom.Rz(geom.princip(psi)).dot(nu)
    #     nu_dot = const.M_inv.dot(const.B(nu).dot(self._input) - const.D(nu).dot(nu) - const.C(nu).dot(nu)) # const.N(nu).dot(nu))
    #     state_dot = np.concatenate([eta_dot, nu_dot])
    #     return state_dot

    def _state_dot(self, state):
        eta = state[:3]
        nu = state[3:]
        # n = state[12:]  # Propeller speeds


        # Input consists of commanded moment forward and commanded torque.
        # We use the Otter's included function for allocating them
        demanded_forward_force = self._input[0]
        demanded_yaw_moment = self._input[1]

        # u1, u2 = self.dynamics_model.controlAllocation(tau_X=demanded_forward_force, tau_N=demanded_yaw_moment)
        # u = np.array([u1, u2])
        u = np.array(self._input)

        state_dot = self.dynamics_model.dynamics(
            eta=eta,
            nu=nu,
            u=u
        )

        if np.linalg.norm(state_dot) > 1000:
            breakpoint()

        return np.array(state_dot)

    def _thrust_surge(self, surge):
        surge = np.clip((surge + 1.0) / 2, 0, 1)
        return surge * self.config.vessel.thrust_max_auv

    def _moment_steer(self, steer):
        steer = np.clip(steer, -1, 1)
        return steer * self.config.vessel.moment_max_auv
