"""
This module implements an AUV that is simulated in the horizontal plane.
"""
import numpy as np
from numba import jit
import numpy.linalg as linalg
from itertools import islice, chain, repeat
import shapely.geometry, shapely.errors, shapely.strtree, shapely.ops, shapely.prepared

import gym_auv.utils.constants as const
import gym_auv.utils.geomutils as geom
from gym_auv.objects.obstacles import *

def odesolver45(f, y, h):
    """Calculate the next step of an IVP of a time-invariant ODE with a RHS
    described by f, with an order 4 approx. and an order 5 approx.
    Parameters:
        f: function. RHS of ODE.
        y: float. Current position.
        h: float. Step length.
    Returns:
        q: float. Order 2 approx.
        w: float. Order 3 approx.
    """
    s1 = f(y)
    s2 = f(y+h*s1/4.0)
    s3 = f(y+3.0*h*s1/32.0+9.0*h*s2/32.0)
    s4 = f(y+1932.0*h*s1/2197.0-7200.0*h*s2/2197.0+7296.0*h*s3/2197.0)
    s5 = f(y+439.0*h*s1/216.0-8.0*h*s2+3680.0*h*s3/513.0-845.0*h*s4/4104.0)
    s6 = f(y-8.0*h*s1/27.0+2*h*s2-3544.0*h*s3/2565+1859.0*h*s4/4104.0-11.0*h*s5/40.0)
    w = y + h*(25.0*s1/216.0+1408.0*s3/2565.0+2197.0*s4/4104.0-s5/5.0)
    q = y + h*(16.0*s1/135.0+6656.0*s3/12825.0+28561.0*s4/56430.0-9.0*s5/50.0+2.0*s6/55.0)
    return w, q

def _standardize_intersect(intersect):
    if intersect.is_empty:
        return []
    elif isinstance(intersect, shapely.geometry.LineString):
        return [shapely.geometry.Point(intersect.coords[0])]
    elif isinstance(intersect, shapely.geometry.Point):
        return [intersect]
    else:
        return list(intersect.geoms)

@jit(nopython=True)
def _feasibility_pooling(x, width, theta):
    N_sensors = x.shape[0]
    sort_idx = np.argsort(x)
    for idx in sort_idx:
        surviving = x > x[idx] + width
        d = x[idx]*theta
        opening_width = 0
        opening_span = 0
        opening_start = -theta*(N_sensors-1)/2
        found_opening = False
        for isensor, sensor_survives in enumerate(surviving):
            if sensor_survives:
                opening_width += d
                opening_span += theta
                if opening_width > width:
                    opening_center = opening_start + opening_span/2
                    if abs(opening_center) < theta*(N_sensors-1)/4:
                        found_opening = True
            else:
                opening_width += 0.5*d
                opening_span += 0.5*theta
                if opening_width > width:
                    opening_center = opening_start + opening_span/2
                    if abs(opening_center) < theta*(N_sensors-1)/4:
                        found_opening = True
                opening_width = 0
                opening_span = 0
                opening_start = -theta*(N_sensors-1)/2 + isensor*theta

        if not found_opening: 
            return max(0, x[idx])

    return max(0, np.max(x))

def _simulate_sensor(sensor_angle, p0_point, sensor_range, obstacles):
    sensor_endpoint = (
        p0_point.x + np.cos(sensor_angle)*sensor_range,
        p0_point.y + np.sin(sensor_angle)*sensor_range
    )
    sector_ray = shapely.geometry.LineString([p0_point, sensor_endpoint])

    obst_intersections = [sector_ray.intersection(elm.boundary) for elm in obstacles]
    obst_intersections = list(map(_standardize_intersect, obst_intersections))
    obst_references = list(chain.from_iterable(repeat(obstacles[i], len(obst_intersections[i])) for i in range(len(obst_intersections))))
    obst_intersections = list(chain(*obst_intersections))

    if obst_intersections:
        measured_distance, intercept_idx = min((float(p0_point.distance(elm)), i) for i, elm in enumerate(obst_intersections))
        obstacle = obst_references[intercept_idx]
        if not obstacle.static:
            obst_speed_homogenous = geom.to_homogeneous([obstacle.dx, obstacle.dy])
            obst_speed_rel_homogenous = geom.Rz(-sensor_angle - np.pi/2).dot(obst_speed_homogenous)
            obst_speed_vec_rel = geom.to_cartesian(obst_speed_rel_homogenous)
        else:
            obst_speed_vec_rel = (0, 0)
    else:
        measured_distance = sensor_range
        obst_speed_vec_rel = (0, 0)

    return (measured_distance, obst_speed_vec_rel)

class Vessel():
    """
    Creates an environment with a vessel, goal and obstacles.
    """

    NAVIGATION_STATES = [
        'surge_velocity',
        'sway_velocity',
        'yaw_rate',
        'look_ahead_course_error',
        'course_error',
        'cross_track_error'
    ]

    def __init__(self, config, init_pos, width=4):
        """
        The __init__ method declares all class atributes.

        Parameters
        ----------
        init_pos : np.array
            The initial position of the veHssel [x, y, psi], where
            psi is the initial heading of the AUV.
        width : float
            The maximum distance from the center of the AUV to its edge
            in meters. Defaults to 2.
        """
        
        self.config = config
        self.width = width

        # Initializing attributes
        self.n_sectors = self.config["n_sectors"]
        self.n_sensors = self.config["n_sensors_per_sector"]*self.config["n_sectors"]
        self.sensor_range = self.config["sensor_range"]
        self.d_sensor_angle = 2*np.pi/(self.n_sensors)
        self.sensor_angles = np.array([-np.pi + (i + 1)*self.d_sensor_angle for i in range(self.n_sensors)])
        self.sector_angles = []
        self.n_sensors_per_sector = [0]*self.n_sectors
        self.sector_start_indeces = [0]*self.n_sectors
        self.sector_end_indeces = [0]*self.n_sectors
        self.sensor_internal_indeces = []
        self._sensor_interval = max(1, int(1/self.config["sensor_frequency"]))

        # Calculating sensor partitioning
        last_isector = -1
        tmp_sector_angle_sum = 0
        tmp_sector_sensor_count = 0
        for isensor in range(self.n_sensors):
            isector = self.config["sector_partition_fun"](self, isensor)
            angle = self.sensor_angles[isensor]
            if isector == last_isector:
                tmp_sector_angle_sum += angle
                tmp_sector_sensor_count += 1
            else:
                if last_isector > -1:
                    self.sector_angles.append(tmp_sector_angle_sum/tmp_sector_sensor_count)
                last_isector = isector
                self.sector_start_indeces[isector] = isensor
                tmp_sector_angle_sum = angle
                tmp_sector_sensor_count = 1
            self.n_sensors_per_sector[isector] += 1
        self.sector_angles.append(tmp_sector_angle_sum/tmp_sector_sensor_count)
        self.sector_angles = np.array(self.sector_angles)

        for isensor in range(self.n_sensors):
            isector = self.config["sector_partition_fun"](self, isensor)
            isensor_internal = isensor - self.sector_start_indeces[isector]
            self.sensor_internal_indeces.append(isensor_internal)

        for isector in range(self.n_sectors):
            self.sector_end_indeces[isector] = self.sector_start_indeces[isector] + self.n_sensors_per_sector[isector]

        # Calculating feasible closeness
        if self.config["sensor_log_transform"]:
            self._get_closeness = lambda x: 1 - np.clip(np.log(1 + x)/np.log(1 + self.sensor_range), 0, 1)
        else:
            self._get_closeness = lambda x: 1 - np.clip(x/self.sensor_range, 0, 1)

        # Initializing vessel to initial position
        self.reset(init_pos)

    def reset(self, init_pos, init_speed=None):
        if init_speed is None:
            init_speed = [0, 0, 0]
        init_pos = np.array(init_pos, dtype=np.float64)
        init_speed = np.array(init_speed, dtype=np.float64)
        self._state = np.hstack([init_pos, init_speed])
        self.prev_states = np.vstack([self._state])
        self.input = [0, 0]
        self.prev_inputs =np.vstack([self.input])
        self.smoothed_torque_change = 0
        self.smoothed_torque = 0
        self.last_sensor_dist_measurements = np.ones((self.n_sensors,))*self.sensor_range
        self.last_sensor_speed_measurements = np.zeros((self.n_sensors,2))
        self.last_sector_dist_measurements = np.zeros((self.n_sectors,))
        self.last_sector_feasible_dists = np.zeros((self.n_sectors,))
        self.last_navi_state_dict = dict((state, 0) for state in Vessel.NAVIGATION_STATES)

        self.sensor_endpoint = [None for isensor in range(self.n_sensors)]

        self._step_counter = 0
        self._perceive_counter = 0
        self._nearby_obstacles = []

    def step(self, action):
        """
        Steps the vessel one step forward

        Parameters
        ----------
        action : np.array
            [propeller_input, rudder_position], where
            0 <= propeller_input <= 1 and -1 <= rudder_position <= 1.
        """
        self.input = np.array([self._thrust_surge(action[0]), self._moment_steer(action[1])])
        w, q = odesolver45(self._state_dot, self._state, self.config["t_step_size"])
        
        self._state = q
        self._state[2] = geom.princip(self._state[2])

        self.prev_states = np.vstack([self.prev_states,self._state])
        self.prev_inputs = np.vstack([self.prev_inputs,self.input])

        torque_change = self.input[1] - self.prev_inputs[-2, 1] if len(self.prev_inputs) > 1 else self.input[1]
        self.smoothed_torque_change = 0.9*self.smoothed_torque_change + 0.1*abs(torque_change)
        self.smoothed_torque = 0.9*self.smoothed_torque + 0.1*abs(self.input[1])

        self._step_counter += 1

    def _state_dot(self, state):
        psi = state[2]
        nu = state[3:]

        tau = np.array([self.input[0], 0, self.input[1]])

        eta_dot = geom.Rzyx(0, 0, geom.princip(psi)).dot(nu)
        nu_dot = const.M_inv.dot(
            tau
            #- const.D.dot(nu)
            - const.N(nu).dot(nu)
        )
        state_dot = np.concatenate([eta_dot, nu_dot])
        return state_dot

    def _thrust_surge(self, surge):
        surge = np.clip(surge, 0, 1)
        return surge*const.THRUST_MAX_AUV

    def _moment_steer(self, steer):
        steer = np.clip(steer, -1, 1)
        return steer*const.MOMENT_MAX_AUV

    def perceive(self, obstacles):
        """
        Calculates and returns the sensor-based observation array of the environment, 
        as well as whether the vessel has collided.
        
        Returns
        -------
        obs : np.array
        collision : bool
        (sector_closenesses, sector_velocities, collision, sector_feasible_distances)
        """

        # Initializing variables
        p0_point = shapely.geometry.Point(*self.position)

        # Loading nearby obstacles, i.e. obstacles within the vessel's detection range
        if self._step_counter % self.config["sensor_interval_load_obstacles"] == 0:
            self._nearby_obstacles = list(filter(
                lambda obst: float(p0_point.distance(obst.boundary)) - self.width < self.sensor_range, obstacles
            ))

        if not self._nearby_obstacles:
            self.last_sensor_dist_measurements = np.ones((self.n_sensors,))*self.sensor_range
            sector_feasible_distances = np.ones((self.n_sectors,))*self.sensor_range
            sector_closenesses = np.zeros((self.n_sectors,))
            sector_velocities = np.zeros((2*self.n_sectors,))
            collision = False

        else:
            # Simulating all sensors using _simulate_sensor subroutine
            sensor_angles_ned = self.sensor_angles + self.heading
            activate_sensor = lambda i: (i % self._sensor_interval) == (self._perceive_counter % self._sensor_interval)
            sensor_sim_args = (p0_point, self.sensor_range, self._nearby_obstacles)
            sensor_output_arrs = list(map(
                lambda i: _simulate_sensor(sensor_angles_ned[i], *sensor_sim_args) if activate_sensor(i) else (
                    self.last_sensor_dist_measurements[i],
                    self.last_sensor_speed_measurements[i]
                ), 
                range(self.n_sensors)
            ))
            sensor_dist_measurements, sensor_speed_measurements = zip(*sensor_output_arrs)
            sensor_dist_measurements = np.array(sensor_dist_measurements)
            sensor_speed_measurements = np.array(sensor_speed_measurements)
            self.last_sensor_dist_measurements = sensor_dist_measurements
            self.last_sensor_speed_measurements = sensor_speed_measurements

            # Partitioning sensor readings into sectors
            sector_dist_measurements = np.split(sensor_dist_measurements, self.sector_start_indeces[1:])
            sector_speed_measurements = np.split(sensor_speed_measurements, self.sector_start_indeces[1:], axis=0)

            # Performing feasibility pooling
            sector_feasible_distances = np.array(list(
                map(lambda x: _feasibility_pooling(x, self.width, self.d_sensor_angle), sector_dist_measurements)
            ))

            # Calculating feasible closeness
            sector_closenesses = self._get_closeness(sector_feasible_distances)

            # Retrieving obstacle speed for closest obstacle within each sector
            closest_obst_sensor_indeces = list(map(np.argmin, sector_dist_measurements))
            sector_velocities = np.concatenate(
                [sector_speed_measurements[i][closest_obst_sensor_indeces[i]] for i in range(self.n_sectors)]
            )

            # Testing if vessel has collided
            collision = any(
                float(p0_point.distance(obst.boundary)) - self.width <= 0 for obst in self._nearby_obstacles
            )

        self.last_sector_dist_measurements = sector_closenesses
        self.last_sector_feasible_dists = sector_feasible_distances
        self._perceive_counter += 1

        return (sector_closenesses, sector_velocities, collision)

    def navigate(self, path):
        """
        Calculates and returns navigation states representing the vessel's attitude
        with respect to the desired path.
        
        Returns
        -------
        states : np.array
        """

        # Calculating path arclength at reference point, i.e. the point closest to the vessel
        vessel_arclength = path.get_closest_arclength(self.position)

        # Calculating tangential path direction at reference point
        path_direction = path.get_direction(vessel_arclength)
        cross_track_error = geom.Rzyx(0, 0, -path_direction).dot(
            np.hstack([path(vessel_arclength) - self.position, 0])
        )[1]

        # Calculating tangential path direction at look-ahead point
        look_ahead_path_direction = path.get_direction(vessel_arclength + self.config["look_ahead_distance"]*path.length) 
        look_ahead_course_error = float(geom.princip(look_ahead_path_direction - self.course))

        # Calculating vector difference between look-ahead point and vessel position
        target_vector = path(vessel_arclength) - self.position

        # Calculating heading error
        target_heading = np.arctan2(target_vector[1], target_vector[0])
        course_error = float(geom.princip(target_heading - self.course))

        # Concatenating states
        self.last_navi_state_dict = {
            'surge_velocity': self.velocity[0],
            'sway_velocity': self.velocity[1],
            'yaw_rate': self.yawrate,
            'look_ahead_course_error': look_ahead_course_error,
            'course_error': course_error,
            'cross_track_error': cross_track_error/path.length,
            'target_heading': target_heading,
            'look_ahead_path_direction': look_ahead_path_direction,
            'path_direction': path_direction
        }
        navigation_states = np.array([self.last_navi_state_dict[state] for state in Vessel.NAVIGATION_STATES])

        # Deciding if vessel has reached the goal
        goal_distance = linalg.norm(path.end - self.position)
        reached_goal = goal_distance <= self.config["min_goal_distance"]
        
        # Calculating path progress
        progress = vessel_arclength/path.length

        return (navigation_states, reached_goal, progress)

    @property
    def position(self):
        """
        Returns an array holding the position of the AUV in cartesian
        coordinates.
        """
        return self._state[0:2]

    @property
    def x(self):
        return self.position[0]

    @property
    def y(self):
        return self.position[1]

    @property
    def init_position(self):
        """
        Returns an array holding the path of the AUV in cartesian
        coordinates.
        """
        return self.prev_states[-1, 0:2]

    @property
    def path_taken(self):
        """
        Returns an array holding the path of the AUV in cartesian
        coordinates.
        """
        return self.prev_states[:, 0:2]

    @property
    def heading(self):
        """
        Returns the heading of the AUV wrt true north.
        """
        return self._state[2]

    @property
    def heading_history(self):
        """
        Returns the heading of the AUV wrt true north.
        """
        return self.prev_states[:, 2]

    @property
    def heading_change(self):
        """
        Returns the change of heading of the AUV wrt true north.
        """
        return geom.princip(self.prev_states[-1, 2] - self.prev_states[-2, 2]) if len(self.prev_states) >= 2 else self.heading

    @property
    def velocity(self):
        """
        Returns the surge and sway velocity of the AUV.
        """
        return self._state[3:5]

    @property
    def speed(self):
        """
        Returns the surge and sway velocity of the AUV.
        """
        return linalg.norm(self.velocity)

    @property
    def yawrate(self):
        """
        Returns the rate of rotation about the z-axis.
        """
        return self._state[5]

    @property
    def max_speed(self):
        """
        Returns the max speed of the AUV.
        """
        return const.MAX_SPEED

    @property
    def crab_angle(self):
        return np.arctan2(self.velocity[1], self.velocity[0])

    @property
    def course(self):
        return self.heading + self.crab_angle