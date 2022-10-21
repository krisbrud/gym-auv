from typing import List, Tuple
import numpy as np
from itertools import chain, repeat
import shapely.geometry, shapely.errors, shapely.strtree, shapely.ops, shapely.prepared
import gym_auv
from gym_auv.objects.obstacles import BaseObstacle, CircleParams

import gym_auv.utils.geomutils as geom


def _standardize_intersect(intersect):
    if intersect.is_empty:
        return []
    elif isinstance(intersect, shapely.geometry.LineString):
        return [shapely.geometry.Point(intersect.coords[0])]
    elif isinstance(intersect, shapely.geometry.Point):
        return [intersect]
    else:
        return list(intersect.geoms)


def _find_feasible_angle_diff(
    obstacle_enclosing_circle: CircleParams,
    p0_point: shapely.geometry.Point,
):
    dist_p0_circle_center = p0_point.distance(obstacle_enclosing_circle.center)

    safe_dist = max(1e-8, dist_p0_circle_center)  # Avoid zero division

    max_angle_from_circle_center = np.arcsin(
        obstacle_enclosing_circle.radius / safe_dist
    )

    if np.isnan(max_angle_from_circle_center):
        # Inside convex hull - return pi as we want to check everywhere
        return np.pi

    return max_angle_from_circle_center


def _find_limit_angle_rays(
    obstacle_enclosing_circle: CircleParams,
    p0_point: shapely.geometry.Point,
    heading: float,
    angle_per_ray: float,  # radians
) -> Tuple[int, int]:
    """Finds the indices of the rays that may collide with an obstacle (so that we do not need to simulate this for every other ray)"""
    obstacle_relative_pos = np.array(obstacle_enclosing_circle.center) - np.array(
        p0_point
    )
    # Find the relative angle from the heading to the obstacle. Use clockwise positive rotation, as this is
    # done in the NED plane
    n, e = obstacle_relative_pos
    obstacle_relative_bearing = np.arctan2(e, n) - heading
    feasible_angle_diff = _find_feasible_angle_diff(obstacle_enclosing_circle, p0_point)

    # Assume seam on back (ray 0 and N meets at the back), and clockwise positive rotation
    idx_min_ray = int(
        np.floor(
            (np.pi + (obstacle_relative_bearing - feasible_angle_diff)) / angle_per_ray
        )
    )
    idx_max_ray = int(
        np.ceil(
            (np.pi + (obstacle_relative_bearing + feasible_angle_diff)) / angle_per_ray
        )
    )

    return (idx_min_ray, idx_max_ray)


def find_rays_to_simulate_for_obstacles(
    obstacles: List[BaseObstacle],
    p0_point: shapely.geometry.Point,
    heading: float,
    angle_per_ray: float,
    n_rays: int,
) -> List[List[BaseObstacle]]:
    # Make a list of obstacles that may intersect per ray.
    # They are passed by reference in python, so it should be pretty fast.
    obstacles_to_simulate_per_ray = [[] for _ in range(n_rays)]

    for obstacle in obstacles:
        idx_min_ray, idx_max_ray = _find_limit_angle_rays(
            obstacle.enclosing_circle, p0_point, heading, angle_per_ray
        )

        # Add obstacle to all rays which may collide with it
        # Because negative indices wrap around (in the same way as the sensor!),
        # they aren't a problem
        for i in range(idx_min_ray - 1, idx_max_ray % n_rays):
            # -1 as first sensor has angle -pi + "angle between rays"
            obstacles_to_simulate_per_ray[i].append(obstacle)

    return obstacles_to_simulate_per_ray


def simulate_sensor_brute_force(sensor_angle, p0_point, sensor_range, obstacles):
    sensor_endpoint = (
        p0_point.x + np.cos(sensor_angle) * sensor_range,
        p0_point.y + np.sin(sensor_angle) * sensor_range,
    )
    sector_ray = shapely.geometry.LineString([p0_point, sensor_endpoint])

    obst_intersections = [sector_ray.intersection(elm.boundary) for elm in obstacles]
    obst_intersections = list(map(_standardize_intersect, obst_intersections))
    obst_references = list(
        chain.from_iterable(
            repeat(obstacles[i], len(obst_intersections[i]))
            for i in range(len(obst_intersections))
        )
    )
    obst_intersections = list(chain(*obst_intersections))

    if obst_intersections:
        measured_distance, intercept_idx = min(
            (float(p0_point.distance(elm)), i)
            for i, elm in enumerate(obst_intersections)
        )
        obstacle = obst_references[intercept_idx]
        if not obstacle.static:
            obst_speed_homogenous = geom.to_homogeneous([obstacle.dx, obstacle.dy])
            obst_speed_rel_homogenous = geom.Rz(-sensor_angle - np.pi / 2).dot(
                obst_speed_homogenous
            )
            obst_speed_vec_rel = geom.to_cartesian(obst_speed_rel_homogenous)
        else:
            obst_speed_vec_rel = (0, 0)
        ray_blocked = True
    else:
        measured_distance = sensor_range
        obst_speed_vec_rel = (0, 0)
        ray_blocked = False

    return (measured_distance, obst_speed_vec_rel, ray_blocked)


def simulate_sensor(sensor_angle, p0_point, sensor_range, obstacles):
    sensor_endpoint = (
        p0_point.x + np.cos(sensor_angle) * sensor_range,
        p0_point.y + np.sin(sensor_angle) * sensor_range,
    )
    sector_ray = shapely.geometry.LineString([p0_point, sensor_endpoint])

    obst_intersections = [sector_ray.intersection(elm.boundary) for elm in obstacles]
    obst_intersections = list(map(_standardize_intersect, obst_intersections))
    obst_intersections = list(chain(*obst_intersections))

    if obst_intersections:
        distances = [p0_point.distance(elm) for elm in obst_intersections]
        measured_distance = np.min(distances)
        ray_blocked = True
    else:
        measured_distance = sensor_range
        ray_blocked = False

    return (measured_distance, (0, 0), ray_blocked)


def make_occupancy_grid(
    lidar_ranges: np.ndarray,
    sensor_angles: np.ndarray,
    sensor_range: np.ndarray,
    grid_size: int,
    blocked_sensors: np.ndarray,
) -> np.ndarray:
    # Each row are (north, east) coordinates of a ray
    pos = (
        np.vstack(
            [lidar_ranges * np.cos(sensor_angles), lidar_ranges * np.sin(sensor_angles)]
        )
    ).T  

    # Only calculate occupancy for positions with measurements
    pos_with_collisions = pos[blocked_sensors, :]

    # Calculate the indices in the grid
    indices_decimals = (pos_with_collisions * (grid_size / 2) / sensor_range) + (
        grid_size / 2
    )
    # Round the indices to integers
    indices = np.floor(indices_decimals).astype(np.int32)

    # Occupancy grid uses (row, col) i.e. (y, x) indexing
    occupancy_grid = np.zeros((grid_size, grid_size))
    occupancy_grid[indices[:, 1], indices[:, 0]] = 1.0

    return occupancy_grid


class LidarPreprocessor:
    """LidarPreprocessor reduces the dimensionality of the Lidar measurements through feasibility pooling"""

    def __init__(self, config: gym_auv.Config, _d_sensor_angle: float):
        self._feasibility_width = (
            config.vessel.vessel_width * config.vessel.feasibility_width_multiplier
        )
        self._n_sectors = config.vessel.n_sectors
        self._sector_angles = []
        self._n_sensors_per_sector = [0] * self._n_sectors
        self._sector_start_indeces = [0] * self._n_sectors
        self._sector_end_indeces = [0] * self._n_sectors
        self._sector_partition_fun = config.vessel.sector_partition_fun
        self._d_sensor_angle = _d_sensor_angle

    def _init_sectors(self) -> None:
        """Initializes the sectors used in for instance feasibility pooling

        Calling this function is not needed if no pooling is done.
        """
        last_isector = -1
        tmp_sector_angle_sum = 0
        tmp_sector_sensor_count = 0
        for isensor in range(self._n_sensors):
            isector = self._sector_partition_fun(self, isensor)
            angle = self._sensor_angles[isensor]
            if isector == last_isector:
                tmp_sector_angle_sum += angle
                tmp_sector_sensor_count += 1
            else:
                if last_isector > -1:
                    self._sector_angles.append(
                        tmp_sector_angle_sum / tmp_sector_sensor_count
                    )
                last_isector = isector
                self._sector_start_indeces[isector] = isensor
                tmp_sector_angle_sum = angle
                tmp_sector_sensor_count = 1
            self._n_sensors_per_sector[isector] += 1
        self._sector_angles.append(tmp_sector_angle_sum / tmp_sector_sensor_count)
        self._sector_angles = np.array(self._sector_angles)

        for isensor in range(self._n_sensors):
            isector = self._sector_partition_fun(self, isensor)
            isensor_internal = isensor - self._sector_start_indeces[isector]
            self._sensor_internal_indeces.append(isensor_internal)

        for isector in range(self._n_sectors):
            self._sector_end_indeces[isector] = (
                self._sector_start_indeces[isector]
                + self._n_sensors_per_sector[isector]
            )

    def preprocess(
        self,
        sensor_dist_measurements: np.ndarray,
        sensor_speed_measurements: np.ndarray,
    ) -> np.ndarray:
        # Partitioning sensor readings into sectors
        sector_dist_measurements = np.split(
            sensor_dist_measurements, self._sector_start_indeces[1:]
        )
        sector_speed_measurements = np.split(
            sensor_speed_measurements, self._sector_start_indeces[1:], axis=0
        )

        # Performing feasibility pooling
        sector_feasible_distances = np.array(
            list(
                map(
                    lambda x: LidarPreprocessor._feasibility_pooling(
                        x, self._feasibility_width, self._d_sensor_angle
                    ),
                    sector_dist_measurements,
                )
            )
        )

        # Retrieving obstacle speed for closest obstacle within each sector
        closest_obst_sensor_indeces = list(map(np.argmin, sector_dist_measurements))
        sector_velocities = np.concatenate(
            [
                sector_speed_measurements[i][closest_obst_sensor_indeces[i]]
                for i in range(self._n_sectors)
            ]
        )

        return sector_feasible_distances, sector_velocities

    @staticmethod
    def _feasibility_pooling(
        measurements: np.ndarray, width: float, theta: float
    ) -> float:
        """Applies feasibility pooling to the sensors in a sector

        Args:
            measurements:   Measurements corresponding to one sector
            width:          The width from the center to the side of the vessel
            theta:          Angle between neighboring sensors
        Returns:
            The maximum distance to a feasible opening for the vessel
        """

        N_sensors = measurements.shape[0]
        sort_idx = np.argsort(measurements)
        for idx in sort_idx:
            surviving = measurements > measurements[idx] + width
            d = measurements[idx] * theta
            opening_width = 0
            opening_span = 0
            opening_start = -theta * (N_sensors - 1) / 2
            found_opening = False
            for isensor, sensor_survives in enumerate(surviving):
                if sensor_survives:
                    opening_width += d
                    opening_span += theta
                    if opening_width > width:
                        opening_center = opening_start + opening_span / 2
                        if abs(opening_center) < theta * (N_sensors - 1) / 4:
                            found_opening = True
                else:
                    opening_width += 0.5 * d
                    opening_span += 0.5 * theta
                    if opening_width > width:
                        opening_center = opening_start + opening_span / 2
                        if abs(opening_center) < theta * (N_sensors - 1) / 4:
                            found_opening = True
                    opening_width = 0
                    opening_span = 0
                    opening_start = -theta * (N_sensors - 1) / 2 + isensor * theta

            if not found_opening:
                return max(0, measurements[idx])

        return max(0, np.max(measurements))
