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


