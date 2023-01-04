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
    obstacle_relative_pos_ned = np.array(obstacle_enclosing_circle.center) - np.array(
        p0_point
    )
    # Find the relative angle from the heading to the obstacle. Use clockwise positive rotation, as this is
    # done in the NED plane
    n, e = obstacle_relative_pos_ned
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
    # They are passed by reference in python, so it should still be pretty fast.
    obstacles_to_simulate_per_ray = [[] for _ in range(n_rays)]

    for obstacle in obstacles:
        idx_min_ray, idx_max_ray = _find_limit_angle_rays(
            obstacle.enclosing_circle, p0_point, heading, angle_per_ray
        )

        # Add obstacle to all rays which may collide with it
        # Because negative indices wrap around (in the same way as the sensor!),
        # they aren't a problem
        for i in range(idx_min_ray - 1, idx_max_ray):
            # -1 as first sensor has angle -pi + "angle between rays"
            try:
                i_safe = i % n_rays
                obstacles_to_simulate_per_ray[i_safe].append(obstacle)
            except IndexError:
                print("Index error")
                print(i)
                print(idx_min_ray)
                print(idx_max_ray)
                print(n_rays)
                print(obstacle.enclosing_circle.center.centroid)
                print("radius", obstacle.enclosing_circle.radius)
                print(p0_point.centroid)
                breakpoint()
                idx_min_ray, idx_max_ray = _find_limit_angle_rays(
                    obstacle.enclosing_circle, p0_point, heading, angle_per_ray
                )

                raise IndexError

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

    return (measured_distance, None, ray_blocked)


def get_relative_positions_of_lidar_measurements(
    lidar_ranges: np.ndarray,
    sensor_angles: np.ndarray,
    indices_to_plot: np.ndarray,
) -> np.ndarray:
    """Gets the positions of the lidar measurements in the body frame.

    Returns
    -------
    np.ndarray of shape (c, 2), where c is the number of lidars with collisions
    """
    # Each row are (north, east) coordinates of a ray
    pos = (
        np.vstack(
            [lidar_ranges * np.cos(sensor_angles), lidar_ranges * np.sin(sensor_angles)]
        )
    ).T

    # Only calculate occupancy for positions with measurements
    pos_with_collisions = pos[indices_to_plot, :]
    return pos_with_collisions


def _filter_valid_indices(indices: np.ndarray, grid_size: int) -> np.ndarray:
    """Filters out the indices that are out of bounds, and returns the valid ones.
    Takes a (N, 2) array as input, and outputs a (M, 2) as output, where M <= N
    """
    invalid_element_mask = (indices < 0) | (grid_size <= indices)

    # As only one of the indices in a coordinate may be out of bounds,
    # e.g. [67, 4] where valid indices are between 0 and 63
    # We do the "OR" operation over columns within rows to find invalid rows
    invalid_row_mask = invalid_element_mask[:, 0] | invalid_element_mask[:, 1]

    if np.all(invalid_row_mask):
        return np.array([])

    # Choose all columns in the rows that are valid (not invalid)
    valid_rows = indices[~invalid_row_mask]
    return valid_rows


def make_occupancy_grid(
    positions_body: np.ndarray,
    sensor_range: np.ndarray,
    grid_size: int,
) -> np.ndarray:
    # Calculate the indices in the grid
    indices_decimals = (positions_body * (grid_size / 2) / sensor_range) + (
        grid_size / 2
    )
    # Round the indices to integers
    indices = np.floor(indices_decimals).astype(np.int32)

    valid_indices = _filter_valid_indices(indices, grid_size)

    occupancy_grid = np.zeros((grid_size, grid_size))
    if len(valid_indices):
        # Avoid error from indexing without any valid indices in the next line
        # Occupancy grid uses (row, col) i.e. (y, x) indexing
        occupancy_grid[valid_indices[:, 0], valid_indices[:, 1]] = 1.0

    # Flip along the y-axis, such that straight ahead (first coordinate in body frame)
    # is up and right (second coordinate in body frame) stays to the right
    flipped_occupancy_grid = np.flipud(occupancy_grid)

    return flipped_occupancy_grid
