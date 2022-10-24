from typing import Any, Union
import numpy as np
import shapely.geometry, shapely.errors, shapely.strtree, shapely.ops, shapely.prepared
import gym_auv
from abc import ABC, abstractmethod
from gym_auv.objects.vessel.lidar_measurements import LidarMeasurements


class BaseLidarPreprocessor(ABC):
    @abstractmethod
    def preprocess(
        self, measurements: LidarMeasurements
    ) -> Union[LidarMeasurements, np.ndarray]:
        """Abstract method for preprocessing lidar measurements"""


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


