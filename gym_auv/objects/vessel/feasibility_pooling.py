import numpy as np
import shapely.geometry, shapely.errors, shapely.strtree, shapely.ops, shapely.prepared
import gym_auv
from abc import ABC, abstractmethod
from gym_auv.objects.vessel.lidar_measurements import LidarMeasurements


class FeasPoolPreprocessor:
    # Currently not guaranteed to work, as it is not used in my (Kristian's) project.
    # Left here in case someone wants to expand on this code base and use this in the
    # future.
    """FeasPoolPreprocessor reduces the dimensionality of the Lidar measurements through feasibility pooling"""

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
                    lambda x: FeasPoolPreprocessor._feasibility_pooling(
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
