# Configuration for gym_auv gym environment

from dataclasses import dataclass
import dataclasses
from functools import cached_property
import math
from typing import Any, Callable, List, Tuple, Union

# import gym_auv
from gym_auv.utils.observe_functions import observe_obstacle_fun
from gym_auv.utils.sector_partitioning import sector_partition_fun


@dataclass
class EpisodeConfig:
    # ---- EPISODE ---- #
    min_cumulative_reward: float = float(
        -2000
    )  # Minimum cumulative reward received before episode ends
    max_timesteps: int = 10000  # Maximum amount of timesteps before episode ends
    min_goal_distance: float = float(
        5
    )  # Minimum aboslute distance to the goal position before episode ends
    min_path_progress: float = 0.99  # Minimum path progress before scenario is considered successful and the episode ended


@dataclass
class SimulationConfig:
    t_step_size: float = 1.0  # Length of simulation timestep [s]
    # sensor_frequency: float = (
    #     1.0  # Sensor execution frequency (0.0 = never execute, 1.0 = always execute)
    # )
    observe_frequency: float = (
        1.0  # Frequency of using actual obstacles instead of virtual ones for detection
    )
    sensor_interval_load_obstacles: int = 25  # Interval for loading nearby obstacles


@dataclass
class VesselConfig:
    thrust_max_auv: float = 2.0  # Maximum thrust of the AUV [N]
    moment_max_auv: float = 0.15  # maximum moment applied to the AUV [Nm]
    vessel_width: float = 1.255  # Width of vessel [m]
    look_ahead_distance: int = 300  # Path look-ahead distance for vessel [m]


@dataclass
class SensorConfig:
    use_dict_observation: bool = True

    n_lidar_rays: int = 180
    range: float = 150.0  # Range of rangefinder sensors [m]
    apply_log_transform: bool = False  # Whether to use a log. transform when calculating closeness                 #

    # use_relative_vectors: bool = True
    observe_proprioceptive: bool = True  # Whether to include navigation states (surge, sway, yaw rate)
    observe_cross_track_error: bool = True  # Whether to include cross-track error in the observation
    observe_heading_error: bool = True  # Whether to include heading error in observation
    observe_la_heading_error: bool = True  # Whether to include the look-ahead heading error in the observation
    use_lidar: bool = (
        True
        # Whether rangefinder sensors for perception should be activated
    )
    use_occupancy_grid: bool = True
    use_velocity_observations: bool = False
    occupancy_grid_size: int = 64
    observe_obstacle_fun: Callable[
        [int, float], bool
    ] = observe_obstacle_fun  
    # Function that outputs whether an obstacle should be observed (True),
    # or if a virtual obstacle based on the latest reading should be used (False).
    # This represents a trade-off between sensor accuracy and computation speed.
    # With real-world terrain, using virtual obstacles is critical for performance.
    

    @property
    def lidar_shape(self) -> Tuple[int, int]:
        lidar_channels = 1

        if self.use_velocity_observations:
            lidar_channels = 3

        return (lidar_channels, self.n_lidar_rays)

    @property
    def n_lidar_observations(self) -> int:
        return math.prod(self.lidar_shape)

    @property
    def dense_observation_size(self) -> int:
        n_dense_observations = 0

        if self.observe_proprioceptive:
            n_dense_observations += 3
        
        if self.observe_cross_track_error:
            n_dense_observations += 1

        if self.observe_heading_error:
            n_dense_observations += 1

        if self.observe_la_heading_error:
            n_dense_observations += 1

        return n_dense_observations
    


@dataclass
class RenderingConfig:
    pass
    # show_indicators: bool = (
    #     True  # Whether to show debug information on screen during 2d rendering.
    # )
    # autocamera3d: bool = (
    #     True  # Whether to let the camera automatically rotate during 3d rendering
    # )
    # render_distance: Union[
    #     int, str
    # ] = 300  # 3D rendering render distance, or "random" [m]


@dataclass
class Config:
    episode: EpisodeConfig = EpisodeConfig()
    sensor: SensorConfig = SensorConfig()
    simulation: SimulationConfig = SimulationConfig()
    vessel: VesselConfig = VesselConfig()
    rendering: RenderingConfig = RenderingConfig()

    def __iter__(self):
        return iter(dataclasses.fields(self))
