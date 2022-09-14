# Configuration for gym_auv gym environment

from dataclasses import dataclass
import dataclasses
from functools import cached_property
from typing import Any, Callable, Union

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
    sensor_frequency: float = (
        1.0  # Sensor execution frequency (0.0 = never execute, 1.0 = always execute)
    )
    observe_frequency: float = (
        1.0  # Frequency of using actual obstacles instead of virtual ones for detection
    )


@dataclass
class VesselConfig:
    thrust_max_auv: float = 2.0  # Maximum thrust of the AUV [N]
    moment_max_auv: float = 0.15  # maximum moment applied to the AUV [Nm]
    vessel_width: float = 1.255  # Width of vessel [m]
    feasibility_width_multiplier: float = (
        5.0  # Multiplier for vessel width in feasibility pooling algorithm
    )
    look_ahead_distance: int = 300  # Path look-ahead distance for vessel [m]
    render_distance: Union[
        int, str
    ] = 300  # 3D rendering render distance, or "random" [m]
    sensing: bool = (
        True  # Whether rangerfinder sensors for perception should be activated
    )
    sensor_interval_load_obstacles: int = 25  # Interval for loading nearby obstacles
    n_sensors_per_sector: int = 20  # Number of rangefinder sensors within each sector
    n_sectors: int = 9  # Number of sensor sectors
    sensor_use_feasibility_pooling: bool = False  # Whether to use the Feasibility pooling preprocessing for LiDAR measurements
    sector_partition_fun: Callable[
        [Any, int], int
    ] = sector_partition_fun  # Function that returns corresponding sector for a given sensor index
    sensor_rotation: bool = False  # Whether to activate the sectors in a rotating pattern (for performance reasons)
    sensor_range: float = 150.0  # Range of rangefinder sensors [m]
    sensor_log_transform: bool = True  # Whether to use a log. transform when calculating closeness                 #
    observe_obstacle_fun: Callable[
        [int, float], bool
    ] = observe_obstacle_fun  # Function that outputs whether an obstacle should be observed (True),
    # or if a virtual obstacle based on the latest reading should be used (False).
    # This represents a trade-off between sensor accuracy and computation speed.
    # With real-world terrain, using virtual obstacles is critical for performance.
    use_dict_observation: bool = True

    @cached_property
    def n_sensors(self) -> int:
        # Calculates the number of sensors in total
        return self.n_sensors_per_sector * self.n_sectors


@dataclass
class RenderingConfig:
    show_indicators: bool = (
        True  # Whether to show debug information on screen during 2d rendering.
    )
    autocamera3d: bool = (
        True  # Whether to let the camera automatically rotate during 3d rendering
    )


@dataclass
class Config:
    episode: EpisodeConfig = EpisodeConfig()
    simulation: SimulationConfig = SimulationConfig()
    vessel: VesselConfig = VesselConfig()
    rendering: RenderingConfig = RenderingConfig()

    def __iter__(self):
        return iter(dataclasses.fields(self))
