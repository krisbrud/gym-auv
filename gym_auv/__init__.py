import numpy as np
from gym.envs.registration import register

DEFAULT_CONFIG = {
    "reward_ds": 3,
    "penalty_negative_ds": 3,
    "reward_speed_error": -1,
    "reward_la_heading_error": 0,
    "reward_heading_error": -1,
    "reward_cross_track_error": 2,
    "reward_d_cross_track_error": -200,
    "reward_closeness": -0.0001,
    "reward_collision": -10,
    "reward_rudderchange": -2,
    "living_penalty": -3,
    "max_closest_point_distance": 10,
    "max_closest_point_heading_error": np.pi/6,
    "nobstacles": 20,
    "lidar_range": 100,
    "lidar_range_log_transform": True,
    "obst_reward_range": 9,
    "t_step_size": 0.1,
    "cruise_speed": 2,
    "min_la_dist": 50,
    "goal_dist": 800,
    "min_reward": -10000,
    "end_on_collision": False,
    "max_timestemps": 10000,
    "sensor_interval_obstacles": 1,
    "sensor_interval_path": 100,
    "n_sensors_per_sector": 9,
    "n_sectors": 25,
    "n_rings": 9,
    "detection_grid": False,
    "lidars": True,
    "lidar_rotation": False,
    "rear_detection": False,
    "sensor_convolution_sigma": 1
}

SCENARIOS = {
    'Colav-v0': {
        'entry_point': 'gym_auv.envs:ColavEnv',
        'config': {
            "reward_ds": 1,
            "reward_closeness": -0.5,
            "reward_speed_error": -0.08,
            "reward_collision": -1000,
            "nobstacles": 20,
            "lidar_range": 40,
            "obst_reward_range": 15,
            "t_step_size": 0.1,
            "cruise_speed": 1.5,
            "goal_dist": 400,
            "reward_rudderchange": 0,
            "min_reward": -500,
            "end_on_collision": True,
            "max_timestemps": 10000,
            "sensor_interval_obstacles": 20,
            "include_sensor_deltas": False,
            "n_sensors": 4,
        }
    },
    'PathFollowing-v0': {
        'entry_point': 'gym_auv.envs:PathFollowingEnv',
        'config': {
            "reward_ds": 1,
            "reward_speed_error": -0.08,
            "reward_cross_track_error": -0.5,
            "t_step_size": 0.1,
            "cruise_speed": 1.5,
            "la_dist": 10,
            "goal_dist": 400,
            "reward_rudderchange": 0,
            "min_reward": -500,
            "max_timestemps": 10000,
            "n_sensors": 0,
        }
    },
    'PathColav-v0': {
        'entry_point': 'gym_auv.envs:PathColavEnv',
        'config': DEFAULT_CONFIG
    },
    'TestScenario1-v0': {
        'entry_point': 'gym_auv.envs:TestScenario1',
        'config': DEFAULT_CONFIG
    }
}

for scenario in SCENARIOS:
    register(
        id=scenario,
        entry_point=SCENARIOS[scenario]['entry_point'],
        #kwargs={'env_config': SCENARIOS[scenario]['config']}
    )