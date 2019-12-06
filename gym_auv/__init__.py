import numpy as np
from gym.envs.registration import register

def sample_lambda():
    log = -np.random.gamma(1, 2)
    y = np.power(10, log)
    return y

DEFAULT_CONFIG = {
    "reward_ds": 0.5,#1.5,
    "penalty_negative_ds": 0,#3,
    "reward_speed_error": 0,#-0.08,
    "reward_la_heading_error": 0,#-0.01,
    "reward_heading_error": 0,#-0.01,
    "reward_cross_track_error": -0.005,#1,
    "reward_d_cross_track_error": 0,#-10,
    "reward_lambda": sample_lambda,
    "reward_gamma_theta": 4,
    "reward_gamma_x": 0.005,
    "reward_gamma_y_e": 0.05,
    "reward_closeness": 1,
    "reward_speed": 0.5,
    "penalty_collision": 100,
    "penalty_rudder_angle_change": 0.2,
    "penalty_rudder_angle": 0.2,
    "min_reward": -2000,
    "max_distance": 250,
    "living_penalty": 0.7,
    "callable_update_interval": None,
    "cross_track_error_sigma": 50,
    "max_closest_point_distance": 10,
    "max_closest_point_heading_error": np.pi/6,
    "nobstacles": 20,
    "obst_reward_range": 18,
    "t_step_size": 0.14,
    "adaptive_step_size": False,
    "cruise_speed": 2,
    "min_la_dist": 100,
    "goal_dist": 800,
    "min_goal_closeness": 5,
    "min_goal_progress": 2,
    "end_on_collision": True,
    "teleport_on_collision": False,
    "max_timestemps": 10000,
    "sensor_interval_obstacles": 1,
    "sensor_noise_std": 0,
    "sector_clear_column": False,
    "sensor_interval_path": 10,
    "closeness_sector_delay": 0,
    "n_sensors_per_sector": 9,
    "n_sectors": 15,
    "n_rings": 9,
    "detection_grid": False,
    "lidars": True,
    "lidar_rotation": False,
    "lidar_range": 150,
    "lidar_range_log_transform": True,
    "security": False,
    "security_margin": 0,
    "security_smoothing_factor": 0.95,
    "rear_detection": False,
    "sensor_convolution_sigma": 0,
    "save_measurements": False,
    "show_indicators": True
}

SCENARIOS = {
    'Colav-v0': {
        'entry_point': 'gym_auv.envs:ColavEnv',
        'config': DEFAULT_CONFIG
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
    'PathGeneration-v0': {
        'entry_point': 'gym_auv.envs:PathGenerationEnv',
        'config': dict(n_obstacles=0, **DEFAULT_CONFIG)
    },
    'TestScenario1-v0': {
        'entry_point': 'gym_auv.envs:TestScenario1',
        'config': DEFAULT_CONFIG
    },
    'TestScenario2-v0': {
        'entry_point': 'gym_auv.envs:TestScenario2',
        'config': DEFAULT_CONFIG
    },
    'TestScenario3-v0': {
        'entry_point': 'gym_auv.envs:TestScenario3',
        'config': DEFAULT_CONFIG
    },
    'TestScenario4-v0': {
        'entry_point': 'gym_auv.envs:TestScenario4',
        'config': DEFAULT_CONFIG
    },
    'DebugScenario-v0': {
        'entry_point': 'gym_auv.envs:DebugScenario',
        'config': DEFAULT_CONFIG
    },
    'PathColavControl-v0': {
        'entry_point': 'gym_auv.envs:PathColavControlEnv',
        'config': DEFAULT_CONFIG
    }
}

for scenario in SCENARIOS:
    register(
        id=scenario,
        entry_point=SCENARIOS[scenario]['entry_point'],
        #kwargs={'env_config': SCENARIOS[scenario]['config']}
    )