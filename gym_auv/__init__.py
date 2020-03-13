import numpy as np
from gym.envs.registration import register
from functools import partial

def sample_lambda(scale):
    log = -np.random.gamma(1, scale)
    y = np.power(10, log)
    return y

def observe_obstacle_fun(t, dist):
    return t % (int(0.0025*dist**1.7) + 1) == 0

DEFAULT_CONFIG = {
    # ---- REWARD ---- #
    "reward_lambda": partial(sample_lambda, 2),     # Function that returns new (random) lambda value
    "reward_gamma_theta": 10,                       # Reward parameter for obstacle angle
    "reward_gamma_x": 0.005,                        # Reward parameter for obstacle distance
    "reward_gamma_y_e": 0.05,                       # Reward parameter for cross-track error
    "reward_speed": 1.0,                            # Reward parameter for speed
    "penalty_yawrate": 0.5,                         # Penalty parameter for yaw rate
    "cruise_speed": 2,                              # Ideal vessel speed [m/s]

    # ---- EPISODE ---- #
    "min_cumulative_reward": -2000,                 # Minimum cumulative reward received before episode ends
    "max_timestemps": 10000,                        # Maximum amount of timesteps before episode ends
    "max_distance": 250,                            # Maximum distance from path before episode ends
    "min_goal_distance": 5,                         # Minimum aboslute distance to the goal position before episode ends 
    "min_goal_progress": 2,                         # Minimum path distance to the goal position before episode ends
    "end_on_collision": True,                       # Whether to end the episode upon collision
    "teleport_on_collision": False,                 # Whether to teleport the vessel back in time on collision
    
    # ---- SIMULATION ---- #
    "t_step_size": 1.0,                             # Length of simulation timestep [s]
    "adaptive_step_size": False,                    # Whether to use an adaptive step-size Runga-Kutta method 

    # ---- VESSEL ---- #
    "vessel_width": 4,                              # Width of vessel [m]
    "look_ahead_distance": 100,                     # Path look-ahead distance for vessel         
    "sensor_interval_obstacles": 2,                 # Interval for simulating rangefinder sensors
    "update_interval_path": 1,                      # Interval for updating path following-related variables
    "sensor_interval_load_obstacles": 100,          # Interval for loading nearby obstacles
    "n_sensors_per_sector": 7,                      # Number of rangefinder sensors within each sector
    "n_sectors": 25,                                # Number of sensor sectors
    "lidar_rotation": False,                        # Whether to activate the sectors in a rotating pattern (for performance reasons)
    "lidar_range": 150,                             # Range of rangefinder sensors [m]
    "lidar_range_log_transform": True,              # Whether to use a log. transform when calculating closeness                 #
    "observe_obstacle_fun": observe_obstacle_fun,   # Function that outputs whether an obstacle should be observed (True),
                                                    # or if a virtual obstacle based on the latest reading should be used (False).
                                                    # This represents a trade-off between sensor accuracy and computation speed.
                                                    # With real-world terrain, using virtual obstacles is critical for performance.
    
    # ---- RENDERING ---- #
    "show_indicators": False,                       # Whether to show debug information on screen during 2d rendering.
    'autocamera3d': False                           # Whether to let the camera automatically rotate during 3d rendering
}

MOVING_CONFIG = DEFAULT_CONFIG.copy()
MOVING_CONFIG['reward_lambda'] = partial(sample_lambda, 0.2)
MOVING_CONFIG['min_reward'] = -1000
MOVING_CONFIG['t_step_size'] = 1.0 #0.3

DEBUG_CONFIG = DEFAULT_CONFIG.copy()
DEBUG_CONFIG['t_step_size'] = 0.5
DEBUG_CONFIG['min_goal_distance'] = 0.1
#DEBUG_CONFIG['n_sensors_per_sector'] = 1

REALWORLD_CONFIG = DEFAULT_CONFIG.copy()
REALWORLD_CONFIG['t_step_size'] = 1.0#0.2#0.1
#REALWORLD_CONFIG["lidar_range"] = 500
#REALWORLD_CONFIG['look_ahead_distance'] = 1500


SCENARIOS = {
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
        'config': DEBUG_CONFIG
    },
    'EmptyScenario-v0': {
        'entry_point': 'gym_auv.envs:EmptyScenario',
        'config': DEBUG_CONFIG
    },
    'Sorbuoya-v0': {
        'entry_point': 'gym_auv.envs:Sorbuoya',
        'config': REALWORLD_CONFIG
    },
    'Trondheimsfjorden-v0': {
        'entry_point': 'gym_auv.envs:Trondheimsfjorden',
        'config': REALWORLD_CONFIG
    },
    'Trondheim-v0': {
        'entry_point': 'gym_auv.envs:Trondheim',
        'config': REALWORLD_CONFIG
    },
    'MovingObstacles-v0': {
        'entry_point': 'gym_auv.envs:MovingObstacles',
        'config': MOVING_CONFIG
    }
}

for scenario in SCENARIOS:
    register(
        id=scenario,
        entry_point=SCENARIOS[scenario]['entry_point'],
        #kwargs={'env_config': SCENARIOS[scenario]['config']}
    )