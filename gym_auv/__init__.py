import numpy as np
from gym.envs.registration import register

def observe_obstacle_fun(t, dist):
    return t % (int(0.0025*dist**1.7) + 1) == 0

def return_true_fun(t, dist):
    return True

def sector_partition_fun(env, isensor, c=0.1):
    a = env.config["n_sensors_per_sector"]*env.config["n_sectors"]
    b = env.config["n_sectors"]
    sigma = lambda x: b / (1 + np.exp((-x + a / 2) / (c * a)))
    return int(np.floor(sigma(isensor) - sigma(0)))

DEFAULT_CONFIG = {
    # ---- EPISODE ---- #
    "min_cumulative_reward": -2000,                 # Minimum cumulative reward received before episode ends
    "max_timesteps": 10000,                         # Maximum amount of timesteps before episode ends
    "min_goal_distance": 5,                         # Minimum aboslute distance to the goal position before episode ends
    
    # ---- SIMULATION ---- #
    "t_step_size": 1.0,                             # Length of simulation timestep [s]
    "sensor_frequency": 1.0,                        # Sensor execution frequency (0.0 = never execute, 1.0 = always execute)

    # ---- VESSEL ---- #
    "vessel_width": 4.0,                            # Width of vessel [m]
    "look_ahead_distance": 100,                     # Path look-ahead distance for vessel [m]
    "sensor_interval_load_obstacles": 25,           # Interval for loading nearby obstacles
    "n_sensors_per_sector": 29,                     # Number of rangefinder sensors within each sector
    "n_sectors": 9,                                 # Number of sensor sectors
    "sector_partition_fun": sector_partition_fun,   # Function that returns corresponding sector for a given sensor index
    "sensor_rotation": False,                       # Whether to activate the sectors in a rotating pattern (for performance reasons)
    "sensor_range": 150.0,                            # Range of rangefinder sensors [m]
    "sensor_log_transform": True,                   # Whether to use a log. transform when calculating closeness                 #
    "observe_obstacle_fun": observe_obstacle_fun,   # Function that outputs whether an obstacle should be observed (True),
                                                    # or if a virtual obstacle based on the latest reading should be used (False).
                                                    # This represents a trade-off between sensor accuracy and computation speed.
                                                    # With real-world terrain, using virtual obstacles is critical for performance.
    
    # ---- RENDERING ---- #
    "show_indicators": True,                        # Whether to show debug information on screen during 2d rendering.
    'autocamera3d': False                           # Whether to let the camera automatically rotate during 3d rendering
}

MOVING_CONFIG = DEFAULT_CONFIG.copy()
MOVING_CONFIG['observe_obstacle_fun'] = return_true_fun

DEBUG_CONFIG = DEFAULT_CONFIG.copy()
DEBUG_CONFIG['t_step_size'] = 0.5
DEBUG_CONFIG['min_goal_distance'] = 0.1

REALWORLD_CONFIG = DEFAULT_CONFIG.copy()
REALWORLD_CONFIG['t_step_size'] = 1.0

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