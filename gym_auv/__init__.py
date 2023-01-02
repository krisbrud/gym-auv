import numpy as np
from gym.envs.registration import register
from gym_auv.config import Config
from gym_auv.envs import (
    TestCrossing,
    TestCrossing1,
    TestScenario1,
    TestScenario2,
    TestScenario3,
    TestScenario4,
)
from gym_auv.envs.testscenario import DebugScenario, EmptyScenario, TestHeadOn

# import gym_auv.config.Config


DEFAULT_CONFIG = Config()

MOVING_CONFIG = Config()
# MOVING_CONFIG.vessel.sector_partition_fun = sector_partition_fun
# MOVING_CONFIG.vessel.observe_obstacle_fun = observe_obstacle_fun
# MOVING_CONFIG["observe_obstacle_fun"] = return_true_fun


DEBUG_CONFIG = Config()
DEBUG_CONFIG.simulation.t_step_size = 0.5
DEBUG_CONFIG.episode.min_goal_distance = 0.1
# DEBUG_CONFIG["t_step_size"] = 0.5
# DEBUG_CONFIG["min_goal_distance"] = 0.1

REALWORLD_CONFIG = Config()

PATHFOLLOW_CONFIG = Config()
PATHFOLLOW_CONFIG.vessel.sensing = False
# PATHFOLLOW_CONFIG.episode.min_cumulative_reward = -1000
# REALWORLD_CONFIG.simulation.t_step_size = 0.2

LOS_COLAV_CONFIG = Config()

# REALWORLD_CONFIG.vessel.render_distance = 300
# REALWORLD_CONFIG = DEFAULT_CONFIG.copy()
# REALWORLD_CONFIG["t_step_size"] = 0.2
# REALWORLD_CONFIG["render_distance"] = 300  # 2000
# REALWORLD_CONFIG["observe_frequency"] = 0.1

SCENARIOS = {
    "TestScenario1-v0": {
        "entry_point": "gym_auv.envs:TestScenario1",
        "constructor": TestScenario1,
        "config": DEFAULT_CONFIG,
    },
    "TestScenario2-v0": {
        "entry_point": "gym_auv.envs:TestScenario2",
        "constructor": TestScenario2,
        "config": DEFAULT_CONFIG,
    },
    "TestScenario3-v0": {
        "entry_point": "gym_auv.envs:TestScenario3",
        "constructor": TestScenario3,
        "config": DEFAULT_CONFIG,
    },
    "TestScenario4-v0": {
        "entry_point": "gym_auv.envs:TestScenario4",
        "constructor": TestScenario4,
        "config": DEFAULT_CONFIG,
    },
    "TestHeadOn-v0": {
        "entry_point": "gym_auv.envs:TestHeadOn",
        "constructor": TestHeadOn,
        "config": DEFAULT_CONFIG,
    },
    "TestCrossing-v0": {
        "entry_point": "gym_auv.envs:TestCrossing",
        "constructor": TestCrossing,
        "config": DEFAULT_CONFIG,
    },
    "TestCrossing1-v0": {
        "entry_point": "gym_auv.envs:TestCrossing1",
        "constructor": TestCrossing1,
        "config": DEFAULT_CONFIG,
    },
    "DebugScenario-v0": {
        "entry_point": "gym_auv.envs:DebugScenario",
        "constructor": DebugScenario,
        "config": DEBUG_CONFIG,
    },
    "EmptyScenario-v0": {
        "entry_point": "gym_auv.envs:EmptyScenario",
        "constructor": EmptyScenario,
        "config": DEBUG_CONFIG,
    },
    # 'Sorbuoya-v0': {
    #     'entry_point': 'gym_auv.envs:Sorbuoya',
    #     'config': REALWORLD_CONFIG
    # },
    # 'Agdenes-v0': {
    #     'entry_point': 'gym_auv.envs:Agdenes',
    #     'config': REALWORLD_CONFIG
    # },
    # 'Trondheim-v0': {
    #     'entry_point': 'gym_auv.envs:Trondheim',
    #     'config': REALWORLD_CONFIG
    # },
    # 'Trondheimsfjorden-v0': {
    #     'entry_point': 'gym_auv.envs:Trondheimsfjorden',
    #     'config': REALWORLD_CONFIG
    # },
    "MovingObstaclesNoRules-v0": {
        "entry_point": "gym_auv.envs:MovingObstaclesNoRules",
        "config": MOVING_CONFIG,
    },
    "MovingObstaclesSimpleRewarder-v0": {
        "entry_point": "gym_auv.envs:MovingObstaclesBasic",
        "config": MOVING_CONFIG,
    },
    "MovingObstaclesLosRewarder-v0": {
        "entry_point": "gym_auv.envs:MovingObstaclesLosRewarder",
        "config": LOS_COLAV_CONFIG,
    },
    "PathFollowNoObstacles-v0": {
        "entry_point": "gym_auv.envs:PathFollowNoObstacles",
        "config": PATHFOLLOW_CONFIG,
    },
    # "MovingObstaclesColreg-v0": {
    #     "entry_point": "gym_auv.envs:MovingObstaclesColreg",
    #     "config": MOVING_CONFIG,
    # },
    # 'FilmScenario-v0':  {
    #     'entry_point': 'gym_auv.envs:FilmScenario',
    #     'config': REALWORLD_CONFIG
    # },
}

for scenario in SCENARIOS:
    register(
        id=scenario,
        entry_point=SCENARIOS[scenario]["entry_point"],
        kwargs={"env_config": SCENARIOS[scenario]["config"]},
    )
