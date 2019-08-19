from gym.envs.registration import register

SCENARIOS = {
    'Colav-v0': {
        'entry_point': 'gym_auv.envs:ColavEnv',
        'config': {
            "reward_ds": 1,
            "reward_closeness": -0.5,
            "reward_speed_error": -0.08,
            "reward_collision": -5000,
            "nobstacles": 20,
            "obst_detection_range": 40,
            "obst_reward_range": 15,
            "t_step_size": 0.1,
            "cruise_speed": 1.5,
            "goal_dist": 400,
            "reward_rudderchange": 0,
            "min_reward": -500,
            "end_on_collision": True
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
            "min_reward": -500
        }
    },
    'PathColav-v0': {
        'entry_point': 'gym_auv.envs:PathColavEnv',
        'config': {
            "reward_ds": 1,
            "reward_speed_error": -0.08,
            "reward_cross_track_error": -0.5,
            "reward_closeness": -0.5,
            "reward_collision": -5000,
            "nobstacles": 20,
            "obst_detection_range": 40,
            "obst_reward_range": 15,
            "t_step_size": 0.1,
            "cruise_speed": 1.5,
            "la_dist": 10,
            "goal_dist": 400,
            "reward_rudderchange": 0,
            "min_reward": -500,
            "end_on_collision": True
        }
    }
}

for scenario in SCENARIOS:
    register(
        id=scenario,
        entry_point=SCENARIOS[scenario]['entry_point'],
        kwargs={'env_config': SCENARIOS[scenario]['config']}
    )