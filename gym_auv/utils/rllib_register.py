# import gym
# from ray.tune import register_env

# import gym_auv


# def register_gym_auv_envs():
#     # Register the gym_auv scenarios so they may be used from rllib
#     scenarios = gym_auv.SCENARIOS

#     for name in scenarios.keys():
#         register_env(name, rllib_env_creator)
