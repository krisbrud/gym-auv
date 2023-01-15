# %% 
import gym
import gym_auv
import copy

env_name = "MovingObstaclesLosRewarder-v0" 

gym_auv_config = copy.deepcopy(gym_auv.MOVING_CONFIG)
gym_auv_config.episode.use_terminated_truncated_step_api = True

env = gym.make(env_name, env_config=gym_auv_config)

# obs = env.reset()

# Make lunar lander env with human rendering
# env = gym.make("LunarLander-v2", render_mode="rgb_array") # , render_mode="human")
# env = gym.make("LunarLander-v2", render_mode="human")

# # Make it a playable game
from gym.utils.play import play, PlayPlot
import numpy as np

# def callback(obs_t, obs_tp1, action, rew, terminated, truncated, info):
#     return [rew,]
# plotter = PlayPlot(callback, 150, ["reward"])

keys_to_action = {
    "w": np.array([1.0, 0]),
    "a": np.array([0, 1.0]),
    "d": np.array([0, -1.0]),
    "s": np.array([0, 0]),
    "wa": np.array([1.0, 1.0]),
    "wd": np.array([1.0, -1.0]),
}

noop = np.array([0, 0])

play(env, keys_to_action=keys_to_action, fps=10, noop=noop), 
# callback=plotter.callback)


# %%
