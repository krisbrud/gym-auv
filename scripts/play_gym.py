# %% 
import gym
# import gym_auv
# Make lunar lander env with human rendering
# env = gym.make("LunarLander-v2", render_mode="rgb_array") # , render_mode="human")
env = gym.make("LunarLander-v2", render_mode="human")

# Make it a playable game
from gym.utils.play import play
keys_to_action = {
    "w": 2,
    "a": 1,
    "d": 3,
    "s": 0,
}

play(env, keys_to_action=keys_to_action, fps=10)

# obs = env.reset()
# env.render()
# input()
# %%

# # Plot the image
# import matplotlib.pyplot as plt
# plt.imshow(img)
# %%
