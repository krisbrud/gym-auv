# %%
import sys
sys.path.append('..')

import os
import pickle
import glob
import numpy as np
import gym_auv
from matplotlib import pyplot as plt
import seaborn as sns

# Import all pkl files in subdirectory play_episodes
episode_paths = glob.glob("./play_episodes/*.pkl") # Get all pkl files in subdirectory play_episodes

episodes = {}

for episode_path in episode_paths:
    episode_name = os.path.basename(episode_path)[:-4]  # Remove .pkl from the end
    with open(episode_path, "rb") as f:
        episodes[episode_name] = pickle.load(f)

# %%

from gym_auv.objects.rewarder import LosColavParams, LOSColavRewarder
params = LosColavParams()
rewarder = LOSColavRewarder()

"""
# Default parameters
class LosColavParams(RewarderParams):
    def __init__(self):
        self.gamma_theta = 10.0
        self.gamma_x = 0.1  # 0.1
        self.gamma_v_y = 1.0
        self.gamma_y_e = 5.0
        self.penalty_yawrate = 0  # 10.0
        self.penalty_torque_change = 0.0
        self.cruise_speed = 0.1
        self.neutral_speed = 0.05
        self.collision = -500  # -2000.0 #  -10000.0
        self.lambda_ = 0.6  # 0.5
        self.eta = 0
        self.negative_multiplier = 2
        self.reward_scale = 0.5
"""

params.lambda_ = 0.6  # Colav/pathfollow trade off
params.gamma_x = 0.1  # Distance weight
params.gamma_v_y = 1.0  # Speed weight towards target
params.gamma_y_e = 5.0   
params.gamma_theta = 10.0  # Weighting of penalty for angle
params.path_reward_scale = 2.5 
params.colav_reward_scale = 0.2
params.negative_multiplier = 2.0


use_tanh = True
if use_tanh:
    elementwise_transform = lambda x: np.tanh(x) 
else:
    elementwise_transform = lambda x: x

# Calculate rewards for all episodes
rewards = {}
for episode_name, episode in episodes.items():
    rewards[episode_name] = []
    for step_idx, vessel_data in enumerate(episode):
        reward = rewarder.calculate(vessel_data=vessel_data, params=params)
        transformed_reward = elementwise_transform(reward)
        rewards[episode_name].append(transformed_reward)


# Calculate cumulative rewards for all episodes
cumulative_rewards = {key: np.cumsum(rewards[key]) for key, reward in rewards.items()}
def prettify_name(name):
    """Prettify episode name, by removing snake case and replacing with spaces"""    
    return name.replace("_", " ").capitalize()

plt.style.use('ggplot')

# Plot cumulative rewards for all episodes, with episode name as label using seaborn

for episode_name, cumulative_reward in cumulative_rewards.items():
    pretty_name = prettify_name(episode_name)

    # Color code the episodes according to the cumulative reward at the end using a cmap


    sns.lineplot(x=range(len(cumulative_reward)), y=cumulative_reward, label=episode_name)

plt.xlabel("Time step")
plt.ylabel(f"Cumulative reward {'(tanh)' if use_tanh else ''}")


# %%

def _make_df_from_cumulative_rewards(cumulative_rewards):
    """Make rows from each element of each dictionary in the cumulative rewards dictionary"""
    import pandas as pd
    df = pd.DataFrame()
    for episode_name, cumulative_reward in cumulative_rewards.items():
        df_episode = pd.DataFrame(cumulative_reward, columns=["time_step", "cumulative_reward"])
        df_episode["episode"] = episode_name
        df = df.append(df_episode, ignore_index=True)
    return df

# %%
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap
import pandas as pd

# Create a dataframe from the cumulative_rewards list
df = pd.DataFrame({'episode': [k for k in cumulative_rewards.keys()], 
                   'cumulative_reward': [v[-1] for v in cumulative_rewards.values()]})

# Sort the dataframe by cumulative_reward
df = df.sort_values(by='cumulative_reward')

# Set ggplot style
plt.style.use('ggplot')
# sns.set_style("ggplot")

# Create colormap
# cmap = ListedColormap(['red','blue'], N=10)

# Plot the data using matplotlib
plt.scatter(df['episode'], df['cumulative_reward'], c=df['cumulative_reward'], cmap=cmap)
plt.xlabel('Episode')
plt.ylabel('Cumulative Reward')
plt.colorbar()
plt.show()

# %%
