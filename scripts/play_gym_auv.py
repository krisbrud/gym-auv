import os
import gym
import gym_auv
import copy
import pickle

from gym.utils.play import play, PlayPlot
from gym_auv.reporting import plot_trajectory
import numpy as np


gym_auv_config = copy.deepcopy(gym_auv.LOS_COLAV_CONFIG)
gym_auv_config.episode.use_terminated_truncated_step_api = True
gym_auv_config.episode.return_latest_data_in_info = True

env_name = "MovingObstaclesLosRewarder-v0" 
# env_name = "TestScenario1-v0"
env = gym.make(env_name, env_config=gym_auv_config)


rewards = []
latest_data = []
done_indices = []
step_idx = 0

def callback(obs_t, obs_tp1, action, rew, terminated, truncated, info):
    global step_idx
    global rewards
    global done_indices
    global latest_data

    done = truncated or terminated
    if done:
        done_indices.append(step_idx)
    
    step_idx += 1

    rewards.append(rew)
    latest_data.append(copy.deepcopy(info["latest_data"]))

    return [rew,]

keys_to_action = {
    "w": np.array([1.0, 0]),
    "a": np.array([0, 1.0]),
    "d": np.array([0, -1.0]),
    "s": np.array([0, 0]),
    "wa": np.array([1.0, 1.0]),
    "wd": np.array([1.0, -1.0]),
}

noop = np.array([0, 0])  # Action if no key is pressed for a frame

play(env, keys_to_action=keys_to_action, fps=30, noop=noop, callback=callback)


# print()
# fig_dir = os.path.join(os.path.dirname(__file__), "test_plots")

# # env_copy = copy.deepcopy(env)
# plot_trajectory(env, fig_dir)


def get_file_path(file_name):
    if not file_name.endswith(".pkl"):
        file_name += ".pkl"

    subdir_name = "play_episodes"  # Subdirectory of the directory this script is in

    # construct the path to the subdirectory
    subdir_path = os.path.join(os.path.dirname(__file__), subdir_name)

    # construct the path to the file in the subdirectory
    file_path = os.path.join(subdir_path, file_name)

    return file_path

file_name = input("Enter file name for saving episode data (press enter to not save): ").strip()
if file_name: # If not empty or only whitespace (removed by strip)
    print("Only saving first episode. Run the script again to save more episodes.")

    file_path = get_file_path(file_name)

    if len(done_indices) > 0:
        episode_latest_data = latest_data[:done_indices[0]]
    else:
        episode_latest_data = latest_data

    with open(file_path, "wb") as f:
        pickle.dump(episode_latest_data, f)