from typing import Dict, Union

import numpy as np
import gym.spaces


def clip_to_space(
    obs: Union[np.ndarray, Dict[str, np.ndarray]], space: gym.spaces.Space
):
    # Normalize a vector or dict to the size defined by the observation spaces of type box or dict
    if isinstance(obs, np.ndarray):
        assert isinstance(
            space, gym.spaces.Box
        ), f"Got a np.ndarray observation, but the observation space was of type {type(space)}, not Box!"

        normalized_obs = np.clip(obs, space.low, space.high)
    elif isinstance(obs, dict):
        assert isinstance(
            space, gym.spaces.Dict
        ), f"Got a dict observation, but the observation space was of type {type(space)}, not Dict!"

        normalized_obs = {}
        for key, observation in obs.items():
            normalized_obs[key] = np.clip(observation, space[key].low, space[key].high)

    return normalized_obs
