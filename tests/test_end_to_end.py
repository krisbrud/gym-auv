import pytest
import numpy as np

import gym
import gym.spaces
import gym_auv


def _make_env(env_name) -> gym.Env:
    return gym.make(env_name)


def _assert_all_within_space_limits(obs: np.ndarray, space: gym.spaces.Box):
    assert isinstance(obs, np.ndarray)

    assert np.all(space.low <= obs)
    assert np.all(space.high >= obs)


@pytest.mark.parametrize("scenario_name", list(gym_auv.SCENARIOS.keys()))
def test_single_step(scenario_name):
    """Simple end-to-end test of environment"""
    # Do a single non-zero action in the environment, see that the observation changes
    env = _make_env(scenario_name)
    first_obs = env.reset()  # Reset

    mock_action = np.array([0.5, 0.6])
    obs, reward, done, info = env.step(mock_action)

    obs_space = env.observation_space
    # Assert that the observations are within valid range
    if isinstance(obs_space, gym.spaces.Box):
        _assert_all_within_space_limits(obs, obs_space)
    elif isinstance(obs_space, gym.spaces.Dict):
        assert isinstance(obs, dict), "Env space is dict but observation is not!"

        # Check each observation
        for key in obs.keys():
            _assert_all_within_space_limits(obs[key], obs_space[key])
    else:
        raise TypeError(f"Unsupported observation space type {type(obs_space)}")

    # Check that other values are of correct type
    assert isinstance(reward, float)
    assert isinstance(done, bool)
    assert isinstance(info, dict)

    # Check that the new observation is different from the previous one.
    # As the observation includes navigation features (velocities etc), this should be true
    if isinstance(obs_space, gym.spaces.Box):
        assert np.any(first_obs != obs)
    elif isinstance(obs_space, gym.spaces.Dict):
        is_different = False
        for key in obs.keys():
            if np.any(obs[key] != first_obs[key]):
                is_different = True

        assert is_different


if __name__ == "__main__":
    test_single_step("TestScenario1-v0")
