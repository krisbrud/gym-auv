import pytest
import numpy as np

import gym
import gym_auv


def _make_env(env_name) -> gym.Env:
    return gym.make(env_name)


@pytest.mark.parametrize("scenario_name", list(gym_auv.SCENARIOS.keys()))
def test_single_step(scenario_name):
    """Simple end-to-end test of environment"""
    # Do a single non-zero action in the environment, see that the observation changes
    env = _make_env(scenario_name)
    first_obs = env.reset()  # Reset

    mock_action = np.array([0.5, 0.6])
    obs, reward, done, info = env.step(mock_action)

    # Assert that the observations are within valid range
    assert np.all(env.observation_space.low <= obs)
    assert np.all(env.observation_space.high >= obs)

    # Check that other values are of correct type
    assert isinstance(reward, float)
    assert isinstance(done, bool)
    assert isinstance(info, dict)

    # Check that the new observation is different from the previous one.
    # As the observation includes navigation features (velocities etc), this should be true
    assert np.any(first_obs != obs)
