import numpy as np
import gym.spaces

from gym_auv.utils.clip_to_space import clip_to_space

# def _get_mock_


def test_clip_box():
    obs = np.array([1.3, -0.2])
    space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,))

    clipped_obs = clip_to_space(obs, space)
    expected_obs = np.array([1.0, -0.2])

    assert np.allclose(clipped_obs, expected_obs)


def test_clip_dict():
    # Test that a dict observation is clipped correctly according to keys
    obs = {"foo": np.array([1.3, 0.2, 0.1]), "bar": np.array([-1.2])}
    space = gym.spaces.Dict(
        {
            "foo": gym.spaces.Box(low=-1.0, high=1, shape=(3,)),
            "bar": gym.spaces.Box(low=-0.5, high=0.5, shape=(1,)),
        }
    )

    clipped_obs = clip_to_space(obs, space)
    expected_obs = {"foo": np.array([1.0, 0.2, 0.1]), "bar": np.array([-0.5])}

    for key in expected_obs.keys():
        assert np.allclose(clipped_obs[key], expected_obs[key])
