import numpy as np
import pytest

from gym_auv.objects.rewarder import los_path_reward

def test_los_path_reward():
    # Test that the reward is calculated correctly
    north_west_unit_vector = np.array([1, -1]) / np.sqrt(2)

    velocity_north_west = 1.0 * north_west_unit_vector
    velocity_north_east = np.array([1, 1]) / np.sqrt(2)

    assert np.allclose(1.0, los_path_reward(velocity_north_west, north_west_unit_vector))
    assert np.allclose(0.0, los_path_reward(velocity_north_east, north_west_unit_vector))  