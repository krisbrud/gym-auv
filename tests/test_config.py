import pytest
import gym_auv
from gym_auv.envs.testscenario import TestScenario1

@pytest.mark.parametrize("use_lidar_velocity", [True, False])
def test_use_lidar_velocity(use_lidar_velocity):
    config = gym_auv.Config()
    config.vessel.sensor_use_velocity_observations = use_lidar_velocity
    env = TestScenario1(env_config=config)
    obs = env.reset()

    expected_obs_shape = (config.vessel.dense_observation_size + config.vessel.n_lidar_observations,)
    assert obs.shape == expected_obs_shape, f"{obs.shape = } != {expected_obs_shape = }"