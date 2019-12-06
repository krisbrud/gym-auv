import numpy as np
import gym
from gym import spaces

from gym_auv.envs.pathcolav import PathColavEnv
import gym_auv.utils.geomutils as geom
from gym_auv.objects.auv import AUV2D, AUVPerfectFollower
from gym_auv.objects.path import RandomCurveThroughOrigin, ParamCurve
from gym_auv.objects.obstacles import StaticObstacle
from gym_auv.environment import BaseShipScenario

class PathGenerationEnv(BaseShipScenario):
    def __init__(self, env_config, test_mode):
        super().__init__(env_config, test_mode)
        self.action_space = gym.spaces.Box(
            low=np.array([-40, -40, -5, 1]),
            high=np.array([40, 40, 5, 20]),
            dtype=np.float32
        )
        
        low_obs = [-1]*self.n_observations
        high_obs = [1]*self.n_observations
        low_obs[self.nstates - 1] = -10000
        high_obs[self.nstates - 1] = 10000
        self.observation_space = gym.spaces.Box(
            low=np.array(low_obs),
            high=np.array(high_obs),
            dtype=np.float32
        )

    def generate(self):
        super().generate()
        init_pos = self.path(0)
        init_angle = self.path.get_direction(0)
        init_pos[0] += 50*(self.np_random.rand()-0.5)
        init_pos[1] += 50*(self.np_random.rand()-0.5)
        init_angle = geom.princip(init_angle + 2*np.pi*(self.np_random.rand()-0.5))

        self.vessel = AUVPerfectFollower(self.config["t_step_size"], np.hstack([init_pos, init_angle]), width=4)

    def step_reward(self, action):
        step_reward = 0

        reward_dt = 1*action[3]
        reward_cte = -1*self.past_errors['d_cross_track'][-1]
        reward_prog = 1*(self.path_prog[-1] - self.path_prog[-2])

        step_reward += reward_dt
        step_reward += reward_cte
        step_reward += reward_prog

        done = False
        info = {}

        return done, step_reward, info