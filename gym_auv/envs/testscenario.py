import numpy as np

from gym_auv.envs.pathcolav import PathColavEnv
import gym_auv.utils.geomutils as geom
from gym_auv.objects.auv import AUV2D
from gym_auv.objects.path import RandomCurveThroughOrigin, ParamCurve
from gym_auv.objects.obstacles import StaticObstacle
from gym_auv.environment import BaseShipScenario

class TestScenario1(PathColavEnv):
    def generate(self):
        self.path = ParamCurve([[0, 100], [0, 100]])

        init_pos = self.path(0)
        init_angle = self.path.get_direction(0)

        self.vessel = AUV2D(self.config["t_step_size"], np.hstack([init_pos, init_angle]))
        prog = self.path.get_closest_arclength(self.vessel.position)
        self.path_prog = np.array([prog])
        self.max_path_prog = prog

        n_obstacles = 1
        for o in range(n_obstacles):
            obst_position = self.path(max(50, (0.15 + 0.6*o/n_obstacles)*self.path.s_max))
            obst_radius = 20*(np.sqrt(o) + 1)
            self.obstacles.append(StaticObstacle(obst_position, obst_radius))
