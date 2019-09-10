import numpy as np

from gym_auv.envs.pathcolav import PathColavEnv
import gym_auv.utils.geomutils as geom
from gym_auv.objects.auv import AUV2D
from gym_auv.objects.path import RandomCurveThroughOrigin, ParamCurve
from gym_auv.objects.obstacles import StaticObstacle
from gym_auv.environment import BaseShipScenario

class TestScenario1(PathColavEnv):
    def generate(self):
        self.path = ParamCurve([[0, 500], [0, 500]])

        init_pos = self.path(0)
        init_angle = self.path.get_direction(0)

        self.vessel = AUV2D(self.config["t_step_size"], np.hstack([init_pos, init_angle]))
        prog = self.path.get_closest_arclength(self.vessel.position)
        self.path_prog = np.array([prog])
        self.max_path_prog = prog

        n_obstacles = 5
        obst_arclength = 30
        for o in range(n_obstacles):
            obst_radius = 10 + 10*o
            obst_arclength += obst_radius*2 + 30
            obst_position = self.path(obst_arclength)
            self.obstacles.append(StaticObstacle(obst_position, obst_radius))
