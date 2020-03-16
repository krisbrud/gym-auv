import numpy as np

import gym_auv.utils.geomutils as geom
from gym_auv.objects.auv import AUV2D
from gym_auv.objects.path import RandomCurveThroughOrigin, ParamCurve
from gym_auv.objects.obstacles import PolygonObstacle, VesselObstacle, CircularObstacle
from gym_auv.environment import BaseShipScenario
import shapely.geometry, shapely.errors

import os 
dir_path = os.path.dirname(os.path.realpath(__file__))

class MovingObstacles(BaseShipScenario):

    def __init__(self, env_config, render_mode, test_mode):
        super().__init__(env_config, test_mode, render_mode, detect_moving=True)

    def generate(self):
        # Initializing path
        nwaypoints = int(np.floor(4*self.np_random.rand() + 2))
        self.path = RandomCurveThroughOrigin(self.np_random, nwaypoints, length=800)

        # Initializing vessel
        init_pos = self.path(0)
        init_angle = self.path.get_direction(0)
        init_pos[0] += 50*(self.np_random.rand()-0.5)
        init_pos[1] += 50*(self.np_random.rand()-0.5)
        init_angle = geom.princip(init_angle + 2*np.pi*(self.np_random.rand()-0.5))
        self.vessel = AUV2D(self.config["t_step_size"], np.hstack([init_pos, init_angle]), width=self.config["vessel_width"], adaptive_step_size=self.config["adaptive_step_size"])
        prog = 0
        self.path_prog_hist = np.array([prog])
        self.max_path_prog = prog
        
        self.obstacles = []

        # Adding moving obstacles
        for _ in range(100):
            other_vessel_trajectory = []

            obst_position, obst_radius = self._generate_obstacle(obst_radius_mean=10, displacement_dist_std=500)
            obst_direction = np.random.random()*2*np.pi
            obst_speed = np.random.random()*1

            for i in range(10000):
                other_vessel_trajectory.append((i, (
                    obst_position[0] + i*obst_speed*np.cos(obst_direction), 
                    obst_position[1] + i*obst_speed*np.sin(obst_direction)
                )))
            other_vessel_obstacle = VesselObstacle(width=obst_radius, trajectory=other_vessel_trajectory)

            self.obstacles.append(other_vessel_obstacle)

        # Adding static obstacles
        for _ in range(25):
            obstacle = CircularObstacle(*self._generate_obstacle())
            self.obstacles.append(obstacle)

        self.update()

