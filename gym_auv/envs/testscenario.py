import numpy as np

from gym_auv.envs.pathcolav import PathColavEnv
import gym_auv.utils.geomutils as geom
from gym_auv.objects.auv import AUV2D
from gym_auv.objects.path import RandomCurveThroughOrigin, ParamCurve
from gym_auv.objects.obstacles import StaticObstacle
from gym_auv.environment import BaseShipScenario

class TestScenario1(PathColavEnv):
    def generate(self):
        self.path = ParamCurve([[0, 1100], [0, 1100]])

        init_pos = self.path(0)
        init_angle = self.path.get_direction(0)

        self.vessel = AUV2D(self.config["t_step_size"], np.hstack([init_pos, init_angle]))
        prog = self.path.get_closest_arclength(self.vessel.position)
        self.path_prog = np.array([prog])
        self.max_path_prog = prog

        n_obstacles = 7
        obst_arclength = 30
        for o in range(n_obstacles):
            obst_radius = 10 + 10*o**1.5
            obst_arclength += obst_radius*2 + 30
            obst_position = self.path(obst_arclength)
            self.obstacles.append(StaticObstacle(obst_position, obst_radius))

class TestScenario2(PathColavEnv):
    def generate(self):

        waypoint_array = []
        for t in range(500):
            x = t*np.cos(t/100)
            y = 2*t
            waypoint_array.append([x, y])

        waypoints = np.vstack(waypoint_array).T
        self.path = ParamCurve(waypoints)

        init_pos = self.path(0)
        init_angle = self.path.get_direction(0)

        self.vessel = AUV2D(self.config["t_step_size"], np.hstack([init_pos, init_angle]))
        prog = self.path.get_closest_arclength(self.vessel.position)
        self.path_prog = np.array([prog])
        self.max_path_prog = prog

        obst_arclength = 30
        obst_radius = 5
        while True:
            obst_arclength += 2*obst_radius
            if (obst_arclength >= self.path.length):
                break

            obst_displacement_dist = 140 - 120 / (1 + np.exp(-0.005*obst_arclength))
            
            obst_position = self.path(obst_arclength)
            obst_displacement_angle = self.path.get_direction(obst_arclength) - np.pi/2
            obst_displacement = obst_displacement_dist*np.array([
                np.cos(obst_displacement_angle), 
                np.sin(obst_displacement_angle)
            ])

            self.obstacles.append(StaticObstacle(obst_position + obst_displacement, obst_radius))
            self.obstacles.append(StaticObstacle(obst_position - obst_displacement, obst_radius))

class TestScenario3(PathColavEnv):
    def generate(self):
        waypoints = np.vstack([[0, 0], [0, 500]]).T
        self.path = ParamCurve(waypoints)

        init_pos = self.path(0)
        init_angle = self.path.get_direction(0)

        self.vessel = AUV2D(self.config["t_step_size"], np.hstack([init_pos, init_angle]))
        prog = self.path.get_closest_arclength(self.vessel.position)
        self.path_prog = np.array([prog])
        self.max_path_prog = prog

        N_obst = 20
        N_dist = 100
        for n in range(N_obst + 1):
            obst_radius = 25
            angle = np.pi/4 +  n/N_obst * np.pi/2
            obst_position = np.array([np.cos(angle)*N_dist, np.sin(angle)*N_dist])
            self.obstacles.append(StaticObstacle(obst_position, obst_radius))

class TestScenario4(PathColavEnv):
    def generate(self):
        waypoints = np.vstack([[0, 0], [0, 500]]).T
        self.path = ParamCurve(waypoints)

        init_pos = self.path(0)
        init_angle = self.path.get_direction(0)

        self.vessel = AUV2D(self.config["t_step_size"], np.hstack([init_pos, init_angle]))
        prog = self.path.get_closest_arclength(self.vessel.position)
        self.path_prog = np.array([prog])
        self.max_path_prog = prog

        N_obst = 20
        N_dist = 100
        for n in range(N_obst+1):
            obst_radius = 25
            angle = n/N_obst * 2*np.pi
            if (abs(angle < 3/2*np.pi) < np.pi/12):
                continue
            obst_position = np.array([np.cos(angle)*N_dist, np.sin(angle)*N_dist])
            self.obstacles.append(StaticObstacle(obst_position, obst_radius))

class DebugScenario(PathColavEnv):
    def generate(self):
        waypoints = np.vstack([[0, 0], [0, 10]]).T
        self.path = ParamCurve(waypoints)

        init_pos = self.path(0)
        init_angle = self.path.get_direction(0)

        self.vessel = AUV2D(self.config["t_step_size"], np.hstack([init_pos, init_angle]))
        prog = self.path.get_closest_arclength(self.vessel.position)
        self.path_prog = np.array([prog])
        self.max_path_prog = prog