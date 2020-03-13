import numpy as np

import gym_auv.utils.geomutils as geom
from gym_auv.objects.auv import AUV2D
from gym_auv.objects.path import RandomCurveThroughOrigin, ParamCurve
from gym_auv.objects.obstacles import CircularObstacle, VesselObstacle
from gym_auv.environment import BaseShipScenario

import os 
dir_path = os.path.dirname(os.path.realpath(__file__))

TERRAIN_DATA_PATH = './resources/terrain.npy'

class TestScenario1(BaseShipScenario):
    def generate(self):
        self.path = ParamCurve([[0, 1100], [0, 1100]])

        init_pos = self.path(0)
        init_angle = self.path.get_direction(0)

        self.vessel = AUV2D(self.config["t_step_size"], np.hstack([init_pos, init_angle]))
        prog = self.path.get_closest_arclength(self.vessel.position)
        self.path_prog_hist = np.array([prog])
        self.max_path_prog = prog

        obst_arclength = 30
        for o in range(20):
            obst_radius = 10 + 10*o**1.5
            obst_arclength += obst_radius*2 + 30
            obst_position = self.path(obst_arclength)
            self.obstacles.append(CircularObstacle(obst_position, obst_radius))

class TestScenario2(BaseShipScenario):
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
        self.path_prog_hist = np.array([prog])
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

            self.obstacles.append(CircularObstacle(obst_position + obst_displacement, obst_radius))
            self.obstacles.append(CircularObstacle(obst_position - obst_displacement, obst_radius))

class TestScenario3(BaseShipScenario):
    def generate(self):
        waypoints = np.vstack([[0, 0], [0, 500]]).T
        self.path = ParamCurve(waypoints)

        init_pos = self.path(0)
        init_angle = self.path.get_direction(0)

        self.vessel = AUV2D(self.config["t_step_size"], np.hstack([init_pos, init_angle]))
        prog = self.path.get_closest_arclength(self.vessel.position)
        self.path_prog_hist = np.array([prog])
        self.max_path_prog = prog

        N_obst = 20
        N_dist = 100
        for n in range(N_obst + 1):
            obst_radius = 25
            angle = np.pi/4 +  n/N_obst * np.pi/2
            obst_position = np.array([np.cos(angle)*N_dist, np.sin(angle)*N_dist])
            self.obstacles.append(CircularObstacle(obst_position, obst_radius))

class TestScenario4(BaseShipScenario):
    def generate(self):
        waypoints = np.vstack([[0, 0], [0, 500]]).T
        self.path = ParamCurve(waypoints)

        init_pos = self.path(0)
        init_angle = self.path.get_direction(0)

        self.vessel = AUV2D(self.config["t_step_size"], np.hstack([init_pos, init_angle]))
        prog = self.path.get_closest_arclength(self.vessel.position)
        self.path_prog_hist = np.array([prog])
        self.max_path_prog = prog

        N_obst = 20
        N_dist = 100
        for n in range(N_obst+1):
            obst_radius = 25
            angle = n/N_obst * 2*np.pi
            if (abs(angle < 3/2*np.pi) < np.pi/12):
                continue
            obst_position = np.array([np.cos(angle)*N_dist, np.sin(angle)*N_dist])
            self.obstacles.append(CircularObstacle(obst_position, obst_radius))

class EmptyScenario(BaseShipScenario):
    def __init__(self, *args, **kwargs):
        super().__init__(detect_moving=True, *args, **kwargs)

    def generate(self):
        waypoints = np.vstack([[25, 10], [25, 40]]).T
        self.path = ParamCurve(waypoints)

        init_pos = self.path(0)
        init_angle = self.path.get_direction(0)

        self.vessel = AUV2D(self.config["t_step_size"], np.hstack([init_pos, init_angle]), width=self.config["vessel_width"])
        prog = self.path.get_closest_arclength(self.vessel.position)
        self.path_prog_hist = np.array([prog])
        self.max_path_prog = prog

        if self.render_mode == '3d':
            self.all_terrain = np.zeros((50, 50), dtype=float)
            self.viewer3d.create_world(self.all_terrain, 0, 0, 50, 50)

class DebugScenario(BaseShipScenario):

    def __init__(self, *args, **kwargs):
        super().__init__(detect_moving=True, *args, **kwargs)

    def generate(self):
        waypoints = np.vstack([[250, 100], [250, 300]]).T
        self.path = ParamCurve(waypoints)

        init_pos = self.path(0)
        init_angle = self.path.get_direction(0)

        self.vessel = AUV2D(self.config["t_step_size"], np.hstack([init_pos, init_angle]), width=self.config["vessel_width"])
        prog = self.path.get_closest_arclength(self.vessel.position)
        self.path_prog_hist = np.array([prog])
        self.max_path_prog = prog
        
        self.obstacles = []
        self.vessel_obstacles = []

        for vessel_idx in range(5):
            other_vessel_trajectory = []
            trajectory_shift = np.random.random()*2*np.pi
            trajectory_radius = np.random.random()*40 + 30
            trajectory_speed = np.random.random()*0.01 + 0.01
            for i in range(10000):
                #other_vessel_trajectory.append((10*i, (250, 400-10*i)))
                other_vessel_trajectory.append((1*i, (
                    250 + trajectory_radius*np.cos(trajectory_speed*i + trajectory_shift), 
                    150 + 70*vessel_idx + trajectory_radius*np.sin(trajectory_speed*i + trajectory_shift)
                )))
            other_vessel_obstacle = VesselObstacle(width=6, trajectory=other_vessel_trajectory)

            self.obstacles.append(other_vessel_obstacle)
            self.vessel_obstacles.append(other_vessel_obstacle)

        for vessel_idx in range(5):
            other_vessel_trajectory = []
            trajectory_start = np.random.random()*200 + 150
            trajectory_speed = np.random.random()*0.1 + 0.1
            trajectory_shift = 10*np.random.random()
            for i in range(10000):
                other_vessel_trajectory.append((i, (245 + 2.5*vessel_idx + trajectory_shift, trajectory_start-10*trajectory_speed*i)))
            other_vessel_obstacle = VesselObstacle(width=6, trajectory=other_vessel_trajectory)

            self.obstacles.append(other_vessel_obstacle)
            self.vessel_obstacles.append(other_vessel_obstacle)

        if self.render_mode == '3d':
            self.all_terrain = np.load(TERRAIN_DATA_PATH)[1950:2450, 5320:5820]/7.5
            #terrain = np.zeros((500, 500), dtype=float)

            # for x in range(10, 40):
            #     for y in range(10, 40):
            #         z = 0.5*np.sqrt(max(0, 15**2 - (25.0-x)**2 - (25.0-y)**2))
            #         terrain[x][y] = z
            self.viewer3d.create_world(self.all_terrain, 0, 0, 500, 500)
