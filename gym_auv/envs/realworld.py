import numpy as np
import pandas as pd

import gym_auv.utils.geomutils as geom
from gym_auv.objects.vessel import Vessel
from gym_auv.objects.path import RandomCurveThroughOrigin, Path
from gym_auv.objects.obstacles import PolygonObstacle, VesselObstacle
from gym_auv.environment import ASV_Scenario
import shapely.geometry, shapely.errors

import os 
dir_path = os.path.dirname(os.path.realpath(__file__))

VIEW_DISTANCE_3D = 1000 #1000#0
UPDATE_WAIT = 100
VESSEL_DATA_PATH = '../resources/vessel_data_local.csv'
TERRAIN_DATA_PATH = '../resources/terrain.npy'
INCLUDED_VESSELS = None

class RealWorldEnv(ASV_Scenario):

    def __init__(self, *args, **kw):
        self.last_scenario_load_coordinates = None
        self.all_terrain = None

        df = pd.read_csv(VESSEL_DATA_PATH)
        vessels = dict(tuple(df.groupby('Vessel_Name')))
        vessel_names = sorted(list(vessels.keys()))

        self.other_vessels = []

        #print('Preprocessing traffic...')
        for vessel_idx, vessel_name in enumerate(vessel_names):
            if vessel_idx > 20:
                break

            vessels[vessel_name]['AIS_Timestamp'] = pd.to_datetime(vessels[vessel_name]['AIS_Timestamp'])
            vessels[vessel_name]['AIS_Timestamp'] -= vessels[vessel_name].iloc[0]['AIS_Timestamp']
            start_timestamp = None

            last_timestamp = pd.to_timedelta(0, unit='D')
            cutoff_dt = pd.to_timedelta(0.1, unit='D')
            path = []
            start_index = np.random.randint(0, len(vessels[vessel_name])-1)
            for _, row in vessels[vessel_name][start_index:].iterrows():
                if len(path) == 0:
                    start_timestamp = row['AIS_Timestamp']
                timedelta = row['AIS_Timestamp'] - last_timestamp
                if timedelta < cutoff_dt:
                    path.append((int((row['AIS_Timestamp']-start_timestamp).total_seconds()), (row['AIS_East']/10.0-self.x0, row['AIS_North']/10.0-self.y0)))
                else:
                    if len(path) > 0 and not np.isnan(row['AIS_Length_Overall']) and row['AIS_Length_Overall'] > 0:
                        self.other_vessels.append((row['AIS_Length_Overall']/10.0, path, vessel_name))
                    path = []
                last_timestamp = row['AIS_Timestamp']

            if self.other_vessels:
                #break
                print(vessel_name, path[0], len(path))
                #break
                #print(vessel_name, self.other_vessels)
                #break
        
        #print('Completed traffic preprocessing')

        super().__init__(*args, **kw)

    def _generate(self):

        init_pos = self.path(0)
        init_angle = self.path.get_direction(0)

        self.vessel = Vessel(self.config, np.hstack([init_pos, init_angle]))
        prog = self.path.get_closest_arclength(self.vessel.position)
        self.path_prog_hist = np.array([prog])
        self.max_path_prog = prog

        self.all_obstacles = []
        self.obstacles = []
        for obstacle_perimeter in self.obstacle_perimeters:
            if len(obstacle_perimeter) > 3:
                obstacle = PolygonObstacle(obstacle_perimeter)
                if obstacle.valid:
                    self.all_obstacles.append(obstacle)

        for vessel_width, vessel_trajectory, vessel_name in self.other_vessels:
            # for k in range(0, len(vessel_trajectory)-1):
            #     vessel_obstacle = VesselObstacle(width=int(vessel_width), trajectory=vessel_trajectory[k:])
            #     self.all_obstacles.append(vessel_obstacle)
            if len(vessel_trajectory) > 2:
                vessel_obstacle = VesselObstacle(width=int(vessel_width), trajectory=vessel_trajectory, name=vessel_name)
                self.all_obstacles.append(vessel_obstacle)

        self._update()

    def _update(self):

        if self.t_step % UPDATE_WAIT == 0:

            travelled_distance = np.linalg.norm(self.vessel.position - self.last_scenario_load_coordinates) if self.last_scenario_load_coordinates is not None else np.inf

            if self.verbose:
                print('Update scheduled with distance travelled {:.2f}.'.format(travelled_distance))
            if travelled_distance > VIEW_DISTANCE_3D/10:
                
                if self.verbose:
                    print('Loading nearby terrain...'.format(len(self.obstacles)))
                vessel_center = shapely.geometry.Point(
                    self.vessel.position[0], 
                    self.vessel.position[1],
                )
                self.obstacles = []
                self.vessel_obstacles = []
                for obstacle in self.all_obstacles:
                    obst_dist = float(vessel_center.distance(obstacle.boundary)) - self.vessel.width
                    if obst_dist <= VIEW_DISTANCE_3D:
                        self.obstacles.append(obstacle)
                        if not obstacle.static:
                            self.vessel_obstacles.append(obstacle)
                    else:
                        if not obstacle.static:
                            obstacle.update(UPDATE_WAIT*self.config["t_step_size"])

                if self.verbose:
                    print('Loaded nearby terrain ({} obstacles).'.format(len(self.obstacles)))
                
                if self.render_mode == '3d':
                    if self.verbose:
                        print('Loading nearby 3D terrain...')
                    x = int(self.vessel.x)
                    y = int(self.vessel.y)
                    xlow = max(0, x-VIEW_DISTANCE_3D)
                    xhigh = min(self.all_terrain.shape[0], x+VIEW_DISTANCE_3D)
                    ylow = max(0, y-VIEW_DISTANCE_3D)
                    yhigh = min(self.all_terrain.shape[1], y+VIEW_DISTANCE_3D)
                    self.viewer3d.create_world(self.all_terrain, xlow, ylow, xhigh, yhigh)
                    if self.verbose:
                        print('Loaded nearby 3D terrain ({}-{}, {}-{})'.format(xlow, xhigh, ylow, yhigh))

                self.last_scenario_load_coordinates = self.vessel.position

        super()._update()


    def reset(self):
        if hasattr(self, 'all_obstacles'):
            self.obstacles = self.all_obstacles
        super().reset()

class Sorbuoya(RealWorldEnv):
    def __init__(self, *args, **kw):
        self.x0 = 0
        self.y0 = 10000
        super().__init__(*args, **kw)

    def _generate(self):
        #self.path = Path([[-50, 1750], [250, 1200]])
        self.path = Path([[650, 1750], [450, 1200]])
        self.obstacle_perimeters = np.load('../resources/obstacles_sorbuoya.npy')
        self.all_terrain = np.load(TERRAIN_DATA_PATH)[0000:2000, 10000:12000]/7.5
        super()._generate()

class Trondheimsfjorden(RealWorldEnv):
    def __init__(self, *args, **kw):
        self.x0 = 3121
        self.y0 = 5890
        super().__init__(*args, **kw)

    def _generate(self):
        #self.path = Path([[520, 1070, 4080, 5473, 10170, 12220], [3330, 5740, 7110, 4560, 7360, 11390]], smooth=False) #South-west -> north-east
        self.path = Path([[4177-self.x0, 4137-self.x0, 3217-self.x0], [6700-self.y0, 7075-self.y0, 6840-self.y0]], smooth=False)
        self.obstacle_perimeters = np.load('../resources/obstacles_entrance.npy')
        self.all_terrain = np.load(TERRAIN_DATA_PATH)[3121:4521, 5890:7390]/7.5
        super()._generate()

class Trondheim(RealWorldEnv):
    def __init__(self, *args, **kw):
        self.x0 = 5000
        self.y0 = 1900
        super().__init__(*args, **kw)

    def _generate(self):
        self.path = Path([[1900, 1000], [2500, 500]])
        self.obstacle_perimeters = np.load('../resources/obstacles_trondheim.npy')
        self.all_terrain = np.load(TERRAIN_DATA_PATH)[5000:8000, 1900:4900]/7.5
        super()._generate()

