from copy import deepcopy
import numpy as np
import numpy.linalg as linalg
import shapely.geometry
from scipy.optimize import minimize

from scipy import interpolate

import gym_auv.utils.geomutils as geom

class Path():
    def __init__(self, waypoints, smooth=True):
        self.waypoints = waypoints.copy()
        forloop_range = 3 if smooth else 1
        for _ in range(forloop_range):
            arclengths = arc_len(waypoints)
            path_coords = interpolate.pchip(x=arclengths, y=waypoints, axis=1)
            path_derivatives = path_coords.derivative()
            path_dderivatives = path_derivatives.derivative()
            waypoints = path_coords(np.linspace(arclengths[0], arclengths[-1], 1000))

        self._path_coords = path_coords
        self._path_derivatives = path_derivatives
        self._path_dderivatives = path_dderivatives

        self.length = arclengths[-1]
        S = np.linspace(0, self.length, 10*self.length)
        self.points = np.transpose(self._path_coords(S))
        self._linestring = shapely.geometry.LineString(self.points)
        self.start = path_coords(0)
        self.end = path_coords(self.length)

    def __call__(self, arclength):
        return self._path_coords(arclength)

    def get_direction(self, arclength):
        derivative = self._path_derivatives(arclength)
        return np.arctan2(derivative[1], derivative[0])

    def get_closest_arclength(self, position, x0=None):
        if x0 is not None:
            x = position[0]
            y = position[1]
            d = minimize(
                    fun=lambda w: linalg.norm(self(w) - position),
                    x0=x0,
                    jac=lambda w: -2*(x - self(w)[0])*self._path_derivatives(w)[0] -2*(y - self(w)[1])*self._path_derivatives(w)[1],
                    hess=lambda w: 2*(-(x - self(w)[0])*self._path_dderivatives(w)[0] -(y - self(w)[1])*self._path_dderivatives(w)[1] + self._path_derivatives(w)[0]**2*self._path_derivatives(w)[1]**2),
                    method='Newton-CG' 
                ).x[0]
            d = np.clip(d, 0, self.length)
        else:
            d = self._linestring.project(shapely.geometry.Point(position))
        return d

    def get_closest_point(self, position, x0=None):
        d = self.get_closest_arclength(position, x0)
        p = self._linestring.interpolate(d)
        closest_point = list(p.coords)[0]
        return closest_point, d

    def get_closest_point_distance(self, position, x0=None):
        closest_point, closest_arclength = self.get_closest_point(position, x0=x0)
        closest_point_distance =  linalg.norm(closest_point - position)
        return closest_point_distance, closest_point, closest_arclength

    def __reversed__(self):
        curve = deepcopy(self)
        path_coords = curve.path_coords
        curve.path_coords = lambda s: path_coords(curve.length-s)
        return curve

    def plot(self, ax, s, *opts):
        s = np.array(s)
        z = self(s)
        ax.plot(-z[1, :], z[0, :], *opts)

class RandomCurveThroughOrigin(Path):
    def __init__(self, rng, nwaypoints, length=400):
        angle_init = 2*np.pi*(rng.rand() - 0.5)
        start = np.array([0.5*length*np.cos(angle_init), 0.5*length*np.sin(angle_init)])
        end = -np.array(start)
        waypoints = np.vstack([start, end])
        for waypoint in range(nwaypoints // 2):
            newpoint1 = ((nwaypoints // 2 - waypoint)
                         * start / (nwaypoints // 2 + 1)
                         + length / (nwaypoints // 2 + 1)
                         * (rng.rand()-0.5))
            newpoint2 = ((nwaypoints // 2 - waypoint)
                         * end / (nwaypoints // 2 + 1)
                         + length / (nwaypoints // 2 + 1)
                         * (rng.rand()-0.5))
            waypoints = np.vstack([waypoints[:waypoint+1, :],
                                   newpoint1,
                                   np.array([0, 0]),
                                   newpoint2,
                                   waypoints[-1*waypoint-1:, :]])
        super().__init__(np.transpose(waypoints))


def arc_len(coords):
    diff = np.diff(coords, axis=1)
    delta_arc = np.sqrt(np.sum(diff ** 2, axis=0))
    return np.concatenate([[0], np.cumsum(delta_arc)])
