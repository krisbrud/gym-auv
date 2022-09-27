import pygame
import numpy as np

from typing import List
from gym_auv.rendering.render2d.geometry import Circle, Geom, PolyLine
from gym_auv.rendering.render2d.state import RenderableState
from gym_auv.objects.obstacles import (
    BaseObstacle,
    CircularObstacle,
    PolygonObstacle,
    VesselObstacle,
)
from gym_auv.objects.vessel import Vessel
from gym_auv.objects.path import Path


def _render_path(path: Path):
    polyline = draw_polyline(path.points, linewidth=1, color=(0.3, 1.0, 0.3))
    return polyline


def _render_vessel(vessel: Vessel) -> List[Geom]:
    path_taken_line = draw_polyline(
        vessel.path_taken, linewidth=1, color=(0.8, 0, 0)
    )  # previous positions
    vertices = [
        (-vessel.width / 2, -vessel.width / 2),
        (-vessel.width / 2, vessel.width / 2),
        (vessel.width / 2, vessel.width / 2),
        (3 / 2 * vessel.width, 0),
        (vessel.width / 2, -vessel.width / 2),
    ]

    vessel_shape = draw_shape(
        vertices, vessel.position, vessel.heading, color=(0, 0, 0.8)
    )

    return [path_taken_line, vessel_shape]


def _render_sensors(vessel: Vessel) -> List[Geom]:
    sensor_lines: List[Geom] = []
    for isensor, sensor_angle in enumerate(vessel._sensor_angles):
        distance = vessel._last_sensor_dist_measurements[isensor]
        p0 = vessel.position
        p1 = (
            p0[0] + np.cos(sensor_angle + vessel.heading) * distance,
            p0[1] + np.sin(sensor_angle + vessel.heading) * distance,
        )

        # closeness = vessel._last_sector_dist_measurements[isector]
        closeness = vessel._last_sensor_dist_measurements[isensor]
        redness = 0.5 + 0.5 * max(0, closeness)
        greenness = 1 - max(0, closeness)
        blueness = 1
        alpha = 0.5
        sensor_lines.append(
            draw_line(p0, p1, color=(redness, greenness, blueness, alpha))
        )

    return sensor_lines


def _render_progress(path: Path, vessel: Vessel) -> List[Geom]:
    color = pygame.Color(200, 77, 77)

    geoms = []
    ref_point = path(vessel._last_navi_state_dict["vessel_arclength"]).flatten()
    geoms.append(Circle(origin=ref_point, radius=1, res=30, color=color))

    target_point = path(vessel._last_navi_state_dict["target_arclength"]).flatten()
    geoms.append(Circle(origin=target_point, radius=1, res=30, color=color))

    return geoms


def _render_obstacles(obstacles: List[BaseObstacle]) -> List[Geom]:
    geoms = []
    for obst in obstacles:
        c = pygame.Color(200, 200, 200)

        if isinstance(obst, CircularObstacle):
            geoms.append(Circle(obst.position, obst.radius, color=c))

        elif isinstance(obst, PolygonObstacle):
            geoms.append(PolyLine(obst.points, color=c))

        elif isinstance(obst, VesselObstacle):
            geoms.append(PolyLine(list(obst.boundary.exterior.coords), color=c))

    return geoms


def make_background(W=env_bg_w, H=env_bg_h) -> Geom:
    color = (37, 150, 190)  # "#2596be" Semi-dark blue
    # TODO: Change: Potentially by using this example
    # https://www.geeksforgeeks.org/how-to-change-screen-background-color-in-pygame/
    # background = pyglet.shapes.Rectangle(x=0, y=0, width=W, height=H, color=color)
    # background.draw()


def make_objects(state: RenderableState):
    t = viewer.transform
    # t.enable()
    _render_sensors(viewer, vessel=state.vessel)
    # _render_interceptions(env)
    if state.path is not None:
        _render_path(viewer, path=state.path)
    _render_vessel(viewer, vessel=state.vessel)
    # _render_tiles(env, win)
    _render_obstacles(viewer=viewer, obstacles=state.obstacles)
    if state.path is not None:
        _render_progress(viewer=viewer, path=state.path, vessel=state.vessel)
    # _render_interceptions(env)

    # Visualise path error (DEBUGGING)
    # p = np.array(env.vessel.position)
    # dir = rotate(env.past_obs[-1][0:2], env.vessel.heading)
    # env._viewer2d.draw_line(p, p + 10*np.array(dir), color=(0.8, 0.3, 0.3))

    for geom in viewer.onetime_geoms:
        geom.render()

    t.disable()

    # if state.show_indicators:
    #     _render_indicators(
    #         viewer=viewer,
    #         W=WINDOW_W,
    #         H=WINDOW_H,
    #         last_reward=state.last_reward,
    #         cumulative_reward=state.cumulative_reward,
    #         t_step=state.t_step,
    #         episode=state.episode,
    #         lambda_tradeoff=state.lambda_tradeoff,
    #         eta=state.eta,
    #     )
