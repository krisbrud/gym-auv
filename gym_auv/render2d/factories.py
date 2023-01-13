import pygame
import numpy as np

from typing import List
from gym_auv.render2d.geometry import (
    Circle,
    FilledPolygon,
    BaseGeom,
    Line,
    PolyLine,
)
from gym_auv.render2d.state import RenderableState
from gym_auv.render2d.utils import clamp_to_uint8
from gym_auv.objects.obstacles import (
    BaseObstacle,
    CircularObstacle,
    PolygonObstacle,
    VesselObstacle,
)
from gym_auv.objects.vessel import Vessel
from gym_auv.objects.path import Path
from gym_auv.render2d import colors
from gym_auv.render2d.utils import ndarray_to_vector2_list


def _render_path(path: Path) -> PolyLine:
    points = ndarray_to_vector2_list(path.points)
    polyline = PolyLine(points, color=colors.LIGHT_GREEN)

    return polyline


def _render_path_taken(vessel: Vessel) -> PolyLine:
    # previous positions
    points = ndarray_to_vector2_list(vessel.path_taken)
    path_taken_line = PolyLine(
        points=points,
        color=colors.BLUE_GREEN,
    )

    return path_taken_line


def _render_vessel(vessel: Vessel) -> FilledPolygon:
    vertices = [
        pygame.Vector2(-vessel.width / 2, -vessel.width / 2),
        pygame.Vector2(-vessel.width / 2, vessel.width / 2),
        pygame.Vector2(vessel.width / 2, vessel.width / 2),
        pygame.Vector2(3 / 2 * vessel.width, 0),
        pygame.Vector2(vessel.width / 2, -vessel.width / 2),
    ]

    vessel_shape = FilledPolygon(vertices, color=colors.ORANGE)

    return vessel_shape


def _render_sensors(vessel: Vessel) -> List[BaseGeom]:
    sensor_lines: List[BaseGeom] = []
    for isensor, sensor_angle in enumerate(vessel._sensor_angles):
        distance = vessel._last_sensor_dist_measurements[isensor]
        p0 = pygame.Vector2(0, 0)
        p1 = (
            pygame.Vector2(
                np.cos(sensor_angle),
                np.sin(sensor_angle),
            )
            * distance
        )

        # closeness = vessel._last_sector_dist_measurements[isector]
        closeness = vessel._last_sensor_dist_measurements[isensor]
        redness = clamp_to_uint8(int(0.5 + 0.5 * max(0, closeness) * 255))
        greenness = clamp_to_uint8(int((1 - max(0, closeness)) * 255))
        blueness = 255
        alpha = 127
        color = pygame.Color(redness, greenness, blueness, alpha)
        sensor_lines.append(Line(start=p0, end=p1, color=color))

    return sensor_lines


def _render_progress(path: Path, vessel: Vessel) -> List[BaseGeom]:
    geoms = []
    ref_point = pygame.Vector2(
        *path(vessel._last_navi_state_dict["vessel_arclength"]).flatten()
    )
    geoms.append(Circle(center=ref_point, radius=1, color=colors.EGG_WHITE))

    target_point = pygame.Vector2(
        *path(vessel._last_navi_state_dict["target_arclength"]).flatten()
    )
    geoms.append(Circle(center=target_point, radius=1, color=colors.EGG_WHITE))

    return geoms


def _render_obstacles(obstacles: List[BaseObstacle]) -> List[BaseGeom]:
    geoms = []
    for obst in obstacles:
        c = colors.EGG_WHITE

        if isinstance(obst, CircularObstacle):
            geoms.append(Circle(pygame.Vector2(*obst.position), obst.radius, color=c))

        elif isinstance(obst, PolygonObstacle):
            points = ndarray_to_vector2_list(obst.points)
            geoms.append(PolyLine(pygame.Vector2(points, color=c)))

        elif isinstance(obst, VesselObstacle):
            points = ndarray_to_vector2_list(obst.boundary.exterior.coords)
            geoms.append(FilledPolygon(points, color=c))

    return geoms


def make_world_frame_geoms(state: RenderableState) -> List[BaseGeom]:
    geoms = []

    if state.path is not None:
        geoms.append(_render_path(path=state.path))
        if len(state.vessel.path_taken) > 1:  # Avoid rendering a single point, which causes a crash
            geoms.append(_render_path_taken(vessel=state.vessel))
    geoms.extend(_render_obstacles(obstacles=state.obstacles))
    if state.path is not None:
        geoms.extend(_render_progress(path=state.path, vessel=state.vessel))

    return geoms


def make_body_frame_geoms(state: RenderableState) -> List[BaseGeom]:
    geoms = []
    geoms.append(_render_vessel(vessel=state.vessel))
    geoms.extend(_render_sensors(vessel=state.vessel))

    return geoms
