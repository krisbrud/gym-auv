"""
2D rendering framework.
Modified version of the classical control module in OpenAI's gym.

Changes:
    - Added an 'origin' argument to the draw_circle() and make_circle() functions to allow drawing of circles anywhere.
    - Added an 'outline' argument to the draw_circle() function, allows a more stylised render

Created by Haakon Robinson, based on OpenAI's gym.base_env.classical.rendering.py
"""

import itertools
import os
from typing import List, Tuple, Union
import six
import sys

# import pyglet
# from pyglet import gl
import pygame
import numpy as np
import math
from numpy import sin, cos, arctan2
from gym import error
from gym_auv.rendering.render2d.state import RenderableState
from gym_auv.utils.sector_partitioning import sector_partition_fun

from gym_auv.rendering.render2d.geometry import (
    BaseGeom,
    Circle,
    Line,
    Transformation,
)
from gym_auv.rendering.render2d import colors
from gym_auv.rendering.render2d.factories import (
    make_body_frame_geoms,
    make_world_frame_geoms,
)

STATE_W = 96
STATE_H = 96
VIDEO_W = 720
VIDEO_H = 600
WINDOW_W = VIDEO_W
WINDOW_H = VIDEO_H

SCALE = 5.0  # Track scale
PLAYFIELD = 5000  # Game over boundary
FPS = 50
ZOOM = 2  # Camera ZOOM
DYNAMIC_ZOOM = False
CAMERA_ROTATION_SPEED = 0.02
env_bg_h = int(2 * PLAYFIELD)
env_bg_w = int(2 * PLAYFIELD)

RAD2DEG = 57.29577951308232


"""
TODO: Refactor.
Environment dependencies of Render2d
- path
- vessel
- sensor_obst_intercepts_transformed_hist
- time_step
- config:
  - sector partition function
"""


# def rad2deg(rad: Union[float, np.array]) -> float:
# return rad * 180 / np.pi


# env_bg = None
# bg = None
# rot_angle = None


class Renderer2d:
    def __init__(
        self,
        width=WINDOW_W,
        height=WINDOW_H,
        render_fps: int = 60,
        zoom: float = 1.5,
    ):
        self.width = width
        self.height = height
        self.render_fps = render_fps

        self.screen = None
        self.clock = None
        self.zoom = zoom

    def render(self, state: RenderableState, render_mode="human"):
        Renderer2d._validate_render_mode(render_mode)

        if self.screen is None:
            pygame.init()
            if render_mode == "human":
                pygame.display.init()
                self.screen = pygame.display.set_mode(self.screen_size)
            else:  # mode == "rgb_array":
                self.screen = pygame.Surface(self.screen_size)
        if self.clock is None:
            self.clock = pygame.time.Clock()

        self.surf = pygame.Surface(self.screen_size)
        self.surf.fill(colors.BLUE)
        # Make geometry
        world_geoms = make_world_frame_geoms(state=state)
        body_geoms = make_body_frame_geoms(state=state)

        # Transform world geoms to body frame
        body_to_world_transformation = Transformation(
            translation=state.vessel.position, angle=state.vessel.heading
        )
        body_geoms.extend(
            list(
                map(
                    lambda geom: geom.transform(body_to_world_transformation),
                    world_geoms,
                )
            )
        )

        # Center camera on vessel by applying transform
        centering_translation = pygame.Vector2(x=-self.width / 2, y=-self.height / 2)
        camera_transformation = Transformation(
            translation=centering_translation,
            angle=0.0,
        )

        centered_geoms = list(
            map(lambda geom: geom.transform(camera_transformation), body_geoms)
        )

        for geom in centered_geoms:
            self.render_geom(geom)

        self.surf = pygame.transform.flip(self.surf, False, True)
        self.screen.blit(self.surf, (0, 0))

        if render_mode == "human":
            pygame.event.pump()
            self.clock.tick(self.render_fps)
            pygame.display.flip()

        elif render_mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )

    def render_geom(self, geom: BaseGeom):
        geom.render(self.surf)

    @property
    def screen_size(self) -> Tuple[int, int]:
        return (self.width, self.height)

    @staticmethod
    def _validate_render_mode(render_mode: str):
        """Raises ValueError if render mode is invalid"""
        valid_render_modes = ("human", "rgb_array")
        if render_mode not in valid_render_modes:
            raise ValueError(
                f"{render_mode}, is not a valid rendering mode!\nValid modes are {valid_render_modes}"
            )