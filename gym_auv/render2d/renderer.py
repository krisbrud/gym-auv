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
from gym_auv.render2d.state import RenderableState
from gym_auv.utils.sector_partitioning import sector_partition_fun

from gym_auv.render2d.geometry import (
    BaseGeom,
    Circle,
    Line,
    Transformation,
)
from gym_auv.render2d import colors
from gym_auv.render2d.factories import (
    make_body_frame_geoms,
    make_world_frame_geoms,
)
from gym_auv.render2d.utils import apply_transformation

WINDOW_W = 720
WINDOW_H = 600
FPS = 60


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

    def render(
        self,
        state: RenderableState,
        render_mode: str = "human",
        image_observation_mode: bool = False,
    ):
        """Renders the environment.

        render_mode: "human" or "rgb_array"
        render_sensors: Whether to render sensor data. Should be False for image observations
        """
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
        self.surf.fill(colors.LIGHT_BLUE)
        # Make geometry
        world_geoms = make_world_frame_geoms(state=state)

        should_render_sensors = not image_observation_mode  # Don't render sensors if we're making an image observation
        body_geoms = make_body_frame_geoms(state=state, render_sensors=should_render_sensors)

        # Transform world geoms to body frame
        world_to_body_transformation = Transformation(
            translation=-state.vessel.position, angle=-state.vessel.heading
        )
        body_geoms.extend(
            apply_transformation(world_to_body_transformation, world_geoms)
        )

        # Center camera on vessel by applying transform
        if image_observation_mode:
            rotation = 2 * state.vessel.heading  # state.vessel.heading   # 0
        else:
            rotation = state.vessel.heading # - (np.pi / 2)

        zoom_transformation = Transformation(
            translation=pygame.Vector2(0, 0),
            # angle=-np.pi / 2 + state.vessel.heading,
            angle=rotation - (np.pi / 2), #  state.vessel.heading,
            scale=self.zoom,
        )
        zoomed_geoms = apply_transformation(zoom_transformation, body_geoms)
       
        # Center camera on vessel, as coordinate (0, 0) is a corner
        centering_translation = pygame.Vector2(x=self.width / 2, y=self.height / 2)
        camera_transformation = Transformation(
            translation=centering_translation,
            angle=0,
        )

        centered_geoms = apply_transformation(camera_transformation, zoomed_geoms)

        for geom in centered_geoms:
            self.render_geom(geom)

        self.screen.blit(self.surf, (0, 0))


        if not image_observation_mode:        
            self.render_text(f"Reward: {state.last_reward:.2f}", (10, 10))
            self.render_text(f"Cumulative reward: {state.cumulative_reward:.2f}", (10, 40))
            

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
    
    def render_text(self, text, pos):
        font = pygame.font.SysFont(None, 24)
        text_img = font.render(text, True, colors.BLACK_BLUE)
        self.screen.blit(text_img, pos)

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
