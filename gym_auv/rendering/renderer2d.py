"""
2D rendering framework.
Modified version of the classical control module in OpenAI's gym.

Changes:
    - Added an 'origin' argument to the draw_circle() and make_circle() functions to allow drawing of circles anywhere.
    - Added an 'outline' argument to the draw_circle() function, allows a more stylised render

Created by Haakon Robinson, based on OpenAI's gym.base_env.classical.rendering.py
"""

import os
from typing import List, Union
import six
import sys

# import pyglet
# from pyglet import gl
import pygame
import numpy as np
import math
from numpy import sin, cos, arctan2
from gym import error

# from gym_auv.environment import BaseEnvironment
import gym_auv.utils.geomutils as geom
from gym_auv.objects.obstacles import (
    BaseObstacle,
    CircularObstacle,
    PolygonObstacle,
    VesselObstacle,
)
from gym_auv.rendering.render2d.state import RenderableState
from gym_auv.objects.path import Path
from gym_auv.objects.vessel import Vessel
from gym_auv.utils.sector_partitioning import sector_partition_fun
from gym_auv.config import RenderingConfig

from gym_auv.rendering.render2d.geometry import (
    Circle,
    Line,
    Transform,
    make_circle,
    _add_attrs,
    make_polygon,
    make_polyline,
)
from gym_auv.rendering.render2d.renderer import render_blue_background, render_objects


if "Apple" in sys.version:
    if "DYLD_FALLBACK_LIBRARY_PATH" in os.environ:
        os.environ["DYLD_FALLBACK_LIBRARY_PATH"] += ":/usr/lib"
        # (JDS 2016/04/15): avoid bug on Anaconda 2.3.0 / Yosemite

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


def rad2deg(rad: Union[float, np.array]) -> float:
    return rad * 180 / np.pi


env_bg = None
bg = None
rot_angle = None


def get_display(spec):
    """Convert a display specification (such as :0) into an actual Display
    object.

    Pyglet only supports multiple Displays on Linux.
    """
    if spec is None:
        return None
    elif isinstance(spec, six.string_types):
        return pyglet.canvas.Display(spec)
    else:
        raise error.Error(
            "Invalid display specification: {}. (Must be a string like :0 or None.)".format(
                spec
            )
        )


class Viewer2D(object):
    def __init__(self, width=WINDOW_W, height=WINDOW_H, display=None, mode="human"):
        display = get_display(display)

        self.width = width
        self.height = height
        self.mode = mode
        # if self.mode == "human":
        # self.window = pyglet.window.Window(
        #     width=width, height=height, display=display
        # )
        # self.window.on_close = self.window_closed_by_user
        self.isopen = True
        self.geoms = []
        self.onetime_geoms = []
        self.fixed_geoms = []
        self.transform = Transform()
        self.camera_zoom = 1.5

        self.screen = None

        # gl.glEnable(gl.GL_BLEND)
        # gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)

        # Initialize text fields
        # self.reward_text_field = pyglet.text.Label(
        #     "0000",
        #     font_size=10,
        #     x=20,
        #     y=WINDOW_H - 30.00,
        #     anchor_x="left",
        #     anchor_y="center",
        #     color=(0, 0, 0, 255),
        # )
        # self.reward_value_field = pyglet.text.Label(
        #     "0000",
        #     font_size=10,
        #     x=260,
        #     y=WINDOW_H - 30.00,
        #     anchor_x="right",
        #     anchor_y="center",
        #     color=(0, 0, 0, 255),
        # )

        # self.cum_reward_text_field = pyglet.text.Label(
        #     "0000",
        #     font_size=10,
        #     x=20,
        #     y=WINDOW_H - 50.00,
        #     anchor_x="left",
        #     anchor_y="center",
        #     color=(0, 0, 0, 255),
        # )

        # self.cum_reward_value_field = pyglet.text.Label(
        #     "0000",
        #     font_size=10,
        #     x=260,
        #     y=WINDOW_H - 50.00,
        #     anchor_x="right",
        #     anchor_y="center",
        #     color=(0, 0, 0, 255),
        # )

        # self.time_step_text_field = pyglet.text.Label(
        #     "0000",
        #     font_size=10,
        #     x=20,
        #     y=WINDOW_H - 70.00,
        #     anchor_x="left",
        #     anchor_y="center",
        #     color=(0, 0, 0, 255),
        # )
        # self.time_step_value_field = pyglet.text.Label(
        #     "0000",
        #     font_size=10,
        #     x=260,
        #     y=WINDOW_H - 70.00,
        #     anchor_x="right",
        #     anchor_y="center",
        #     color=(0, 0, 0, 255),
        # )

        # self.episode_text_field = pyglet.text.Label(
        #     "0000",
        #     font_size=10,
        #     x=20,
        #     y=WINDOW_H - 90.00,
        #     anchor_x="left",
        #     anchor_y="center",
        #     color=(0, 0, 0, 255),
        # )
        # self.episode_value_field = pyglet.text.Label(
        #     "0000",
        #     font_size=10,
        #     x=260,
        #     y=WINDOW_H - 90.00,
        #     anchor_x="right",
        #     anchor_y="center",
        #     color=(0, 0, 0, 255),
        # )

        # self.lambda_text_field = pyglet.text.Label(
        #     "0000",
        #     font_size=10,
        #     x=20,
        #     y=WINDOW_H - 110.00,
        #     anchor_x="left",
        #     anchor_y="center",
        #     color=(0, 0, 0, 255),
        # )
        # self.lambda_value_field = pyglet.text.Label(
        #     "0000",
        #     font_size=10,
        #     x=260,
        #     y=WINDOW_H - 110.00,
        #     anchor_x="right",
        #     anchor_y="center",
        #     color=(0, 0, 0, 255),
        # )

        # self.eta_text_field = pyglet.text.Label(
        #     "0000",
        #     font_size=10,
        #     x=20,
        #     y=WINDOW_H - 130.00,
        #     anchor_x="left",
        #     anchor_y="center",
        #     color=(0, 0, 0, 255),
        # )
        # self.eta_value_field = pyglet.text.Label(
        #     "0000",
        #     font_size=10,
        #     x=260,
        #     y=WINDOW_H - 130.00,
        #     anchor_x="right",
        #     anchor_y="center",
        #     color=(0, 0, 0, 255),
        # )

        # print("Initialized 2D viewer")

    # def close(self):
    #     self.window.close()

    # def window_closed_by_user(self):
    #     self.isopen = False

    # def set_bounds(self, left, right, bottom, top):
    #     assert right > left and top > bottom
    #     scalex = self.width / (right - left)
    #     scaley = self.height / (top - bottom)
    #     self.transform = Transform(
    #         translation=(-left * scalex, -bottom * scaley), scale=(scalex, scaley)
    #     )

    def add_geom(self, geom):
        self.geoms.append(geom)

    def add_onetime(self, geom):
        self.onetime_geoms.append(geom)

    def add_fixed(self, geom):
        self.fixed_geoms.append(geom)

    def render(self, return_rgb_array=False):
        # gl.glClearColor(1, 1, 1, 1)

        # TODO: Move to init
        if self.screen is None:
            pygame.init()
            if self.render_mode == "human":
                pygame.display.init()
                self.screen = pygame.display.set_mode(
                    (self.screen_width, self.screen_height)
                )
            else:  # mode == "rgb_array":
                self.screen = pygame.Surface((self.screen_width, self.screen_height))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        # self.window.clear()
        # self.window.switch_to()
        # self.window.dispatch_events()
        self.transform.enable()
        for geom in self.geoms:
            geom.render()
        for geom in self.onetime_geoms:
            geom.render()
        self.transform.disable()
        for geom in self.fixed_geoms:
            geom.render()
        arr = None

        if self.render_mode == "human":
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()

        elif self.render_mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )
        if return_rgb_array:
            # buffer = pyglet.image.get_buffer_manager().get_color_buffer()
            # image_data = buffer.get_image_data()
            # arr = np.fromstring(image_data.data, dtype=np.uint8, sep="")
            # arr = arr.reshape(buffer.height, buffer.width, 4)
            # arr = arr[::-1, :, 0:3]
            pass
        # self.window.flip()
        self.onetime_geoms = []
        return arr if return_rgb_array else self.isopen

    def draw_circle(
        self,
        center=(0, 0),
        radius=10,
        res=30,
        filled=True,
        outline=True,
        start_angle=0,
        end_angle=2 * np.pi,
        **attrs
    ):
        geom = Circle(origin=center, radius=radius)
        _add_attrs(geom, attrs)
        self.add_onetime(geom)
        # if filled and outline:
        #     outl = make_circle(center=center, radius=radius, res=res, filled=False)
        #     _add_attrs(outl, {"color": (0, 0, 0), "linewidth": 1})
        #     self.add_onetime(outl)
        return geom

    def draw_polygon(self, v, filled=True, **attrs):
        geom = make_polygon(v=v, filled=filled)
        _add_attrs(geom, attrs)
        self.add_onetime(geom)
        return geom

    def draw_polyline(self, v, **attrs):
        geom = make_polyline(v=v)
        _add_attrs(geom, attrs)

        self.add_onetime(geom)
        return geom

    def draw_line(self, start, end, **attrs):
        geom = Line(start, end)
        _add_attrs(geom, attrs)
        self.add_onetime(geom)
        return geom

    # def get_array(self):
    #     self.window.flip()
    #     image_data = (
    #         pyglet.image.get_buffer_manager().get_color_buffer().get_image_data()
    #     )
    #     self.window.flip()
    #     arr = np.fromstring(image_data.data, dtype=np.uint8, sep="")
    #     arr = arr.reshape(self.height, self.width, 4)
    #     return arr[::-1, :, 0:3]

    def transform_vertices(self, points, translation, rotation, scale=1):
        res = []
        for p in points:
            res.append(
                (
                    cos(rotation) * p[0] * scale
                    - sin(rotation) * p[1] * scale
                    + translation[0],
                    sin(rotation) * p[0] * scale
                    + cos(rotation) * p[1] * scale
                    + translation[1],
                )
            )
        return res

    def draw_arrow(self, base, angle, length, **attrs):
        TRIANGLE_POLY = ((-1, -1), (1, -1), (0, 1))
        head = (base[0] + length * cos(angle), base[1] + length * sin(angle))
        tri = self.transform_vertices(TRIANGLE_POLY, head, angle - np.pi / 2, scale=0.7)
        self.draw_polyline([base, head], linewidth=2, **attrs)
        self.draw_polygon(tri, **attrs)

    # def draw_shape(
    #     self,
    #     vertices,
    #     position=None,
    #     angle=None,
    #     color=(1, 1, 1),
    #     filled=True,
    #     border=True,
    # ):
    #     if position is not None:
    #         poly_path = self.transform_vertices(vertices, position, angle)
    #     else:
    #         poly_path = vertices
    #     if filled:
    #         self.draw_polygon(poly_path + [poly_path[0]], color=color)
    #     if border:
    #         border_color = (0, 0, 0) if type(border) == bool else border
    #         self.draw_polyline(
    #             poly_path + [poly_path[0]],
    #             linewidth=1,
    #             color=border_color if filled else color,
    #         )

    def __del__(self):
        self.close()

    def render_env(self, mode, state: RenderableState, render_config: RenderingConfig):
        global rot_angle

        # print("Render env called!")

        scroll_x = state.vessel.position[0]
        scroll_y = state.vessel.position[1]
        ship_angle = -state.vessel.heading + np.pi / 2

        if rot_angle is None:
            rot_angle = ship_angle
        else:
            rot_angle += CAMERA_ROTATION_SPEED * geom.princip(ship_angle - rot_angle)

        if DYNAMIC_ZOOM:
            if int(state.t_step / 1000) % 2 == 0:
                self.camera_zoom = 0.999 * self.camera_zoom + 0.001 * (
                    ZOOM - self.camera_zoom
                )
            else:
                self.camera_zoom = 0.999 * self.camera_zoom + 0.001 * (
                    1 - self.camera_zoom
                )

        self.transform.set_scale(self.camera_zoom, self.camera_zoom)
        self.transform.set_translation(
            WINDOW_W / 2
            - (
                scroll_x * self.camera_zoom * cos(rot_angle)
                - scroll_y * self.camera_zoom * sin(rot_angle)
            ),
            WINDOW_H / 2
            - (
                scroll_x * self.camera_zoom * sin(rot_angle)
                + scroll_y * self.camera_zoom * cos(rot_angle)
            ),
        )
        self.transform.set_rotation(rot_angle)

        # win = viewer.window
        # win.switch_to()
        # x = win.dispatch_events()
        # win.clear()
        # gl.glViewport(0, 0, WINDOW_W, WINDOW_H)
        render_blue_background()
        render_objects(viewer=viewer, state=state)
        arr = None

        # if mode == "rgb_array":
        #     image_data = (
        #         pyglet.image.get_buffer_manager().get_color_buffer().get_image_data()
        #     )
        #     arr = np.fromstring(image_data.data, dtype=np.uint8, sep="")
        #     arr = arr.reshape(WINDOW_H, WINDOW_W, 4)
        #     arr = arr[::-1, :, 0:3]

        # win.flip()

        viewer.onetime_geoms = []

        return arr


# def render_env(env: BaseEnvironment, mode):
