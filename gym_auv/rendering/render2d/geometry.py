from dataclasses import dataclass
import math
from tkinter import CENTER
from turtle import circle
from typing import List

import numpy as np
import pygame


def _add_attrs(geom, attrs):
    if "color" in attrs:
        geom.set_color(*attrs["color"])
    if "linewidth" in attrs:
        geom.set_linewidth(attrs["linewidth"])


@dataclass
class Transformation:
    translation: pygame.Vector2
    angle: float  # radians
    scale: float = 1

    def apply_to(self, point: pygame.Vector2) -> pygame.Vector2:
        rotated = point.rotate_rad(self.angle)
        translated = rotated - self.translation
        scaled = translated * self.scale

        return scaled


class Geom:
    def __init__(self, points: List[pygame.Vector2] = []):
        # self._color = Color((0, 0, 0, 1.0))
        self.points = points
        # self.attrs: List[Attr] = [self._color]

    def render(self, surf: pygame.Surface):
        raise NotImplementedError

        # for attr in reversed(self.attrs):
        #     attr.enable()
        # self.render1()
        # for attr in self.attrs:
        #     attr.disable()

    @property
    def color(self):
        return self._color

    def transform(self, transformation: Transformation):
        self.points = map(lambda p: transformation.apply_to(p), self.points)

    # def render1(self):
    #     raise NotImplementedError

    # def add_attr(self, attr):
    #     self.attrs.append(attr)

    def set_color(self, r, g, b, alpha=1):
        self._color.vec4 = (r, g, b, alpha)


class Attr(object):
    def enable(self):
        raise NotImplementedError

    def disable(self):
        pass


# class Color(Attr):
#     def __init__(self, vec4):
#         self.vec4 = vec4

#     def enable(self):
#         # gl.glColor4f(*self.vec4)


# class LineStyle(Attr):
#     def __init__(self, style):
#         self.style = style

#     def enable(self):
#         gl.glEnable(gl.GL_LINE_STIPPLE)
#         gl.glLineStipple(1, self.style)

#     def disable(self):
#         gl.glDisable(gl.GL_LINE_STIPPLE)


# class LineWidth(Attr):
#     def __init__(self, stroke):
#         self.stroke = stroke

#     def enable(self):
#         gl.glLineWidth(self.stroke)


# class Point(Geom):
#     def __init__(self):
#         Geom.__init__(self)

#     def render1(self):
#         gl.glBegin(gl.GL_POINTS)  # draw point
#         gl.glVertex3f(0.0, 0.0, 0.0)
#         gl.glEnd()


class FilledPolygon(Geom):
    def __init__(self, points):
        super().__init__(points)

    def render(self, surf):
        pygame.draw.polygon(surf, color=self.color, points=self.points)


class Circle(Geom):
    def __init__(self, center: pygame.Vector2, radius: float = 10):
        super().__init__()
        self.center = center
        self.radius = radius

    @property
    def center(self) -> pygame.Vector2:
        # Allows for inheriting transformation from parent
        return self.points[0]

    @center.setter
    def center(self, center: pygame.Vector2):
        self.points = [center]

    def render(self, surf):
        pygame.draw.circle(
            surf, color=self.color, center=self.center, radius=self.radius
        )


def make_polygon(v, filled=True):
    if filled:
        return FilledPolygon(v)
    else:
        return PolyLine(v, True)


def make_polyline(v):
    return PolyLine(v, False)


def make_capsule(length, width):
    # TODO: Remove
    l, r, t, b = 0, length, width / 2, -width / 2
    box = make_polygon([(l, b), (l, t), (r, t), (r, b)])
    circ0 = Circle(width / 2)
    circ1 = Circle(width / 2)
    # circ1.add_attr(Transform(translation=(length, 0)))
    geom = Compound([box, circ0, circ1])
    return geom


class Compound(Geom):
    def __init__(self, geoms: List[Geom]):
        super().__init__(self)
        self.geoms = geoms
        # for g in self.geoms:
        #     g.attrs = [a for a in g.attrs if not isinstance(a, Color)]

    def render(self, surf: pygame.Surface):
        for geom in self.geoms:
            surf = geom.render()

        return surf

    def transform(self, transformation):
        self.geoms = map(lambda geom: geom.transform(transformation), self.geoms)


class PolyLine(Geom):
    def __init__(self, points: List[pygame.Vector2]):
        super().__init__(points)

    def render(self, surf: pygame.Surface):
        gray = (127, 127, 127)
        return pygame.draw.lines(surface=surf, points=self.points, color=gray)


class Line(Geom):
    def __init__(self, start: pygame.Vector2, end: pygame.Vector2):
        super().__init__(self, [start, end])

    @property
    def start(self) -> pygame.Vector2:
        return self.points[0]

    @property
    def end(self) -> pygame.Vector2:
        return self.points[1]

    def render(self, surf: pygame.Surface):
        gray = (127, 127, 127)

        return pygame.draw.line(surf, gray, self.start, self.end)
