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


class Geom:
    def __init__(self):
        self._color = Color((0, 0, 0, 1.0))
        self.attrs: List[Attr] = [self._color]

    def render(self):
        for attr in reversed(self.attrs):
            attr.enable()
        self.render1()
        for attr in self.attrs:
            attr.disable()

    @property
    def color(self):
        return self._color

    def render1(self):
        raise NotImplementedError

    def add_attr(self, attr):
        self.attrs.append(attr)

    def set_color(self, r, g, b, alpha=1):
        self._color.vec4 = (r, g, b, alpha)


class Attr(object):
    def enable(self):
        raise NotImplementedError

    def disable(self):
        pass


class Transform:
    def __init__(self, translation=(0.0, 0.0), rotation=0.0, scale=(1, 1)):
        self.set_translation(*translation)
        self.set_rotation(rotation)
        self.set_scale(*scale)

    def apply(self, geom):
        pass
        # gl.glPushMatrix()
        # gl.glTranslatef(
        #     self.translation[0], self.translation[1], 0
        # )  # translate to GL loc ppint
        # gl.glRotatef(rad2deg(self.rotation), 0, 0, 1.0)
        # gl.glScalef(self.scale[0], self.scale[1], 1)

    def disable(self):
        gl.glPopMatrix()

    def set_translation(self, newx, newy):
        self.translation = (float(newx), float(newy))

    def set_rotation(self, new):
        self.rotation = float(new)

    def set_scale(self, newx, newy):
        self.scale = (float(newx), float(newy))


class Color(Attr):
    def __init__(self, vec4):
        self.vec4 = vec4

    def enable(self):
        gl.glColor4f(*self.vec4)


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
    def __init__(self, v):
        Geom.__init__(self)
        self.v = v

    def render(self, surf):
        pygame.draw.polygon(surf, color=self.color, points=self.v)

        # if len(self.v) == 4:
        #     gl.glBegin(gl.GL_QUADS)
        # elif len(self.v) > 4:
        #     gl.glBegin(gl.GL_POLYGON)
        # else:
        #     gl.glBegin(gl.GL_TRIANGLES)
        # for p in self.v:
        #     gl.glVertex3f(p[0], p[1], 0)  # draw each vertex
        # gl.glEnd()


class Circle(Geom):
    def __init__(self, center: float = 0.0, radius: float = 10):
        Geom.__init__(self)
        self.center = center
        self.radius = radius

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
    circ1.add_attr(Transform(translation=(length, 0)))
    geom = Compound([box, circ0, circ1])
    return geom


class Compound(Geom):
    def __init__(self, gs):
        Geom.__init__(self)
        self.gs = gs
        for g in self.gs:
            g.attrs = [a for a in g.attrs if not isinstance(a, Color)]

    def render(self, surf: pygame.Surface):
        for g in self.gs:
            surf = g.render()

        return surf


class PolyLine(Geom):
    def __init__(self, v, close):
        Geom.__init__(self)
        self.v = v
        self.close = close
        # self.linewidth = LineWidth(1)
        self.add_attr(self.linewidth)

    def render(self, surf: pygame.Surface):
        gray = (127, 127, 127)
        return pygame.draw.lines(surface=surf, points=self.v, color=gray)


class Line(Geom):
    def __init__(self, start=(0.0, 0.0), end=(0.0, 0.0), linewidth=1):
        Geom.__init__(self)
        self.start = start
        self.end = end
        # self.linewidth = LineWidth(linewidth)
        self.add_attr(self.linewidth)

    def render(self, surf: pygame.Surface):
        gray = (127, 127, 127)
        start = pygame.Vector2(*self.start)
        end = pygame.Vector2(*self.end)

        return pygame.draw.line(surf, gray, start, end)
