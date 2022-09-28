from abc import abstractmethod
from dataclasses import dataclass
from typing import List, Union
from typing_extensions import Self

import numpy as np
import pygame

from gym_auv.rendering.render2d import colors


@dataclass
class Transformation:
    translation: pygame.Vector2
    angle: float  # radians
    scale: float = 1

    def apply_to(self, point: pygame.Vector2) -> pygame.Vector2:
        translated = point + self.translation
        rotated = translated.rotate_rad(self.angle)
        scaled = rotated * self.scale

        return scaled


class BaseGeom:
    @abstractmethod
    def render(self, surf: pygame.Surface) -> Union[np.ndarray, None]:
        """Renders the shape(s) to the pygame surface"""
        pass

    @abstractmethod
    def transform(self, transformation: Transformation):
        pass


class BasePointGeom(BaseGeom):
    def __init__(
        self,
        points: List[pygame.Vector2],
        color: pygame.Color,
    ):
        self._points = points
        self._color = color

    @abstractmethod
    def render(self, surf: pygame.Surface) -> Union[np.ndarray, None]:
        """Renders the shape(s) to the pygame surface"""
        pass

    @property
    def color(self):
        return self._color

    @property
    def points(self):
        return self._points

    def transform(self, transformation: Transformation) -> Self:
        self._points = list(map(lambda p: transformation.apply_to(p), self.points))

        return self


class FilledPolygon(BasePointGeom):
    def __init__(self, points: List[pygame.Vector2], color: pygame.Color):
        super().__init__(points, color)

    def render(self, surf):
        pygame.draw.polygon(surf, color=self.color, points=self.points)


class Circle(BaseGeom):
    def __init__(
        self,
        center: pygame.Vector2,
        radius: float = 10,
        color: pygame.Color = colors.BLACK,
    ):
        super().__init__()
        self.center = center
        self.radius = radius
        self.color = color

    @property
    def center(self) -> pygame.Vector2:
        # Allows for inheriting transformation from parent
        return self.points[0]

    @center.setter
    def center(self, center: pygame.Vector2):
        self.points = [center]

    def transform(self, transformation: Transformation):
        self.center = transformation.apply_to(self.center)
        self.radius *= transformation.scale
        return self

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


class Compound(BaseGeom):
    def __init__(self, geoms: List[BaseGeom]):
        self.geoms = geoms

    def render(self, surf: pygame.Surface):
        for geom in self.geoms:
            geom.render(surf)

    def transform(self, transformation):
        self.geoms = list(map(lambda geom: geom.transform(transformation), self.geoms))


class PolyLine(BasePointGeom):
    def __init__(self, points: List[pygame.Vector2], color: pygame.Color):
        super().__init__(points, color)

    def render(self, surf: pygame.Surface):
        pygame.draw.lines(
            surface=surf, color=self.color, points=self.points, closed=False
        )


class Line(BasePointGeom):
    def __init__(
        self,
        start: pygame.Vector2,
        end: pygame.Vector2,
        color: pygame.Color = colors.GRAY,
    ):
        points = [start, end]
        super().__init__(points, color)

    @property
    def start(self) -> pygame.Vector2:
        return self.points[0]

    @property
    def end(self) -> pygame.Vector2:
        return self.points[1]

    def render(self, surf: pygame.Surface):
        pygame.draw.line(surf, self.color, self.start, self.end)
