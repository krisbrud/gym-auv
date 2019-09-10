import numpy as np
import shapely.geometry

class StaticObstacle():
    def __init__(self, position, radius, color=(0.6, 0, 0)):
        self.color = color
        if not isinstance(position, np.ndarray):
            position = np.array(position)
        if radius < 0:
            raise ValueError
        self.radius = radius
        self.position = position.flatten()
        self.observed = False
        self.collided = False
        self.circle = shapely.geometry.Point(*self.position).buffer(self.radius).boundary

    def step(self):
        pass
