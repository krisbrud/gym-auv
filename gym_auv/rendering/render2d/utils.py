from typing import List
import numpy as np
import pygame


def _clamp(x: int, min_val: int, max_val: int) -> int:
    return min(max(min_val, x), max_val)


def clamp_to_uint8(x: int) -> int:
    return _clamp(x, 0, 255)


def ndarray_to_vector2_list(arr: np.ndarray) -> List[pygame.Vector2]:
    """Converts an array of shape (N, 2) to a list of N Vector2 objects"""
    points = [pygame.Vector2(*p) for _, p in enumerate(arr)]
    return points
