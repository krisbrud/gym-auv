# from re import I
import pytest
import numpy as np

from gym_auv.objects.vessel import Vessel
from gym_auv.objects.obstacles import CircularObstacle
from gym_auv import DEFAULT_CONFIG


@pytest.fixture
def my_vessel() -> Vessel:
    init_state = np.array([5, -5, np.deg2rad(45)])
    vessel = Vessel(DEFAULT_CONFIG, init_state)
    return vessel


@pytest.fixture
def obst_behind_vessel() -> CircularObstacle:
    pos = np.array([0, -9.5])
    # pos = np.array([0, 0])
    radius = 1.5 

    obst = CircularObstacle(pos, radius)
    return obst


@pytest.fixture
def ranges(my_vessel: Vessel, obst_behind_vessel: CircularObstacle) -> np.ndarray:
    obstacles = [obst_behind_vessel]
    ranges, velocities = my_vessel.perceive(obstacles=obstacles)
    return ranges


def _is_intercepted(range: float):
    return 0 < range < 1


def test_no_obst_in_front(ranges: np.ndarray):
    idx_front = len(ranges) // 2
    assert not _is_intercepted(ranges[idx_front])


def test_obst_in_last_sensor(ranges: np.ndarray):
    assert _is_intercepted(ranges[-1])


def test_obst_in_first_sensor(ranges: np.ndarray):
    assert _is_intercepted(ranges[0])
