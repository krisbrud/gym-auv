# from re import I
import pytest
import numpy as np

from gym_auv.objects.vessel import Vessel
from gym_auv.objects.obstacles import CircularObstacle, VesselObstacle
from gym_auv import DEFAULT_CONFIG


@pytest.fixture
def my_vessel() -> Vessel:
    init_state = np.array([5, -5, np.deg2rad(45)])
    vessel = Vessel(DEFAULT_CONFIG, init_state)
    return vessel


@pytest.fixture(params=[True, False])
def obst_behind_vessel(request) -> CircularObstacle:
    if request.param:
        pos = np.array([0, -9.5])
    else:
        pos = np.array([0, -10.5])
    radius = 1.5 

    obst = CircularObstacle(pos, radius)
    return obst

# @pytest.fixture
# def vessel_obst() -> VesselObstacle:
#     obst_position = [5, -15]
#     # obst_position = [5, 10]
#     obst_speed = 2
#     obst_direction = -np.pi / 2
#     other_vessel_trajectory = []
#     for i in range(10000):
#         other_vessel_trajectory.append(
#             (
#                 i,
#                 (
#                     obst_position[0] + i * obst_speed * np.cos(obst_direction),
#                     obst_position[1] + i * obst_speed * np.sin(obst_direction),
#                 ),
#             )
#         )
    
#     obst_radius = 0.01
#     other_vessel_obstacle = VesselObstacle(
#         width=obst_radius, trajectory=other_vessel_trajectory
#     )
#     return other_vessel_obstacle

@pytest.fixture
def ranges(my_vessel: Vessel, obst_behind_vessel: CircularObstacle) -> np.ndarray:
    obstacles = [obst_behind_vessel]
    ranges, velocities = my_vessel.perceive(obstacles=obstacles)
    return ranges

@pytest.fixture
def max_range(my_vessel) -> float:
    return my_vessel.config.sensor.range

def _is_intercepted(range: float, max_range: float):
    return 0 < range < max_range


def test_no_obst_in_front(ranges: np.ndarray, max_range: float):
    idx_front = len(ranges) // 2
    assert not _is_intercepted(ranges[idx_front], max_range)


def test_obst_in_last_sensor(ranges: np.ndarray, max_range: float):
    assert _is_intercepted(ranges[-1], max_range)


def test_obst_in_first_sensor(ranges: np.ndarray, max_range: float):
    assert _is_intercepted(ranges[0], max_range)


# def test_vessel_obstacle_velocities(my_vessel: Vessel, vessel_obst: VesselObstacle):
#     obstacles = [vessel_obst]
#     my_vessel.config.sensor.use_velocity_observations = True
#     ranges, velocities = my_vessel.perceive(obstacles=obstacles)
#     # assert velocities[0] == vessel_obst.speed
#     argmax_0 = np.argmax(velocities[0, :])
#     argmax_1 = np.argmax(velocities[1, :])
#     argmin_0 = np.argmin(velocities[0, :])
#     argmin_1 = np.argmin(velocities[1, :])
#     print("velocities", velocities)