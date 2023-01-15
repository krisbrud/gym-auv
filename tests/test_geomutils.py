import pytest
import numpy as np

from gym_auv.utils.geomutils import transform_ned_to_body, transform_body_to_ned


def test_transform_ned_to_body():
    # Test that the unit vector in the north direction 
    # is transformed correctly when rotated 45 degrees clockwise

    # Define a unit vector in the north direction
    north_unit_vector = np.array([1, 0])

    heading = np.pi / 4  # 45 degrees clockwise

    # Transform the unit vector to the body frame
    north_unit_vector_body = transform_ned_to_body(north_unit_vector, np.zeros(2), heading)

    assert np.allclose(north_unit_vector_body, np.array([1, -1]) / np.sqrt(2))


def test_transform_body_to_ned():
    # Test that a straight-ahead velocity vector in the body frame is transformed correctly

    straight_ahead_unit_vector_body = np.array([1, 0])
    left_unit_vector_body = np.array([0, -1])

    heading = np.pi / 4  # 45 degrees clockwise

    # Transform the vector to the NED frame
    straight_ahead_unit_vector_ned = transform_body_to_ned(straight_ahead_unit_vector_body, np.zeros(2), heading)
    assert np.allclose(straight_ahead_unit_vector_ned, np.array([1, 1]) / np.sqrt(2))

    left_unit_vector_ned = transform_body_to_ned(left_unit_vector_body, np.zeros(2), heading)
    assert np.allclose(left_unit_vector_ned, np.array([1, -1]) / np.sqrt(2))

def test_transform_body_to_ned2():
    # Test that a straight-ahead velocity vector in the body frame is transformed correctly

    straight_ahead_unit_vector_body = np.array([1, 0])
    left_unit_vector_body = np.array([0, -1])

    heading = -np.pi / 4  # 45 degrees clockwise

    # Transform the vector to the NED frame
    straight_ahead_unit_vector_ned = transform_body_to_ned(straight_ahead_unit_vector_body, np.zeros(2), heading)
    assert np.allclose(straight_ahead_unit_vector_ned, np.array([1, -1]) / np.sqrt(2))

    left_unit_vector_ned = transform_body_to_ned(left_unit_vector_body, np.zeros(2), heading)
    assert np.allclose(left_unit_vector_ned, np.array([-1, -1]) / np.sqrt(2))