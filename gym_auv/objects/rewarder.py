import numpy as np
from abc import ABC, abstractmethod
# from gym_auv.objects.vessel import Vessel
from dataclasses import dataclass
from typing import Optional

# TODO remove these
deg2rad = np.pi / 180
rad2deg = 180 / np.pi


def _sample_lambda(scale):
    log = -np.random.gamma(1, scale)
    y = np.power(10, log)
    return y


def _sample_eta():
    y = np.random.gamma(shape=1.9, scale=0.6)
    return y


@dataclass
class RewarderParams:
    # Colav rewarder
    gamma_theta: Optional[float] = None
    gamma_x: Optional[float] = None
    gamma_v_y: Optional[float] = None
    gamma_y_e: Optional[float] = None
    penalty_yawrate: Optional[float] = None
    penalty_torque_change: Optional[float] = None
    cruise_speed: Optional[float] = None
    neutral_speed: Optional[float] = None
    collision: Optional[float] = None
    lambda_: Optional[float] = None
    eta: Optional[float] = None
    negative_multiplier: Optional[float] = None
    reward_scale: Optional[float] = None

    # COLREG rewarder
    gamma_x_stat: Optional[float] = None
    gamma_x_starboard: Optional[float] = None
    gamma_x_port: Optional[float] = None
    alpha_lambda: Optional[float] = None
    gamma_min_x: Optional[float] = None
    gamma_weight: Optional[float] = None


class ColavParams(RewarderParams):
    def __init__(self):
        self.gamma_theta = 10.0
        self.gamma_x = 0.1  # 0.1
        self.gamma_v_y = 1.0
        self.gamma_y_e = 5.0
        self.penalty_yawrate = 0  # 10.0
        self.penalty_torque_change = 0.0
        self.cruise_speed = 0.1
        self.neutral_speed = 0.05
        self.collision = -500  # -2000.0 #  -10000.0
        self.lambda_ = 0.6  # 0.5
        self.eta = 0  # _sample_eta()
        self.negative_multiplier = 2
        self.reward_scale = 0.5


class ColregParams(RewarderParams):
    def __init__(self):
        self.gamma_theta = 10.0
        self.gamma_x_stat = 0.09
        self.gamma_x_starboard = 0.07
        self.gamma_x_port = 0.09
        self.alpha_lambda = 3.5
        self.gamma_min_x = 0.04
        self.gamma_weight = 2
        self.gamma_v_y = 2.0
        self.gamma_y_e = 5.0
        self.penalty_yawrate = 0.0
        self.penalty_torque_change = 0.01
        self.cruise_speed = 0.1
        self.neutral_speed = 0.1
        self.negative_multiplier = 2.0
        self.collision = -10000.0
        self.lambda_ = 0.5
        self.eta = 0.2


class PathFollowParams(RewarderParams):
    def __init__(self):
        self.gamma_theta = 10.0
        self.gamma_x = 0.1
        self.gamma_v_y = 1.0
        self.gamma_y_e = 5.0
        self.penalty_yawrate = 0  # 10.0 # 20  # 0.0
        self.penalty_torque_change = 0.0
        self.cruise_speed = 0.1
        self.neutral_speed = 0.05  # 0.05
        self.negative_multiplier = 2.0
        self.collision = -10000.0
        self.lambda_ = 0.5
        self.eta = 0


class LosColavParams(RewarderParams):
    def __init__(self):
        self.gamma_theta = 10.0
        self.gamma_x = 0.1  # 0.1
        self.gamma_v_y = 1.0
        self.gamma_y_e = 5.0
        self.penalty_yawrate = 0  # 10.0
        self.penalty_torque_change = 0.0
        self.cruise_speed = 0.1
        self.neutral_speed = 0.05
        self.collision = -500  # -2000.0 #  -10000.0
        self.lambda_ = 0.6  # 0.5
        self.eta = 0
        self.negative_multiplier = 2
        self.reward_scale = 0.5


class BaseRewarder(ABC):
    @abstractmethod
    def calculate(self, vessel_data, params: RewarderParams) -> float:
        """
        Calculates the step reward and decides whether the episode
        should be ended.

        Parameters
        ----------
        vessel_data : dict
            Dictionary containing the latest data of the vessel.
            Returned by vessel.req_latest_data()
        parameters : RewarderParams
            Dataclass containing the parameters of the rewarder.

        Returns
        -------
        reward : float
            The reward for performing action at this timestep.
        """

class PathFollowRewarder(BaseRewarder):
    """Path following rewarder, no collision avoidance."""
    def calculate(self, vessel_data, params: PathFollowParams):
        # latest_data = vessel_data["req_latest_data()
        
        nav_states = vessel_data["navigation"]
        # measured_distances = vessel_data["distance_measurements"]
        # measured_speeds = vessel_data["speed_measurements"]
        collision = vessel_data["collision"]

        if collision:
            reward = params.collision * (1 - params.lambda_)
            return reward

        reward = 0

        # Extracting navigation states
        cross_track_error = nav_states["cross_track_error"]
        heading_error = nav_states["heading_error"]

        # Calculating path following reward component
        cross_track_performance = np.exp(
            -params.gamma_y_e * np.abs(cross_track_error)
        )
        # path_reward = (np.cos(heading_error) * vessel_data["speed / vessel_data["max_speed) * cross_track_performance
        path_reward = (
            1 + np.cos(heading_error) * vessel_data["speed"] / vessel_data["max_speed"]
        ) * (1 + cross_track_performance) - 1

        # Extra penalty for going backwards, as going continously in a circle is
        # a local minima giving an average positive reward with this strategy.
        # gamma_backwards = 2.5
        # path_reward = (
        #     1
        #     + min(np.cos(heading_error), gamma_backwards * np.cos(heading_error))
        #     * vessel_data["speed / vessel_data["max_speed
        # ) * (1 + cross_track_performance) - 1

        # Calculating living penalty
        living_penalty = (
            params.lambda_ * (2 * params.neutral_speed + 1)
            + params.eta * params.neutral_speed
        )

        # Calculating total reward
        reward = (
            path_reward
            - living_penalty
            + params.eta * vessel_data["speed"] / vessel_data["max_speed"]
            - params.penalty_yawrate * abs(vessel_data["yaw_rate"])
        )

        # if reward < 0:
        #     reward *= params.negative_multiplier

        return reward


def meyer_colav_reward(
    n_sensors: int,
    sensor_angles: np.ndarray,
    measured_distances: np.ndarray,
    measured_speeds: np.ndarray,
    sensor_range: float,
    gamma_theta: float,
    gamma_x: float,
    gamma_v_y: float,
):
    """Calculate reward/penalty based on how close the vessel is to obstacles according to the sensors."""
    closeness_penalty_num = 0
    closeness_penalty_den = 0
    if n_sensors > 0:
        for isensor in range(n_sensors):
            angle = sensor_angles[isensor]
            x = measured_distances[isensor]
            speed_vec = measured_speeds[:, isensor]
            weight = 1 / (1 + np.abs(gamma_theta * angle))
            raw_penalty = sensor_range * np.exp(
                -gamma_x * x + gamma_v_y * max(0, speed_vec[1])
            )
            weighted_penalty = weight * raw_penalty
            closeness_penalty_num += weighted_penalty
            closeness_penalty_den += weight

        closeness_reward = -closeness_penalty_num / closeness_penalty_den
    else:
        closeness_reward = 0

    return closeness_reward


def los_path_reward(
    velocity_ned: np.ndarray, lookahead_unit_vec_ned: np.ndarray, coeff: float = 1.0
):
    """Calculates the path following reward based on Fossen-style Line-of-Sight Guidance

    Parameters
    ----------
    velocity_ned : np.ndarray
        The velocity vector in NED coordinates
    lookahead_unit_vec_ned : np.ndarray
        The unit vector pointing in the direction of the lookahead point
    coeff : float, optional

    Returns
    -------
    float
        The path following reward
    """
    path_reward = np.dot(velocity_ned, lookahead_unit_vec_ned) * coeff

    return path_reward


class LOSColavRewarder(BaseRewarder):
    def calculate(self, vessel_data, params: LosColavParams):
        # latest_data = vessel_data["req_latest_data()
        nav_states = vessel_data["navigation"]
        measured_distances = vessel_data["distance_measurements"]
        measured_speeds = vessel_data["speed_measurements"]
        collision = vessel_data["collision"]

        if collision:
            reward = params.collision * (1 - params.lambda_)
            return reward * params.reward_scale

        # Calculating path following reward component
        lookahead_vector_normalized_ned = nav_states["lookahead_vector_normalized_ned"]
        velocity_ned = nav_states["velocity_ned"]

        path_reward = los_path_reward(
            lookahead_unit_vec_ned=lookahead_vector_normalized_ned,
            velocity_ned=velocity_ned,
            coeff=0.8,  # 0.5,
        )

        if path_reward < 0:
            # Double the penalty of going backwards
            # This is intended to discourage the vessel from going in circles,
            # without also multiplying slight negative rewards from closeness to obstacles
            # while going in the right direction
            path_reward *= params.negative_multiplier

        # Calculating obstacle avoidance reward component
        closeness_reward = meyer_colav_reward(
            n_sensors=vessel_data["n_sensors"],
            sensor_angles=vessel_data["sensor_angles"],
            measured_distances=measured_distances,
            measured_speeds=measured_speeds,
            sensor_range=vessel_data["sensor_range"],
            gamma_theta=params.gamma_theta,
            gamma_x=params.gamma_x,
            gamma_v_y=params.gamma_v_y,
        )

        # Calculating living penalty
        living_penalty = (
            params.lambda_ * (2 * params.neutral_speed + 1)
            + params.eta * params.neutral_speed
        )

        reached_goal_reward = 0
        if vessel_data["reached_goal"]:
            reached_goal_reward = 10

        # Calculating total reward
        reward = (
            params.lambda_ * path_reward
            + reached_goal_reward
            + (1 - params.lambda_) * closeness_reward
            - living_penalty
            # + params.eta * vessel_data["speed / vessel_data["max_speed
            # - params.penalty_yawrate * abs(vessel_data["yaw_rate)
        )

        # if reward < 0:
        #     reward *= params.negative_multiplier

        return reward * params.reward_scale


class ColavRewarder(BaseRewarder):
    def calculate(self, vessel_data, params: ColavParams):
        # vessel_data = vessel_data["req_latest_data()
        nav_states = vessel_data["navigation"]
        measured_distances = vessel_data["distance_measurements"]
        measured_speeds = vessel_data["speed_measurements"]
        collision = vessel_data["collision"]

        if collision:
            reward = params.collision * (1 - params.lambda_)
            return reward

        reward = 0

        # Extracting navigation states
        cross_track_error = nav_states["cross_track_error"]
        heading_error = nav_states["heading_error"]

        # Calculating path following reward component
        cross_track_performance = np.exp(
            -params.gamma_y_e * np.abs(cross_track_error)
        )
        path_reward = (
            1 + np.cos(heading_error) * vessel_data["speed"] / vessel_data["max_speed"]
        ) * (1 + cross_track_performance) - 1

        # Calculating obstacle avoidance reward component
        closeness_penalty_num = 0
        closeness_penalty_den = 0
        if vessel_data["n_sensors"] > 0:
            for isensor in range(vessel_data["n_sensors"]):
                angle = vessel_data["sensor_angles"][isensor]
                x = measured_distances[isensor]
                speed_vec = measured_speeds[:, isensor]
                weight = 1 / (1 + np.abs(params.gamma_theta * angle))
                raw_penalty = vessel_data["sensor_range"] * np.exp(
                    -params.gamma_x * x
                    + params.gamma_v_y * max(0, speed_vec[1])
                )
                weighted_penalty = weight * raw_penalty
                closeness_penalty_num += weighted_penalty
                closeness_penalty_den += weight

            closeness_reward = -closeness_penalty_num / closeness_penalty_den
        else:
            closeness_reward = 0

        closeness_reward *= (
            180 / 256
        )  # Scale closeness reward down because of more sensors

        # Calculating living penalty
        living_penalty = (
            params.lambda_ * (2 * params.neutral_speed + 1)
            + params.eta * params.neutral_speed
        )

        # Calculating total reward
        reward = (
            params.lambda_ * path_reward
            + (1 - params.lambda_) * closeness_reward
            - living_penalty
            + params.eta * vessel_data["speed"] / vessel_data["max_speed"]
            - params.penalty_yawrate * abs(vessel_data["yaw_rate"])
        )

        if reward < 0:
            reward *= params.negative_multiplier

        return reward


class ColregRewarder(BaseRewarder):
    def calculate(self, vessel_data, params: ColregParams):
        # latest_data = vessel_data["req_latest_data()
        nav_states = vessel_data["navigation"]
        measured_distances = vessel_data["distance_measurements"]
        measured_speeds = vessel_data["speed_measurements"]
        collision = vessel_data["collision"]
        # print([x[1] for x in measured_speeds])
        if collision:
            reward = params.collision  # *(1-params['lambda'])
            return reward

        reward = 0

        # Extracting navigation states
        cross_track_error = nav_states["cross_track_error"]
        heading_error = nav_states["heading_error"]

        # Calculating path following reward component
        cross_track_performance = np.exp(
            -params.gamma_y_e * np.abs(cross_track_error)
        )
        path_reward = (
            1 + np.cos(heading_error) * vessel_data["speed"] / vessel_data["max_speed"]
        ) * (1 + cross_track_performance) - 1

        # Calculating obstacle avoidance reward component
        closeness_penalty_num = 0
        closeness_penalty_den = 0
        static_closeness_penalty_num = 0
        static_closeness_penalty_den = 0
        closeness_reward = 0
        static_closeness_reward = 0
        moving_distances = []
        lambdas = []

        speed_weight = 2

        if vessel_data["n_sensors > 0"]:
            for isensor in range(vessel_data["n_sensors"]):
                angle = vessel_data["sensor_angles"][isensor]
                x = measured_distances[isensor]
                speed_vec = measured_speeds[isensor]  # TODO

                if speed_vec.any():

                    if speed_vec[1] > 0:
                        params.lambda_ = 1 / (1 + np.exp(-0.04 * x + 4))
                    if speed_vec[1] < 0:
                        params.lambda_ = 1 / (1 + np.exp(-0.06 * x + 3))
                    lambdas.append(params.lambda_)

                    weight = 2 / (
                        1 + np.exp(params.gamma_weight * np.abs(angle))
                    )
                    moving_distances.append(x)

                    if (
                        angle < 0 * deg2rad and angle > -112.5 * deg2rad
                    ):  # straffer høyre side

                        raw_penalty = 100 * np.exp(
                            -params.gamma_x_starboard * x
                            + speed_weight * speed_vec[1]
                        )
                    else:
                        raw_penalty = 100 * np.exp(
                            -params.gamma_x_port * x
                            + speed_weight * speed_vec[1]
                        )
                    # else:
                    #    raw_penalty = 100*np.exp(-params['gamma_x_stat']*x)

                    weighted_penalty = (
                        (1 - params.lambda_) * weight * raw_penalty
                    )
                    closeness_penalty_num += weighted_penalty
                    closeness_penalty_den += weight

                else:

                    weight = 1 / (1 + np.abs(params.gamma_theta * angle))
                    raw_penalty = 100 * np.exp(-params.gamma_x_stat * x)
                    weighted_penalty = weight * raw_penalty
                    static_closeness_penalty_num += weighted_penalty
                    static_closeness_penalty_den += weight

            if closeness_penalty_num:
                closeness_reward = -closeness_penalty_num / closeness_penalty_den

            if static_closeness_penalty_num:
                static_closeness_reward = (
                    -static_closeness_penalty_num / static_closeness_penalty_den
                )

        # if len(moving_distances) != 0:
        #    min_dist = np.amin(moving_distances)
        #    params['lambda'] = 1/(1+np.exp(-0.04*min_dist+4))
        # params['lambda'] = 1/(1+np.exp(-(params['gamma_min_x']*min_dist-params['alpha_lambda'])))
        # else:
        #    params['lambda'] = 1

        if len(lambdas):
            path_lambda = np.amin(lambdas)
        else:
            path_lambda = 1

        # if path_reward > 0:
        #    path_reward = path_lambda*path_reward
        # Calculating living penalty
        living_penalty = 1  # .2*(params['neutral_speed']+1) + params['eta']*params['neutral_speed']

        # Calculating total reward
        reward = (
            path_lambda * path_reward
            + static_closeness_reward
            + closeness_reward
            - living_penalty
            + params.eta * vessel_data["speed"] / vessel_data["max_speed"]
        )  # - \
        # params['penalty_yawrate']*abs(vessel_data["yaw_rate)

        if reward < 0:
            reward *= params.negative_multiplier

        return reward
