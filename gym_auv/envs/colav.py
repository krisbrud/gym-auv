"""
This module implements the AUV gym environment through the AUVenv class.
"""
import numpy as np
import numpy.linalg as linalg

import gym
from gym.utils import seeding

import gym_auv.utils.geomutils as geom
from gym_auv.objects.auv import AUV2D
from gym_auv.objects.obstacles import StaticObstacle
from gym_auv.environment import BaseShipScenario

class ColavEnv(BaseShipScenario):
    """
    Creates an environment with a vessel, goal and obstacles.

    Attributes
    ----------
    config : dict
        The configuration dictionary.
    nstates : int
        The number of state variables passed to the agent.
    nsectors : int
        The number of obstacle detection sectors around the vessel.
    vessel : gym_auv.objects.auv.AUV2D
        The AUV that is controlled by the agent.
    goal : np.array
        The goal position.
    np_random : np.random.RandomState
        Random number generator.
    obstacles : list
        List of obstacles of type
        gym_auv.objects.obstacles.StaticObstacle.
    reward : float
        The accumulated reward.
    goal_dist : float
        The distance to the goal.
    last_action : np.array
        The last action that was preformed.
    steps : int
        Number of timesteps passed.
    action_space : gym.spaces.Box
        The action space. Consists of two floats, where the first, the
        propeller input can be between 0 and 1, and the second, the
        rudder position, can be between -1 and 1.
    observation_space : gym.spaces.Box
        The observation space. Consists of
        self.nstates + self.nsectors floats that must be between
        0 and 1.
    """

    def step_reward(self):
        """
        Calculates the step_reward and decides whether the episode
        should be ended.

        Returns
        -------
        done : bool
            If True the episode is ended.
        step_reward : double
            The reward for performing action at his timestep.
        """
        done = False
        step_reward = 0
        info = {"collision": False}

        last_dist = self.goal_dist
        self.goal_dist = linalg.norm(self.vessel.position - self.goal)
        progress = last_dist - self.goal_dist

        max_prog = self.config["cruise_speed"]*self.config["t_step_size"]
        speed_error = ((linalg.norm(self.vessel.velocity) - self.config["cruise_speed"])/self.vessel.max_speed)
        self.past_errors['speed'] = np.append(self.past_errors['speed'], speed_error)

        step_reward += (np.clip(progress/max_prog, -1, 1)*self.config["reward_ds"])
        step_reward += (max(speed_error, 0)*self.config["reward_speed_error"])

        for sector in range(self.nsectors):
            closeness = self.past_obs[-1, self.nstates + sector]
            dist = self.config["obst_detection_range"]*(1 - closeness)
            reward_range = self.config["obst_reward_range"]
            if closeness >= 1:
                step_reward = self.config["reward_collision"]
                info["collision"] = True
                if self.config["end_on_collision"]:
                    done = True
            elif dist < reward_range:
                step_reward += ((1 - dist/reward_range)*self.config["reward_closeness"])

        if (self.cumulative_reward < self.config["min_reward"]
            or self.goal_dist > 3*self.config["goal_dist"]
            or self.past_actions.shape[0] >= 20000
            or abs(self.goal_dist) < 5
        ):
            done = True

        return done, step_reward, info

    def reset(self):
        self.goal = None
        self.goal_dist = self.config["goal_dist"]
        return super().reset()

    def generate(self):
        """
        Sets up the environment. Places the goal and obstacles and
        creates the AUV.
        """
        init_pos = np.array([0, 0])
        init_angle = 2*np.pi*(self.np_random.rand()-0.5)
        goal_angle = 2*np.pi*(self.np_random.rand()-0.5)

        self.goal = self.config["goal_dist"]*np.array([np.cos(goal_angle), np.sin(goal_angle)])
        self.vessel = AUV2D(self.config["t_step_size"], np.hstack([init_pos, init_angle]))

        for _ in range(self.config["nobstacles"]):
            obst_dist = (0.75*self.config["goal_dist"]*(self.np_random.rand() + 0.2))
            obst_ang = (goal_angle + 2*np.pi*(self.np_random.rand()-0.5))
            position = (obst_dist*np.array([np.cos(obst_ang), np.sin(obst_ang)]))
            if linalg.norm(position) < 50:
                position[0] = np.sign(position[0])*50
                if position[0] == 0:
                    position[0] = 50
            radius = 15*(self.np_random.rand()+0.5)
            self.obstacles.append(StaticObstacle(position, radius))

    def observe(self):
        """
        Generates the observation of the environment.
        Parameters
        ----------
        action : np.array
            [propeller_input, rudder_position].

        Returns
        -------
        obs : np.array
            [
            surge velocity,
            sway velocity,
            heading error,
            distance to goal,
            propeller_input,
            rudder_positione,
            self.nsectors*[closeness to closest obstacle in sector]
            ]
            All observations are between -1 and 1.
        """
        obst_range = self.config["obst_detection_range"]

        goal_vector = self.goal - self.vessel.position
        goal_direction = np.arctan2(goal_vector[1], goal_vector[0])
        heading_error = float(geom.princip(goal_direction - self.vessel.heading))
        obs = np.zeros((self.nstates + self.nsectors,))

        obs[0] = np.clip(self.vessel.velocity[0] /self.vessel.max_speed, -1, 1)
        obs[1] = np.clip(self.vessel.velocity[1] / 0.26, -1, 1)
        obs[2] = np.clip(self.vessel.yawrate / 0.55, -1, 1)
        obs[3] = np.clip(heading_error / np.pi, -1, 1)
        for obst in self.obstacles:
            distance_vec = geom.Rzyx(0, 0, -self.vessel.heading).dot(
                np.hstack([obst.position - self.vessel.position, 0])
            )
            dist = linalg.norm(distance_vec)
            if dist < obst_range + obst.radius + self.vessel.width:
                ang = ((float(np.arctan2(distance_vec[1], distance_vec[0])) + np.pi) / (2*np.pi))
                closeness = 1 - np.clip(np.log(1 + max(dist - self.vessel.width - obst.radius, 0))/np.log(obst_range), 0, 1)
                isector = int(np.floor(ang*self.nsectors + np.pi/self.nsectors)) % self.nsectors
                if obs[self.nstates + isector] < closeness:
                    obs[self.nstates + isector] = closeness
        return obs
