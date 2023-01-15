import math
from typing import Tuple, Union
import gym
import numpy as np
from gym.utils import seeding
import gym_auv
from gym_auv.objects.path import Path

from gym_auv.objects.vessel import Vessel
from gym_auv.objects.rewarder import ColavRewarder
from gym_auv.render2d.state import RenderableState
from gym_auv.objects.vessel.sensor import (
    make_occupancy_grid,
    get_relative_positions_of_lidar_measurements,
)
from gym_auv.render2d.renderer import Renderer2d
from gym_auv.render2d.renderer import FPS as rendererFPS2D
import gym_auv.render2d.renderer

# import gym_auv.rendering.render3d as render3d
from gym_auv.utils.clip_to_space import clip_to_space
import gym_auv.utils.geomutils as geom
import gym.spaces
from abc import ABC, abstractmethod


class BaseEnvironment(gym.Env, ABC):
    """Creates an environment with a vessel and a path."""

    metadata = {
        "render.modes": ["human", "rgb_array", "state_pixels"],
        "video.frames_per_second": rendererFPS2D,
    }

    def __init__(
        self,
        env_config: Union[gym_auv.Config, dict],
        test_mode: bool = False,
        renderer: Union[str, None] = "2d",
        render_mode: str = "rgb_array",  # "human",
        verbose: bool = False,
        **kwargs,
    ):
        """The __init__ method declares all class atributes and calls
        the self.reset() to intialize them properly.

        Parameters
        ----------
            env_config : dict
                Configuration parameters for the environment.
                The default values are set in __init__.py
            test_mode : bool
                If test_mode is True, the environment will not be autonatically reset
                due to too low cumulative reward or too large distance from the path.
            renderer : {'2d', '3d', 'both'}
                Whether to use 2d or 3d rendering. 'both' is currently broken.
            verbose
                Whether to print debugging information.
        """

        if not hasattr(self, "_rewarder_class"):
            self._rewarder_class = ColavRewarder
            self._n_moving_obst = 10
            self._n_moving_stat = 10

        self.test_mode = test_mode
        self.renderer = renderer
        self.verbose = verbose

        # As some RL frameworks (RLlib) requires the config object to be a dictionary,
        # a workaround for this is wrapping our `Config` object in a dict, while keeping type safety of
        # using dataclasses for config
        if isinstance(env_config, dict):
            self.config = env_config["config"]
            assert isinstance(self.config, gym_auv.Config), (
                "Expected config attribute of env_config to be"
                f"gym_auv.Config, but got type {type(self.config)}!"
            )
        elif isinstance(env_config, gym_auv.Config):
            # Config argument should already be the right type
            self.config = env_config

        print("self.config", self.config)
        # print("self.config.vessel", self.config.vessel)

        self.episode = 0
        self.total_t_steps = 0
        self.t_step = 0
        self.cumulative_reward = 0
        self.history = []
        self.rewarder = None

        # Declaring attributes
        self.obstacles = []
        self.vessel = None
        self.path = None

        self.reached_goal = None
        self.collision = None
        self.progress = None
        self.last_reward = None
        self.last_episode = None
        self.rng = None
        self.seed()
        self._tmp_storage = None
        self._last_image_frame = None

        self._action_space = gym.spaces.Box(
            low=np.array([0.0, -1.0]),
            high=np.array([1.0, 1.0]),
            dtype=np.float32,
        )

        self._build_observation_space()

        # Initializing rendering
        self._renderer2d = None
        self._viewer3d = None
        self.render_mode = render_mode
        if self.renderer == "2d" or self.renderer == "both":
            self._renderer2d = Renderer2d(
                render_fps=self.metadata["video.frames_per_second"]
            )

        self.reset()

    def _build_observation_space(self) -> None:
        """Initializes the observation space according to the config"""

        obs_space_dict = {}

        if self.config.sensor.dense_observation_size > 0:
            obs_space_dict["dense"] = gym.spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(self.config.sensor.dense_observation_size,),
                dtype=np.float32,
            )

        if self.config.sensor.use_image_observation:
            width, height = self.config.sensor.image_shape
            self.image_output_renderer = Renderer2d(width=width, height=height)

            image_channels = 3  # RGB
            if self.config.sensor.image_channel_first:
                image_shape_with_channels = (
                    image_channels,
                    *self.config.sensor.image_shape,
                )
            else:
                image_shape_with_channels = (
                    *self.config.sensor.image_shape,
                    image_channels,
                )

            assert (
                len(image_shape_with_channels) == 3
            ), "Image shape must be 3-dimensional!"

            obs_space_dict["image"] = gym.spaces.Box(
                low=0,
                high=255,
                shape=image_shape_with_channels,
                dtype=np.uint8,
            )

        if self.config.sensor.use_lidar:
            if self.config.sensor.use_occupancy_grid:
                # Add an occupancy grid encoding lidar measurements as well
                # as the path to the output
                grid_size = self.config.sensor.occupancy_grid_size
                obs_space_dict["occupancy"] = gym.spaces.Box(
                    low=-1.0,
                    high=1.0,
                    shape=(2, grid_size, grid_size),
                    dtype=np.float32,
                )
            else:
                # Add lidar measurements to the output instead
                obs_space_dict["lidar"] = gym.spaces.Box(
                    low=-1.0,
                    high=1.0,
                    shape=self.config.sensor.lidar_shape,
                    dtype=np.float32,
                )

        if self.config.sensor.use_dict_observation:
            self._observation_space = gym.spaces.Dict(obs_space_dict)
        else:
            # Make a flattened observation, but use the dictionary that is built as a
            # guide to which observations should be a part of it
            observation_shapes = [space.shape for space in obs_space_dict]

            # Calculate the flattened size of the observation spaces by:
            # 1. Calculating the flat size of individual spaces by multiplying them out
            # e.g. (2, 64, 64) -> 2 * 64 * 64 = 8192, or (6,) -> 6
            observation_sizes = [math.prod(shape) for shape in observation_shapes]

            # 2. Summing the individual contributions
            observation_size = sum(observation_sizes)

            self._observation_space = gym.spaces.Box(
                low=-1, high=1, shape=(observation_size), dtype=np.float32
            )

    @property
    def action_space(self) -> gym.spaces.Box:
        """Array defining the shape and bounds of the agent's action."""
        return self._action_space

    @property
    def observation_space(self) -> gym.spaces.Box:
        """Array defining the shape and bounds of the agent's observations."""
        return self._observation_space

    def reset(self, save_history=True, **kwargs):
        """Reset the environment's state. Returns observation.

        Returns
        -------
        obs : np.ndarray
            The initial observation of the environment.
        """

        # print('Resetting environment')

        if self.verbose:
            print(
                "Resetting environment... Last reward was {:.2f}".format(
                    self.cumulative_reward
                )
            )

        # Seeding
        if self.rng is None:
            self.seed()

        # Saving information about episode
        if self.t_step:
            self.save_latest_episode(save_history=save_history)

        # Incrementing counters
        self.episode += 1
        self.total_t_steps += self.t_step

        # Resetting all internal variables
        self.cumulative_reward = 0
        self.t_step = 0
        self.last_reward = 0
        self.reached_goal = False
        self.collision = False
        self.progress = 0
        self._last_image_frame = None

        # Generating a new environment
        if self.verbose:
            print("Generating scenario...")
        self._generate()
        if self.rewarder is None:
            if self.verbose:
                print(
                    "Warning: Rewarder not initialized by _generate() method call. Selecting default ColavRewarder instead."
                )
            self.rewarder = self._rewarder_class()
            # self.rewarder = ColavRewarder() 
        if self.verbose:
            print("Generated scenario")

        # Getting initial observation vector
        obs = self.observe()
        if self.verbose:
            print("Calculated initial observation")

        # Resetting temporary data storage
        self._tmp_storage = {
            "cross_track_error": [],
        }

        if self.config.episode.use_terminated_truncated_step_api:
            return obs, {}
        else:
            return obs

    def observe(self) -> np.ndarray:
        """Returns the array of observations at the current time-step.

        Returns
        -------
        obs : np.ndarray
            The observation of the environment.
        """
        observations = {}

        navigation_states = self.vessel.navigate(self.path)
        observations["dense"] = navigation_states

        if self.config.sensor.use_image_observation:
            image = self._make_image()
            observations["image"] = image

        if bool(self.config.sensor.use_lidar):
            sensor_closenesses, sensor_velocities = self.vessel.perceive(self.obstacles)

            if self.config.sensor.use_occupancy_grid:
                occupancy_grids = self._make_occupancy_grids(
                    sensor_closenesses, self.path
                )
                observations["occupancy"] = occupancy_grids
            else:
                if self.config.sensor.use_velocity_observations:
                    # Add the velocity of the obstacles to the lidar observations
                    observations["lidar"] = np.vstack(
                        (sensor_closenesses, sensor_velocities)
                    )
                else:
                    observations["lidar"] = sensor_closenesses

        # Clamp/clip the observation to the valid domain as specified by the Space
        if isinstance(self.observation_space, gym.spaces.Box):
            flat_obs = []
            for o in observations.values():
                flat_obs.append(o.flatten())

            obs = np.clip(
                flat_obs,
                a_min=self.observation_space.low,
                a_max=self.observation_space.high,
            )
        elif isinstance(self.observation_space, gym.spaces.Dict):
            # TODO: Remove this code if it works
            # sensor_closenesses = sensor_closenesses.reshape(1, -1)  # Add leading axis
            # raw_lidar_obs = np.vstack((sensor_closenesses, sector_velocities))
            # raw_obs = {
            #     "proprioceptive": navigation_states,
            #     "lidar": raw_lidar_obs,
            # }
            obs = clip_to_space(observations, self.observation_space)

        return obs

    def _make_image(self) -> np.ndarray:
        """Returns the image observation at the current time-step.

        Returns
        -------
        image : np.ndarray
            The image observation of the environment.
        """
        width, height = self.config.sensor.image_shape
        zoom = (width / 2) / self.config.sensor.range
        renderer = Renderer2d(width=width, height=height, zoom=zoom)

        image = renderer.render(
            state=self.renderable_state,
            render_mode="rgb_array",
            image_observation_mode=True,
        )
        return image

    def step(self, action: list) -> Tuple[np.ndarray, float, bool, dict]:
        """
        Steps the environment by one timestep. Returns observation, reward, done, info.

        Parameters
        ----------
        action : np.ndarray
            [thrust_input, torque_input].

        Returns
        -------
        obs : np.ndarray
            Observation of the environment after action is performed.
        reward : double
            The reward for performing action at his timestep.
        done : bool
            If True the episode is ended, due to either a collision or having reached the goal position.
        info : dict
            Dictionary with data used for reporting or debugging
        """

        # action[0] = (action[0] + 1)/2 # Done to be compatible with RL algorithms that require symmetric action spaces
        if np.isnan(action).any():
            action = np.zeros(action.shape)

        # If the environment is dynamic, calling self.update will change it.
        self._update()

        # Updating vessel state from its dynamics model
        self.vessel.step(action)

        # Getting observation vector
        obs = self.observe()
        vessel_data = self.vessel.req_latest_data()
        self.collision = vessel_data["collision"]
        self.reached_goal = vessel_data["reached_goal"]
        self.goal_distance = vessel_data["navigation"]["goal_distance"]
        self.progress = vessel_data["progress"]

        # Receiving agent's reward
        # reward = self.rewarder.calculate(vessel_data, parameters=self.config.rewarder.params)
        # reward = self.rewarder.calculate(vessel_data=vessel_data, parameters=self.config.rewarder.params)
        reward = self.rewarder.calculate(vessel_data=vessel_data, params=self.config.rewarder.params)
        self.last_reward = reward
        self.cumulative_reward += reward

        info = {}
        info["collision"] = self.collision
        info["reached_goal"] = self.reached_goal
        info["goal_distance"] = self.goal_distance
        info["progress"] = self.progress
        info["collision_or_reached_goal"] = self.collision or self.reached_goal

        if self.config.episode.return_latest_data_in_info:
            info["latest_data"] = self.vessel.req_latest_data()
        # Testing criteria for ending the episode
        # done = self._isdone()
        truncated = self._is_truncated()
        terminated = self._is_terminal()

        done = terminated or truncated

        self._save_latest_step()

        self.t_step += 1

        if self.t_step % 1000 == 0:
            self._print_info()

        if done:
            print(f"Done after {self.t_step} steps. Info: {info}")
            self._print_info()

        if self.config.episode.use_terminated_truncated_step_api:
            # use new api from gym v0.26.0
            return (obs, reward, terminated, truncated, info)
        else:
            # use old api
            return (obs, reward, done, info)
    
    def _make_rewarder(self):
        """Creates the rewarder for the environment."""
        rewarder = Rewarder(self.config.rewarder)
        return rewarder

    def _print_info(self) -> None:
        print(
            f"time step = {self.t_step}, progress = {self.progress}, cumulative reward = {self.cumulative_reward}"
        )
        actions_taken: np.ndarray = self.vessel.actions_taken
        print("Mean of actions this episode:", np.mean(actions_taken, axis=0))
        print("Std of actions this episode:", np.std(actions_taken, axis=0))

    @property
    def episode_action_mean(self) -> np.ndarray:
        """Returns the mean of actions taken this episode"""
        return np.mean(self.vessel.actions_taken, axis=0)

    @property
    def episode_action_std(self) -> np.ndarray:
        """Returns the standard deviation of actions taken this episode"""
        return np.std(self.vessel.actions_taken, axis=0)

    def _make_occupancy_grids(
        self, lidar_distances: np.ndarray, path: Path
    ) -> np.ndarray:
        """Makes an occupancy grid containing the lidar measurements and path in cartesian coordinates

        Returns
        -------
        occupancy_grid: numpy ndarray of shape (2, G, G), where G is the grid size
        """
        grid_size = self.config.sensor.occupancy_grid_size

        # Make lidar occupancy grid
        sensor_range = self.config.sensor.range
        indices_to_plot = (
            lidar_distances < self.config.sensor.range
        )  # Plot only indices with measurements
        lidar_positions_body = get_relative_positions_of_lidar_measurements(
            lidar_ranges=lidar_distances,
            sensor_angles=self.vessel.sensor_angles,
            indices_to_plot=indices_to_plot,
        )
        lidar_occupancy_grid = make_occupancy_grid(
            positions_body=lidar_positions_body,
            grid_size=grid_size,
            sensor_range=sensor_range,
        )

        # Make path occupancy grid
        every_n_path_point = (
            10  # Path points are pretty close, only use every 10th one.
        )
        path_positions_ned = path.points[::every_n_path_point]

        path_coordinates_body = geom.transform_ned_to_body(
            path_positions_ned, self.vessel.position, self.vessel.heading
        )
        path_occupancy_grid = make_occupancy_grid(
            path_coordinates_body,
            sensor_range=sensor_range,
            grid_size=grid_size,
        )

        occupancy_grid = np.stack([lidar_occupancy_grid, path_occupancy_grid])

        return occupancy_grid

    def _is_terminal(self) -> bool:
        """Returns True if the episode is done due to a terminal event, False otherwise."""
        return self.collision or self.reached_goal

    def _is_truncated(self) -> bool:
        """Returns True if the episode is done due to a truncated event, False otherwise."""

        is_timelimit_reached = (self.t_step >= self.config.episode.max_timesteps - 1)
        is_cumulative_reward_too_low = (
            self.cumulative_reward < self.config.episode.min_cumulative_reward
        )

        return bool(is_timelimit_reached or is_cumulative_reward_too_low)

    # def _isdone(self) -> bool:
    #     return any(
    #         [
    #             self.collision,
    #             self.reached_goal,
    #             # self.t_step >= self.config.episode.max_timesteps - 1
    #             # and not self.test_mode,
    #             # self.cumulative_reward < self.config.episode.min_cumulative_reward
    #             # and not self.test_mode,
    #         ]
    #     )

    def _update(self) -> None:
        """Updates the environment at each time-step. Can be customized in sub-classes."""
        for obst in self.obstacles:
            if not obst.static:
                obst.update(dt=self.config.simulation.t_step_size)

    @abstractmethod
    def _generate(self) -> None:
        """Create new, stochastically genereated scenario.
        To be implemented in extensions of BaseEnvironment. Must set the
        'vessel', 'path' and 'obstacles' attributes.
        """

    def close(self):
        """Closes the environment. To be called after usage."""
        pass
        if self._renderer2d is not None:
            # self._renderer2d.close()
            pass
        if self._viewer3d is not None:
            self._viewer3d.close()

    # def render(self, mode="rgb_array", **kwargs):
    def render(self):
        """Render one frame of the environment.
        The default mode will do something human friendly, such as pop up a window.

        As the OrderEnforcing wrapper is used in RLlib, an ignored **kwargs is added here."""
        image_arr = None
        # print("inside env.render()!")
        try:
            if self.renderer == "2d" or self.renderer == "both":
                image_arr = self._renderer2d.render(
                    state=self.renderable_state, render_mode=self.render_mode
                )
            # if self.renderer == "3d" or self.renderer == "both":
            #     image_arr = render3d.render_env(
            #         self, mode, self.config.simulation.t_step_size
            #     )
        except OSError:
            image_arr = self._last_image_frame

        if image_arr is None:
            image_arr = self._last_image_frame
        else:
            self._last_image_frame = image_arr

        if image_arr is None and self.render_mode == "rgb_array":
            print("Warning: image_arr is None -> video is likely broken")

        return image_arr

    def seed(self, seed=None):
        """Reseeds the random number generator used in the environment"""
        self.rng, seed = seeding.np_random(seed)
        return [seed]

    @property
    def renderable_state(self) -> RenderableState:
        renderable_state = RenderableState(
            last_reward=self.last_reward,
            cumulative_reward=self.cumulative_reward,
            t_step=self.t_step,
            episode=self.episode,
            lambda_tradeoff=self.config.rewarder.params.lambda_,
            eta=self.config.rewarder.params.eta,
            obstacles=self.obstacles,
            path=self.path,
            vessel=self.vessel,
            # show_indicators=self.config.rendering.show_indicators,
        )
        return renderable_state

    def _save_latest_step(self):
        latest_data = self.vessel.req_latest_data()
        self._tmp_storage["cross_track_error"].append(
            abs(latest_data["navigation"]["cross_track_error"]) * 100
        )

    def save_latest_episode(self, save_history=True):
        # print('Saving latest episode with save_history = ' + str(save_history))
        self.last_episode = {
            "path": self.path(np.linspace(0, self.path.length, 1000))
            if self.path is not None
            else None,
            "path_taken": self.vessel.path_taken,
            "obstacles": self.obstacles,
        }
        if save_history:
            self.history.append(
                {
                    "cross_track_error": np.array(
                        self._tmp_storage["cross_track_error"]
                    ).mean(),
                    "reached_goal": int(self.reached_goal),
                    "collision": int(self.collision),
                    "reward": self.cumulative_reward,
                    "timesteps": self.t_step,
                    "duration": self.t_step * self.config.simulation.t_step_size,
                    "progress": self.progress,
                    "pathlength": self.path.length,
                }
            )
