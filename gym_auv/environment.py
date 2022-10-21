from typing import Tuple, Union
import gym
import numpy as np
from gym.utils import seeding
import gym_auv

from gym_auv.objects.vessel import Vessel
from gym_auv.objects.rewarder import ColavRewarder
from gym_auv.render2d.state import RenderableState

from gym_auv.render2d.renderer import Renderer2d
from gym_auv.render2d.renderer import FPS as rendererFPS2D
import gym_auv.render2d.renderer

# import gym_auv.rendering.render3d as render3d
from gym_auv.utils.clip_to_space import clip_to_space
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
        # render_mode: str = "rgb_array",  # "human",
        verbose: bool = False,
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
            # low=np.array([0, -1]),
            low=np.array([-1, -0.15]), # [-1, -1]
            high=np.array([1, 0.15]), # [1, 1]
            dtype=np.float32,
        )
        # Setting dimension of observation vector
        # self.n_observations = (
        #     len(Vessel.NAVIGATION_FEATURES)
        #     + self.config.vessel.n_lidar_observations + self._rewarder_class.N_INSIGHTS
        # )
        self.n_observations = self.config.vessel.dense_observation_size
        if self.config.vessel.use_lidar:
            self.n_observations += self.config.vessel.n_lidar_observations

        if self.config.vessel.use_dict_observation:
            # Use a dictionary observation, as we want to encode the proprioceptive sensors (velocities etc)
            # and LiDAR measurements differently, and keep the lidar measurements as a multi-channel "image",
            # making it easier to apply Convolutional NNs etc. to
            n_navigation_observations = len(Vessel.NAVIGATION_FEATURES)

            # The LiDAR has a distance/closeness measurements, as well as two measurements
            # that correspond to the planar velocity of an object obstructing the sensor (if there is one).

            self._observation_space = gym.spaces.Dict(
                {
                    "proprioceptive": gym.spaces.Box(
                        low=-1.0,
                        high=1.0,
                        shape=(n_navigation_observations,),
                        dtype=np.float32,
                    ),
                    "lidar": gym.spaces.Box(
                        low=-1.0, high=1.0, shape=self.config.vessel.lidar_shape
                    ),
                }
            )
        else:
            self._observation_space = gym.spaces.Box(
                low=np.array([-1] * self.n_observations),
                high=np.array([1] * self.n_observations),
                dtype=np.float32,
            )

        # Initializing rendering
        self._renderer2d = None
        self._viewer3d = None
        # self.render_mode = render_mode
        if self.renderer == "2d" or self.renderer == "both":
            self._renderer2d = Renderer2d(
                render_fps=self.metadata["video.frames_per_second"]
            )
        if self.renderer == "3d" or self.renderer == "both":
            if self.config.vessel.render_distance == "random":
                self.render_distance = self.rng.randint(300, 2000)
            else:
                self.render_distance = self.config.vessel.render_distance
            # render3d.init_env_viewer(
            #     self,
            #     autocamera=self.config.rendering.autocamera3d,
            #     render_dist=self.render_distance,
            # )

        self.reset()

    @property
    def action_space(self) -> gym.spaces.Box:
        """Array defining the shape and bounds of the agent's action."""
        return self._action_space

    @property
    def observation_space(self) -> gym.spaces.Box:
        """Array defining the shape and bounds of the agent's observations."""
        return self._observation_space

    def reset(self, save_history=True):
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
            self.rewarder = self._rewarder_class(self.vessel, self.test_mode)
        if self.verbose:
            print("Generated scenario")

        # Initializing 3d viewer
        # if self.renderer == "3d":
        # render3d.init_boat_model(self)
        # self._viewer3d.create_path(self.path)

        # Getting initial observation vector
        obs = self.observe()
        if self.verbose:
            print("Calculated initial observation")

        # Resetting temporary data storage
        self._tmp_storage = {
            "cross_track_error": [],
        }

        # print('Reset environment')

        return obs

    def observe(self) -> np.ndarray:
        """Returns the array of observations at the current time-step.

        Returns
        -------
        obs : np.ndarray
            The observation of the environment.
        """
        reward_insight = self.rewarder.insight()
        navigation_states = self.vessel.navigate(self.path)
        if bool(self.config.vessel.use_lidar):
            sector_closenesses, sector_velocities = self.vessel.perceive(self.obstacles)
        else:
            sector_closenesses, sector_velocities = [], []

        # Clamp/clip the observation to the valid domain as specified by the Space
        if isinstance(self.observation_space, gym.spaces.Box):
            raw_obs = [
                    reward_insight,
                    navigation_states,
                ]

            if self.config.vessel.use_lidar:
                raw_obs.append(sector_closenesses.flatten())
            if self.config.vessel.sensor_use_velocity_observations:
                raw_obs.append(sector_velocities.flatten())

            raw_obs = np.hstack(raw_obs)

            obs = np.clip(
                raw_obs,
                a_min=self.observation_space.low,
                a_max=self.observation_space.high,
            )
        elif isinstance(self.observation_space, gym.spaces.Dict):
            sector_closenesses = sector_closenesses.reshape(1, -1)  # Add leading axis
            raw_lidar_obs = np.vstack((sector_closenesses, sector_velocities))
            raw_obs = {
                "proprioceptive": navigation_states,
                "lidar": raw_lidar_obs,
            }
            obs = clip_to_space(raw_obs, self.observation_space)

        return obs

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
        reward = self.rewarder.calculate()
        self.last_reward = reward
        self.cumulative_reward += reward

        info = {}
        info["collision"] = self.collision
        info["reached_goal"] = self.reached_goal
        info["goal_distance"] = self.goal_distance
        info["progress"] = self.progress

        # Testing criteria for ending the episode
        done = self._isdone()

        self._save_latest_step()

        self.t_step += 1
        # if self.t_step % 50 == 1:
        #     print("timestep:", self.t_step)
        if self.t_step % 1000 == 0:
            # print(f"time step = {self.t_step}, progress = {self.progress}, cumulative reward = {self.cumulative_reward}")
            # # self.vessel.
            # print(f"yaw rate: {self.vessel.yaw_rate}, cross-track-error: {vessel_data['navigation']['cross_track_error']}")
            # print(f"velocity: {self.vessel.velocity}")
            self._print_info()

        if done:
            print(f"Done after {self.t_step} steps. Info: {info}")
            # print(f"Cumulative reward: {self.cumulative_reward}")
            # print(f"yaw rate: {self.vessel.yaw_rate}")
            # actions_taken: np.ndarray = self.vessel.actions_taken
            # print("Mean of actions this episode:", np.mean(actions_taken, axis=0))
            # print("Std of actions this episode:", np.std(actions_taken, axis=0))
            self._print_info()

        return (obs, reward, done, info)

    def _print_info(self) -> None:
            print(f"time step = {self.t_step}, progress = {self.progress}, cumulative reward = {self.cumulative_reward}")
            actions_taken: np.ndarray = self.vessel.actions_taken
            print("Mean of actions this episode:", np.mean(actions_taken, axis=0))
            print("Std of actions this episode:", np.std(actions_taken, axis=0))


    def _isdone(self) -> bool:
        return any(
            [
                self.collision,
                self.reached_goal,
                self.t_step >= self.config.episode.max_timesteps - 1 and not self.test_mode,
                self.cumulative_reward < self.config.episode.min_cumulative_reward
                and not self.test_mode,
            ]
        )

    def _update(self) -> None:
        """Updates the environment at each time-step. Can be customized in sub-classes."""
        [
            obst.update(dt=self.config.simulation.t_step_size)
            for obst in self.obstacles
            if not obst.static
        ]

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

    def render(self, mode="rgb_array", **kwargs):
        """Render one frame of the environment.
        The default mode will do something human friendly, such as pop up a window.

        As the OrderEnforcing wrapper is used in RLlib, an ignored **kwargs is added here."""
        image_arr = None
        # print("inside env.render()!")
        try:
            if self.renderer == "2d" or self.renderer == "both":
                image_arr = self._renderer2d.render(
                    state=self.renderable_state, render_mode=mode
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

        if image_arr is None and mode == "rgb_array":
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
            lambda_tradeoff=self.rewarder.params["lambda"],
            eta=self.rewarder.params["eta"],
            obstacles=self.obstacles,
            path=self.path,
            vessel=self.vessel,
            show_indicators=self.config.rendering.show_indicators,
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
