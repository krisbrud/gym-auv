from gym import ObservationWrapper
import gym.spaces

class ImageChannelLastWrapper(ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        # self.observation_space = env.observation_space
        original_observation_space = env.observation_space
        
        assert isinstance(original_observation_space, gym.spaces.Dict)
        transformed_observation_space_dict = {}

        for key, space in original_observation_space.spaces.items():
            if key == "image":
                old_shape = space.shape
                new_shape = (old_shape[2], old_shape[0], old_shape[1])
                space.shape = new_shape

                new_obs_space = gym.spaces.Box(low=space.low, high=space.high, shape=new_shape, dtype=space.dtype)

                transformed_observation_space_dict[key] = new_obs_space
            else:
                transformed_observation_space_dict[key] = space

        self.observation_space = gym.spaces.Dict(transformed_observation_space_dict)


    def observation(self, observation: dict):
        image_observation = observation["image"]
        image_observation = image_observation.transpose(2, 0, 1)

