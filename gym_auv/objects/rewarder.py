import numpy as np

def _sample_lambda(scale):
    log = -np.random.gamma(1, scale)
    y = np.power(10, log)
    return y

def _sample_eta():
    y = np.random.gamma(shape=1.9, scale=0.6)
    return y

class BaseRewarder:
    def __init__(self):
        self.params = {}

    def reset(self, vessel):
        """
        Resets rewarder instance
        """
        self.vessel = vessel

    def calculate(self, *args, **kwargs):
        """
        Calculates the step reward and decides whether the episode
        should be ended.

        Returns
        -------
        reward : float
            The reward for performing action at his timestep.
        info : dict
            Dictionary with extra information.
        """
        raise NotImplementedError

class ColavRewarder(BaseRewarder):
    def __init__(self):
        super().__init__()
        self.params['gamma_theta'] = 5
        self.params['gamma_x'] = 0.5
        self.params['gamma_y_e'] = 0.05
        self.params['penalty_yawrate'] = 1.0
        self.params['penalty_torque_change'] = 2
        self.params['cruise_speed'] = 0.1
        self.params['neutral_speed'] = 0.1
        self.params['negative_multiplier'] = 2
  
    def reset(self, vessel):
        super().reset(vessel)
        self.params['lambda'] =  _sample_lambda(scale=0.2)
        self.params['eta'] = _sample_eta()
    
    def calculate(self):
        nav_states = self.vessel.last_navi_state_dict
        feasible_distances = self.vessel.last_sector_feasible_dists
         
        reward = 0

        # Extracting navigation states
        cross_track_error = nav_states['cross_track_error']
        course_error = nav_states['course_error']

        # Calculating path following reward component
        cross_track_performance = np.exp(-self.params['gamma_y_e']*np.abs(cross_track_error))
        path_reward = (1 + np.cos(course_error)*self.vessel.speed/self.vessel.max_speed)*(1 + cross_track_performance) - 1
        
        # Calculating obstacle avoidance reward component
        closeness_penalty_num = 0
        closeness_penalty_den = 0
        if self.vessel.n_sensors > 0:
            for isector in range(self.vessel.n_sectors):
                angle = self.vessel.sector_angles[isector]
                x = feasible_distances[isector]
                weight = 1 / (1 + np.abs(self.params['gamma_theta']*angle))
                raw_penalty = (self.vessel.sensor_range/max(x, 1))**self.params['gamma_x'] - 1
                weighted_penalty = weight*raw_penalty
                closeness_penalty_num += weighted_penalty
                closeness_penalty_den += weight

            closeness_reward = -closeness_penalty_num/closeness_penalty_den
        else:
            closeness_reward = 0

        # Calculating living penalty
        living_penalty = self.params['lambda']*(2*self.params["neutral_speed"]+1) + self.params["eta"]*self.params["neutral_speed"]
        
        # Calculating total reward
        reward = self.params['lambda']*path_reward + \
            (1-self.params['lambda'])*closeness_reward - \
            living_penalty + \
            self.params["eta"]*self.vessel.speed/self.vessel.max_speed - \
            self.params["penalty_yawrate"]*abs(self.vessel.yawrate) - \
            self.params["penalty_torque_change"]*abs(self.vessel.smoothed_torque_change)

        if reward < 0:
            reward *= self.params['negative_multiplier']

        return reward
