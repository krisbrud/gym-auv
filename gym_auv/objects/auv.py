"""
This module implements an AUV that is simulated in the horizontal plane.
"""
import numpy as np
import numpy.linalg as linalg
from scipy.optimize import minimize

import gym_auv.utils.constants as const
import gym_auv.utils.geomutils as geom

def odesolver45(f, y, h):
    """Calculate the next step of an IVP of a time-invariant ODE with a RHS
    described by f, with an order 4 approx. and an order 5 approx.
    Parameters:
        f: function. RHS of ODE.
        y: float. Current position.
        h: float. Step length.
    Returns:
        q: float. Order 2 approx.
        w: float. Order 3 approx.
    """
    s1 = f(y)
    s2 = f(y+h*s1/4.0)
    s3 = f(y+3.0*h*s1/32.0+9.0*h*s2/32.0)
    s4 = f(y+1932.0*h*s1/2197.0-7200.0*h*s2/2197.0+7296.0*h*s3/2197.0)
    s5 = f(y+439.0*h*s1/216.0-8.0*h*s2+3680.0*h*s3/513.0-845.0*h*s4/4104.0)
    s6 = f(y-8.0*h*s1/27.0+2*h*s2-3544.0*h*s3/2565+1859.0*h*s4/4104.0-11.0*h*s5/40.0)
    w = y + h*(25.0*s1/216.0+1408.0*s3/2565.0+2197.0*s4/4104.0-s5/5.0)
    q = y + h*(16.0*s1/135.0+6656.0*s3/12825.0+28561.0*s4/56430.0-9.0*s5/50.0+2.0*s6/55.0)
    return w, q

class AUV2D():
    """
    Creates an environment with a vessel, goal and obstacles.

    Attributes
    ----------
    path_taken : np.array
        Array of size (?, 2) discribing the path the AUV has taken.
    radius : float
        The maximum distance from the center of the AUV to its edge
        in meters.
    t_step : float
        The simulation timestep.
    input : np.array
        The current input. [propeller_input, rudder_position].
    """
    def __init__(self, t_step, init_pos, width=4, adaptive_step_size=False):
        """
        The __init__ method declares all class atributes.

        Parameters
        ----------
        t_step : float
            The simulation timestep to be used to simulate this AUV.
        init_pos : np.array
            The initial position of the vessel [x, y, psi], where
            psi is the initial heading of the AUV.
        width : float
            The maximum distance from the center of the AUV to its edge
            in meters. Defaults to 2.
        """
        self.width = width
        self.t_step = t_step
        self.adaptive_step_size = adaptive_step_size
        self.reset(init_pos)

    def step(self, action):
        """
        Steps the vessel self.t_step seconds forward.

        Parameters
        ----------
        action : np.array
            [propeller_input, rudder_position], where
            0 <= propeller_input <= 1 and -1 <= rudder_position <= 1.
        """
        self.input = np.array([self._thrust_surge(action[0]), self._moment_steer(action[1])])
        self._sim()

        self.prev_states = np.vstack([self.prev_states,self._state])
        self.prev_inputs = np.vstack([self.prev_inputs,self.input])

        torque_change = self.input[1] - self.prev_inputs[-2, 1] if len(self.prev_inputs) > 1 else self.input[1]
        self.smoothed_torque_change = 0.9*self.smoothed_torque_change + 0.1*abs(torque_change)
        self.smoothed_torque = 0.9*self.smoothed_torque + 0.1*abs(self.input[1])

    def reset(self, init_pos, init_speed=None):
        if (init_speed is None):
            init_speed = [0, 0, 0]
        init_pos = np.array(init_pos, dtype=np.float64)
        init_speed = np.array(init_speed, dtype=np.float64)
        self._state = np.hstack([init_pos, init_speed])
        self.prev_states = np.vstack([self._state])
        self.input = [0, 0]
        self.prev_inputs =np.vstack([self.input])
        self.smoothed_torque_change = 0
        self.smoothed_torque = 0
        self.planned_path = None

    def _state_dot(self, state):
        psi = state[2]
        nu = state[3:]

        tau = np.array([self.input[0], 0, self.input[1]])

        eta_dot = geom.Rzyx(0, 0, geom.princip(psi)).dot(nu)
        nu_dot = const.M_inv.dot(
            tau
            #- const.D.dot(nu)
            - const.N(nu).dot(nu)
        )
        state_dot = np.concatenate([eta_dot, nu_dot])
        return state_dot

    def _sim(self):
        # k_1 = self._state_dot(self._state)
        # k_2 = self._state_dot(self._state + k_1*self.t_step)
        # self._state += self.t_step*(k_1+k_2)/2

        if (self.adaptive_step_size):
            self.t_step = 1
            err = np.inf
            while err > np.pi/1800:
                try:
                    w, q = odesolver45(self._state_dot, self._state, self.t_step)
                    err = abs(w[2]-q[2])
                except OverflowError:
                    pass
                self.t_step*=0.9
        else:
            w, q = odesolver45(self._state_dot, self._state, self.t_step)
        
        self._state = q
        self._state[2] = geom.princip(self._state[2])

    def teleport_back(self, timesteps):
        self._state = self.prev_states[max(-len(self.prev_states), -(timesteps+1))]

    @property
    def position(self):
        """
        Returns an array holding the position of the AUV in cartesian
        coordinates.
        """
        return self._state[0:2]

    @property
    def x(self):
        return self.position[0]

    @property
    def y(self):
        return self.position[1]

    @property
    def init_position(self):
        """
        Returns an array holding the path of the AUV in cartesian
        coordinates.
        """
        return self.prev_states[-1, 0:2]

    @property
    def path_taken(self):
        """
        Returns an array holding the path of the AUV in cartesian
        coordinates.
        """
        return self.prev_states[:, 0:2]

    @property
    def heading(self):
        """
        Returns the heading of the AUV wrt true north.
        """
        return self._state[2]

    @property
    def heading_history(self):
        """
        Returns the heading of the AUV wrt true north.
        """
        return self.prev_states[:, 2]

    @property
    def heading_change(self):
        """
        Returns the change of heading of the AUV wrt true north.
        """
        return geom.princip(self.prev_states[-1, 2] - self.prev_states[-2, 2]) if len(self.prev_states) >= 2 else self.heading

    @property
    def velocity(self):
        """
        Returns the surge and sway velocity of the AUV.
        """
        return self._state[3:5]

    @property
    def speed(self):
        """
        Returns the surge and sway velocity of the AUV.
        """
        return linalg.norm(self.velocity)

    @property
    def yawrate(self):
        """
        Returns the rate of rotation about the z-axis.
        """
        return self._state[5]

    @property
    def max_speed(self):
        """
        Returns the max speed of the AUV.
        """
        return const.MAX_SPEED

    @property
    def crab_angle(self):
        return np.arctan2(self.velocity[1], self.velocity[0])

    @property
    def course(self):
        return self.heading + self.crab_angle

    def _thrust_surge(self, surge):
        surge = np.clip(surge, 0, 1)
        return surge*const.THRUST_MAX_AUV

    def _moment_steer(self, steer):
        steer = np.clip(steer, -1, 1)
        return steer*const.MOMENT_MAX_AUV

class AUVPerfectFollower(AUV2D):
    def step(self, action):
        a = action[0]
        b = action[1]
        c = action[2]
        dt = action[3]

        def curve(t):
            return (
                t*a*np.sin(c*t),
                t*b,
                0
            )

        def dcurve(t):
            return (
                t*a*c*np.cos(c*t)+a*np.sin(c*t),
                b,
                0
            )

        N_LINSPACE = 1000
        S = np.linspace(0, 20, N_LINSPACE)
        psi = self._state[2]

        self.planned_path = np.transpose(curve(S))
        self.planned_path = geom.Rzyx(0, 0, psi-np.pi/2).dot(self.planned_path)
        self.planned_path[0] += self._state[0]
        self.planned_path[1] += self._state[1]
        self.planned_path = self.planned_path[:2]

        planned_path_d = np.transpose(dcurve(S))
        planned_path_d = geom.Rzyx(0, 0, psi).dot(planned_path_d)
        
        n_dt = int(dt/20*N_LINSPACE)
        pos = np.array([self.planned_path[0][n_dt], self.planned_path[1][n_dt]])
        der = np.array([planned_path_d[0][n_dt], planned_path_d[1][n_dt]])
        
        path_angle = np.arctan2(der[1], der[0])
        self._state[0] = pos[0]
        self._state[1] = pos[1]
        self._state[2] = geom.princip(path_angle - np.pi/2)