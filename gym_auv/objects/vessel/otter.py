"""Copyright (c) 2018 The Python Packaging Authority

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software. 

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE."""

# Based on an implementation from Thor Inge Fossen
# https://github.com/cybergalactic/PythonVehicleSimulator/blob/master/src/python_vehicle_simulator/vehicles/otter.py

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import math

from gym_auv.objects.vessel.otterutils import crossFlowDrag, Hmtrx, m2c, Smtrx, eta_dot
import gym_auv.utils.geomutils as geom

# TODO: Find maximum force from reference model - max propeller rotation given
# TODO: Find maximum torque from reference model
# TODO: Implement clipping


class Otter:
    """
    otter()                                           Propeller step inputs
    otter('headingAutopilot',psi_d,V_c,beta_c,tau_X)  Heading autopilot

    Inputs:
        psi_d: desired heading angle (deg)
        V_c: current speed (m/s)
        beta_c: current direction (deg)
        tau_X: surge force, pilot input (N)
    """

    def __init__(
        self,
        r=0,
        V_current=0,
        beta_current=0,
        # tau_X = 120
    ):

        # Constants
        D2R = math.pi / 180  # deg2rad
        g = 9.81  # acceleration of gravity (m/s^2)
        rho = 1026  # density of water (kg/m^3)

        self.ref = r
        self.V_c = V_current
        self.beta_c = beta_current * D2R
        # self.tauX = tau_X  # surge force (N)

        # Initialize the Otter USV model
        self.T_n = 1.0  # propeller time constants (s)
        self.L = 2.0  # Length (m)
        self.B = 1.08  # beam (m)
        self.u = np.array([0, 0, 0, 0, 0, 0], float)  # velocity vector
        self.u_actual = np.array([0, 0], float)  # propeller revolution states
        self.name = "Otter USV (see 'otter.py' for more details)"

        self.controls = [
            "Left propeller shaft speed (rad/s)",
            "Right propeller shaft speed (rad/s)",
        ]
        self.dimU = len(self.controls)

        # Vehicle parameters
        m = 55.0  # mass (kg)
        mp = 25.0  # Payload (kg)
        self.m_total = m + mp
        rp = np.array([0, 0, -0.35], float)  # location of payload (m)
        rg = np.array([0.2, 0, -0.2], float)  # CG for hull only (m)
        rg = (m * rg + mp * rp) / (m + mp)  # CG corrected for payload
        self.S_rg = Smtrx(rg)
        self.H_rg = Hmtrx(rg)
        self.S_rp = Smtrx(rp)

        R44 = 0.4 * self.B  # radii of gyration (m)
        R55 = 0.25 * self.L
        R66 = 0.25 * self.L
        T_yaw = 1.0  # time constant in yaw (s)
        Umax = 6 * 0.5144  # max forward speed (m/s)

        # Data for one pontoon
        self.B_pont = 0.25  # beam of one pontoon (m)
        y_pont = 0.395  # distance from centerline to waterline centroid (m)
        Cw_pont = 0.75  # waterline area coefficient (-)
        Cb_pont = 0.4  # block coefficient, computed from m = 55 kg

        # Inertia dyadic, volume displacement and draft
        nabla = (m + mp) / rho  # volume
        self.T = nabla / (2 * Cb_pont * self.B_pont * self.L)  # draft
        Ig_CG = m * np.diag(np.array([R44**2, R55**2, R66**2]))
        self.Ig = Ig_CG - m * self.S_rg @ self.S_rg - mp * self.S_rp @ self.S_rp

        # Experimental propeller data including lever arms
        self.l1 = -y_pont  # lever arm, left propeller (m)
        self.l2 = y_pont  # lever arm, right propeller (m)
        self.k_pos = 0.02216 / 2  # Positive Bollard, one propeller
        self.k_neg = 0.01289 / 2  # Negative Bollard, one propeller
        self.n_max = math.sqrt((0.5 * 24.4 * g) / self.k_pos)  # max. prop. rev.
        self.n_min = -math.sqrt((0.5 * 13.6 * g) / self.k_neg)  # min. prop. rev.

        # MRB_CG = [ (m+mp) * I3  O3      (Fossen 2021, Chapter 3)
        #               O3       Ig ]
        MRB_CG = np.zeros((6, 6))
        MRB_CG[0:3, 0:3] = (m + mp) * np.identity(3)
        MRB_CG[3:6, 3:6] = self.Ig
        MRB = self.H_rg.T @ MRB_CG @ self.H_rg

        # Hydrodynamic added mass (best practice)
        Xudot = -0.1 * m
        Yvdot = -1.5 * m
        Zwdot = -1.0 * m
        Kpdot = -0.2 * self.Ig[0, 0]
        Mqdot = -0.8 * self.Ig[1, 1]
        Nrdot = -1.7 * self.Ig[2, 2]

        self.MA = -np.diag([Xudot, Yvdot, Zwdot, Kpdot, Mqdot, Nrdot])

        # System mass matrix
        self.M = MRB + self.MA
        self.Minv = np.linalg.inv(self.M)

        # Hydrostatic quantities (Fossen 2021, Chapter 4)
        Aw_pont = Cw_pont * self.L * self.B_pont  # waterline area, one pontoon
        I_T = (
            2
            * (1 / 12)
            * self.L
            * self.B_pont**3
            * (6 * Cw_pont**3 / ((1 + Cw_pont) * (1 + 2 * Cw_pont)))
            + 2 * Aw_pont * y_pont**2
        )
        I_L = 0.8 * 2 * (1 / 12) * self.B_pont * self.L**3
        KB = (1 / 3) * (5 * self.T / 2 - 0.5 * nabla / (self.L * self.B_pont))
        BM_T = I_T / nabla  # BM values
        BM_L = I_L / nabla
        KM_T = KB + BM_T  # KM values
        KM_L = KB + BM_L
        KG = self.T - rg[2]
        GM_T = KM_T - KG  # GM values
        GM_L = KM_L - KG

        G33 = rho * g * (2 * Aw_pont)  # spring stiffness
        G44 = rho * g * nabla * GM_T
        G55 = rho * g * nabla * GM_L
        G_CF = np.diag([0, 0, G33, G44, G55, 0])  # spring stiff. matrix in CF
        LCF = -0.2
        H = Hmtrx(np.array([LCF, 0.0, 0.0]))  # transform G_CF from CF to CO
        self.G = H.T @ G_CF @ H

        # Natural frequencies
        w3 = math.sqrt(G33 / self.M[2, 2])
        w4 = math.sqrt(G44 / self.M[3, 3])
        w5 = math.sqrt(G55 / self.M[4, 4])

        # Linear damping terms (hydrodynamic derivatives)
        Xu = -24.4 * g / Umax  # specified using the maximum speed
        Yv = 0
        Zw = -2 * 0.3 * w3 * self.M[2, 2]  # specified using relative damping
        Kp = -2 * 0.2 * w4 * self.M[3, 3]
        Mq = -2 * 0.4 * w5 * self.M[4, 4]
        Nr = -self.M[5, 5] / T_yaw  # specified by the time constant T_yaw

        self.D = -np.diag([Xu, Yv, Zw, Kp, Mq, Nr])

        # Trim: theta = -7.5 deg corresponds to 13.5 cm less height aft
        self.trim_moment = 0
        self.trim_setpoint = 280

        # Propeller configuration/input matrix
        B = self.k_pos * np.array([[1, 1], [-self.l1, -self.l2]])
        self.Binv = np.linalg.inv(B)

    def dynamics(self, eta, nu, u_actual, u_control):
        """
        Returns the derivatives of the vessel.

        Params:
        -------
        eta (np.ndarray): 6DoF pose of the vehicle
        nu  (np.ndarray): 6DoF velocities of the vehicle
        n_actual (np.ndarray): (2,) array of the actual propeller speeds
        force_command

        [nu,u_actual] = dynamics(eta,nu,u_actual,u_control,sampleTime) integrates
        the Otter USV equations of motion using Euler's method.
        """

        # Input vector
        n = np.array([u_actual[0], u_actual[1]])

        # Current velocities
        u_c = self.V_c * math.cos(self.beta_c - eta[5])  # current surge vel.
        v_c = self.V_c * math.sin(self.beta_c - eta[5])  # current sway vel.

        nu_c = np.array([u_c, v_c, 0, 0, 0, 0], float)  # current velocity vector
        nu_r = nu - nu_c  # relative velocity vector

        # Rigid body and added mass Coriolis and centripetal matrices
        # CRB_CG = [ (m+mp) * Smtrx(nu2)          O3   (Fossen 2021, Chapter 6)
        #              O3                   -Smtrx(Ig*nu2)  ]
        CRB_CG = np.zeros((6, 6))
        CRB_CG[0:3, 0:3] = self.m_total * Smtrx(nu[3:6])
        CRB_CG[3:6, 3:6] = -Smtrx(np.matmul(self.Ig, nu[3:6]))
        CRB = self.H_rg.T @ CRB_CG @ self.H_rg  # transform CRB from CG to CO

        CA = m2c(self.MA, nu_r)
        CA[5, 0] = 0  # assume that the Munk moment in yaw can be neglected
        CA[5, 1] = 0  # if nonzero, must be balanced by adding nonlinear damping

        C = CRB + CA

        # Ballast
        g_0 = np.array([0.0, 0.0, 0.0, 0.0, self.trim_moment, 0.0])

        # Control forces and moments - with propeller revolution saturation
        thrust = np.zeros(2)
        for i in range(0, 2):

            n[i] = np.clip(n[i], self.n_min, self.n_max)  # saturation, physical limits

            if n[i] > 0:  # positive thrust
                thrust[i] = self.k_pos * n[i] * abs(n[i])
            else:  # negative thrust
                thrust[i] = self.k_neg * n[i] * abs(n[i])

        # Control forces and moments
        tau = np.array(
            [
                thrust[0] + thrust[1],
                0,
                0,
                0,
                0,
                -self.l1 * thrust[0] - self.l2 * thrust[1],
            ]
        )

        # Hydrodynamic linear damping + nonlinear yaw damping
        tau_damp = -np.matmul(self.D, nu_r)
        tau_damp[5] = tau_damp[5] - 10 * self.D[5, 5] * abs(nu_r[5]) * nu_r[5]

        # State derivatives (with dimension)
        tau_crossflow = crossFlowDrag(self.L, self.B_pont, self.T, nu_r)
        sum_tau = (
            tau
            + tau_damp
            + tau_crossflow
            - np.matmul(C, nu_r)
            - np.matmul(self.G, eta)
            - g_0
        )

        print(f"{tau = }")
        print(f"{tau_damp = }")
        print(f"{tau_crossflow = }")
        print(f"{np.matmul(C, nu_r) = }")
        print(f"{np.matmul(self.G, eta) = }")
        print("-" * 20)
        print("\n")

        nu_dot = np.matmul(self.Minv, sum_tau)  # USV dynamics

        eta_dot_ = eta_dot(eta, nu)

        n_dot = (u_control - n) / self.T_n  # propeller dynamics
        # trim_dot = (self.trim_setpoint - self.trim_moment) / 5  # trim dynamics

        # Forward Euler integration [k+1]
        # nu = nu + sampleTime * nu_dot
        # n = n + sampleTime * n_dot
        # self.trim_moment = self.trim_moment + sampleTime * trim_dot

        # u_actual = np.array(n, float)

        return np.concatenate([eta_dot_, nu_dot, n_dot])

    def controlAllocation(self, tau_X, tau_N):
        """
        [n1, n2] = controlAllocation(tau_X, tau_N)
        """
        tau = np.array([tau_X, tau_N])  # tau = B * u_alloc
        u_alloc = np.matmul(self.Binv, tau)  # u_alloc = inv(B) * tau

        # u_alloc = abs(n) * n --> n = sign(u_alloc) * sqrt(u_alloc)
        n1 = np.sign(u_alloc[0]) * math.sqrt(abs(u_alloc[0]))
        n2 = np.sign(u_alloc[1]) * math.sqrt(abs(u_alloc[1]))

        return n1, n2


class Otter3DoF:
    def __init__(self):
        # TODO = 123
        self.mass = 55.0
        self.length = 2.0
        self.xg = 0.2
        R66 = 0.25 * self.length  # Radii of gyration (m)
        self.Iz = self.mass * (R66**2)  # Moment of inertia around z-axis

        self.X_udot = -0.1 * self.mass
        self.Y_vdot = -1.5 * self.mass
        self.N_rdot = -1.7 * self.Iz

        M_RB = np.diag([self.mass, self.mass, self.Iz])

        self.M = M_RB  # + M_A
        self.M_inv = np.linalg.inv(self.M)

        y_pont = 0.395  # distance from centerline to waterline centroid (m)
        self.l1 = -y_pont
        self.l2 = y_pont
        self.B = np.array([[1, 1], [0, 0], [-self.l1, self.l2]])

        Umax = 6 * 0.5144
        g = 9.81
        self.X_u = -24.4 * g / Umax
        self.Y_v = 0
        T_yaw = 1.0
        self.N_r = -1.7 * self.Iz / T_yaw

        self.D_L = np.diag([self.X_u, self.Y_v, self.N_r])

    def dynamics(self, eta, nu, u):
        """Calculates the derivatives of eta and nu"""
        r = nu[2]  # Yaw rate
        psi = eta[2]  # Heading
        C = np.array(
            [
                [0, -self.mass * r, -self.mass * self.xg * r],
                [self.mass * r, 0, 0],
                [self.mass * self.xg * r, 0, 0],
            ]
        )

        D_N = np.zeros((3, 3))
        D_N[2, 2] = -10 * self.N_r * abs(r)

        D = self.D_L + D_N

        eta_dot = geom.Rz(psi).dot(nu)
        nu_dot = self.M_inv.dot(self.B.dot(u) - C.dot(nu) - D.dot(nu))

        state_dot = np.concatenate(eta_dot, nu_dot)

        return state_dot
