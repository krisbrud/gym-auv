import numpy as np


m = 23.8
x_g = 0.046
I_z = 1.760
X_udot = -2.0
Y_vdot = -10.0
Y_rdot = 0.0
N_rdot = -1.0
N_vdot = 0.0
X_u = -0.72253
Y_v = -0.88965
Y_r = -0.1
N_v = 0.03130
N_r = -0.5
X_uu = -1.32742
X_uuu = -5.86643
Y_vv = -36.47287
Y_rr = 0.3
Y_rv = -0.805
Y_vr = -0.845
N_vv = 3.95645
N_rr = -9.1
N_rv = 0.13
N_vr = 0.080
Y_uvb = -0.5*1000*np.pi*1.24*(0.15/2)**2
Y_uvf = -1000*3*0.0064
Y_urf = -0.4*Y_uvf
N_uvb = (-0.65*1.08 + 0.4)*Y_uvb
N_uvf = -0.4*Y_uvf
N_urf = -0.4*Y_urf
Y_uudr = 19.2
N_uudr = -0.4*Y_uudr

MAX_SPEED = 2

M =  np.array([[m, 0, 0],
    [0, m, m*x_g],
    [0, m*x_g, I_z]]
)  
# Added mass 
M_A = np.array([[-X_udot, 0, 0],
                [0, -Y_vdot, -Y_rdot],
                [0, -N_vdot, -N_rdot]])
M_inv = np.linalg.inv(M + M_A)

def D(nu):
    # Use second order modulus model (Clarke 2003) - see Cybership paper
    D_linear =  np.array([
        [-X_u, 0, 0],
        [0, -Y_v, -Y_r],
        [0, -Y_r, -N_r]
    ])
    
    u, v, r = nu[0], nu[1], nu[2]
    d11 = X_uu * abs(u) + X_uuu * (u **2)
    d22 = Y_vv * abs(v) + Y_rv * abs(r)
    d23 = Y_vr * abs(v) + Y_rr * abs(r)
    d32 = N_vv * abs(v) + N_rv * abs(r)
    d33 = N_vr * abs(v) + N_rr * abs(r)

    D_nonlinear = np.array([
        [-d11,0, 0],
        [0, -d22, -d23],
        [0, -d32, -d33]
    ])

    return D_linear + D_nonlinear

def B(nu):
    return np.array([
        [1, 0],
        [0, -1.7244],
        [0, 1],
    ])

def C(nu):
    u = nu[0]
    v = nu[1]
    r = nu[2]
    C_RB = np.array([
        [0, 0, -m(x_g * r + v)],
        [0, 0, m * u],
        [m(x_g * r + v), -m * u, 0]
    ])  
    c13 = Y_vdot + 0.5 * (N_vdot + Y_rdot) * r
    c23 = -X_udot * u
    C_A = np.array([
        [0, 0, c13],
        [0, 0, c23],
        [-c13, -c23, 0]])

    return C_RB + C_A

def N(nu):
    u = nu[0]
    v = nu[1]
    r = nu[2]
    N = np.array([
        [-X_u, 0, 0],
        [0, -Y_v, m*u - Y_r],
        [0, -N_v, m*x_g*u-N_r]
    ])  
    return N
