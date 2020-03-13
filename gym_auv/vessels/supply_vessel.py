import numpy as np

M =  np.array([
    [7.22e6, 0, 0],
    [0, 1.21e7, -3.63e7],
    [0, -3.63e7, 4.75e9]
]) 
M_inv = np.linalg.inv(M)

B = np.array([
    [1.0, 0],
    [0, -1.13e6],
    [0, 9.63e7]
])

D =  np.array([
    [1.74e5, 0, 0],
    [0, 1.25e6, 2.14e6],
    [0, -6.24e7, 1.35e9]
])

def C(nu):
    u = nu[0]
    v = nu[1]
    r = nu[2]
    C = np.array([
        [0, 0, -1.21e7*v + 3.63e7*r],
        [0, 0, 7.22e6*u],
        [1.21e7*v - 3.63e7*r, -7.22e6*u, 0]
    ])  
    return C

MAX_SPEED = 7
MAX_THRUST = 1600*1000
MAX_RUDDER_ANGLE = 30/180*np.pi
LENGTH = 82.45
BEAM = 18.8