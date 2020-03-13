import numpy as np


def princip(angle):
    return ((angle + np.pi) % (2*np.pi)) - np.pi

def Rzyx(phi, theta, psi):
    cphi = np.cos(phi)
    sphi = np.sin(phi)
    cth = np.cos(theta)
    sth = np.sin(theta)
    cpsi = np.cos(psi)
    spsi = np.sin(psi)

    return np.vstack([
        np.hstack([cpsi*cth, -spsi*cphi+cpsi*sth*sphi, spsi*sphi+cpsi*cphi*sth]),
        np.hstack([spsi*cth, cpsi*cphi+sphi*sth*spsi, -cpsi*sphi+sth*spsi*cphi]),
        np.hstack([-sth, cth*sphi, cth*cphi])
    ])

def Rz(psi):
    cpsi = np.cos(psi)
    spsi = np.sin(psi)

    return np.vstack([
        np.hstack([cpsi, -spsi, 0]),
        np.hstack([spsi, cpsi, -0]),
        np.hstack([0, 0, 1])
    ])

def Rzyx_dpsi(phi, theta, psi):
    cphi = np.cos(phi)
    sphi = np.sin(phi)
    cth = np.cos(theta)
    sth = np.sin(theta)
    cpsi = np.cos(psi)
    spsi = np.sin(psi)

    return np.vstack([
        np.hstack([-spsi*cth, -cpsi*cphi-spsi*sth*sphi, cpsi*sphi-spsi*cphi*sth]),
        np.hstack([cpsi*cth, -spsi*cphi+sphi*sth*cpsi, spsi*sphi+sth*cpsi*cphi]),
        np.hstack([0, 0, 0])
    ])

def to_homogeneous(x):
    return np.array([x[0], x[1], 1])

def to_cartesian(x):
    return np.array([x[0], x[1]])

def feasibility_pooling(x, W, theta, N_sensors):
    sort_idx = np.argsort(x, axis=None)
    for idx in sort_idx:
        surviving = x > x[idx] + W
        d = x[idx]*theta
        opening_width = 0
        opening_span = 0
        opening_start = -theta*(N_sensors-1)/2
        found_opening = False
        for isensor, lidar_surviving in enumerate(surviving):
            if (lidar_surviving):
                opening_width += d
                opening_span += theta
                if (opening_width > W):
                    opening_center = opening_start + opening_span/2
                    if (abs(opening_center) < theta*(N_sensors-1)/4):
                        found_opening = True
            else:
                opening_width += 0.5*d
                opening_span += 0.5*theta
                if (opening_width > W):
                    opening_center = opening_start + opening_span/2
                    if (abs(opening_center) < theta*(N_sensors-1)/4):
                        found_opening = True
                opening_width = 0
                opening_span = 0
                opening_start = -theta*(N_sensors-1)/2 + isensor*theta

        if (not found_opening): 
            return (max(0, x[idx]), idx)

    return (max(0, np.max(x)), N_sensors//2)