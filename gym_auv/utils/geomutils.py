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