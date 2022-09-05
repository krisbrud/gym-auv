import numpy as np


def sector_partition_fun(env, isensor, c=0.1) -> int:
    # TODO: Remove reference to env, pass vessel config etc instead
    a = env.config.vessel.n_sensors_per_sector * env.config.vessel.n_sectors
    b = env.config.vessel.n_sectors
    sigma = lambda x: b / (1 + np.exp((-x + a / 2) / (c * a)))
    return int(np.floor(sigma(isensor) - sigma(0)))
