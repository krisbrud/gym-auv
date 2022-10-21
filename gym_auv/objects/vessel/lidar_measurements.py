import numpy as np
from typing import Union
from dataclasses import dataclass

@dataclass
class LidarMeasurements:
    distances: np.ndarray
    blocked: np.ndarray