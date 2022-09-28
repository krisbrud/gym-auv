from dataclasses import dataclass
from typing import List

from numpy import ndarray
from gym_auv.objects.path import Path
from gym_auv.objects.vessel import Vessel
from gym_auv.objects.obstacles import BaseObstacle


@dataclass
class RenderableState:
    """RenderableState contains all the information the renderer needs to render the environment"""

    # Text
    last_reward: float
    cumulative_reward: float
    t_step: float
    episode: int
    lambda_tradeoff: float
    eta: float

    # Vessel
    vessel: Vessel
    obstacles: List[BaseObstacle]

    # Path
    path: Path

    # Config
    show_indicators: bool
