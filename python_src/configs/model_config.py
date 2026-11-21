r"""
Model Configuration
===================
Dimensions and hyperparameters of the SuperNet and Final Model.
"""

from dataclasses import dataclass
from configs.base_config import BaseConfig


@dataclass
class ModelConfig(BaseConfig):
    """Dimensions and hyperparameters of the SuperNet and Final Model."""
    d_model: int = 128                              # Hidden feature embedding dimension across encoders and operators
    hz: int = 2                                     # Temporal sampling frequency in Hertz (2 Hz = 0.5s per timestep)
    past_sec: float = 2.0                           # Historical observation window in seconds
    future_sec: float = 6.0                         # Future trajectory forecasting horizon in seconds
    past_len: int = 4                               # Number of past observation timesteps (past_sec * hz)
    future_len: int = 12                            # Number of future prediction timesteps (future_sec * hz)
    num_modes: int = 5                              # Number of multi-modal trajectory forecasts (K=5 for nuScenes benchmark)
    max_agents: int = 20                            # Maximum surrounding dynamic neighbor agents tracked for social interaction
    max_obstacles: int = 10                         # Maximum static obstacles and parked entities preserved for spatial context
    max_lanes: int = 30                             # Maximum HD map lane polylines encoded in the map branch
    lane_pts: int = 15                              # Equidistant 2D waypoint points sampled per lane centerline polyline
    dropout: float = 0.1                            # Dropout probability applied for layer regularization

    def __post_init__(self):
        # Keep model tensor dimensions synchronized with the sampling grid.
        if self.hz is None or self.hz <= 0:
            raise ValueError(f"ModelConfig.hz must be positive, got {self.hz}.")
        if self.past_sec is None or self.past_sec <= 0 or self.future_sec is None or self.future_sec <= 0:
            raise ValueError("ModelConfig past_sec and future_sec must be positive.")
        self.past_len = int(round(self.past_sec * self.hz))
        self.future_len = int(round(self.future_sec * self.hz))
        for name in ("d_model", "past_len", "future_len", "num_modes", "max_agents", "lane_pts"):
            if getattr(self, name) <= 0:
                raise ValueError(f"ModelConfig.{name} must be positive, got {getattr(self, name)}.")
        if self.max_lanes < 0:
            raise ValueError(f"ModelConfig.max_lanes cannot be negative, got {self.max_lanes}.")
        if self.max_obstacles < 0:
            raise ValueError(f"ModelConfig.max_obstacles cannot be negative, got {self.max_obstacles}.")

