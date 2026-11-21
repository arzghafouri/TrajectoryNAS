r"""
Search Configuration for TrajectoryLNAS
=======================================
Defines hyperparameter configuration tuned for maximum DARTS accuracy and stable search convergence.
"""

from dataclasses import dataclass, field
from configs.base_config import BaseConfig
from configs.data_config import DataConfig


DEFAULT_TEMPORAL_OPS: list[str] = [
    "GRUEncoder",
    "LSTMEncoder",
    "TemporalConvEncoder",
    "TemporalSelfAttention",
]

DEFAULT_INTERACTION_OPS: list[str] = [
    "GCNLayer",
    "GATLayer",
    "SocialPooling",
    "PairwiseAttention",
]

DEFAULT_MAP_OPS: list[str] = [
    "LaneAttention",
    "LaneGraphEncoder",
    "MapFeatureFusion",
]

# Backward-compatible global aliases
TEMPORAL_NAMES: list[str] = DEFAULT_TEMPORAL_OPS
INTERACTION_NAMES: list[str] = DEFAULT_INTERACTION_OPS
MAP_NAMES: list[str] = DEFAULT_MAP_OPS


@dataclass
class SearchConfig(BaseConfig):
    """Hyperparameters for DARTS Bi-Level Optimization tuned for top accuracy."""
    dataset: str = "nuscenes"                       # Target dataset name ('nuscenes' or 'synthetic')
    train_path: str | None = None                   # Custom training sequences path (defaults to DataConfig)
    val_path: str | None = None                     # Custom validation sequences path (defaults to DataConfig)
    seed: int = 42                                  # Reproducibility seed for deterministic execution
    warmup_epochs: int = 10                         # Warmup epochs where alpha is frozen before search updates
    epochs: int = 30                                # Total differentiable DARTS search epochs
    batch_size: int = 64                            # Physical batch size per GPU forward pass
    grad_accum_steps: int = 4                       # Gradient accumulation steps (Effective batch size = 64 * 2 = 128)
    use_amp: bool = True                            # Automatic Mixed Precision FP16 acceleration
    lr_weights: float = 1e-3                        # Learning rate for SuperNet weights w (Adam)
    lr_alphas: float = 1e-2                         # Learning rate for architectural parameters alpha
    weight_decay: float = 1e-4                      # L2 weight decay for network weights
    alpha_weight_decay: float = 0.0                 # L2 weight decay for architectural parameters alpha
    val_batches_for_alpha: int = 1                  # Validation batches evaluated per alpha optimization step
    lambda_flops: float = 0.0005                    # Accuracy-first hardware FLOPs penalty coefficient
    temperature_start: float = 1.0                  # Initial temperature for continuous operator mixture softmax
    temperature_min: float = 0.2                    # Minimum final temperature for discrete operator convergence
    search_samples_percent: float | None = 50.0     # Subsampling dataset percentage to accelerate search phase
    gradient_checkpointing: bool = False            # Gradient checkpointing toggle for memory reduction
    device: str = "cuda"                            # Target computation device ('cuda' or 'cpu')
    optimization_strategy: str = "Bilevel / First-Order DARTS" # Differentiable bilevel architecture search policy

    @property
    def alpha_lr(self) -> float:
        """Alias for lr_alphas."""
        return self.lr_alphas

    @alpha_lr.setter
    def alpha_lr(self, value: float):
        self.lr_alphas = float(value)

    @property
    def search_epochs(self) -> int:
        """Alias for epochs in search phase."""
        return self.epochs

    @search_epochs.setter
    def search_epochs(self, value: int):
        self.epochs = int(value)

    # Dynamically configured candidate operators per branch (no hard-coding)
    temporal_ops: list[str] = field(default_factory=lambda: list(DEFAULT_TEMPORAL_OPS))
    interaction_ops: list[str] = field(default_factory=lambda: list(DEFAULT_INTERACTION_OPS))
    map_ops: list[str] = field(default_factory=lambda: list(DEFAULT_MAP_OPS))

    @property
    def num_temporal(self) -> int:
        """Dynamic number of temporal candidate operators."""
        return len(self.temporal_ops)

    @property
    def num_interaction(self) -> int:
        """Dynamic number of interaction candidate operators."""
        return len(self.interaction_ops)

    @property
    def num_map(self) -> int:
        """Dynamic number of map candidate operators."""
        return len(self.map_ops)

    def __post_init__(self):
        self.temporal_ops = list(self.temporal_ops)
        self.interaction_ops = list(self.interaction_ops)
        self.map_ops = list(self.map_ops)

        if not self.temporal_ops:
            raise ValueError("SearchConfig.temporal_ops cannot be empty.")
        if not self.interaction_ops:
            raise ValueError("SearchConfig.interaction_ops cannot be empty.")
        if not self.map_ops:
            raise ValueError("SearchConfig.map_ops cannot be empty.")

        if self.dataset == "nuscenes":
            d_cfg = DataConfig()
            if self.train_path is None:
                self.train_path = d_cfg.output_dir
            if self.val_path is None:
                self.val_path = d_cfg.output_dir

