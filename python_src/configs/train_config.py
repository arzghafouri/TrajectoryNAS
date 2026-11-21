r"""
Training Configuration for TrajectoryLNAS
=========================================
Hyperparameters for Standalone Final Model Retraining.
"""

from dataclasses import dataclass
from configs.base_config import BaseConfig
from configs.data_config import DataConfig


@dataclass
class TrainConfig(BaseConfig):
    """Hyperparameters for Standalone Retraining Phase."""
    dataset: str = "nuscenes"                       # Target dataset name ('nuscenes' or 'synthetic')
    train_path: str | None = None                   # Custom training sequences path (defaults to DataConfig)
    val_path: str | None = None                     # Custom validation sequences path (defaults to DataConfig)
    seed: int = 42                                  # Reproducibility seed for deterministic retraining
    epochs: int = 60                                # Maximum retraining epoch count ceiling
    batch_size: int = 128                           # Physical batch size on GPU per forward pass
    grad_accum_steps: int = 2                       # Gradient accumulation steps (Effective batch size = 64 * 2 = 128)
    use_amp: bool = True                            # Automatic Mixed Precision FP16 acceleration
    lr: float = 7.5e-4                              # Base learning rate for Adam optimizer (supports 1e-3 scaling)
    min_lr: float = 1e-5                            # Minimum learning rate floor for Cosine Annealing scheduler
    weight_decay: float = 1e-4                      # L2 weight decay regularizer (supports 5e-4 for anti-overfitting)
    clip_grad_norm: float = 5.0                     # Maximum gradient clipping L2 norm for recurrent stability
    early_stopping_patience: int = 8                # Early stopping patience epochs without validation ADE improvement
    early_stopping_min_delta: float = 0.002         # Minimum ADE improvement threshold to register new best checkpoint (2mm)
    train_samples_percent: float | None = None      # Dataset training percentage (None = 100% full dataset of 488,643 samples)
    hard_sample_mining: float = 0.3                 # Hard Sample Mining sensitivity [0.0, 1.0]. 0.0 = disabled (uniform), >0 = error-weighted sampling
    hard_sample_ema: float = 0.7                    # Exponential moving average momentum for sample error tracking across epochs
    save_path: str = "checkpoints/best_trajectory_lnas.pt"  # Destination path for optimal model checkpoint
    device: str = "cuda"                            # Target computation device ('cuda' or 'cpu')

    @property
    def hard_sample_sensitivity(self) -> float:
        """Alias for hard_sample_mining sensitivity parameter."""
        return self.hard_sample_mining

    @hard_sample_sensitivity.setter
    def hard_sample_sensitivity(self, value: float):
        self.hard_sample_mining = float(value)

    @property
    def hard_mining_sensitivity(self) -> float:
        """Alias for hard_sample_mining sensitivity parameter."""
        return self.hard_sample_mining

    @hard_mining_sensitivity.setter
    def hard_mining_sensitivity(self, value: float):
        self.hard_sample_mining = float(value)

    def __post_init__(self):
        if self.dataset == "nuscenes":
            d_cfg = DataConfig()
            if self.train_path is None:
                self.train_path = d_cfg.output_dir
            if self.val_path is None:
                self.val_path = d_cfg.output_dir

        # Validate hard sample mining sensitivity in range [0.0, 1.0]
        if self.hard_sample_mining is None:
            self.hard_sample_mining = 0.0
        else:
            self.hard_sample_mining = float(self.hard_sample_mining)

        if not (0.0 <= self.hard_sample_mining <= 1.0):
            raise ValueError(
                f"TrainConfig.hard_sample_mining must be in range [0.0, 1.0], but got {self.hard_sample_mining}. "
                "0.0 disables hard sample mining (uniform sampling), while values up to 1.0 increase sensitivity to challenging cases."
            )
