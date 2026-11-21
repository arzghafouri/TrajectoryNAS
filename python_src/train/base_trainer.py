r"""
Abstract Base Trainer for Trajectory Prediction Models
======================================================
Provides standard lifecycle methods, device resolution, FP16/AMP, and batch helper utilities.
"""

import os
from abc import ABC, abstractmethod
from typing import Any
import torch

from models.base_operator import normalize_future_tensor


def unpack_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    """Transfers all tensor elements in a batch dict to target device and normalizes ground truth."""
    unpacked = {}
    for key, val in batch.items():
        if val is not None and hasattr(val, "to"):
            unpacked[key] = val.to(device)
        else:
            unpacked[key] = val

    if "future" in unpacked and unpacked["future"] is not None:
        unpacked["future"] = normalize_future_tensor(unpacked["future"])

    return unpacked


class BaseTrainer(ABC):
    """Abstract Base Class for training, searching, and validating models."""

    def __init__(self, device: str | None = None, use_amp: bool = True):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, str):
            self.device = torch.device(device if (device == "cpu" or torch.cuda.is_available()) else "cpu")
        else:
            self.device = device

        self.use_amp = use_amp and (self.device.type == "cuda")
        try:
            self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        except (AttributeError, TypeError):
            self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

    @abstractmethod
    def train_epoch(self, epoch: int) -> dict | float:
        """Runs one training epoch over the training data."""

    @abstractmethod
    def validate(self, epoch: int) -> dict:
        """Runs evaluation over the validation set."""

    def clip_gradients(self, parameters, max_norm: float = 5.0):
        """Clips parameter gradients with AMP unscaling support."""
        if self.use_amp:
            self.scaler.unscale_(parameters)
        torch.nn.utils.clip_grad_norm_(parameters, max_norm)

    def save_checkpoint(self, state_dict: dict, file_path: str):
        """Saves model weights safely, creating parent folders if needed."""
        os.makedirs(os.path.dirname(file_path) if os.path.dirname(file_path) else ".", exist_ok=True)
        torch.save(state_dict, file_path)
        print(f"[Checkpoint] Successfully saved to: {file_path}")

    def load_checkpoint(self, file_path: str) -> dict:
        """Loads checkpoint from disk."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Checkpoint file not found: {file_path}")
        return torch.load(file_path, map_location=self.device)
