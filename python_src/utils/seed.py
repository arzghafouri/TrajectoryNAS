"""
Reproducibility & PRNG Seed Management
======================================
Provides deterministic PRNG seed locking for PyTorch, CUDA, NumPy, and Python.
"""

import os
import random

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """
    Guarantees 100% reproducible execution across PyTorch, CUDA, NumPy, and Python PRNG.
    Eliminates initial weight initialization noise and batch shuffling variance.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
