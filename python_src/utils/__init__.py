"""
Utils Package Module
====================
System utilities, PRNG seed management, data loaders, and stream logging.
"""

try:
    from utils.dataloaders import get_dataloaders
except ImportError:
    get_dataloaders = None

from utils.logger import DualLogger

try:
    from utils.seed import set_seed
except ImportError:
    def set_seed(seed: int = 42) -> None:
        import os
        import random
        random.seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)

__all__ = [
    "DualLogger",
    "SyntheticTrajectoryDataset",
    "ade",
    "endpoint_miss_rate_k",
    "fde",
    "get_dataloaders",
    "get_supernet_complexity",
    "get_supernet_costs",
    "get_supernet_flops",
    "max_displacement_miss_rate_k",
    "miss_rate",
    "set_seed",
    "simple_complexity_proxy",
    "snip_score",
    "synflow_score",
]


def __getattr__(name: str):
    # Lazy imports for backward-compatibility re-exports to break circular dependencies
    if name in ("ade", "fde", "miss_rate", "endpoint_miss_rate_k", "max_displacement_miss_rate_k"):
        from evaluations.metrics import (
            ade,
            endpoint_miss_rate_k,
            fde,
            max_displacement_miss_rate_k,
            miss_rate,
        )
        return {
            "ade": ade,
            "fde": fde,
            "miss_rate": miss_rate,
            "endpoint_miss_rate_k": endpoint_miss_rate_k,
            "max_displacement_miss_rate_k": max_displacement_miss_rate_k,
        }[name]
    elif name in ("snip_score", "synflow_score"):
        from evaluations.metrics import snip_score, synflow_score
        return {"snip_score": snip_score, "synflow_score": synflow_score}[name]
    elif name in ("get_supernet_costs", "get_supernet_flops", "get_supernet_complexity", "simple_complexity_proxy"):
        from evaluations.metrics import (
            get_supernet_complexity,
            get_supernet_costs,
            get_supernet_flops,
            simple_complexity_proxy,
        )
        return {
            "get_supernet_costs": get_supernet_costs,
            "get_supernet_flops": get_supernet_flops,
            "get_supernet_complexity": get_supernet_complexity,
            "simple_complexity_proxy": simple_complexity_proxy
        }[name]
    elif name == "SyntheticTrajectoryDataset":
        from data.synthetic_dataset import SyntheticTrajectoryDataset
        return SyntheticTrajectoryDataset
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

