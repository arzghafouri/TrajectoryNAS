from train.base_trainer import BaseTrainer
from train.final_trainer import FinalModelTrainer
from train.train_final import run_final_training

__all__ = [
    "BaseTrainer",
    "FinalModelTrainer",
    "run_architecture_search",
    "run_final_training",
]


def __getattr__(name: str):
    if name == "run_architecture_search":
        from core.progressive_search import run_architecture_search
        return run_architecture_search
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

