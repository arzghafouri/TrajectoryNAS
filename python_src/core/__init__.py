__all__ = [
    "BaseTrainer",
    "DARTSAlgorithm",
    "DARTSSearcher",
    "DualLogger",
    "FinalModelTrainer",
    "run_architecture_search",
]


def __getattr__(name: str):
    if name in ("DARTSAlgorithm", "DARTSSearcher"):
        from core.darts_algorithm import DARTSAlgorithm
        return DARTSAlgorithm
    elif name == "run_architecture_search":
        from core.progressive_search import run_architecture_search
        return run_architecture_search
    elif name == "BaseTrainer":
        from train.base_trainer import BaseTrainer
        return BaseTrainer
    elif name == "FinalModelTrainer":
        from train.final_trainer import FinalModelTrainer
        return FinalModelTrainer
    elif name == "DualLogger":
        from utils.logger import DualLogger
        return DualLogger
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

