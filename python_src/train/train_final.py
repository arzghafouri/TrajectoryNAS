"""
TrajectoryLNAS Standalone Final Retraining
==========================================
Sole Responsibility: Trains the discretized FinalTrajectoryModel discovered by DARTS.

Author: Alireza Ghafouri (Student ID: 4031631209)
"""

import argparse
import os
import sys

# Ensure python_src directory is in sys.path
_curr_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.dirname(_curr_dir)
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

from configs.train_config import TrainConfig
from models.base_operator import resolve_architecture_indices
from models.final_model import FinalTrajectoryModel
from train.final_trainer import FinalModelTrainer


def run_final_training(
    best_op_indices: tuple[int, int, int] | None = None,
    train_cfg: TrainConfig | None = None,
    dataset: str = "nuscenes",
    train_path: str | None = None,
    val_path: str | None = None,
    epochs: int = 50,
    batch_size: int | None = None,
    grad_accum_steps: int | None = None,
    lr: float = 5e-4,
    patience: int = 8,
    min_delta: float = 0.002,
    train_samples_percent: float | None = None,
    hard_sample_mining: float = 0.0,
    save_path: str = "checkpoints/best_trajectory_lnas.pt"
) -> tuple[FinalTrajectoryModel, FinalModelTrainer]:
    """
    Executes standalone retraining of the winning architecture with Early Stopping and Hard Sample Mining.
    Follows Dependency Injection: accepts pre-configured TrainConfig object directly.
    """
    resolved_ops, ops_source = resolve_architecture_indices(best_ops=best_op_indices)
    print(f"[Final Training] Resolved operator indices {resolved_ops} (source: {ops_source})")

    cfg = train_cfg or TrainConfig(
        dataset=dataset,
        train_path=train_path,
        val_path=val_path,
        epochs=epochs,
        lr=lr,
        early_stopping_patience=patience,
        early_stopping_min_delta=min_delta,
        train_samples_percent=train_samples_percent,
        hard_sample_mining=hard_sample_mining,
        save_path=save_path
    )
    if train_cfg is None:
        if batch_size is not None:
            cfg.batch_size = batch_size
        if grad_accum_steps is not None:
            cfg.grad_accum_steps = grad_accum_steps

    trainer = FinalModelTrainer(
        best_op_indices=resolved_ops,
        train_config=cfg
    )
    final_model = trainer.train_and_evaluate()
    return final_model, trainer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TrajectoryLNAS Standalone Final Training")
    parser.add_argument("--dataset", type=str, default="nuscenes", choices=["nuscenes", "synthetic"])
    parser.add_argument("--train_path", type=str, default=None)
    parser.add_argument("--val_path", type=str, default=None)
    parser.add_argument("--temp_op", type=int, default=None)
    parser.add_argument("--inter_op", type=int, default=None)
    parser.add_argument("--map_op", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch_size", type=int, default=None, help="Defaults to TrainConfig.batch_size")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hard_sample_mining", "--hsm", type=float, default=0.0, help="Sensitivity parameter [0.0, 1.0] for Weighted Sampling of hard cases")
    parser.add_argument("--save_path", type=str, default="checkpoints/best_trajectory_lnas.pt")
    args = parser.parse_args()

    cli_ops = None
    if any(x is not None for x in (args.temp_op, args.inter_op, args.map_op)):
        default_ops, _ = resolve_architecture_indices()
        cli_ops = (
            args.temp_op if args.temp_op is not None else default_ops[0],
            args.inter_op if args.inter_op is not None else default_ops[1],
            args.map_op if args.map_op is not None else default_ops[2],
        )

    run_final_training(
        best_op_indices=cli_ops,
        dataset=args.dataset,
        train_path=args.train_path,
        val_path=args.val_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        hard_sample_mining=args.hard_sample_mining,
        save_path=args.save_path
    )
