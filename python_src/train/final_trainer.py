r"""
Standalone Final Model Trainer
==============================
Trains the discretized FinalTrajectoryModel discovered by TrajectoryLNAS.
Implements multi-agent interactions, HD maps, multi-modal loss, AMP FP16, and Early Stopping.
"""

import os
import sys
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

_curr_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.dirname(_curr_dir)
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

from configs.model_config import ModelConfig
from configs.train_config import TrainConfig
from evaluations.metrics import (
    best_of_k_loss,
    endpoint_miss_rate_k,
    evaluate_collision_risk,
    max_displacement_miss_rate_k,
    min_ade_k,
    min_fde_k,
    safe_norm,
)
from models.final_model import FinalTrajectoryModel
from train.base_trainer import BaseTrainer, unpack_batch_to_device
from utils import get_dataloaders, set_seed


class FinalModelTrainer(BaseTrainer):
    """Object-Oriented Standalone Trainer for the Discovered Architecture."""

    def __init__(
        self,
        best_op_indices: tuple[int, int, int],
        train_config: TrainConfig,
        model_config: ModelConfig = None
    ):
        super().__init__(device=train_config.device, use_amp=train_config.use_amp)
        self.cfg = train_config
        self.model_cfg = model_config or ModelConfig()
        self.best_t, self.best_i, self.best_m = [int(x) for x in best_op_indices]

        set_seed(getattr(self.cfg, "seed", 42))

        self.model = FinalTrajectoryModel(
            best_temporal_idx=self.best_t,
            best_interaction_idx=self.best_i,
            best_map_idx=self.best_m,
            d_model=self.model_cfg.d_model,
            past_len=self.model_cfg.past_len,
            future_len=self.model_cfg.future_len,
            num_modes=self.model_cfg.num_modes,
            max_agents=self.model_cfg.max_agents,
            max_obstacles=self.model_cfg.max_obstacles,
            max_lanes=self.model_cfg.max_lanes,
            lane_pts=self.model_cfg.lane_pts,
            hz=self.model_cfg.hz,
        ).to(self.device)

        self.optimizer = Adam(self.model.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=self.cfg.epochs, eta_min=self.cfg.min_lr)

        train_pct = getattr(self.cfg, "train_samples_percent", None)
        self.hard_sample_mining = float(getattr(self.cfg, "hard_sample_mining", 0.0) or 0.0)
        self.hard_sample_ema = float(getattr(self.cfg, "hard_sample_ema", 0.7) or 0.7)

        self.train_loader, self.val_loader = get_dataloaders(
            train_path=self.cfg.train_path,
            val_path=self.cfg.val_path,
            batch_size=self.cfg.batch_size,
            dataset_type=self.cfg.dataset,
            sample_percent=train_pct,
            model_config=self.model_cfg,
            hard_sample_mining=self.hard_sample_mining,
            hard_sample_ema=self.hard_sample_ema,
            train_config=self.cfg,
        )

        self.best_val_ade = float("inf")
        self.best_val_fde = float("inf")
        self.grad_accum_steps = max(1, self.cfg.grad_accum_steps)

    def train_epoch(self, epoch: int) -> float:
        """Trains for one full epoch with AMP and Gradient Accumulation."""
        self.model.train()
        total_loss = 0.0
        step_count = 0
        self.optimizer.zero_grad()

        num_batches = len(self.train_loader)
        accum_counter = 0

        for batch_idx, batch in enumerate(self.train_loader):
            b = unpack_batch_to_device(batch, self.device)

            try:
                autocast_ctx = torch.amp.autocast(device_type="cuda", enabled=self.use_amp)
            except (AttributeError, TypeError):
                autocast_ctx = torch.cuda.amp.autocast(enabled=self.use_amp)

            with autocast_ctx:
                pred_trajs, mode_logits = self.model(
                    b["past"], agents_past=b.get("agents_past"), map_lanes=b.get("map_lanes"),
                    category_id=b.get("category_id"), map_lane_types=b.get("map_lane_types"),
                    neighbor_categories=b.get("neighbor_categories"),
                    agent_mask=b.get("agent_mask"), lane_mask=b.get("lane_mask"),
                    static_obstacles=b.get("static_obstacles"),
                    static_obstacle_types=b.get("static_obstacle_types"),
                    static_obstacle_mask=b.get("static_obstacle_mask")
                )
                loss_reg, loss_cls = best_of_k_loss(pred_trajs, b["future"], mode_logits)

                # --------------------------------------------------------------------------
                # Kinematic Physics Regularization (Jerk & Acceleration Smoothness Loss)
                # --------------------------------------------------------------------------
                # Computes the discrete second-order finite difference across predicted time steps:
                #   \Delta^2 p_t = p_{t+2} - 2 * p_{t+1} + p_t \approx \frac{d^2 p}{dt^2} \Delta t^2
                # This penalizes unnatural path jitter, sudden lateral spikes, and excessive jerk,
                # ensuring that forecasted trajectories adhere to realistic vehicle dynamics
                # and non-holonomic kinematic motion constraints.
                diff2 = pred_trajs[:, :, 2:, :] - 2 * pred_trajs[:, :, 1:-1, :] + pred_trajs[:, :, :-2, :]
                loss_physics = torch.mean(torch.norm(diff2, p=2, dim=-1))

                # Composite multi-modal loss: Winner-Takes-All regression + mode classification + physics regularizer
                loss_batch = loss_reg + (0.1 * loss_cls) + (0.05 * loss_physics)
                loss_scaled = loss_batch / self.grad_accum_steps

            # Update sample errors for Hard Sample Mining
            if self.hard_sample_mining > 0 and "sample_idx" in b:
                with torch.no_grad():
                    step_dists = safe_norm(pred_trajs.detach() - b["future"].unsqueeze(1), dim=-1).mean(dim=-1)
                    per_sample_ade = step_dists.min(dim=-1).values
                    if hasattr(self.train_loader, "sampler") and hasattr(self.train_loader.sampler, "update_sample_errors"):
                        self.train_loader.sampler.update_sample_errors(b["sample_idx"], per_sample_ade)

            self.scaler.scale(loss_scaled).backward()
            accum_counter += 1

            if accum_counter % self.grad_accum_steps == 0 or (batch_idx + 1) == num_batches:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.cfg.clip_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                accum_counter = 0

            total_loss += loss_batch.item()
            step_count += 1

        if step_count > 0:
            self.scheduler.step()

        if self.hard_sample_mining > 0 and hasattr(self.train_loader, "sampler") and hasattr(self.train_loader.sampler, "get_summary"):
            summary_hsm = self.train_loader.sampler.get_summary()
            if summary_hsm.get("active"):
                print(
                    f"  -> [Hard Sample Mining] Tracked: {summary_hsm['samples_visited']}/{summary_hsm['total_samples']} "
                    f"({summary_hsm['coverage_percent']}%) | Error Mean: {summary_hsm['mean_error']:.3f}m "
                    f"[Min: {summary_hsm['min_error']:.3f}m, Max: {summary_hsm['max_error']:.3f}m] | "
                    f"Top/Bottom Weight Ratio: {summary_hsm['hard_to_easy_weight_ratio']:.2f}x"
                )

        return total_loss / max(1, step_count)

    def validate(self, epoch: int) -> dict[str, object]:
        """Evaluate model performance on validation set using exact sample weighting."""
        if len(self.val_loader) == 0:
            raise RuntimeError("Validation DataLoader is empty; cannot validate FinalTrajectoryModel.")

        self.model.eval()
        sum_ade = 0.0
        sum_fde = 0.0
        sum_endpoint_mr = 0.0
        sum_max_disp_mr = 0.0
        total_samples = 0

        collision_scenarios = 0
        collided_scenarios = 0
        last_unavailable_reason: str | None = None

        k_eval = int(self.model.num_modes)
        effective_k = k_eval

        with torch.inference_mode():
            for batch in self.val_loader:
                b = unpack_batch_to_device(batch, self.device)

                pred_trajs, mode_logits = self.model(
                    b["past"], agents_past=b.get("agents_past"), map_lanes=b.get("map_lanes"),
                    category_id=b.get("category_id"), map_lane_types=b.get("map_lane_types"),
                    neighbor_categories=b.get("neighbor_categories"),
                    agent_mask=b.get("agent_mask"), lane_mask=b.get("lane_mask"),
                    static_obstacles=b.get("static_obstacles"),
                    static_obstacle_types=b.get("static_obstacle_types"),
                    static_obstacle_mask=b.get("static_obstacle_mask")
                )
                B = b["past"].shape[0]
                total_samples += B
                effective_k = min(k_eval, pred_trajs.shape[1])

                sum_ade += min_ade_k(pred_trajs, b["future"], k=effective_k, mode_logits=mode_logits).item() * B
                sum_fde += min_fde_k(pred_trajs, b["future"], k=effective_k, mode_logits=mode_logits).item() * B
                sum_endpoint_mr += endpoint_miss_rate_k(
                    pred_trajs, b["future"], k=effective_k, threshold=2.0, mode_logits=mode_logits
                ) * B
                sum_max_disp_mr += max_displacement_miss_rate_k(
                    pred_trajs, b["future"], k=effective_k, threshold=2.0, mode_logits=mode_logits
                ) * B

                cr = evaluate_collision_risk(
                    pred_trajs,
                    neighbors_future=b.get("neighbors_future"),
                    neighbor_future_mask=b.get("neighbor_future_mask"),
                    mode_logits=mode_logits,
                )
                if cr.get("available"):
                    collision_scenarios += int(cr.get("scenarios_evaluated", 0))
                    collided_scenarios += int(cr.get("collided_scenarios", 0))
                else:
                    last_unavailable_reason = cr.get("reason")

        if total_samples == 0:
            raise RuntimeError("Validation DataLoader yielded zero samples.")

        if collision_scenarios > 0:
            cr_pct = round((collided_scenarios / collision_scenarios) * 100.0, 2)
            collision_summary = {
                "available": True,
                "status": "Computed",
                "collision_rate_percent": cr_pct,
                "safety_compliance_percent": round(100.0 - cr_pct, 2),
                "scenarios_evaluated": collision_scenarios,
                "collided_scenarios": collided_scenarios,
            }
        else:
            collision_summary = {
                "available": False,
                "status": "Unavailable",
                "reason": last_unavailable_reason or "neighbors_future not present in validation sequences.",
                "collision_rate_percent": None,
                "safety_compliance_percent": None,
                "scenarios_evaluated": 0,
                "collided_scenarios": 0,
            }

        endpoint_mr_pct = (sum_endpoint_mr / total_samples) * 100.0
        max_disp_mr_pct = (sum_max_disp_mr / total_samples) * 100.0
        return {
            "val_ade": sum_ade / total_samples,
            "val_fde": sum_fde / total_samples,
            "miss_rate": endpoint_mr_pct,
            "endpoint_miss_rate": endpoint_mr_pct,
            "max_displacement_miss_rate": max_disp_mr_pct,
            "k_modes": effective_k,
            "collision_risk": collision_summary,
        }

    def train_and_evaluate(self) -> FinalTrajectoryModel:
        """Runs the complete training loop with Early Stopping and best weight restoration."""
        summary = self.model.get_summary()
        print("=" * 80)
        print(" Standalone Training of Discovered TrajectoryLNAS Architecture")
        print(f" Target Dataset:       {self.cfg.dataset.upper()}")
        print(f" Temporal Operator:    {summary['temporal_op']} (idx {summary['best_temporal_idx']})")
        print(f" Interaction Operator: {summary['interaction_op']} (idx {summary['best_interaction_idx']})")
        print(f" HD Map Operator:      {summary['map_op']} (idx {summary['best_map_idx']})")
        print(f" Modes (K):            {summary['num_modes']}")
        print(f" Total Parameters:     {summary['parameters']:,}")
        print(f" Theoretical FLOPs:    {summary['theoretical_flops_g']} GFLOPs ({summary['theoretical_macs_m']} MMACs)")
        hsm_status = f"Active (gamma = {self.hard_sample_mining:.2f})" if self.hard_sample_mining > 0 else "Disabled (gamma = 0.0, Uniform Sampling)"
        print(f" Hard Sample Mining:   {hsm_status}")
        print(f" Early Stopping:       Patience = {self.cfg.early_stopping_patience} epochs | Min Delta = {self.cfg.early_stopping_min_delta}m")
        print("=" * 80)

        self.training_history = []
        patience_counter = 0
        best_epoch = 0

        for epoch in range(1, self.cfg.epochs + 1):
            train_loss = self.train_epoch(epoch)
            metrics = self.validate(epoch)

            mean_ade = metrics["val_ade"]
            mean_fde = metrics["val_fde"]
            mean_mr = metrics["miss_rate"]

            improvement = (self.best_val_ade - mean_ade)
            is_best = improvement > self.cfg.early_stopping_min_delta

            if is_best:
                self.best_val_ade = mean_ade
                self.best_val_fde = mean_fde
                best_epoch = epoch
                patience_counter = 0
                self.save_checkpoint({
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "best_ade": self.best_val_ade,
                    "best_fde": self.best_val_fde,
                    "config": summary
                }, self.cfg.save_path)
                flag = " [BEST CHECKPOINT SAVED]"
            else:
                patience_counter += 1
                flag = f" (Patience: {patience_counter}/{self.cfg.early_stopping_patience})"

            self.training_history.append({
                "epoch": epoch,
                "train_loss": round(train_loss, 4),
                "val_ade": round(mean_ade, 4),
                "val_fde": round(mean_fde, 4),
                "miss_rate_2m": round(mean_mr, 2),
                "max_displacement_miss_rate_2m": round(metrics["max_displacement_miss_rate"], 2),
                "collision_risk": metrics["collision_risk"],
                "is_best": is_best
            })

            k_eval = int(metrics.get("k_modes", self.model.num_modes))
            cr = metrics["collision_risk"]
            cr_str = f"{cr['collision_rate_percent']:.2f}%" if cr.get("available") else "Unavailable (Dataset)"
            print(
                f"Epoch [{epoch:02d}/{self.cfg.epochs:02d}] "
                f"Train Loss: {train_loss:.3f} | "
                f"Val minADE@{k_eval}: {mean_ade:.3f}m | "
                f"Val minFDE@{k_eval}: {mean_fde:.3f}m | "
                f"Endpoint MR@{k_eval}: {mean_mr:.1f}% | "
                f"CR: {cr_str}{flag}"
            )

            if self.device.type == "cuda":
                torch.cuda.empty_cache()

            if patience_counter >= self.cfg.early_stopping_patience:
                print(f"\n[Early Stopping Triggered] No validation ADE improvement for {self.cfg.early_stopping_patience} consecutive epochs.")
                print(f"  -> Early stopping activated at epoch {epoch}. Restoring optimal weights from epoch {best_epoch}...")
                break

        k_eval = int(self.model.num_modes)
        if os.path.exists(self.cfg.save_path):
            try:
                try:
                    ckpt = torch.load(self.cfg.save_path, map_location=self.device, weights_only=False)
                except TypeError:
                    ckpt = torch.load(self.cfg.save_path, map_location=self.device)
                self.model.load_state_dict(ckpt["model_state_dict"])
                print(f"[Model Restored] Successfully reloaded optimal weights from epoch {best_epoch} (Val minADE@{k_eval}: {self.best_val_ade:.3f}m)")
            except Exception as e:
                print(f"[Model Restore Warning] Could not reload checkpoint ({e}).")

        print("=" * 80)
        print(" Final Model Training Complete.")
        print(f" Best Validation minADE@{k_eval}: {self.best_val_ade:.3f} meters (Epoch {best_epoch})")
        print(f" Best Validation minFDE@{k_eval}: {self.best_val_fde:.3f} meters")
        print(f" Model Weights Saved: {self.cfg.save_path}")
        print("=" * 80)

        return self.model
