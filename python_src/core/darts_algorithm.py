r"""
Differentiable Neural Architecture Searcher (DARTS)
===================================================
Encapsulates bi-level optimization for continuous SuperNet search with hardware FLOPs penalty.
Supports multi-agent interactions, vectorized HD maps, and multi-modal trajectory forecasting.
"""

import os
import sys
import time
from typing import Any

import torch
from torch.optim import Adam

_curr_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.dirname(_curr_dir)
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

from configs.model_config import ModelConfig
from configs.search_config import SearchConfig
from evaluations.metrics import (
    best_of_k_loss,
    endpoint_miss_rate_k,
    get_supernet_costs,
    min_ade_k,
    min_fde_k,
    snip_score,
    synflow_score,
)
from models.search_supernet import ArchParams, SearchSuperNet
from train.base_trainer import BaseTrainer, unpack_batch_to_device
from utils import get_dataloaders, set_seed


class DARTSAlgorithm(BaseTrainer):
    """
    Object-Oriented DARTS Search Engine for Trajectory Prediction.

    Solves the bi-level optimization problem:
      min_alpha L_val(w*(alpha), alpha) + lambda * FLOPs(alpha)
      s.t. w*(alpha) = argmin_w L_train(w, alpha)
    """

    def __init__(self, search_config: SearchConfig, model_config: ModelConfig = None):
        super().__init__(device=search_config.device, use_amp=search_config.use_amp)
        self.cfg = search_config
        self.model_cfg = model_config or ModelConfig()

        set_seed(getattr(self.cfg, "seed", 42))

        self.supernet = SearchSuperNet(
            d_model=self.model_cfg.d_model,
            past_len=self.model_cfg.past_len,
            future_len=self.model_cfg.future_len,
            num_modes=self.model_cfg.num_modes,
            max_agents=getattr(self.model_cfg, "max_agents", 16),
            max_obstacles=getattr(self.model_cfg, "max_obstacles", 20),
            max_lanes=getattr(self.model_cfg, "max_lanes", 20),
            lane_pts=getattr(self.model_cfg, "lane_pts", 10),
            hz=self.model_cfg.hz,
            temporal_ops=self.cfg.temporal_ops,
            interaction_ops=self.cfg.interaction_ops,
            map_ops=self.cfg.map_ops,
            search_config=self.cfg,
            model_config=self.model_cfg,
        ).to(self.device)

        self.arch_params = ArchParams(
            num_temporal=self.cfg.num_temporal,
            num_interaction=self.cfg.num_interaction,
            num_map=self.cfg.num_map,
            search_config=self.cfg,
        ).to(self.device)

        self.opt_w = Adam(
            self.supernet.parameters(),
            lr=self.cfg.lr_weights,
            weight_decay=self.cfg.weight_decay,
        )
        self.opt_a = Adam(
            self.arch_params.parameters(),
            lr=self.cfg.lr_alphas,
            betas=(0.5, 0.999),
            weight_decay=0.0,
        )

        search_pct = getattr(self.cfg, "search_samples_percent", 20.0)
        self.train_loader, self.val_loader = get_dataloaders(
            train_path=self.cfg.train_path,
            val_path=self.cfg.val_path,
            batch_size=self.cfg.batch_size,
            dataset_type=self.cfg.dataset,
            sample_percent=search_pct,
            model_config=self.model_cfg,
        )
        self.val_iter = None
        self.all_costs = self.supernet.get_ops_flops()
        self.elapsed_seconds = 0.0

        self.temp_ratio = (self.cfg.temperature_min / self.cfg.temperature_start) ** (
            1.0 / max(1, self.cfg.epochs)
        )
        self.grad_accum_steps = max(1, self.cfg.grad_accum_steps)
        if self.cfg.gradient_checkpointing:
            self.supernet.enable_gradient_checkpointing(True)

    def _get_current_temperature(self, epoch: int) -> float:
        """Anneals temperature to encourage sharp discrete operator selection."""
        return max(
            self.cfg.temperature_min,
            self.cfg.temperature_start * (self.temp_ratio**epoch),
        )

    def profile_zero_cost_proxies(self) -> dict[str, Any]:
        """Calculate pre-training diagnostic ZCP metrics on a training batch (no val leakage, excluded from L_arch)."""
        if len(self.train_loader) == 0:
            raise RuntimeError("Training DataLoader is empty; cannot compute SNIP proxy.")
        sample_batch = next(iter(self.train_loader))
        snip = snip_score(self.supernet, sample_batch, self.device)
        synflow = synflow_score(self.supernet, self.device)
        return {
            "snip": snip,
            "synflow": synflow,
            "source_split": "train",
            "diagnostic_only": True,
        }

    def train_epoch(self, epoch: int) -> dict[str, float]:
        """Inner loop: Updates network weights (w) with Gradient Accumulation & AMP."""
        self.supernet.train()
        total_w_loss = 0.0
        step_count = 0
        temperature = self._get_current_temperature(epoch)
        self.opt_w.zero_grad()

        for batch_idx, batch in enumerate(self.train_loader):
            b = unpack_batch_to_device(batch, self.device)
            alphas_dict = self.arch_params.softmax_all(temperature=temperature)

            try:
                autocast_ctx = torch.amp.autocast(
                    device_type="cuda", enabled=self.use_amp
                )
            except (AttributeError, TypeError):
                autocast_ctx = torch.cuda.amp.autocast(enabled=self.use_amp)

            with autocast_ctx:
                pred_trajs, mode_logits = self.supernet(
                    b["past"],
                    alphas_dict,
                    agents_past=b.get("agents_past"),
                    map_lanes=b.get("map_lanes"),
                    category_id=b.get("category_id"),
                    map_lane_types=b.get("map_lane_types"),
                    neighbor_categories=b.get("neighbor_categories"),
                    agent_mask=b.get("agent_mask"),
                    lane_mask=b.get("lane_mask"),
                    static_obstacles=b.get("static_obstacles"),
                    static_obstacle_types=b.get("static_obstacle_types"),
                    static_obstacle_mask=b.get("static_obstacle_mask"),
                )
                loss_reg, loss_cls = best_of_k_loss(
                    pred_trajs, b["future"], mode_logits
                )
                loss_w = (loss_reg + 0.1 * loss_cls) / self.grad_accum_steps

            self.scaler.scale(loss_w).backward()

            if (batch_idx + 1) % self.grad_accum_steps == 0 or (batch_idx + 1) == len(
                self.train_loader
            ):
                self.scaler.unscale_(self.opt_w)
                for p in self.supernet.parameters():
                    if p.grad is not None and (
                        torch.isnan(p.grad).any() or torch.isinf(p.grad).any()
                    ):
                        p.grad.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
                torch.nn.utils.clip_grad_norm_(self.supernet.parameters(), max_norm=5.0)
                self.scaler.step(self.opt_w)
                self.scaler.update()
                self.opt_w.zero_grad()

            total_w_loss += loss_w.item() * self.grad_accum_steps
            step_count += 1

        avg_loss = total_w_loss / max(1, step_count)
        return {"train_w_loss": avg_loss, "step_count": step_count}

    def validate(self, epoch: int, is_warmup: bool = False) -> dict[str, Any]:
        """
        Outer loop: Updates architectural parameters (alpha) across validation batches.
        During warmup (is_warmup=True), architectural parameters are frozen.
        """
        temperature = self._get_current_temperature(epoch)

        # Number of validation batches per epoch for low-noise alpha gradient
        alpha_val_batches = self.grad_accum_steps

        if len(self.val_loader) == 0:
            raise RuntimeError("Validation DataLoader is empty; cannot run DARTS validation step.")

        val_ades = []
        val_fdes = []
        val_mrs = []
        k_eval = int(self.supernet.num_modes)
        effective_k = k_eval

        if not is_warmup:
            self.supernet.train()
            self.opt_a.zero_grad()
        else:
            self.supernet.eval()

        total_arch_loss = 0.0

        for _ in range(alpha_val_batches):
            try:
                val_batch = next(self.val_iter)
            except (TypeError, StopIteration):
                self.val_iter = iter(self.val_loader)
                val_batch = next(self.val_iter)

            b = unpack_batch_to_device(val_batch, self.device)
            alphas_dict = self.arch_params.softmax_all(temperature=temperature)

            try:
                autocast_val = torch.amp.autocast(device_type="cuda", enabled=False)
            except (AttributeError, TypeError):
                autocast_val = torch.cuda.amp.autocast(enabled=False)

            with torch.set_grad_enabled(not is_warmup), autocast_val:
                pred_trajs, mode_logits = self.supernet(
                    b["past"],
                    alphas_dict,
                    agents_past=b.get("agents_past"),
                    map_lanes=b.get("map_lanes"),
                    category_id=b.get("category_id"),
                    map_lane_types=b.get("map_lane_types"),
                    neighbor_categories=b.get("neighbor_categories"),
                    agent_mask=b.get("agent_mask"),
                    lane_mask=b.get("lane_mask"),
                    static_obstacles=b.get("static_obstacles"),
                    static_obstacle_types=b.get("static_obstacle_types"),
                    static_obstacle_mask=b.get("static_obstacle_mask"),
                )
                effective_k = min(k_eval, pred_trajs.shape[1])
                loss_reg, loss_cls = best_of_k_loss(
                    pred_trajs, b["future"], mode_logits
                )
                val_loss_ade = min_ade_k(
                    pred_trajs, b["future"], k=effective_k, mode_logits=mode_logits
                )
                val_loss_fde = min_fde_k(
                    pred_trajs, b["future"], k=effective_k, mode_logits=mode_logits
                )
                val_mr = endpoint_miss_rate_k(
                    pred_trajs, b["future"], k=effective_k, threshold=2.0, mode_logits=mode_logits
                )

                flops_cost = get_supernet_costs(alphas_dict, self.all_costs)
                batch_arch_loss = (
                    loss_reg + (0.1 * loss_cls) + (self.cfg.lambda_flops * flops_cost)
                )

                if not is_warmup:
                    (batch_arch_loss / alpha_val_batches).backward()

                total_arch_loss += batch_arch_loss.item()
                val_ades.append(val_loss_ade.item())
                val_fdes.append(val_loss_fde.item())
                val_mrs.append(val_mr)

        grad_norms = {"temporal": 0.0, "interaction": 0.0, "map": 0.0}
        if not is_warmup:
            for name, param in [
                ("temporal", self.arch_params.alpha_temporal),
                ("interaction", self.arch_params.alpha_interaction),
                ("map", self.arch_params.alpha_map),
            ]:
                if param.grad is not None:
                    if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                        param.grad.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
                    grad_norms[name] = round(param.grad.norm().item(), 6)

            torch.nn.utils.clip_grad_norm_(self.arch_params.parameters(), max_norm=1.0)

            self.opt_a.step()

        self.supernet.eval()

        a_temp, a_inter, a_map = alphas_dict
        mean_ade = float(sum(val_ades) / max(1, len(val_ades)))
        mean_fde = float(sum(val_fdes) / max(1, len(val_fdes)))
        mean_mr = float(sum(val_mrs) / max(1, len(val_mrs)))

        return {
            "val_ade": mean_ade,
            "val_fde": mean_fde,
            "val_miss_rate": mean_mr,
            "k_modes": effective_k,
            "flops": flops_cost.item(),
            "temperature": temperature,
            "grad_norms": grad_norms,
            "alphas": {
                "temporal": [round(x.item(), 4) for x in a_temp],
                "interaction": [round(x.item(), 4) for x in a_inter],
                "map": [round(x.item(), 4) for x in a_map],
            },
        }

    def run_search(self) -> tuple[int, int, int]:
        """Runs complete bi-level DARTS search loop."""
        t_start = time.time()
        warmup_epochs = self.cfg.warmup_epochs
        k_eval = int(self.supernet.num_modes)
        search_pct = getattr(self.cfg, "search_samples_percent", 100.0)
        has_maps = False
        if len(self.train_loader) > 0:
            peek = next(iter(self.train_loader))
            if peek.get("lane_mask") is not None:
                has_maps = bool(peek["lane_mask"].any().item())
            elif peek.get("map_lanes") is not None:
                has_maps = bool((peek["map_lanes"].abs().sum() > 0).item())

        print("=" * 80)
        print(" TrajectoryLNAS: Differentiable Neural Architecture Search (DARTS)")
        print(f" Dataset:        {self.cfg.dataset} (Subsampling: {search_pct}%)")
        print(f" Splits:         Train Batches={len(self.train_loader)} | Val Batches={len(self.val_loader)}")
        print(f" Modes (K):      {k_eval} | Map Context Active: {has_maps}")
        print(f" Temporal Ops:   {self.cfg.temporal_ops}")
        print(f" Social Ops:     {self.cfg.interaction_ops}")
        print(f" HD Map Ops:     {self.cfg.map_ops}")
        print(f" Device:         {self.device} (AMP={self.use_amp})")
        print(f" Batch Size:     {self.cfg.batch_size} (Grad Accum={self.grad_accum_steps})")
        print(f" Search Epochs:  {self.cfg.epochs} (Warmup={warmup_epochs})")
        print(f" FLOPs Penalty:  λ = {self.cfg.lambda_flops}")
        print("=" * 80)

        self.search_history = []
        self.proxies = self.profile_zero_cost_proxies()
        print(
            f"[Zero-Cost Proxies (Diagnostic Only, Split={self.proxies['source_split']}, Excluded from L_arch)] "
            f"SNIP: {self.proxies['snip']:.4f} | SynFlow: {self.proxies['synflow']:.4f}"
        )

        for epoch in range(1, self.cfg.epochs + 1):
            is_warmup = epoch <= warmup_epochs
            train_res = self.train_epoch(epoch)
            val_res = self.validate(epoch, is_warmup=is_warmup)
            effective_k = val_res.get("k_modes", k_eval)

            record = {
                "epoch": epoch,
                "is_warmup": is_warmup,
                "k_modes": effective_k,
                "train_loss": round(train_res["train_w_loss"], 4),
                "val_ade": round(val_res["val_ade"], 4),
                "val_fde": round(val_res["val_fde"], 4),
                "val_miss_rate": round(val_res["val_miss_rate"] * 100.0, 2),
                "flops_g": round(val_res["flops"], 5),
                "temperature": round(val_res["temperature"], 3),
                "alphas": val_res["alphas"],
            }
            self.search_history.append(record)
            prefix = f"Epoch [{epoch:02d}/{self.cfg.epochs:02d}]"
            tag = f" [Warmup {epoch}/{warmup_epochs}]" if is_warmup else ""
            print(
                f"{prefix}{tag} "
                f"Train Loss: {train_res['train_w_loss']:.3f} | "
                f"Val minADE@{effective_k}: {val_res['val_ade']:.3f}m | "
                f"Val minFDE@{effective_k}: {val_res['val_fde']:.3f}m | "
                f"FLOPs: {val_res['flops']:.4f}G | "
                f"Temp: {val_res['temperature']:.2f}" + (" (Alpha Frozen)" if is_warmup else "")
            )
            print(f"  -> Temporal Alphas:     {val_res['alphas']['temporal']}")
            print(f"  -> Interaction Alphas:  {val_res['alphas']['interaction']}")
            print(f"  -> Map Alphas:          {val_res['alphas']['map']}")

            if self.device.type == "cuda":
                torch.cuda.empty_cache()

        self.elapsed_seconds = time.time() - t_start
        print(
            f"\n[DARTSAlgorithm] Architecture search completed in {self.elapsed_seconds:.2f} seconds."
        )
        return self.discretize_architecture()

    def discretize_architecture(self) -> tuple[int, int, int]:
        """Extracts the discrete winning operators via argmax(alpha)."""
        print("=" * 80)
        print(" DARTS Search Completed. Discretizing Optimal Architecture...")
        print("=" * 80)

        temporal_ops = self.cfg.temporal_ops
        interaction_ops = self.cfg.interaction_ops
        map_ops = self.cfg.map_ops

        with torch.no_grad():
            a_temp, a_inter, a_map = self.arch_params.softmax_all()
            best_t = torch.argmax(a_temp).item()
            best_i = torch.argmax(a_inter).item()
            best_m = torch.argmax(a_map).item()

        t_name = temporal_ops[best_t] if best_t < len(temporal_ops) else f"Op_{best_t}"
        i_name = interaction_ops[best_i] if best_i < len(interaction_ops) else f"Op_{best_i}"
        m_name = map_ops[best_m] if best_m < len(map_ops) else f"Op_{best_m}"

        print(
            f"[*] Optimal Temporal Operator:    Index {best_t} -> {t_name}"
        )
        print(
            f"[*] Optimal Interaction Operator: Index {best_i} -> {i_name}"
        )
        print(f"[*] Optimal HD Map Operator:      Index {best_m} -> {m_name}")

        return best_t, best_i, best_m
