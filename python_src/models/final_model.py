import torch
from torch import nn

from typing import Any
from configs.model_config import ModelConfig
from models.base_operator import (
    compute_base_cost_breakdown,
    trajectory_kinematics,
)
from models.supernet_operators import (
    PolylineEncoder,
    build_interaction_operator,
    build_map_operator,
    build_temporal_operator,
)


class FinalTrajectoryModel(nn.Module):
    """
    Extracted standalone multi-modal trajectory prediction model instantiated from the
    best discrete operators discovered by TrajectoryLNAS.
    
    Predicts K candidate future trajectory modes (default K=6) along with confidence probabilities.
    Consumes target kinematics, dynamic agents, static obstacle context, and vectorized HD map lanes.
    """
    def __init__(
        self,
        best_temporal_idx: int | str = None,
        best_interaction_idx: int | str = None,
        best_map_idx: int | str = None,
        d_model: int | None = None,
        past_len: int | None = None,
        future_len: int | None = None,
        num_modes: int | None = None,
        max_agents: int | None = None,
        max_lanes: int | None = None,
        lane_pts: int | None = None,
        max_obstacles: int | None = None,
        hz: float | None = None,
        temporal_ops: list[str] | None = None,
        interaction_ops: list[str] | None = None,
        map_ops: list[str] | None = None,
        search_config: Any | None = None,
        model_config: ModelConfig | None = None,
        **kwargs
    ):
        super().__init__()
        from configs.search_config import SearchConfig
        cfg = model_config or ModelConfig()
        s_cfg = search_config or SearchConfig()

        self.temporal_ops = list(temporal_ops if temporal_ops is not None else s_cfg.temporal_ops)
        self.interaction_ops = list(interaction_ops if interaction_ops is not None else s_cfg.interaction_ops)
        self.map_ops = list(map_ops if map_ops is not None else s_cfg.map_ops)

        d_model = d_model if d_model is not None else cfg.d_model
        past_len = past_len if past_len is not None else cfg.past_len
        future_len = future_len if future_len is not None else cfg.future_len
        num_modes = num_modes if num_modes is not None else cfg.num_modes
        max_agents = max_agents if max_agents is not None else cfg.max_agents
        max_obstacles = max_obstacles if max_obstacles is not None else cfg.max_obstacles
        max_lanes = max_lanes if max_lanes is not None else cfg.max_lanes
        lane_pts = lane_pts if lane_pts is not None else cfg.lane_pts
        hz = float(hz if hz is not None else cfg.hz)
        self.past_len = past_len
        self.hz = hz
        self.future_len = future_len
        self.d_model = d_model
        self.num_modes = num_modes
        self.max_agents = max_agents
        self.max_obstacles = max_obstacles
        self.max_lanes = max_lanes
        self.lane_pts = lane_pts

        # Resolve temporal operator selection
        if best_temporal_idx is None:
            best_temporal_idx = kwargs.get("temporal_op_idx", 0)
        if isinstance(best_temporal_idx, str):
            t_op_name = best_temporal_idx
            self.best_temporal_idx = self.temporal_ops.index(t_op_name) if t_op_name in self.temporal_ops else -1
        else:
            self.best_temporal_idx = int(best_temporal_idx)
            if not (0 <= self.best_temporal_idx < len(self.temporal_ops)):
                raise ValueError(
                    f"best_temporal_idx={self.best_temporal_idx} is out of range for temporal_ops "
                    f"(valid indices: 0..{len(self.temporal_ops) - 1}, ops={self.temporal_ops})."
                )
            t_op_name = self.temporal_ops[self.best_temporal_idx]

        # Resolve interaction operator selection
        if best_interaction_idx is None:
            best_interaction_idx = kwargs.get("interaction_op_idx", 0)
        if isinstance(best_interaction_idx, str):
            i_op_name = best_interaction_idx
            self.best_interaction_idx = self.interaction_ops.index(i_op_name) if i_op_name in self.interaction_ops else -1
        else:
            self.best_interaction_idx = int(best_interaction_idx)
            if not (0 <= self.best_interaction_idx < len(self.interaction_ops)):
                raise ValueError(
                    f"best_interaction_idx={self.best_interaction_idx} is out of range for interaction_ops "
                    f"(valid indices: 0..{len(self.interaction_ops) - 1}, ops={self.interaction_ops})."
                )
            i_op_name = self.interaction_ops[self.best_interaction_idx]

        # Resolve map operator selection
        if best_map_idx is None:
            best_map_idx = kwargs.get("map_op_idx", 0)
        if isinstance(best_map_idx, str):
            m_op_name = best_map_idx
            self.best_map_idx = self.map_ops.index(m_op_name) if m_op_name in self.map_ops else -1
        else:
            self.best_map_idx = int(best_map_idx)
            if not (0 <= self.best_map_idx < len(self.map_ops)):
                raise ValueError(
                    f"best_map_idx={self.best_map_idx} is out of range for map_ops "
                    f"(valid indices: 0..{len(self.map_ops) - 1}, ops={self.map_ops})."
                )
            m_op_name = self.map_ops[self.best_map_idx]

        # 1. Coordinate and Kinematic Embedding Layers
        self.input_proj = nn.Linear(8, d_model)
        self.neighbor_proj = nn.Linear(8, d_model)
        self.category_emb = nn.Embedding(10, d_model)
        self.obstacle_proj = nn.Linear(4, d_model)
        self.obstacle_type_emb = nn.Embedding(6, d_model)
        self.map_encoder = PolylineEncoder(in_channels=4, d_model=d_model)

        # 2, 3, 4. Dynamically Instantiate Selected Operators via Registry
        self.temporal_op = build_temporal_operator(t_op_name, d_model)
        self.interaction_op = build_interaction_operator(i_op_name, d_model)
        self.map_op = build_map_operator(m_op_name, d_model)

        # 5. Tri-Branch Fusion Layer
        self.fusion = nn.Sequential(
            nn.Linear(d_model * 3, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU()
        )

        # 6. Multi-Modal Recurrent Decoder & Mode Classification Head
        self.decoder = nn.GRU(d_model, d_model, batch_first=True)
        self.out_proj = nn.Linear(d_model, num_modes * 2)
        self.prob_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, num_modes)
        )

    def forward(
        self,
        past: torch.Tensor,
        agents_past: torch.Tensor = None,
        map_lanes: torch.Tensor = None,
        category_id: torch.Tensor = None,
        map_lane_types: torch.Tensor = None,
        neighbor_categories: torch.Tensor = None,
        agent_mask: torch.Tensor = None,
        lane_mask: torch.Tensor = None,
        static_obstacles: torch.Tensor = None,
        static_obstacle_types: torch.Tensor = None,
        static_obstacle_mask: torch.Tensor = None
    ):
        """
        Forward multi-modal trajectory inference pass.
        past: [B, 20, 2 or 4] or [B, 20, N, 2 or 4]
        agents_past: [B, 20, N_agents, 2 or 4] (optional surrounding agents history)
        map_lanes: [B, N_lanes, lane_pts, 2 or 4] (optional vectorized lane polylines)
        category_id: [B] (optional agent class ID)
        map_lane_types: [B, N_lanes] (optional lane type ID)
        neighbor_categories: [B, N_agents] (optional neighbor class IDs)
        agent_mask: [B, N_agents] (optional boolean validity mask)
        lane_mask: [B, N_lanes] (optional boolean validity mask)
        
        Returns:
            pred_trajectories: [B, K, 30, 2]
            mode_probs: [B, K] (logits)
        """
        if past.dim() == 4:
            if agents_past is None:
                agents_past = past
            past = past[:, :, 0, :]

        B, T_past, C_past = past.shape

        # 1. Target ego coordinate and kinematic velocity embedding
        if C_past in (2, 4):
            kinematic_past = trajectory_kinematics(past, hz=self.hz) # [B, T_past, 8]
        else:
            kinematic_past = past

        target_feat = self.input_proj(kinematic_past)  # [B, T_past, d_model]

        # Inject Agent Category semantics if available
        if category_id is not None:
            cat_idx = category_id.view(B).clamp(0, 9)
            cat_feat = self.category_emb(cat_idx).unsqueeze(1) # [B, 1, d_model]
            target_feat = target_feat + cat_feat

        # 2. Surrounding agents processing
        neighbor_feats = None
        agent_pos = None

        if agents_past is None:
            agents_past = past.unsqueeze(2)
            agent_mask = torch.ones(B, 1, dtype=torch.bool, device=past.device)
            neighbor_categories = category_id.view(B, 1) if category_id is not None else None

        if agents_past is not None and agents_past.dim() == 4:
            N = agents_past.shape[2]
            if agent_mask is None:
                agent_mask = (agents_past.abs().sum(dim=(1, 3)) > 1e-4) # [B, N]
            else:
                agent_mask = agent_mask.to(device=past.device, dtype=torch.bool).clone()
            agent_mask[:, 0] = True

            if agents_past.shape[-1] in (2, 4):
                kinematic_agents = trajectory_kinematics(agents_past, hz=self.hz) # [B, T, N, 8]
            else:
                kinematic_agents = agents_past

            emb_neighbors = self.neighbor_proj(kinematic_agents)
            n_mean = emb_neighbors.mean(dim=1)
            n_last = emb_neighbors[:, -1, :, :]
            neighbor_feats = 0.5 * (n_mean + n_last) # [B, N, d_model]

            # Inject Neighbor Category semantics if available
            if neighbor_categories is not None and neighbor_categories.shape == (B, N):
                n_cat_feat = self.category_emb(neighbor_categories.clamp(0, 9)) # [B, N, d_model]
                neighbor_feats = neighbor_feats + n_cat_feat

            neighbor_feats = neighbor_feats * agent_mask.unsqueeze(-1).float()
            agent_pos = agents_past[:, -1, :, :2]
        else:
            # Explicit missing modality representation (strictly no feature spoofing/leakage from target)
            neighbor_feats = torch.zeros(B, 1, self.d_model, device=past.device)
            agent_mask = torch.zeros(B, 1, dtype=torch.bool, device=past.device)
            agent_pos = torch.zeros(B, 1, 2, device=past.device)

        # Static obstacles join the social context with type, size, and relative position.
        if static_obstacles is not None and static_obstacles.dim() == 3:
            obstacle_features = static_obstacles.to(device=past.device, dtype=past.dtype).clone()
            obstacle_features[..., :2] = obstacle_features[..., :2] / 50.0
            obstacle_features[..., 2:] = obstacle_features[..., 2:] / 10.0
            obstacle_feats = self.obstacle_proj(obstacle_features)
            if static_obstacle_types is not None:
                obstacle_types = static_obstacle_types.to(device=past.device, dtype=torch.long).clamp(0, 5)
                obstacle_feats = obstacle_feats + self.obstacle_type_emb(obstacle_types)
            if static_obstacle_mask is None:
                obstacle_mask = static_obstacles.abs().sum(dim=-1) > 1e-4
            else:
                obstacle_mask = static_obstacle_mask.to(device=past.device, dtype=torch.bool)
            obstacle_feats = obstacle_feats * obstacle_mask.unsqueeze(-1).to(obstacle_feats.dtype)
            obstacle_pos = static_obstacles[..., :2].to(device=past.device, dtype=past.dtype)
            neighbor_feats = torch.cat((neighbor_feats, obstacle_feats), dim=1)
            agent_mask = torch.cat((agent_mask, obstacle_mask), dim=1)
            agent_pos = torch.cat((agent_pos, obstacle_pos), dim=1)

        # 3. Vectorized HD Map processing
        lane_feats = None

        if map_lanes is not None and map_lanes.dim() == 4 and map_lanes.shape[1] > 0:
            if lane_mask is None:
                lane_mask = (map_lanes.abs().sum(dim=(2, 3)) > 1e-4)
            else:
                lane_mask = lane_mask.to(device=past.device, dtype=torch.bool)
            lane_feats = self.map_encoder(map_lanes, lane_types=map_lane_types, mask=lane_mask)
            lane_feats = lane_feats * lane_mask.unsqueeze(-1).float()
        else:
            # Explicit missing map representation (strictly no feature spoofing/leakage from target)
            lane_feats = torch.zeros(B, 1, self.d_model, device=past.device)
            lane_mask = torch.zeros(B, 1, dtype=torch.bool, device=past.device)

        # 4. Extract features through discovered optimal operators
        feat_temp = self.temporal_op(target_feat)
        feat_inter = self.interaction_op(
            target_feat, neighbor_feats=neighbor_feats, agent_mask=agent_mask, agent_pos=agent_pos
        )
        feat_map = self.map_op(
            target_feat, lane_feats=lane_feats, lane_mask=lane_mask
        )

        # 5. Multi-modal fusion
        combined = torch.cat([feat_temp, feat_inter, feat_map], dim=-1)
        x_fused = self.fusion(combined)

        # 6. Decode future positions via cumulative displacement (kinematically consistent)
        dec_in = x_fused[:, -1:, :].repeat(1, self.future_len, 1)
        dec_out, _ = self.decoder(dec_in)

        # Predict K candidate displacements
        deltas = self.out_proj(dec_out)  # [B, future_len, K * 2]
        deltas = deltas.view(B, self.future_len, self.num_modes, 2).permute(0, 2, 1, 3)
        pred = torch.cumsum(deltas, dim=2)  # [B, K, future_len, 2]

        # Mode classification confidence logits
        mode_logits = self.prob_head(x_fused[:, -1, :])  # [B, K]

        return pred, mode_logits

    def get_summary(self, batch_size: int = 1):
        t_name = self.temporal_ops[self.best_temporal_idx] if 0 <= self.best_temporal_idx < len(self.temporal_ops) else type(self.temporal_op).__name__
        i_name = self.interaction_ops[self.best_interaction_idx] if 0 <= self.best_interaction_idx < len(self.interaction_ops) else type(self.interaction_op).__name__
        m_name = self.map_ops[self.best_map_idx] if 0 <= self.best_map_idx < len(self.map_ops) else type(self.map_op).__name__

        map_enc_breakdown = self.map_encoder.get_cost_breakdown(
            num_lanes=self.max_lanes, lane_pts=self.lane_pts, d_model=self.d_model, batch_size=1
        )
        base_breakdown = compute_base_cost_breakdown(
            past_len=self.past_len,
            future_len=self.future_len,
            d_model=self.d_model,
            num_modes=self.num_modes,
            max_agents=self.max_agents,
            max_obstacles=self.max_obstacles,
            max_lanes=self.max_lanes,
            lane_pts=self.lane_pts,
            map_encoder_flops=map_enc_breakdown["theoretical_flops"],
            batch_size=batch_size,
            map_encoder_params=map_enc_breakdown["trainable_params"],
        )
        temp_breakdown = self.temporal_op.get_cost_breakdown(
            seq_len=self.past_len, d_model=self.d_model, batch_size=batch_size
        )
        inter_breakdown = self.interaction_op.get_cost_breakdown(
            seq_len=self.past_len,
            d_model=self.d_model,
            num_agents=self.max_agents + self.max_obstacles,
            batch_size=batch_size,
        )
        map_breakdown = self.map_op.get_cost_breakdown(
            seq_len=self.past_len, d_model=self.d_model, num_lanes=self.max_lanes, batch_size=batch_size
        )

        base_flops = base_breakdown["theoretical_flops"]
        op_flops = (
            temp_breakdown["theoretical_flops"]
            + inter_breakdown["theoretical_flops"]
            + map_breakdown["theoretical_flops"]
        )
        total_flops = base_flops + op_flops
        total_macs = (
            base_breakdown["theoretical_macs"]
            + temp_breakdown["theoretical_macs"]
            + inter_breakdown["theoretical_macs"]
            + map_breakdown["theoretical_macs"]
        )
        total_flops_g = round(total_flops / 1e9, 4)
        total_flops_m = round(total_flops / 1e6, 3)
        total_macs_m = round((total_flops / 2.0) / 1e6, 3)

        return {
            "temporal_op": t_name,
            "interaction_op": i_name,
            "map_op": m_name,
            "best_temporal_idx": self.best_temporal_idx,
            "best_interaction_idx": self.best_interaction_idx,
            "best_map_idx": self.best_map_idx,
            "d_model": self.d_model,
            "num_modes": self.num_modes,
            "max_agents": self.max_agents,
            "max_obstacles": self.max_obstacles,
            "max_lanes": self.max_lanes,
            "lane_pts": self.lane_pts,
            "past_len": self.past_len,
            "future_len": self.future_len,
            "batch_size": base_breakdown["batch_size"],
            "counting_convention": "1 MAC = 2 FLOPs",
            "theoretical_flops": total_flops,
            "theoretical_macs": total_macs,
            "theoretical_flops_m": total_flops_m,
            "theoretical_flops_g": total_flops_g,
            "theoretical_macs_m": total_macs_m,
            "base_flops": base_flops,
            "op_flops": op_flops,
            "base_flops_m": round(base_flops / 1e6, 3),
            "op_flops_m": round(op_flops / 1e6, 3),
            "parameters": sum(p.numel() for p in self.parameters()),
            "trainable_parameters": sum(p.numel() for p in self.parameters() if p.requires_grad),
            "cost_breakdown": {
                "base": base_breakdown,
                "temporal": temp_breakdown,
                "interaction": inter_breakdown,
                "map": map_breakdown,
            },
        }


# Compatibility alias for renamed model class
FinalTrajectoryLNAS = FinalTrajectoryModel

__all__ = ["FinalTrajectoryLNAS", "FinalTrajectoryModel"]
