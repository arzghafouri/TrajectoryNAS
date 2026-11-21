import math
import torch
from torch import nn
from typing import Any
from configs.model_config import ModelConfig
from models.base_operator import (
    compute_base_cost_breakdown,
    compute_base_flops,
    trajectory_kinematics,
)
from models.supernet_operators import (
    InteractionMixedOp,
    MapMixedOp,
    PolylineEncoder,
    TemporalMixedOp,
)


class ArchParams(nn.Module):
    """
    Architectural parameters (alpha) controlling continuous operator weighting in DARTS.
    Maintains separate parameter vectors for Temporal, Interaction, and Map branches.
    Dynamically parameterized from SearchConfig or explicit operator counts.
    """
    def __init__(
        self,
        num_temporal: int | None = None,
        num_interaction: int | None = None,
        num_map: int | None = None,
        search_config: Any | None = None,
    ):
        super().__init__()
        from configs.search_config import SearchConfig
        cfg = search_config or SearchConfig()
        n_t = num_temporal if num_temporal is not None else getattr(cfg, "num_temporal", len(cfg.temporal_ops))
        n_i = num_interaction if num_interaction is not None else getattr(cfg, "num_interaction", len(cfg.interaction_ops))
        n_m = num_map if num_map is not None else getattr(cfg, "num_map", len(cfg.map_ops))
        self.alpha_temporal = nn.Parameter(torch.zeros(n_t))
        self.alpha_interaction = nn.Parameter(torch.zeros(n_i))
        self.alpha_map = nn.Parameter(torch.zeros(n_m))

    @property
    def num_temporal(self) -> int:
        return len(self.alpha_temporal)

    @property
    def num_interaction(self) -> int:
        return len(self.alpha_interaction)

    @property
    def num_map(self) -> int:
        return len(self.alpha_map)

    def softmax_all(self, temperature: float = 1.0) -> tuple:
        """Compute stable temperature-scaled probabilities without clipping logit differences."""
        temperature = float(temperature)
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("Architecture temperature must be finite and positive.")
        probabilities = []
        for parameter in (self.alpha_temporal, self.alpha_interaction, self.alpha_map):
            if not torch.isfinite(parameter).all():
                raise ValueError("Architecture logits must be finite.")
            logits = parameter.float()
            scaled_logits = (logits - logits.max().detach()) / temperature
            probabilities.append(torch.softmax(scaled_logits, dim=-1))
        return tuple(probabilities)


class SearchSuperNet(nn.Module):
    """
    TrajectoryLNAS Bi-Level Differentiable SuperNet.
    
    Contains continuous operator mixtures across 3 distinct domain branches:
      1. Temporal Branch: Single-agent sequential kinematics (GRU, LSTM, TCN, Self-Attention, etc.)
      2. Interaction Branch: Multi-agent social interactions (GCN, GAT, SocialPooling, PairwiseAttention, etc.)
      3. HD Map Branch: Vectorized road geometry (LaneAttention, LaneGraphEncoder, MapFeatureFusion, etc.)
      
    Multi-Modal Trajectory Prediction:
      Predicts K candidate future trajectories along with confidence probabilities
      to benchmark directly against state-of-the-art models on nuScenes / Argoverse.
    """
    def __init__(
        self,
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
        self.d_model = d_model
        self.hz = hz
        self.past_len = past_len
        self.future_len = future_len
        self.num_modes = num_modes
        self.max_agents = max_agents
        self.max_obstacles = max_obstacles
        self.max_lanes = max_lanes
        self.lane_pts = lane_pts
        self.use_checkpointing = False

        # Coordinate and kinematic embedding layers
        self.input_proj = nn.Linear(8, d_model)
        self.neighbor_proj = nn.Linear(8, d_model)
        self.category_emb = nn.Embedding(10, d_model)
        self.obstacle_proj = nn.Linear(4, d_model)
        self.obstacle_type_emb = nn.Embedding(6, d_model)
        self.map_encoder = PolylineEncoder(in_channels=4, d_model=d_model)

        # 3 Dedicated Domain-Aware Search Branches (dynamically configured from op lists)
        self.temporal_branch = TemporalMixedOp(d_model, op_names=self.temporal_ops)
        self.interaction_branch = InteractionMixedOp(d_model, op_names=self.interaction_ops)
        self.map_branch = MapMixedOp(d_model, op_names=self.map_ops)

        # Tri-Branch fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(d_model * 3, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU()
        )

        # Multi-Modal Future Trajectory Decoder
        self.decoder = nn.GRU(d_model, d_model, batch_first=True)
        # Outputs K displacements for each future timestep: [B, future_len, K * 2]
        self.out_proj = nn.Linear(d_model, num_modes * 2)
        # Confidence classification head over the K modes
        self.prob_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, num_modes)
        )

    def enable_gradient_checkpointing(self, enabled: bool = True):
        """Toggles gradient checkpointing across mixed op branches to reduce peak VRAM on RTX 3050 (4GB)."""
        self.use_checkpointing = enabled

    def forward(
        self,
        past: torch.Tensor,
        alphas_tuple: tuple,
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
        Continuous forward pass weighting each operator by its architectural alpha.
        past: [B, T_past, 2 or 4] or [B, T_past, Agents, 2 or 4]
        alphas_tuple: (alpha_temp, alpha_inter, alpha_map)
        agents_past: [B, T_past, N_agents, 2 or 4] (optional surrounding agents history)
        map_lanes: [B, N_lanes, lane_pts, 2 or 4] (optional vectorized lane polylines)
        category_id: [B] (optional agent class ID)
        map_lane_types: [B, N_lanes] (optional lane type ID)
        neighbor_categories: [B, N_agents] (optional neighbor class IDs)
        agent_mask: [B, N_agents] (optional boolean validity mask)
        lane_mask: [B, N_lanes] (optional boolean validity mask)
        
        Returns:
            pred_trajectories: [B, K, future_len, 2]
            mode_probs: [B, K] (logits)
        """
        # If past is 4D [B, T, N, 2], split target ego and agents
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
            # agents_past: [B, T_past, N, C]
            N = agents_past.shape[2]
            if agent_mask is None:
                agent_mask = (agents_past.abs().sum(dim=(1, 3)) > 1e-4) # [B, N]
            else:
                agent_mask = agent_mask.to(device=past.device, dtype=torch.bool).clone()
            agent_mask[:, 0] = True # Target ego is always valid

            if agents_past.shape[-1] in (2, 4):
                kinematic_agents = trajectory_kinematics(agents_past, hz=self.hz) # [B, T, N, 8]
            else:
                kinematic_agents = agents_past

            # Embed neighbor coordinates and kinematics
            emb_neighbors = self.neighbor_proj(kinematic_agents) # [B, T, N, d_model]
            # Combine historical trajectory mean with final kinematic observation state
            n_mean = emb_neighbors.mean(dim=1) # [B, N, d_model]
            n_last = emb_neighbors[:, -1, :, :] # [B, N, d_model]
            neighbor_feats = 0.5 * (n_mean + n_last) # [B, N, d_model]

            # Inject Neighbor Category semantics if available
            if neighbor_categories is not None and neighbor_categories.shape == (B, N):
                n_cat_feat = self.category_emb(neighbor_categories.clamp(0, 9)) # [B, N, d_model]
                neighbor_feats = neighbor_feats + n_cat_feat

            # Strictly zero-out padded/inactive neighbors to prevent any feature leakage
            neighbor_feats = neighbor_feats * agent_mask.unsqueeze(-1).float()
            agent_pos = agents_past[:, -1, :, :2] # [B, N, 2] positions at observation t=19
        else:
            # Explicit missing modality representation (no feature spoofing/leakage from target_feat)
            neighbor_feats = torch.zeros(B, 1, self.d_model, device=past.device)
            agent_mask = torch.zeros(B, 1, dtype=torch.bool, device=past.device)
            agent_pos = torch.zeros(B, 1, 2, device=past.device)

        # Encode static obstacles as spatial context tokens for the interaction branch.
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
            # map_lanes: [B, N_lanes, lane_pts, 2 or 4]
            if lane_mask is None:
                lane_mask = (map_lanes.abs().sum(dim=(2, 3)) > 1e-4) # [B, N_lanes]
            else:
                lane_mask = lane_mask.to(device=past.device, dtype=torch.bool)
            lane_feats = self.map_encoder(map_lanes, lane_types=map_lane_types, mask=lane_mask) # [B, N_lanes, d_model]
            lane_feats = lane_feats * lane_mask.unsqueeze(-1).float()
        else:
            # Explicit missing map representation (no feature spoofing/leakage from target_feat)
            lane_feats = torch.zeros(B, 1, self.d_model, device=past.device)
            lane_mask = torch.zeros(B, 1, dtype=torch.bool, device=past.device)

        alpha_temp, alpha_inter, alpha_map = alphas_tuple

        # 4. Evaluate three branches concurrently (with optional gradient checkpointing for 4GB VRAM)
        if self.use_checkpointing and self.training:
            from torch.utils.checkpoint import checkpoint
            feat_temp = checkpoint(self.temporal_branch, target_feat, alpha_temp, use_reentrant=False)
            feat_inter = checkpoint(
                self.interaction_branch, target_feat, neighbor_feats, alpha_inter,
                agent_mask, agent_pos, use_reentrant=False
            )
            feat_map = checkpoint(
                self.map_branch, target_feat, lane_feats, alpha_map,
                None, lane_mask, use_reentrant=False
            )
        else:
            feat_temp = self.temporal_branch(target_feat, alpha_temp)
            feat_inter = self.interaction_branch(
                target_feat, neighbor_feats, alpha_inter, agent_mask=agent_mask, agent_pos=agent_pos
            )
            feat_map = self.map_branch(
                target_feat, lane_feats, alpha_map, lane_mask=lane_mask
            )

        # 5. Tri-Branch Fusion
        fused = torch.cat([feat_temp, feat_inter, feat_map], dim=-1)  # [B, T_past, 3 * d_model]
        x_fused = self.fusion(fused)  # [B, T_past, d_model]

        # 6. Multi-Modal Autoregressive Decoding
        # Replicate final temporal state over future prediction horizon
        dec_in = x_fused[:, -1:, :].repeat(1, self.future_len, 1)  # [B, future_len, d_model]
        dec_out, _ = self.decoder(dec_in)  # [B, future_len, d_model]

        # Multi-step displacement prediction for K modes
        deltas = self.out_proj(dec_out)  # [B, future_len, K * 2]
        deltas = deltas.view(B, self.future_len, self.num_modes, 2).permute(0, 2, 1, 3)  # [B, K, future_len, 2]
        pred_trajs = torch.cumsum(deltas, dim=2)  # [B, K, future_len, 2] cumulative displacement

        # Mode classification logits from ego trajectory context
        mode_logits = self.prob_head(x_fused[:, -1, :])  # [B, K]

        return pred_trajs, mode_logits

    def get_ops_flops(self, batch_size: int = 1):
        """Calculates theoretical FLOPs for all candidate operators across branches."""
        map_enc_flops = self.map_encoder.get_flops(
            self.max_lanes, self.lane_pts, self.d_model, batch_size=1
        )
        return {
            "temporal": self.temporal_branch.get_ops_flops(self.past_len, batch_size=batch_size),
            "interaction": self.interaction_branch.get_ops_flops(
                self.past_len, num_agents=self.max_agents + self.max_obstacles, batch_size=batch_size
            ),
            "map": self.map_branch.get_ops_flops(
                self.past_len, num_lanes=self.max_lanes, batch_size=batch_size
            ),
            "base": compute_base_flops(
                past_len=self.past_len,
                future_len=self.future_len,
                d_model=self.d_model,
                num_modes=self.num_modes,
                max_agents=self.max_agents,
                max_obstacles=self.max_obstacles,
                max_lanes=self.max_lanes,
                lane_pts=self.lane_pts,
                map_encoder_flops=map_enc_flops,
                batch_size=batch_size,
            ),
        }

    def get_cost_breakdown(self, batch_size: int = 1) -> dict[str, Any]:
        """Return detailed MACs/FLOPs/non-MAC breakdown for base network and all candidate operators."""
        map_enc_breakdown = self.map_encoder.get_cost_breakdown(
            num_lanes=self.max_lanes, lane_pts=self.lane_pts, d_model=self.d_model, batch_size=1
        )
        return {
            "batch_size": batch_size,
            "counting_convention": "1 MAC = 2 FLOPs",
            "base": compute_base_cost_breakdown(
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
            ),
            "temporal": self.temporal_branch.get_ops_cost_breakdowns(self.past_len, batch_size=batch_size),
            "interaction": self.interaction_branch.get_ops_cost_breakdowns(
                self.past_len, num_agents=self.max_agents + self.max_obstacles, batch_size=batch_size
            ),
            "map": self.map_branch.get_ops_cost_breakdowns(
                self.past_len, num_lanes=self.max_lanes, batch_size=batch_size
            ),
        }


# Class alias for backward compatibility and case variations
SearchSupernet = SearchSuperNet

