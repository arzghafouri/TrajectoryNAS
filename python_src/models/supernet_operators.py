r"""
SuperNet Operators for TrajectoryLNAS
====================================
Defines candidates across Temporal, Interaction, and Map branches with standard FLOPs interfaces.
"""

import math
import torch
import torch.nn.functional as F
from torch import nn

from models.base_operator import (
    BaseOperator,
    batch_scale as _batch_scale,
    safe_cross_attention,
    valid_non_ego_mask as _valid_non_ego_mask,
)

# =========================================================================
# 1. Temporal Operators (Single-Agent Trajectory Dynamics)
# =========================================================================

class GRUEncoder(BaseOperator):
    """Recurrent GRU encoder for capturing sequential motion history."""

    def __init__(self, d_model: int):
        super().__init__(d_model)
        self.gru = nn.GRU(d_model, d_model, batch_first=True)

    def forward(self, x, *args, **kwargs):
        out, _ = self.gru(x)
        return out

    def get_cost_breakdown(self, seq_len=20, d_model=128, batch_size=1, **kwargs):
        scale = _batch_scale(batch_size)
        d_model = d_model if d_model is not None else self.d_model
        per_sample = seq_len * (12 * d_model * d_model + 18 * d_model)
        flops = float(scale * per_sample)
        return {
            "operator": type(self).__name__,
            "trainable_params": sum(p.numel() for p in self.parameters() if p.requires_grad),
            "total_params": sum(p.numel() for p in self.parameters()),
            "batch_size": scale,
            "theoretical_macs": float(scale * seq_len * 6 * d_model * d_model),
            "theoretical_flops": flops,
            "counting_convention": "1 MAC = 2 FLOPs (plus 18*d_model elementwise gate FLOPs/step)",
            "input_dimensions": {"seq_len": int(seq_len), "d_model": int(d_model)},
            "non_mac_ops": ["sigmoid_gates", "tanh_activation", "gate_elementwise_products"],
        }

    def get_flops(self, seq_len=20, d_model=128, batch_size=1, **kwargs):
        return self.get_cost_breakdown(seq_len=seq_len, d_model=d_model, batch_size=batch_size)["theoretical_flops"]


class LSTMEncoder(BaseOperator):
    """LSTM encoder with cell states for long-term temporal dependencies."""

    def __init__(self, d_model: int):
        super().__init__(d_model)
        self.lstm = nn.LSTM(d_model, d_model, batch_first=True)

    def forward(self, x, *args, **kwargs):
        out, _ = self.lstm(x)
        return out

    def get_cost_breakdown(self, seq_len=20, d_model=128, batch_size=1, **kwargs):
        scale = _batch_scale(batch_size)
        d_model = d_model if d_model is not None else self.d_model
        per_sample = seq_len * (16 * d_model * d_model + 24 * d_model)
        flops = float(scale * per_sample)
        return {
            "operator": type(self).__name__,
            "trainable_params": sum(p.numel() for p in self.parameters() if p.requires_grad),
            "total_params": sum(p.numel() for p in self.parameters()),
            "batch_size": scale,
            "theoretical_macs": float(scale * seq_len * 8 * d_model * d_model),
            "theoretical_flops": flops,
            "counting_convention": "1 MAC = 2 FLOPs (plus 24*d_model elementwise cell/gate FLOPs/step)",
            "input_dimensions": {"seq_len": int(seq_len), "d_model": int(d_model)},
            "non_mac_ops": ["sigmoid_gates", "tanh_cell_state", "gate_elementwise_products"],
        }

    def get_flops(self, seq_len=20, d_model=128, batch_size=1, **kwargs):
        return self.get_cost_breakdown(seq_len=seq_len, d_model=d_model, batch_size=batch_size)["theoretical_flops"]


class TemporalConvEncoder(BaseOperator):
    """Temporal Convolutional Network (TCN) 1D with dilated receptive field."""

    def __init__(self, d_model: int, kernel: int = 3, dilation: int = 1):
        super().__init__(d_model)
        self.kernel = kernel
        self.dilation = dilation
        padding = (kernel - 1) * dilation // 2
        self.conv = nn.Conv1d(d_model, d_model, kernel_size=kernel, padding=padding, dilation=dilation)
        self.norm = nn.BatchNorm1d(d_model)
        self.act = nn.ReLU()

    def forward(self, x, *args, **kwargs):
        residual = x
        x_t = x.transpose(1, 2)
        out = self.act(self.norm(self.conv(x_t)))
        out = out.transpose(1, 2)
        return out + residual

    def get_cost_breakdown(self, seq_len=20, d_model=128, batch_size=1, **kwargs):
        scale = _batch_scale(batch_size)
        d_model = d_model if d_model is not None else self.d_model
        per_sample = 2 * seq_len * self.kernel * d_model * d_model
        flops = float(scale * per_sample)
        return {
            "operator": type(self).__name__,
            "trainable_params": sum(p.numel() for p in self.parameters() if p.requires_grad),
            "total_params": sum(p.numel() for p in self.parameters()),
            "batch_size": scale,
            "theoretical_macs": flops / 2.0,
            "theoretical_flops": flops,
            "counting_convention": "1 MAC = 2 FLOPs",
            "input_dimensions": {
                "seq_len": int(seq_len),
                "d_model": int(d_model),
                "kernel": int(self.kernel),
                "dilation": int(self.dilation),
            },
            "non_mac_ops": ["BatchNorm1d", "ReLU", "residual_add"],
        }

    def get_flops(self, seq_len=20, d_model=128, batch_size=1, **kwargs):
        return self.get_cost_breakdown(seq_len=seq_len, d_model=d_model, batch_size=batch_size)["theoretical_flops"]


class TemporalSelfAttention(BaseOperator):
    """Multi-Head Self-Attention over the trajectory temporal dimension."""

    def __init__(self, d_model: int, heads: int = 4):
        super().__init__(d_model)
        self.heads = heads
        self.attn = nn.MultiheadAttention(d_model, heads, batch_first=True)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, *args, **kwargs):
        attn_out, _ = self.attn(x, x, x)
        return self.norm(x + attn_out)

    def get_cost_breakdown(self, seq_len=20, d_model=128, batch_size=1, **kwargs):
        scale = _batch_scale(batch_size)
        d_model = d_model if d_model is not None else self.d_model
        per_sample = 8 * seq_len * d_model * d_model + 4 * seq_len * seq_len * d_model
        flops = float(scale * per_sample)
        return {
            "operator": type(self).__name__,
            "trainable_params": sum(p.numel() for p in self.parameters() if p.requires_grad),
            "total_params": sum(p.numel() for p in self.parameters()),
            "batch_size": scale,
            "theoretical_macs": flops / 2.0,
            "theoretical_flops": flops,
            "counting_convention": "1 MAC = 2 FLOPs",
            "input_dimensions": {"seq_len": int(seq_len), "d_model": int(d_model), "heads": int(self.heads)},
            "non_mac_ops": ["attention_softmax", "attention_scaling", "LayerNorm", "residual_add"],
        }

    def get_flops(self, seq_len=20, d_model=128, batch_size=1, **kwargs):
        return self.get_cost_breakdown(seq_len=seq_len, d_model=d_model, batch_size=batch_size)["theoretical_flops"]


# =========================================================================
# 2. Multi-Agent Social Interaction Operators (Agent-to-Agent Domain)
# =========================================================================

class GCNLayer(BaseOperator):
    """Graph Convolutional Network modeling spatial adjacency between ego and neighbors."""

    def __init__(self, d_model: int):
        super().__init__(d_model)
        self.fc = nn.Linear(d_model, d_model)
        self.act = nn.ReLU()
        self.norm = nn.LayerNorm(d_model)

    def forward(self, target_feat, neighbor_feats=None, agent_mask=None, agent_pos=None, *args, **kwargs):
        if neighbor_feats is None:
            return target_feat

        B, T, D = target_feat.shape
        N = neighbor_feats.shape[1]
        if N == 0:
            return target_feat

        if agent_pos is not None:
            diff = agent_pos.unsqueeze(2) - agent_pos.unsqueeze(1)
            dist = torch.sqrt(torch.sum(diff ** 2, dim=-1) + 1e-8)
            adj = torch.exp(-dist / 10.0)
        else:
            adj = torch.eye(N, device=neighbor_feats.device, dtype=neighbor_feats.dtype).unsqueeze(0).expand(B, -1, -1).clone()
            if N > 1:
                norm_feats = F.normalize(neighbor_feats, p=2, dim=-1)
                sim = torch.bmm(norm_feats, norm_feats.transpose(1, 2)).clamp(min=0.0)
                adj = 0.5 * adj + 0.5 * sim

        if agent_mask is not None:
            agent_mask = agent_mask.to(device=adj.device, dtype=torch.bool)
            mask_2d = agent_mask.unsqueeze(1) & agent_mask.unsqueeze(2)
            adj = adj * mask_2d.to(adj.dtype)

        deg = adj.sum(dim=-1, keepdim=True)
        deg_safe = deg.clamp(min=1e-5)
        d_inv = torch.where(deg > 1e-5, 1.0 / deg_safe, torch.zeros_like(deg))
        norm_adj = adj * d_inv

        gcn_out = self.act(self.fc(torch.bmm(norm_adj, neighbor_feats)))
        ego_updated = gcn_out[:, 0:1, :]

        result = self.norm(target_feat + ego_updated.expand(-1, T, -1))
        if agent_mask is not None:
            has_agents = agent_mask.to(device=target_feat.device, dtype=torch.bool).any(dim=1)
            result = torch.where(has_agents.view(B, 1, 1), result, target_feat)
        return result

    def get_cost_breakdown(self, seq_len=20, d_model=128, num_agents=16, batch_size=1, **kwargs):
        scale = _batch_scale(batch_size)
        d_model = d_model if d_model is not None else self.d_model
        per_sample = 2 * num_agents * num_agents * d_model + 2 * num_agents * d_model * d_model
        flops = float(scale * per_sample)
        return {
            "operator": type(self).__name__,
            "trainable_params": sum(p.numel() for p in self.parameters() if p.requires_grad),
            "total_params": sum(p.numel() for p in self.parameters()),
            "batch_size": scale,
            "theoretical_macs": flops / 2.0,
            "theoretical_flops": flops,
            "counting_convention": "1 MAC = 2 FLOPs",
            "input_dimensions": {"seq_len": int(seq_len), "d_model": int(d_model), "num_agents": int(num_agents)},
            "non_mac_ops": ["pairwise_euclidean_dist", "exp_adjacency", "degree_norm", "ReLU", "LayerNorm"],
        }

    def get_flops(self, seq_len=20, d_model=128, num_agents=16, batch_size=1, **kwargs):
        return self.get_cost_breakdown(
            seq_len=seq_len, d_model=d_model, num_agents=num_agents, batch_size=batch_size
        )["theoretical_flops"]


class GATLayer(BaseOperator):
    """Graph Attention Network modeling dynamic directed attention from ego to neighbors."""

    def __init__(self, d_model: int, heads: int = 4):
        super().__init__(d_model)
        self.heads = heads
        self.attn = nn.MultiheadAttention(d_model, heads, batch_first=True)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, target_feat, neighbor_feats=None, agent_mask=None, *args, **kwargs):
        if neighbor_feats is None or neighbor_feats.shape[1] == 0:
            return target_feat

        valid_agents = _valid_non_ego_mask(agent_mask, neighbor_feats.shape[1], neighbor_feats.device)
        key_padding_mask = (~valid_agents) if valid_agents is not None else None
        if key_padding_mask is None and neighbor_feats.shape[1] > 0:
            key_padding_mask = torch.zeros(neighbor_feats.shape[:2], dtype=torch.bool, device=neighbor_feats.device)
            key_padding_mask[:, 0] = True

        attn_out = safe_cross_attention(self.attn, target_feat, neighbor_feats, neighbor_feats, key_padding_mask)
        result = self.norm(target_feat + attn_out)
        has_social = (~key_padding_mask).any(dim=1)
        return torch.where(has_social.view(-1, 1, 1), result, target_feat)

    def get_cost_breakdown(self, seq_len=20, d_model=128, num_agents=16, batch_size=1, **kwargs):
        scale = _batch_scale(batch_size)
        d_model = d_model if d_model is not None else self.d_model
        per_sample = 4 * (seq_len + num_agents) * d_model * d_model + 4 * seq_len * num_agents * d_model
        flops = float(scale * per_sample)
        return {
            "operator": type(self).__name__,
            "trainable_params": sum(p.numel() for p in self.parameters() if p.requires_grad),
            "total_params": sum(p.numel() for p in self.parameters()),
            "batch_size": scale,
            "theoretical_macs": flops / 2.0,
            "theoretical_flops": flops,
            "counting_convention": "1 MAC = 2 FLOPs",
            "input_dimensions": {
                "seq_len": int(seq_len),
                "d_model": int(d_model),
                "num_agents": int(num_agents),
                "heads": int(self.heads),
            },
            "non_mac_ops": ["attention_softmax", "key_padding_masking", "LayerNorm", "residual_add"],
        }

    def get_flops(self, seq_len=20, d_model=128, num_agents=16, batch_size=1, **kwargs):
        return self.get_cost_breakdown(
            seq_len=seq_len, d_model=d_model, num_agents=num_agents, batch_size=batch_size
        )["theoretical_flops"]


class SocialPooling(BaseOperator):
    """Social Pooling aggregating neighbor agents' spatial context across surrounding vehicles."""

    def __init__(self, d_model: int):
        super().__init__(d_model)
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, target_feat, neighbor_feats=None, agent_mask=None, *args, **kwargs):
        if neighbor_feats is None:
            return target_feat

        B, T, D = target_feat.shape
        N = neighbor_feats.shape[1]
        if N == 0:
            return target_feat

        valid_agents = _valid_non_ego_mask(agent_mask, N, neighbor_feats.device)
        if valid_agents is None:
            valid_agents = torch.ones((B, N), dtype=torch.bool, device=neighbor_feats.device)
            valid_agents[:, 0] = False

        mask = valid_agents.unsqueeze(-1).to(dtype=neighbor_feats.dtype)
        denom = mask.sum(dim=1, keepdim=True).clamp(min=1.0)
        pooled = (neighbor_feats * mask).sum(dim=1, keepdim=True) / denom
        has_neighbors = valid_agents.any(dim=1).to(dtype=pooled.dtype).view(B, 1, 1)
        pooled = pooled * has_neighbors

        social_context = self.fc(pooled) * has_neighbors
        result = self.norm(target_feat + social_context.expand(-1, T, -1))
        return torch.where(has_neighbors.bool(), result, target_feat)

    def get_cost_breakdown(self, seq_len=20, d_model=128, num_agents=16, batch_size=1, **kwargs):
        scale = _batch_scale(batch_size)
        d_model = d_model if d_model is not None else self.d_model
        per_sample = num_agents * d_model + 4 * d_model * d_model
        flops = float(scale * per_sample)
        return {
            "operator": type(self).__name__,
            "trainable_params": sum(p.numel() for p in self.parameters() if p.requires_grad),
            "total_params": sum(p.numel() for p in self.parameters()),
            "batch_size": scale,
            "theoretical_macs": float(scale * 2 * d_model * d_model),
            "theoretical_flops": flops,
            "counting_convention": "1 MAC = 2 FLOPs (plus num_agents*d_model pooling sum FLOPs)",
            "input_dimensions": {"seq_len": int(seq_len), "d_model": int(d_model), "num_agents": int(num_agents)},
            "non_mac_ops": ["masked_mean_pooling", "ReLU", "LayerNorm", "residual_add"],
        }

    def get_flops(self, seq_len=20, d_model=128, num_agents=16, batch_size=1, **kwargs):
        return self.get_cost_breakdown(
            seq_len=seq_len, d_model=d_model, num_agents=num_agents, batch_size=batch_size
        )["theoretical_flops"]


class PairwiseAttention(BaseOperator):
    """Pairwise Spatial-Gated Attention estimating influence between ego vehicle and each neighbor."""

    def __init__(self, d_model: int):
        super().__init__(d_model)
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dist_scale = nn.Parameter(torch.tensor(0.1))
        self.norm = nn.LayerNorm(d_model)

    def forward(self, target_feat, neighbor_feats=None, agent_mask=None, agent_pos=None, *args, **kwargs):
        if neighbor_feats is None:
            return target_feat

        B, T, D = target_feat.shape
        N = neighbor_feats.shape[1]
        if N == 0:
            return target_feat

        Q = self.q_proj(target_feat)
        K = self.k_proj(neighbor_feats)
        V = self.v_proj(neighbor_feats)

        scores = torch.bmm(Q, K.transpose(1, 2)) / math.sqrt(D)

        if agent_pos is not None:
            ego_pos = agent_pos[:, 0:1, :]
            dist_to_ego = torch.sqrt(torch.sum((agent_pos - ego_pos) ** 2, dim=-1, keepdim=True) + 1e-8)
            spatial_bias = -torch.abs(self.dist_scale) * dist_to_ego.transpose(1, 2)
            scores = scores + spatial_bias

        valid_agents = _valid_non_ego_mask(agent_mask, N, neighbor_feats.device)
        if valid_agents is None:
            valid_agents = torch.ones((B, N), dtype=torch.bool, device=neighbor_feats.device)
            valid_agents[:, 0] = False

        all_masked = ~valid_agents.any(dim=-1, keepdim=True)
        safe_mask = valid_agents.clone()
        if all_masked.any():
            safe_mask[all_masked.squeeze(-1), 0] = True

        scores = scores.masked_fill(~safe_mask.unsqueeze(1), -1e4)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = attn_weights * (~all_masked).to(attn_weights.dtype).unsqueeze(-1)

        context = torch.bmm(attn_weights, V)
        out = self.out_proj(context)
        out = out * (~all_masked).to(out.dtype).view(B, 1, 1)
        result = self.norm(target_feat + out)
        return torch.where((~all_masked).view(B, 1, 1), result, target_feat)

    def get_cost_breakdown(self, seq_len=20, d_model=128, num_agents=16, batch_size=1, **kwargs):
        scale = _batch_scale(batch_size)
        d_model = d_model if d_model is not None else self.d_model
        q_proj = 2 * seq_len * d_model * d_model
        kv_proj = 4 * num_agents * d_model * d_model
        attn_scores = 2 * seq_len * num_agents * d_model
        attn_values = 2 * seq_len * num_agents * d_model
        out_proj = 2 * seq_len * d_model * d_model
        flops = float(scale * (q_proj + kv_proj + attn_scores + attn_values + out_proj))
        return {
            "operator": type(self).__name__,
            "trainable_params": sum(p.numel() for p in self.parameters() if p.requires_grad),
            "total_params": sum(p.numel() for p in self.parameters()),
            "batch_size": scale,
            "theoretical_macs": flops / 2.0,
            "theoretical_flops": flops,
            "counting_convention": "1 MAC = 2 FLOPs",
            "input_dimensions": {"seq_len": int(seq_len), "d_model": int(d_model), "num_agents": int(num_agents)},
            "non_mac_ops": ["pairwise_dist_to_ego", "spatial_bias_add", "softmax", "LayerNorm"],
        }

    def get_flops(self, seq_len=20, d_model=128, num_agents=16, batch_size=1, **kwargs):
        return self.get_cost_breakdown(
            seq_len=seq_len, d_model=d_model, num_agents=num_agents, batch_size=batch_size
        )["theoretical_flops"]


# =========================================================================
# 3. Vectorized HD Map Operators (Map & Polyline Topology Domain)
# =========================================================================

class LaneAttention(BaseOperator):
    """Cross-attention mechanism between vehicle trajectory state and HD Map lane vectors."""

    def __init__(self, d_model: int, heads: int = 4):
        super().__init__(d_model)
        self.heads = heads
        self.attn = nn.MultiheadAttention(d_model, heads, batch_first=True)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, target_feat, lane_feats=None, lane_mask=None, *args, **kwargs):
        if lane_feats is None or lane_feats.shape[1] == 0:
            return target_feat

        key_padding_mask = (~lane_mask) if lane_mask is not None else None
        attn_out = safe_cross_attention(self.attn, target_feat, lane_feats, lane_feats, key_padding_mask)
        result = self.norm(target_feat + attn_out)
        if lane_mask is not None:
            has_lanes = lane_mask.to(device=target_feat.device, dtype=torch.bool).any(dim=1)
            result = torch.where(has_lanes.view(-1, 1, 1), result, target_feat)
        return result

    def get_cost_breakdown(self, seq_len=None, d_model=None, num_lanes=None, batch_size=1, **kwargs):
        from configs.model_config import ModelConfig
        cfg = ModelConfig()
        scale = _batch_scale(batch_size)
        seq_len = seq_len if seq_len is not None else cfg.past_len
        d_model = d_model if d_model is not None else self.d_model
        num_lanes = num_lanes if num_lanes is not None else cfg.max_lanes
        q_proj = 2 * seq_len * d_model * d_model
        kv_proj = 4 * num_lanes * d_model * d_model
        attn_matrix = 2 * seq_len * num_lanes * d_model
        attn_value = 2 * seq_len * num_lanes * d_model
        out_proj = 2 * seq_len * d_model * d_model
        flops = float(scale * (q_proj + kv_proj + attn_matrix + attn_value + out_proj))
        return {
            "operator": type(self).__name__,
            "trainable_params": sum(p.numel() for p in self.parameters() if p.requires_grad),
            "total_params": sum(p.numel() for p in self.parameters()),
            "batch_size": scale,
            "theoretical_macs": flops / 2.0,
            "theoretical_flops": flops,
            "counting_convention": "1 MAC = 2 FLOPs",
            "input_dimensions": {
                "seq_len": int(seq_len),
                "d_model": int(d_model),
                "num_lanes": int(num_lanes),
                "heads": int(self.heads),
            },
            "non_mac_ops": ["attention_softmax", "key_padding_masking", "LayerNorm", "residual_add"],
        }

    def get_flops(self, seq_len=None, d_model=None, num_lanes=None, batch_size=1, **kwargs):
        return self.get_cost_breakdown(
            seq_len=seq_len, d_model=d_model, num_lanes=num_lanes, batch_size=batch_size
        )["theoretical_flops"]


class PolylineEncoder(BaseOperator):
    """PointNet / VectorNet-style Polyline Subgraph Encoder for HD Map Centerlines."""

    def __init__(self, in_channels: int = 4, d_model: int = 128):
        super().__init__(d_model)
        self.in_channels = in_channels
        self.point_mlp = nn.Sequential(
            nn.Linear(in_channels, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU()
        )
        self.lane_type_emb = nn.Embedding(5, d_model)
        self.lane_proj = nn.Linear(d_model, d_model)

    def forward(self, map_lanes, lane_types=None, mask=None, **kwargs):
        polylines = map_lanes
        if polylines.dim() == 4:
            B, N, P, C = polylines.shape
            if C == 2 and self.in_channels == 4:
                tangents = torch.zeros_like(polylines)
                if P > 1:
                    tangents[:, :, 1:, :] = polylines[:, :, 1:, :] - polylines[:, :, :-1, :]
                    if mask is not None and mask.dim() == 3:
                        valid_segments = mask[:, :, 1:].bool() & mask[:, :, :-1].bool()
                        tangents[:, :, 1:, :] = tangents[:, :, 1:, :].masked_fill(~valid_segments.unsqueeze(-1), 0.0)
                    tangents[:, :, 0, :] = tangents[:, :, 1, :]
                polylines = torch.cat([polylines, tangents], dim=-1)
                C = polylines.shape[-1]

            pt_feat = self.point_mlp(polylines.reshape(B * N * P, C)).view(B, N, P, self.d_model)

            if mask is not None and mask.dim() == 3:
                point_valid = mask.to(device=pt_feat.device, dtype=torch.bool)
                lane_valid = point_valid.any(dim=2)
                lane_feat = pt_feat.masked_fill(~point_valid.unsqueeze(-1), float("-inf")).amax(dim=2)
                lane_feat = lane_feat.masked_fill(~lane_valid.unsqueeze(-1), 0.0)
            else:
                lane_feat = pt_feat.max(dim=2).values

            if lane_types is not None:
                lane_feat = lane_feat + self.lane_type_emb(lane_types.clamp(0, 4))

            encoded = self.lane_proj(lane_feat)
            if mask is not None:
                if mask.dim() == 2:
                    lane_valid = mask.to(device=encoded.device, dtype=torch.bool)
                encoded = encoded.masked_fill(~lane_valid.unsqueeze(-1), 0.0)
            return encoded

        elif polylines.dim() == 3:
            B, N, C = polylines.shape
            lane_feat = self.point_mlp(polylines.reshape(B * N, C)).view(B, N, self.d_model)
            if lane_types is not None:
                lane_feat = lane_feat + self.lane_type_emb(lane_types.clamp(0, 4))
            encoded = self.lane_proj(lane_feat)
            if mask is not None and mask.dim() == 2:
                lane_valid = mask.to(device=encoded.device, dtype=torch.bool)
                encoded = encoded.masked_fill(~lane_valid.unsqueeze(-1), 0.0)
            return encoded
        else:
            return self.lane_proj(self.point_mlp(polylines))

    def get_cost_breakdown(self, num_lanes=None, lane_pts=None, d_model=None, batch_size=1, **kwargs):
        from configs.model_config import ModelConfig
        cfg = ModelConfig()
        scale = _batch_scale(batch_size)
        num_lanes = num_lanes if num_lanes is not None else cfg.max_lanes
        lane_pts = lane_pts if lane_pts is not None else cfg.lane_pts
        d_model = d_model if d_model is not None else self.d_model
        per_sample = (
            num_lanes * lane_pts * (2 * self.in_channels * (d_model // 2) + 2 * (d_model // 2) * d_model)
            + num_lanes * 2 * d_model * d_model
        )
        flops = float(scale * per_sample)
        return {
            "operator": type(self).__name__,
            "trainable_params": sum(p.numel() for p in self.parameters() if p.requires_grad),
            "total_params": sum(p.numel() for p in self.parameters()),
            "batch_size": scale,
            "theoretical_macs": flops / 2.0,
            "theoretical_flops": flops,
            "counting_convention": "1 MAC = 2 FLOPs",
            "input_dimensions": {
                "num_lanes": int(num_lanes),
                "lane_pts": int(lane_pts),
                "in_channels": int(self.in_channels),
                "d_model": int(d_model),
            },
            "non_mac_ops": ["LayerNorm", "ReLU", "max_pooling", "embedding_lookup"],
        }

    def get_flops(self, num_lanes=None, lane_pts=None, d_model=None, batch_size=1, **kwargs):
        return self.get_cost_breakdown(
            num_lanes=num_lanes, lane_pts=lane_pts, d_model=d_model, batch_size=batch_size
        )["theoretical_flops"]


class LaneGraphEncoder(BaseOperator):
    """Encodes road graph lane connectivity using Graph Convolutions."""

    def __init__(self, d_model: int):
        super().__init__(d_model)
        self.gcn = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        self.cross_attn = nn.MultiheadAttention(d_model, num_heads=2, batch_first=True)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, target_feat, lane_feats=None, lane_adj=None, lane_mask=None, *args, **kwargs):
        if lane_feats is None:
            return target_feat

        B, N, D = lane_feats.shape
        if N == 0:
            return target_feat

        if lane_adj is None:
            norm_lanes = F.normalize(lane_feats, p=2, dim=-1)
            lane_adj = torch.bmm(norm_lanes, norm_lanes.transpose(1, 2)).clamp(min=0.0)
            lane_adj = lane_adj + torch.eye(N, device=lane_feats.device, dtype=lane_feats.dtype).unsqueeze(0)

        if lane_mask is not None:
            lane_mask = lane_mask.to(device=lane_adj.device, dtype=torch.bool)
            mask_2d = lane_mask.unsqueeze(1) & lane_mask.unsqueeze(2)
            lane_adj = lane_adj * mask_2d.to(lane_adj.dtype)

        deg = lane_adj.sum(dim=-1, keepdim=True)
        deg_safe = deg.clamp(min=1e-5)
        d_inv = torch.where(deg > 1e-5, 1.0 / deg_safe, torch.zeros_like(deg))
        norm_adj = lane_adj * d_inv
        graph_lanes = self.gcn(torch.bmm(norm_adj, lane_feats))

        key_padding_mask = (~lane_mask) if lane_mask is not None else None
        attn_out = safe_cross_attention(self.cross_attn, target_feat, graph_lanes, graph_lanes, key_padding_mask)
        result = self.norm(target_feat + attn_out)
        if lane_mask is not None:
            has_lanes = lane_mask.to(device=target_feat.device, dtype=torch.bool).any(dim=1)
            result = torch.where(has_lanes.view(-1, 1, 1), result, target_feat)
        return result

    def get_cost_breakdown(self, seq_len=None, d_model=None, num_lanes=None, batch_size=1, **kwargs):
        from configs.model_config import ModelConfig
        cfg = ModelConfig()
        scale = _batch_scale(batch_size)
        seq_len = seq_len if seq_len is not None else cfg.past_len
        d_model = d_model if d_model is not None else self.d_model
        num_lanes = num_lanes if num_lanes is not None else cfg.max_lanes
        gcn_flops = 2 * num_lanes * num_lanes * d_model + 4 * num_lanes * d_model * d_model
        attn_flops = 4 * (seq_len + num_lanes) * d_model * d_model + 4 * seq_len * num_lanes * d_model
        flops = float(scale * (gcn_flops + attn_flops))
        return {
            "operator": type(self).__name__,
            "trainable_params": sum(p.numel() for p in self.parameters() if p.requires_grad),
            "total_params": sum(p.numel() for p in self.parameters()),
            "batch_size": scale,
            "theoretical_macs": flops / 2.0,
            "theoretical_flops": flops,
            "counting_convention": "1 MAC = 2 FLOPs",
            "input_dimensions": {"seq_len": int(seq_len), "d_model": int(d_model), "num_lanes": int(num_lanes)},
            "non_mac_ops": ["l2_normalize", "degree_norm", "ReLU", "attention_softmax", "LayerNorm"],
        }

    def get_flops(self, seq_len=None, d_model=None, num_lanes=None, batch_size=1, **kwargs):
        return self.get_cost_breakdown(
            seq_len=seq_len, d_model=d_model, num_lanes=num_lanes, batch_size=batch_size
        )["theoretical_flops"]


class MapFeatureFusion(BaseOperator):
    """Direct spatial pooling and non-linear MLP projection of lane geometry with agent state."""

    def __init__(self, d_model: int):
        super().__init__(d_model)
        self.fc = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, target_feat, lane_feats=None, lane_mask=None, *args, **kwargs):
        if lane_feats is None or lane_feats.shape[1] == 0:
            return target_feat

        B, T, D = target_feat.shape
        if lane_mask is not None:
            mask = lane_mask.unsqueeze(-1).to(device=lane_feats.device, dtype=lane_feats.dtype)
            denom = mask.sum(dim=1, keepdim=True).clamp(min=1.0)
            pooled_lane = (lane_feats * mask).sum(dim=1, keepdim=True) / denom
            has_lanes = lane_mask.any(dim=1).to(lane_feats.dtype).view(B, 1, 1)
            pooled_lane = pooled_lane * has_lanes
        else:
            pooled_lane = lane_feats.mean(dim=1, keepdim=True)

        expanded_lane = pooled_lane.expand(-1, T, -1)
        fused = self.fc(torch.cat([target_feat, expanded_lane], dim=-1))
        result = self.norm(target_feat + fused)
        if lane_mask is not None:
            result = torch.where(has_lanes.bool(), result, target_feat)
        return result

    def get_cost_breakdown(self, seq_len=None, d_model=None, num_lanes=None, batch_size=1, **kwargs):
        from configs.model_config import ModelConfig
        cfg = ModelConfig()
        scale = _batch_scale(batch_size)
        seq_len = seq_len if seq_len is not None else cfg.past_len
        d_model = d_model if d_model is not None else self.d_model
        num_lanes = num_lanes if num_lanes is not None else cfg.max_lanes
        per_sample = num_lanes * d_model + 2 * seq_len * (2 * d_model * d_model + d_model * d_model)
        flops = float(scale * per_sample)
        return {
            "operator": type(self).__name__,
            "trainable_params": sum(p.numel() for p in self.parameters() if p.requires_grad),
            "total_params": sum(p.numel() for p in self.parameters()),
            "batch_size": scale,
            "theoretical_macs": float(scale * seq_len * 3 * d_model * d_model),
            "theoretical_flops": flops,
            "counting_convention": "1 MAC = 2 FLOPs (plus num_lanes*d_model pooling sum FLOPs)",
            "input_dimensions": {"seq_len": int(seq_len), "d_model": int(d_model), "num_lanes": int(num_lanes)},
            "non_mac_ops": ["masked_mean_pooling", "LayerNorm", "ReLU", "residual_add"],
        }

    def get_flops(self, seq_len=None, d_model=None, num_lanes=None, batch_size=1, **kwargs):
        return self.get_cost_breakdown(
            seq_len=seq_len, d_model=d_model, num_lanes=num_lanes, batch_size=batch_size
        )["theoretical_flops"]


class IdentityOperator(BaseOperator):
    """Pass-through identity operator (zero parameters, zero extra FLOPs)."""

    def __init__(self, d_model: int = 128):
        super().__init__(d_model)

    def forward(self, x, *args, **kwargs):
        return x

    def get_cost_breakdown(self, batch_size=1, **kwargs):
        scale = _batch_scale(batch_size)
        return {
            "operator": type(self).__name__,
            "trainable_params": 0,
            "total_params": 0,
            "batch_size": scale,
            "theoretical_macs": 0.0,
            "theoretical_flops": 0.0,
            "counting_convention": "1 MAC = 2 FLOPs",
            "input_dimensions": {},
            "non_mac_ops": [],
        }

    def get_flops(self, batch_size=1, *args, **kwargs):
        _batch_scale(batch_size)
        return 0.0


# =========================================================================
# Operator Registries and Dynamic Builders
# =========================================================================

TEMPORAL_OPERATOR_REGISTRY: dict[str, type[BaseOperator]] = {
    "GRUEncoder": GRUEncoder,
    "LSTMEncoder": LSTMEncoder,
    "TemporalConvEncoder": TemporalConvEncoder,
    "TemporalSelfAttention": TemporalSelfAttention,
    "IdentityOperator": IdentityOperator,
    "Identity": IdentityOperator,
    "GRU": GRUEncoder,
    "LSTM": LSTMEncoder,
    "TemporalConv": TemporalConvEncoder,
}

INTERACTION_OPERATOR_REGISTRY: dict[str, type[BaseOperator]] = {
    "GCNLayer": GCNLayer,
    "GATLayer": GATLayer,
    "SocialPooling": SocialPooling,
    "PairwiseAttention": PairwiseAttention,
    "IdentityOperator": IdentityOperator,
    "Identity": IdentityOperator,
    "GCN": GCNLayer,
    "GAT": GATLayer,
}

MAP_OPERATOR_REGISTRY: dict[str, type[BaseOperator]] = {
    "LaneAttention": LaneAttention,
    "LaneGraphEncoder": LaneGraphEncoder,
    "MapFeatureFusion": MapFeatureFusion,
    "IdentityOperator": IdentityOperator,
    "Identity": IdentityOperator,
}


def build_temporal_operator(name: str, d_model: int) -> BaseOperator:
    cls = TEMPORAL_OPERATOR_REGISTRY.get(name)
    if cls is None:
        for k, v in TEMPORAL_OPERATOR_REGISTRY.items():
            if k.lower() == name.lower():
                cls = v
                break
    if cls is None:
        raise ValueError(f"Unknown temporal operator: '{name}'. Supported: {list(TEMPORAL_OPERATOR_REGISTRY.keys())}")
    return cls(d_model)


def build_interaction_operator(name: str, d_model: int) -> BaseOperator:
    cls = INTERACTION_OPERATOR_REGISTRY.get(name)
    if cls is None:
        for k, v in INTERACTION_OPERATOR_REGISTRY.items():
            if k.lower() == name.lower():
                cls = v
                break
    if cls is None:
        raise ValueError(f"Unknown interaction operator: '{name}'. Supported: {list(INTERACTION_OPERATOR_REGISTRY.keys())}")
    return cls(d_model)


def build_map_operator(name: str, d_model: int) -> BaseOperator:
    cls = MAP_OPERATOR_REGISTRY.get(name)
    if cls is None:
        for k, v in MAP_OPERATOR_REGISTRY.items():
            if k.lower() == name.lower():
                cls = v
                break
    if cls is None:
        raise ValueError(f"Unknown map operator: '{name}'. Supported: {list(MAP_OPERATOR_REGISTRY.keys())}")
    return cls(d_model)


# =========================================================================
# 4. Continuous Mixed Operations for DARTS
# =========================================================================

class MapMixedOp(nn.Module):
    """Continuous relaxation of HD Map / Road topology branch operators dynamically built from configuration."""

    def __init__(self, d_model: int, op_names: list[str] | None = None):
        super().__init__()
        self.d_model = d_model
        if op_names is None:
            from configs.search_config import DEFAULT_MAP_OPS
            op_names = DEFAULT_MAP_OPS
        self.op_names = list(op_names)
        self.ops = nn.ModuleList([build_map_operator(name, d_model) for name in self.op_names])

    def forward(self, target_feat, lane_feats, alpha_map, lane_adj=None, lane_mask=None):
        return sum(
            w * op(target_feat, lane_feats=lane_feats, lane_adj=lane_adj, lane_mask=lane_mask)
            for w, op in zip(alpha_map, self.ops)
        )

    def get_ops_cost_breakdowns(self, seq_len: int | None = None, num_lanes: int | None = None, batch_size: int = 1):
        from configs.model_config import ModelConfig
        cfg = ModelConfig()
        if seq_len is None:
            seq_len = cfg.past_len
        if num_lanes is None:
            num_lanes = cfg.max_lanes
        return [op.get_cost_breakdown(seq_len=seq_len, d_model=self.d_model, num_lanes=num_lanes, batch_size=batch_size) for op in self.ops]

    def get_ops_flops(self, seq_len: int | None = None, num_lanes: int | None = None, batch_size: int = 1):
        return [b["theoretical_flops"] for b in self.get_ops_cost_breakdowns(seq_len=seq_len, num_lanes=num_lanes, batch_size=batch_size)]


class TemporalMixedOp(nn.Module):
    """Continuous relaxation of Temporal branch operators dynamically built from configuration."""

    def __init__(self, d_model: int, op_names: list[str] | None = None):
        super().__init__()
        self.d_model = d_model
        if op_names is None:
            from configs.search_config import DEFAULT_TEMPORAL_OPS
            op_names = DEFAULT_TEMPORAL_OPS
        self.op_names = list(op_names)
        self.ops = nn.ModuleList([build_temporal_operator(name, d_model) for name in self.op_names])

    def forward(self, x, alpha_temp):
        return sum(w * op(x) for w, op in zip(alpha_temp, self.ops))

    def get_ops_cost_breakdowns(self, seq_len: int | None = None, batch_size: int = 1):
        if seq_len is None:
            from configs.model_config import ModelConfig
            seq_len = ModelConfig().past_len
        return [op.get_cost_breakdown(seq_len=seq_len, d_model=self.d_model, batch_size=batch_size) for op in self.ops]

    def get_ops_flops(self, seq_len: int | None = None, batch_size: int = 1):
        return [b["theoretical_flops"] for b in self.get_ops_cost_breakdowns(seq_len=seq_len, batch_size=batch_size)]


class InteractionMixedOp(nn.Module):
    """Continuous relaxation of Multi-agent Interaction branch operators dynamically built from configuration."""

    def __init__(self, d_model: int, op_names: list[str] | None = None):
        super().__init__()
        self.d_model = d_model
        if op_names is None:
            from configs.search_config import DEFAULT_INTERACTION_OPS
            op_names = DEFAULT_INTERACTION_OPS
        self.op_names = list(op_names)
        self.ops = nn.ModuleList([build_interaction_operator(name, d_model) for name in self.op_names])

    def forward(self, target_feat, neighbor_feats, alpha_inter, agent_mask=None, agent_pos=None):
        return sum(
            w * op(target_feat, neighbor_feats=neighbor_feats, agent_mask=agent_mask, agent_pos=agent_pos)
            for w, op in zip(alpha_inter, self.ops)
        )

    def get_ops_cost_breakdowns(self, seq_len: int | None = None, num_agents: int | None = None, batch_size: int = 1):
        from configs.model_config import ModelConfig
        cfg = ModelConfig()
        if seq_len is None:
            seq_len = cfg.past_len
        if num_agents is None:
            num_agents = cfg.max_agents + cfg.max_obstacles
        return [op.get_cost_breakdown(seq_len=seq_len, d_model=self.d_model, num_agents=num_agents, batch_size=batch_size) for op in self.ops]

    def get_ops_flops(self, seq_len: int | None = None, num_agents: int | None = None, batch_size: int = 1):
        return [b["theoretical_flops"] for b in self.get_ops_cost_breakdowns(seq_len=seq_len, num_agents=num_agents, batch_size=batch_size)]


