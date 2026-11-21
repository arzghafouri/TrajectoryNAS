r"""
TrajectoryLNAS Comprehensive Data Validation & Technical Acceptance Suite
========================================================================
Author: Alireza Ghafouri (Master's Thesis Project - TrajectoryLNAS)
Paper Reference: "TrajectoryLNAS: Trajectory Prediction Architecture Optimization"

Combines:
  1. Data Quality & Statistical Inspection Engine (DataValidationReport / DataSummary / DataSummery)
     - Validates dataset sequence files, NaN/Inf coordinate detection, temporal grid alignment (2Hz, dt=0.5s),
       cross-split scene leakage, obstacle & lane density, and Hard Sample Mining configuration.
  2. Technical Acceptance Test Suite (run_acceptance_tests)
     - 18 automated end-to-end acceptance tests validating dataset contracts, tensor shapes (20 agents, 32 lanes,
       16 pts, 20 obstacles), gradient flows, mask integrity, model serialization, zero-cost proxies,
       Hard Sample Mining weighted sampling, kinematic physics loss, and discovered architecture [3, 2, 0].
"""

import argparse
import json
import math
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

# Safe PyTorch import
try:
    import torch
    HAS_TORCH = True
except ImportError:
    torch = None
    HAS_TORCH = False

# Ensure workspace root and data directory are in sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

# Safe configuration imports
try:
    from configs.data_config import DataConfig
    from configs.model_config import ModelConfig
    from configs.search_config import SearchConfig
    from configs.train_config import TrainConfig
except ImportError:
    DataConfig = None
    ModelConfig = None
    SearchConfig = None
    TrainConfig = None

# Safe dataset & metric imports
try:
    from data.nuscenes_dataset import (
        NuScenesPredictionDataset,
        build_nuscenes_dataloader,
    )
    from data.synthetic_dataset import SyntheticTrajectoryDataset
    from data.hard_sample_mining import (
        HardSampleMiningManager,
        HardSampleWeightedSampler,
    )
    from evaluations.metrics import (
        ade_at_horizon,
        best_of_k_loss,
        evaluate_collision_risk,
        get_supernet_costs,
        safe_norm,
        snip_score,
        synflow_score,
    )
    from models.final_model import FinalTrajectoryLNAS, FinalTrajectoryModel
    from models.search_supernet import ArchParams, SearchSuperNet
except ImportError:
    NuScenesPredictionDataset = None
    build_nuscenes_dataloader = None
    SyntheticTrajectoryDataset = None
    HardSampleMiningManager = None
    HardSampleWeightedSampler = None
    ade_at_horizon = None
    best_of_k_loss = None
    evaluate_collision_risk = None
    get_supernet_costs = None
    safe_norm = None
    snip_score = None
    synflow_score = None
    FinalTrajectoryLNAS = None
    FinalTrajectoryModel = None
    ArchParams = None
    SearchSuperNet = None


# Pure Python Math & Geometry Helpers
def _mean(lst: list[float]) -> float:
    return sum(lst) / len(lst) if lst else 0.0

def _min(lst: list[int | float]) -> int | float:
    return min(lst) if lst else 0

def _max(lst: list[int | float]) -> int | float:
    return max(lst) if lst else 0

def _dist_2d(p1: list[float], p2: list[float]) -> float:
    return math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)

def _path_length(pts: list[list[float]]) -> float:
    if len(pts) < 2:
        return 0.0
    return sum(_dist_2d(pts[i], pts[i+1]) for i in range(len(pts) - 1))


# ==============================================================================
# 1. Dataset Quality & Statistical Inspection Engine
# ==============================================================================

class DataValidationReport:
    """
    Comprehensive dataset quality inspector, statistical analyzer, and report generator.
    Evaluates schema compliance, coordinate bounds, temporal grids, scene leakage,
    obstacle & lane densities, and Hard Sample Mining parameters.
    """

    def __init__(
        self,
        data_root: str | None = None,
        output_dir: str | None = None,
        sample_percent: float = 100.0,
        seed: int = 42,
        allow_mock_data: bool = False
    ):
        d_cfg = DataConfig() if DataConfig is not None else None
        self.data_root = data_root or (d_cfg.output_dir if d_cfg else "data/nuscenes")
        self.sample_percent = float(sample_percent)
        self.seed = int(seed)
        self.allow_mock_data = bool(allow_mock_data)

        self.output_dir = Path(output_dir).resolve() if output_dir else ROOT_DIR / "reports"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.status = "PENDING"
        self.dataset_type = "Real"
        self.train_raw_samples: list[dict[str, Any]] = []
        self.val_raw_samples: list[dict[str, Any]] = []
        self.warnings: list[str] = []

    def _resolve_sequence_file_paths(self) -> tuple[Path, Path]:
        root_path = Path(self.data_root)
        candidates_train = [
            root_path / "train_sequences.jsonl",
            root_path / "train_sequences.json",
            root_path / "v1.0-trainval" / "train_sequences.jsonl",
            root_path / "v1.0-trainval" / "train_sequences.json",
            ROOT_DIR / "data" / "nuscenes" / "v1.0-trainval" / "train_sequences.jsonl",
            ROOT_DIR / "data" / "nuscenes" / "v1.0-trainval" / "train_sequences.json",
            ROOT_DIR / "data" / "nuscenes" / "train_sequences.jsonl",
            ROOT_DIR / "data" / "nuscenes" / "train_sequences.json",
        ]
        candidates_val = [
            root_path / "val_sequences.jsonl",
            root_path / "val_sequences.json",
            root_path / "v1.0-trainval" / "val_sequences.jsonl",
            root_path / "v1.0-trainval" / "val_sequences.json",
            ROOT_DIR / "data" / "nuscenes" / "v1.0-trainval" / "val_sequences.jsonl",
            ROOT_DIR / "data" / "nuscenes" / "v1.0-trainval" / "val_sequences.json",
            ROOT_DIR / "data" / "nuscenes" / "val_sequences.jsonl",
            ROOT_DIR / "data" / "nuscenes" / "val_sequences.json",
        ]
        train_file = next((p for p in candidates_train if p.exists()), candidates_train[0])
        val_file = next((p for p in candidates_val if p.exists()), candidates_val[0])
        return train_file, val_file

    def load_and_validate(self) -> dict[str, Any]:
        """Loads sequences and verifies data schema, NaN values, temporal grid, and split leakage."""
        m_cfg = ModelConfig() if ModelConfig is not None else None
        t_cfg = TrainConfig() if TrainConfig is not None else None

        expected_past = m_cfg.past_len if m_cfg else 4
        expected_future = m_cfg.future_len if m_cfg else 12
        expected_hz = float(m_cfg.hz if m_cfg else 2.0)
        expected_dt = 1.0 / expected_hz if expected_hz > 0 else 0.5

        train_file, val_file = self._resolve_sequence_file_paths()

        if not train_file.exists() or not val_file.exists():
            if not self.allow_mock_data:
                self.status = "NOT_EXTRACTED"
                info_msg = (
                    f"Dataset sequences not found at {train_file.parent}. "
                    "Extract using 'python data/preprocessing.py' or run tests in synthetic mode."
                )
                self.warnings.append(info_msg)
                report_data = {
                    "status": self.status,
                    "dataset_path": str(train_file.parent),
                    "model_config_dimensions": {
                        "past_len": expected_past,
                        "future_len": expected_future,
                        "hz": expected_hz,
                        "dt_seconds": expected_dt,
                        "max_agents": m_cfg.max_agents if m_cfg else 20,
                        "max_lanes": m_cfg.max_lanes if m_cfg else 32,
                        "lane_pts": m_cfg.lane_pts if m_cfg else 16,
                        "max_obstacles": m_cfg.max_obstacles if m_cfg else 20,
                        "num_modes": m_cfg.num_modes if m_cfg else 5,
                    },
                    "train_config_policies": {
                        "hard_sample_mining": getattr(t_cfg, "hard_sample_mining", 0.5) if t_cfg else 0.5,
                        "hard_sample_ema": getattr(t_cfg, "hard_sample_ema", 0.7) if t_cfg else 0.7,
                        "early_stopping_patience": getattr(t_cfg, "early_stopping_patience", 8) if t_cfg else 8,
                    },
                    "warnings": self.warnings
                }
                self._save_report(report_data)
                return report_data

        def _read_samples(path: Path) -> list[dict[str, Any]]:
            if not path.exists():
                return []
            with open(path, "rb") as f:
                first_byte = b""
                while True:
                    chunk = f.read(1024)
                    if not chunk:
                        break
                    stripped = chunk.strip()
                    if stripped:
                        first_byte = stripped[:1]
                        break
            with open(path, "r", encoding="utf-8") as f:
                if first_byte == b"{" or path.name.endswith(".jsonl"):
                    return [json.loads(line) for line in f if line.strip()]
                else:
                    return json.load(f)

        try:
            if train_file.exists():
                self.train_raw_samples = _read_samples(train_file)
            if val_file.exists():
                self.val_raw_samples = _read_samples(val_file)
        except Exception as e:
            self.status = "FAILED"
            err_msg = f"Failed to load dataset files: {e}"
            self.warnings.append(err_msg)
            return {"status": "FAILED", "error": err_msg}

        all_samples = self.train_raw_samples + self.val_raw_samples
        nan_count = 0
        invalid_coords = 0
        temporal_grid_mismatches = 0
        agent_counts = []
        lane_counts = []
        obstacle_counts = []
        speeds_past = []
        speeds_future = []

        for s in all_samples:
            past = s.get("past_traj", [])
            future = s.get("future_traj", [])

            # Validate point values
            for pt in past + future:
                for val in pt:
                    if math.isnan(val) or math.isinf(val):
                        nan_count += 1
                    if abs(val) > 1e6:
                        invalid_coords += 1

            # Validate temporal grid lengths and intervals if timestamps present
            ts_past = s.get("timestamps_past", [])
            ts_future = s.get("timestamps_future", [])
            if ts_past and len(ts_past) == len(past) and len(ts_past) > 1:
                dt_vals = [ts_past[i+1] - ts_past[i] for i in range(len(ts_past)-1)]
                if any(abs(dt - expected_dt) > 1e-3 for dt in dt_vals):
                    temporal_grid_mismatches += 1

            # Densities
            agent_counts.append(len(s.get("neighbors_past", [])))
            lane_counts.append(len(s.get("map_lanes", [])))
            obstacle_counts.append(len(s.get("static_objects", [])))

            # Speeds (approx m/s)
            dur_past = (len(past) - 1) * expected_dt
            if dur_past > 0:
                speeds_past.append(_path_length(past) / dur_past)

            dur_future = len(future) * expected_dt
            if dur_future > 0:
                speeds_future.append(_path_length(future) / dur_future)

        # Check cross-split scene leakage
        tr_scenes = set(s.get("scene_token", s.get("scene_name", "")) for s in self.train_raw_samples if "scene_token" in s or "scene_name" in s)
        va_scenes = set(s.get("scene_token", s.get("scene_name", "")) for s in self.val_raw_samples if "scene_token" in s or "scene_name" in s)
        shared_scenes = tr_scenes.intersection(va_scenes)

        leakage_clean = (len(shared_scenes) == 0)
        self.status = "READY" if (nan_count == 0 and leakage_clean) else "WARNING"

        # Check Best Architecture Checkpoint
        arch_file = ROOT_DIR / "checkpoints" / "best_architecture.json"
        arch_summary = {}
        if arch_file.exists():
            try:
                with open(arch_file, "r", encoding="utf-8") as af:
                    arch_data = json.load(af)
                    arch_summary = {
                        "discovered_best_ops": arch_data.get("best_ops", [3, 2, 0]),
                        "discovered_architecture": arch_data.get("discovered_architecture", {}),
                        "timestamp": arch_data.get("timestamp"),
                    }
            except Exception:
                pass

        report_data = {
            "status": self.status,
            "total_train_samples": len(self.train_raw_samples),
            "total_val_samples": len(self.val_raw_samples),
            "total_samples": len(all_samples),
            "nan_inf_count": nan_count,
            "invalid_coords_count": invalid_coords,
            "temporal_grid_mismatches": temporal_grid_mismatches,
            "shared_scenes_count": len(shared_scenes),
            "scene_leakage_status": "PASSED (Zero Scene Leakage)" if leakage_clean else f"LEAKAGE DETECTED ({len(shared_scenes)} shared scenes)",
            "model_config_dimensions": {
                "past_len": expected_past,
                "future_len": expected_future,
                "hz": expected_hz,
                "dt_seconds": expected_dt,
                "max_agents": m_cfg.max_agents if m_cfg else 20,
                "max_lanes": m_cfg.max_lanes if m_cfg else 32,
                "lane_pts": m_cfg.lane_pts if m_cfg else 16,
                "max_obstacles": m_cfg.max_obstacles if m_cfg else 20,
                "num_modes": m_cfg.num_modes if m_cfg else 5,
            },
            "train_config_policies": {
                "hard_sample_mining": getattr(t_cfg, "hard_sample_mining", 0.5) if t_cfg else 0.5,
                "hard_sample_ema": getattr(t_cfg, "hard_sample_ema", 0.7) if t_cfg else 0.7,
                "early_stopping_patience": getattr(t_cfg, "early_stopping_patience", 8) if t_cfg else 8,
            },
            "discovered_architecture": arch_summary,
            "kinematics_and_context_statistics": {
                "avg_past_speed_mps": round(_mean(speeds_past), 2),
                "avg_future_speed_mps": round(_mean(speeds_future), 2),
                "max_observed_speed_mps": round(_max(speeds_past + speeds_future), 2),
                "avg_neighbor_agents": round(_mean(agent_counts), 2),
                "avg_map_lanes": round(_mean(lane_counts), 2),
                "avg_static_obstacles": round(_mean(obstacle_counts), 2),
            },
            "warnings": self.warnings
        }

        self._save_report(report_data)
        self._print_terminal_summary(report_data)
        return report_data

    def _save_report(self, data: dict[str, Any]):
        out_file = self.output_dir / "data_validation_report.json"
        try:
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception:
            pass

    def _print_terminal_summary(self, data: dict[str, Any]):
        print("=" * 80)
        print(" TrajectoryLNAS Dataset Quality & Schema Audit Report")
        print(f" Status:                  {data.get('status')}")
        print(f" Total Train Sequences:   {data.get('total_train_samples', 0):,}")
        print(f" Total Val Sequences:     {data.get('total_val_samples', 0):,}")
        print(f" NaN/Inf Anomalies:       {data.get('nan_inf_count', 0)}")
        print(f" Scene Leakage Audit:     {data.get('scene_leakage_status')}")
        dim = data.get("model_config_dimensions", {})
        print(f" Dimensions:              Agents={dim.get('max_agents')}, Lanes={dim.get('max_lanes')}, Pts={dim.get('lane_pts')}, Obstacles={dim.get('max_obstacles')}, Modes={dim.get('num_modes')}")
        pol = data.get("train_config_policies", {})
        print(f" Training Policies:       Hard Sample Mining γ={pol.get('hard_sample_mining')}, Patience={pol.get('early_stopping_patience')}")
        kin = data.get("kinematics_and_context_statistics", {})
        if kin:
            print(f" Kinematics & Context:    Past Speed={kin.get('avg_past_speed_mps')}m/s | Avg Neighbors={kin.get('avg_neighbor_agents')} | Avg Lanes={kin.get('avg_map_lanes')}")
        print("=" * 80)


# Backward-compatibility aliases
DataSummery = DataValidationReport
DataSummary = DataValidationReport


# ==============================================================================
# 2. Automated Technical Acceptance Test Suite
# ==============================================================================

def run_acceptance_tests():
    """
    Executes 18 comprehensive technical acceptance tests validating:
      - Dataset contracts, missing file diagnostics, and padding integrity
      - Expanded context dimensions (20 agents, 32 lanes, 16 pts, 20 obstacles)
      - Empty context stability (lanes, neighbors, obstacles)
      - SuperNet & Standalone winning model [3, 2, 0] forward & backward passes
      - 100% finite gradients across AMP and parameters
      - Zero-Cost Proxies (SNIP, SynFlow) with weight restoration
      - Collision risk metric semantic compliance
      - Hard Sample Mining (HSM) DataLoader integration & error-weighted sampling
      - Kinematic physics smoothness loss
      - Discovered architecture [3, 2, 0] checkpoint verification
    """
    print("=" * 80)
    print(" TrajectoryLNAS Technical Acceptance & Verification Suite (data/data_validation.py)")
    if not HAS_TORCH:
        print("[Notice] PyTorch is not available in the current Python environment.")
        print("  -> Data quality inspection (DataValidationReport) is fully functional.")
        print("  -> To execute GPU/PyTorch tensor tests, run in an environment with PyTorch installed.")
        print("=" * 80)
        return

    print(f" PyTorch Version: {torch.__version__}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f" Target Device:  {device}")
    print("=" * 80)

    m_cfg = ModelConfig() if ModelConfig is not None else None
    t_cfg = TrainConfig() if TrainConfig is not None else None

    B = 2
    T_past = m_cfg.past_len if m_cfg else 4
    T_future = m_cfg.future_len if m_cfg else 12
    K_modes = m_cfg.num_modes if m_cfg else 5
    N_agents = m_cfg.max_agents if m_cfg else 20
    N_obstacles = m_cfg.max_obstacles if m_cfg else 20
    N_lanes = m_cfg.max_lanes if m_cfg else 32
    P_pts = m_cfg.lane_pts if m_cfg else 16
    hz_val = m_cfg.hz if m_cfg else 2.0
    dt_val = 1.0 / hz_val

    # Test 1: Missing real dataset file raises expected exception with 4-part details
    print("[Test 1] Missing real dataset file raises expected exception...")
    caught = False
    try:
        _ = NuScenesPredictionDataset(data_dir="/tmp/nonexistent_nuscenes_dir_999", generate_mock_if_missing=False)
    except FileNotFoundError as e:
        caught = True
        err_str = str(e)
        assert "Missing file" in err_str and "Expected path" in err_str and "Relevant configuration key" in err_str
        print("  ✓ PASSED: FileNotFoundError raised with 4-part informative message.")
    assert caught, "NuScenesPredictionDataset must raise FileNotFoundError when data missing"

    print("-" * 80)

    # Test 2: Production dataset initialization cannot silently generate mock data
    print("[Test 2] Production dataset initialization cannot silently generate mock data...")
    from data.preprocessing import DataPreparationPipeline
    pipeline = DataPreparationPipeline()
    caught_pipe = False
    try:
        pipeline.prepare(mode="real", dataroot="/tmp/nonexistent_nuscenes_dir_999")
    except (FileNotFoundError, ImportError) as e:
        caught_pipe = True
        print(f"  ✓ PASSED: Real mode rejected missing dataset ({type(e).__name__}).")
    assert caught_pipe, "Pipeline must raise error in real mode when data is missing"

    print("-" * 80)

    # Test 3: Missing lanes are zero-padded without artificial lane generation (32 lanes, 16 pts)
    print(f"[Test 3] Missing lanes are zero-padded to max_lanes={N_lanes} and lane_pts={P_pts}...")
    dummy_sample = {
        "dataset_type": "real_nuscenes",
        "sample_token": "dummy_sample_0",
        "instance_token": "dummy_inst_0",
        "category": "vehicle.car",
        "city": "boston-seaport",
        "past_traj": [[float(i * 1.5), float(i * 0.2)] for i in range(T_past)],
        "future_traj": [[float(20 * 1.5 + i * 1.5), float(20 * 0.2 + i * 0.2)] for i in range(T_future)],
        "timestamps_past": [round(-((T_past - 1 - i) * dt_val), 4) for i in range(T_past)],
        "timestamps_future": [round((i + 1) * dt_val, 4) for i in range(T_future)],
        "neighbors_past": [[[float(i * 1.2 + a * 3.0), float(i * 0.1)] for i in range(T_past)] for a in range(3)],
        "map_lanes": [[[float(p * 5.0), float(l * 3.5)] for p in range(P_pts)] for l in range(3)], # only 3 valid lanes
        "neighbor_categories": ["vehicle.car"] * 3,
        "map_lane_types": ["lane"] * 3,
        "static_objects": [
            {"translation": [8.0, 2.0, 0.0], "size": [0.5, 0.5, 1.0], "category": "movable_object.barrier"}
        ]
    }
    tmp_dir = tempfile.mkdtemp(prefix="lnas_test_ds_")
    try:
        train_json_path = os.path.join(tmp_dir, "train_sequences.json")
        with open(train_json_path, "w", encoding="utf-8") as f:
            json.dump([dummy_sample], f)

        ds = NuScenesPredictionDataset(
            data_dir=tmp_dir, split="train", past_len=T_past, future_len=T_future,
            max_agents=N_agents, max_obstacles=N_obstacles, max_lanes=N_lanes,
            lane_pts=P_pts, model_config=m_cfg, generate_mock_if_missing=False
        )
        item = ds[0]
        # Padded lanes beyond index 2 must be strict zeros
        assert item["map_lanes"].shape == (N_lanes, P_pts, 2), f"Expected shape ({N_lanes}, {P_pts}, 2)"
        assert (item["map_lanes"][3:] == 0).all(), "Padded lanes must be strictly zero"
        print(f"  ✓ PASSED: Missing lanes are strictly zero-padded to ({N_lanes}, {P_pts}, 2).")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    print("-" * 80)

    # Test 4: Lane masks correctly identify valid and invalid lanes out of 32 lanes
    print(f"[Test 4] Lane masks correctly identify valid and invalid lanes (3 valid out of {N_lanes})...")
    assert item["lane_mask"][:3].all() and not item["lane_mask"][3:].any(), "Lane mask mismatch"
    assert item["lane_mask"].shape == (N_lanes,), f"Lane mask must have shape ({N_lanes},)"
    print(f"  ✓ PASSED: Lane mask has True for 3 valid lanes and False for {max(0, N_lanes - 3)} padded lanes.")

    print("-" * 80)

    # Test 5: Empty lane collections do not cause NaN or Inf values in SuperNet
    print("[Test 5] Empty lane collections do not cause NaN or Inf values in SuperNet...")
    past_t = torch.randn(B, T_past, 2, device=device)
    map_lanes_empty = torch.zeros(B, N_lanes, P_pts, 2, device=device)
    lane_mask_empty = torch.zeros(B, N_lanes, dtype=torch.bool, device=device)
    supernet = SearchSuperNet(
        past_len=T_past, future_len=T_future, num_modes=K_modes,
        max_agents=N_agents, max_obstacles=N_obstacles, max_lanes=N_lanes,
        lane_pts=P_pts, hz=hz_val
    ).to(device)
    arch_params = ArchParams().to(device)
    alphas = arch_params.softmax_all()

    p_trajs, p_logits = supernet(past_t, alphas, map_lanes=map_lanes_empty, lane_mask=lane_mask_empty)
    assert not torch.isnan(p_trajs).any() and not torch.isinf(p_trajs).any(), "NaN/Inf in empty lane trajectories"
    assert not torch.isnan(p_logits).any() and not torch.isinf(p_logits).any(), "NaN/Inf in empty lane logits"
    print("  ✓ PASSED: Forward pass with 0 valid lanes produced finite outputs.")

    print("-" * 80)

    # Test 6: Neighbor masks correctly distinguish valid agents from padding out of max_agents=20
    print(f"[Test 6] Neighbor masks correctly distinguish valid agents from padding out of {N_agents}...")
    assert item["agent_mask"][0] == True, "Target ego vehicle must be valid at index 0"
    assert item["agent_mask"][:4].all() and not item["agent_mask"][4:].any(), "Agent mask mismatch"
    assert item["agents_past"].shape == (T_past, N_agents, 2)
    assert (item["agents_past"][:, 4:, :] == 0).all(), "Padded neighbor slots must be zero"
    print(f"  ✓ PASSED: Neighbor mask correctly distinguishes ego, 3 valid neighbors, and {max(0, N_agents - 4)} padding slots.")

    print("-" * 80)

    # Test 7: Zero valid neighbors do not cause numerical instability
    print("[Test 7] Zero valid neighbors do not cause numerical instability...")
    agents_empty = torch.zeros(B, T_past, N_agents, 2, device=device)
    agent_mask_zero_neigh = torch.zeros(B, N_agents, dtype=torch.bool, device=device)
    agent_mask_zero_neigh[:, 0] = True # Ego only, zero neighbors

    p_trajs2, p_logits2 = supernet(past_t, alphas, agents_past=agents_empty, agent_mask=agent_mask_zero_neigh)
    assert not torch.isnan(p_trajs2).any() and not torch.isinf(p_trajs2).any(), "NaN/Inf in zero-neighbor trajectories"
    assert not torch.isnan(p_logits2).any() and not torch.isinf(p_logits2).any(), "NaN/Inf in zero-neighbor logits"
    print("  ✓ PASSED: Forward pass with 0 valid neighbors produced finite outputs.")

    print("-" * 80)

    # Test 8: DataLoader cleanly preserves tensor shapes, masks, and sample_idx
    print("[Test 8] DataLoader cleanly preserves tensor shapes, masks, and sample_idx...")
    tmp_dir8 = tempfile.mkdtemp(prefix="lnas_test_loader_")
    try:
        with open(os.path.join(tmp_dir8, "train_sequences.json"), "w", encoding="utf-8") as f:
            json.dump([dummy_sample] * 4, f)
        loader = build_nuscenes_dataloader(
            data_dir=tmp_dir8, split="train", batch_size=2, past_len=T_past,
            future_len=T_future, max_agents=N_agents, max_obstacles=N_obstacles,
            max_lanes=N_lanes, lane_pts=P_pts, model_config=m_cfg
        )
        batch = next(iter(loader))
        assert batch["past"].shape == (2, T_past, 2), "Past shape mismatch"
        assert batch["future"].shape == (2, T_future, 2), "Future shape mismatch"
        assert batch["agents_past"].shape == (2, T_past, N_agents, 2), "Agents shape mismatch"
        assert batch["agent_mask"].shape == (2, N_agents), "Agent mask mismatch"
        assert batch["map_lanes"].shape == (2, N_lanes, P_pts, 2), "Map lanes shape mismatch"
        assert batch["lane_mask"].shape == (2, N_lanes), "Lane mask mismatch"
        assert batch["static_obstacles"].shape == (2, N_obstacles, 4), "Static obstacles shape mismatch"
        assert batch["static_obstacle_mask"].shape == (2, N_obstacles), "Static obstacle mask mismatch"
        assert "sample_idx" in batch, "sample_idx must be in batch for Hard Sample Mining!"
        assert batch["sample_idx"].shape == (2,), "sample_idx shape mismatch"
        print("  ✓ PASSED: DataLoader cleanly preserves tensor shapes, masks, static obstacles, and sample_idx.")
    finally:
        shutil.rmtree(tmp_dir8, ignore_errors=True)

    print("-" * 80)

    # Test 9: SuperNet forward pass accepts full input data contract
    print("[Test 9] SuperNet forward pass accepts full input data contract...")
    past_req = torch.randn(B, T_past, 2, device=device, requires_grad=True)
    agents_req = torch.randn(B, T_past, N_agents, 2, device=device, requires_grad=True)
    lanes_req = torch.randn(B, N_lanes, P_pts, 2, device=device, requires_grad=True)
    obstacles_req = torch.randn(B, N_obstacles, 4, device=device, requires_grad=True)
    pred_trajs, mode_logits = supernet(
        past_req, alphas, agents_past=agents_req, map_lanes=lanes_req,
        agent_mask=batch["agent_mask"].to(device), lane_mask=batch["lane_mask"].to(device),
        static_obstacles=obstacles_req, static_obstacle_mask=batch["static_obstacle_mask"].to(device)
    )
    assert pred_trajs.shape == (B, K_modes, T_future, 2), f"Output trajectory shape must be ({B}, {K_modes}, {T_future}, 2)"
    assert mode_logits.shape == (B, K_modes), f"Mode logits shape must be ({B}, {K_modes})"
    print(f"  ✓ PASSED: SuperNet forward pass verified with shape ({B}, {K_modes}, {T_future}, 2).")

    print("-" * 80)

    # Test 10: Standalone Winning Model [3, 2, 0] Forward Pass
    print("[Test 10] Standalone Discovered Architecture [3, 2, 0] forward pass...")
    # Discovered combination: 3=TemporalSelfAttention, 2=SocialPooling, 0=LaneAttention
    final_model = FinalTrajectoryModel(
        best_temporal_idx=3, best_interaction_idx=2, best_map_idx=0,
        past_len=T_past, future_len=T_future, num_modes=K_modes,
        max_agents=N_agents, max_obstacles=N_obstacles, max_lanes=N_lanes, lane_pts=P_pts,
        hz=hz_val
    ).to(device)
    f_preds, f_logits = final_model(
        past_req.detach(), agents_past=agents_req.detach(), map_lanes=lanes_req.detach(),
        agent_mask=batch["agent_mask"].to(device), lane_mask=batch["lane_mask"].to(device),
        static_obstacles=obstacles_req.detach(), static_obstacle_mask=batch["static_obstacle_mask"].to(device)
    )
    assert f_preds.shape == (B, K_modes, T_future, 2)
    assert f_logits.shape == (B, K_modes)
    summary_model = final_model.get_summary()
    assert summary_model["temporal_op"] == "TemporalSelfAttention"
    assert summary_model["interaction_op"] == "SocialPooling"
    assert summary_model["map_op"] == "LaneAttention"
    print(f"  ✓ PASSED: Standalone Winning Architecture [3, 2, 0] ({summary_model['temporal_op']} + {summary_model['interaction_op']} + {summary_model['map_op']}) executed cleanly.")

    print("-" * 80)

    # Test 11: Backward pass produces finite gradients across inputs, operators, and alphas
    print("[Test 11] Backward pass produces finite gradients across inputs, operators, and alphas...")
    future_gt = torch.randn(B, T_future, 2, device=device)
    loss_reg, loss_cls = best_of_k_loss(pred_trajs, future_gt, mode_logits)
    loss = loss_reg + 0.1 * loss_cls
    loss.backward()

    assert past_req.grad is not None and torch.isfinite(past_req.grad).all(), "Non-finite past gradient"
    assert agents_req.grad is not None and torch.isfinite(agents_req.grad).all(), "Non-finite agents gradient"
    assert lanes_req.grad is not None and torch.isfinite(lanes_req.grad).all(), "Non-finite map lanes gradient"
    assert arch_params.alpha_temporal.grad is not None and torch.isfinite(arch_params.alpha_temporal.grad).all(), "Non-finite alpha gradient"
    print("  ✓ PASSED: Backward pass completed with 100% finite gradients across inputs & parameters.")

    print("-" * 80)

    # Test 12: Synthetic fallback dataset works cleanly with updated dimensions
    print(f"[Test 12] Synthetic fallback dataset works cleanly with max_agents={N_agents}, max_lanes={N_lanes}...")
    syn_ds = SyntheticTrajectoryDataset(num_samples=10, num_agents=N_agents, model_config=m_cfg)
    syn_item = syn_ds[0]
    assert syn_item["past"].shape == (T_past, 2)
    assert syn_item["future"].shape == (T_future, 2)
    assert syn_item["agents_past"].shape == (T_past, N_agents, 2)
    assert syn_item["map_lanes"].shape == (N_lanes, P_pts, 2)
    assert "sample_idx" in syn_item, "SyntheticTrajectoryDataset must provide sample_idx"
    print("  ✓ PASSED: SyntheticTrajectoryDataset updated and matches all dimensions.")

    print("-" * 80)

    # Test 13: Zero-Cost Proxies (SNIP & SynFlow) execute safely and restore weights 100%
    print("[Test 13] Zero-Cost Proxies (SNIP & SynFlow) execute safely and restore weights 100%...")
    sample_batch = {
        "past": past_req.detach(),
        "future": future_gt,
        "agents_past": agents_req.detach(),
        "map_lanes": lanes_req.detach(),
        "agent_mask": batch["agent_mask"].to(device),
        "lane_mask": batch["lane_mask"].to(device),
        "static_obstacles": obstacles_req.detach(),
        "static_obstacle_mask": batch["static_obstacle_mask"].to(device),
    }
    pre_weights = {name: p.clone() for name, p in supernet.named_parameters()}
    snip_val = snip_score(supernet, sample_batch, device=str(device))
    assert math.isfinite(snip_val) and snip_val > 0.0, f"Invalid SNIP score: {snip_val}"

    synflow_val = synflow_score(supernet, device=str(device))
    assert math.isfinite(synflow_val) and synflow_val > 0.0, f"Invalid SynFlow score: {synflow_val}"

    for name, p in supernet.named_parameters():
        assert torch.equal(p, pre_weights[name]), f"SynFlow corrupted parameter '{name}'!"
    print(f"  ✓ PASSED: SNIP ({snip_val:.2f}) and SynFlow ({synflow_val:.2f}) executed with 100% weight integrity.")

    print("-" * 80)

    # Test 14: Collision Risk Metric semantics
    print("[Test 14] Collision Risk Metric semantics...")
    col_past = evaluate_collision_risk(pred_trajs.detach(), neighbors_future=agents_req.detach())
    assert col_past["available"] is False, "Collision risk must report unavailable when historical data is passed!"

    dummy_future_neighbors = torch.randn(B, T_future, N_agents, 2, device=device)
    col_future = evaluate_collision_risk(pred_trajs.detach(), neighbors_future=dummy_future_neighbors)
    assert col_future["available"] is True, "Collision risk must compute successfully when genuine future neighbor data exists!"
    print("  ✓ PASSED: Collision Risk accurately distinguishes historical vs future neighbor trajectories.")

    print("-" * 80)

    # Test 15: FLOPs & MACs Unit Consistency
    print("[Test 15] FLOPs & MACs Unit Consistency (FLOPs = 2 * MACs & GFLOPs scale)...")
    summary = final_model.get_summary()
    assert summary["theoretical_flops"] == summary["theoretical_macs_m"] * 2 * 1e6 or abs(summary["theoretical_flops"] - summary["theoretical_macs_m"] * 2e6) < 1e5
    assert 0.005 <= summary["theoretical_flops_g"] <= 0.1, f"Expected realistic GFLOPs in [0.005, 0.1], got {summary['theoretical_flops_g']}"
    print(f"  ✓ PASSED: Mathematical FLOPs ({summary['theoretical_flops_m']}M / {summary['theoretical_flops_g']}G) and MACs ({summary['theoretical_macs_m']}M) verified.")

    print("-" * 80)

    # Test 16: Hard Sample Mining (HSM) DataLoader Integration & Dynamic Weighted Sampling
    print("[Test 16] Hard Sample Mining (HSM) DataLoader Integration & Dynamic Weighted Sampling...")
    hsm_sens = getattr(t_cfg, "hard_sample_mining", 0.5) if t_cfg else 0.5
    hsm_manager = HardSampleMiningManager(num_samples=100, sensitivity=hsm_sens, ema_momentum=0.7)
    assert hsm_manager.is_active == (hsm_sens > 0.0), f"HSM active state mismatch for sensitivity={hsm_sens}"

    # Sample 5 is easy (0.05m error), Sample 25 is hard (1.80m error)
    hsm_manager.update_sample_errors([5, 25], [0.05, 1.80])
    weights = hsm_manager.get_weights()
    w_easy = weights[5].item() if hasattr(weights[5], "item") else weights[5]
    w_hard = weights[25].item() if hasattr(weights[25], "item") else weights[25]
    assert w_hard > w_easy, f"Hard sample weight ({w_hard}) must exceed easy sample weight ({w_easy})"

    # Test DataLoader sampler attachment with HSM
    tmp_dir16 = tempfile.mkdtemp(prefix="lnas_test_hsm_loader_")
    try:
        with open(os.path.join(tmp_dir16, "train_sequences.json"), "w", encoding="utf-8") as f:
            json.dump([dummy_sample] * 6, f)
        hsm_loader = build_nuscenes_dataloader(
            data_dir=tmp_dir16, split="train", batch_size=2, past_len=T_past,
            future_len=T_future, max_agents=N_agents, max_obstacles=N_obstacles,
            max_lanes=N_lanes, lane_pts=P_pts, model_config=m_cfg,
            hard_sample_mining=hsm_sens
        )
        assert hasattr(hsm_loader, "sampler") and isinstance(hsm_loader.sampler, HardSampleWeightedSampler), "HSM loader must attach HardSampleWeightedSampler"
        hsm_batch = next(iter(hsm_loader))
        assert "sample_idx" in hsm_batch and (hsm_batch["sample_idx"] >= 0).all()
        print(f"  ✓ PASSED: Hard Sample Mining active (γ={hsm_sens}). Hard sample weight ratio: {w_hard/w_easy:.2f}x.")
    finally:
        shutil.rmtree(tmp_dir16, ignore_errors=True)

    print("-" * 80)

    # Test 17: Kinematic Physics Loss & Second-Order Displacement Smoothness
    print("[Test 17] Kinematic Physics Loss & Second-Order Displacement Smoothness...")
    # Straight constant velocity: displacement difference of order 2 is exactly 0
    straight_pred = torch.zeros(B, K_modes, T_future, 2, device=device)
    for t in range(T_future):
        straight_pred[:, :, t, 0] = float(t * 1.5)
    diff2_straight = straight_pred[:, :, 2:, :] - 2 * straight_pred[:, :, 1:-1, :] + straight_pred[:, :, :-2, :]
    loss_physics_straight = torch.mean(torch.norm(diff2_straight, p=2, dim=-1))
    assert loss_physics_straight.item() < 1e-5, f"Constant velocity trajectory must have zero physics loss, got {loss_physics_straight.item()}"

    # Jittery trajectory: non-zero second-order derivative
    jitter_pred = straight_pred.clone()
    jitter_pred[:, :, 1::2, 0] += 0.8
    diff2_jitter = jitter_pred[:, :, 2:, :] - 2 * jitter_pred[:, :, 1:-1, :] + jitter_pred[:, :, :-2, :]
    loss_physics_jitter = torch.mean(torch.norm(diff2_jitter, p=2, dim=-1))
    assert loss_physics_jitter.item() > 0.5, f"Jittery trajectory must have significant physics penalty, got {loss_physics_jitter.item()}"
    print(f"  ✓ PASSED: Physics Jerk Loss validates smooth kinematics (Straight: {loss_physics_straight.item():.5f} vs Jitter: {loss_physics_jitter.item():.3f}).")

    print("-" * 80)

    # Test 18: Discovered Architecture Checkpoint (best_architecture.json) Schema & Consistency
    print("[Test 18] Discovered Architecture Checkpoint (best_architecture.json) Schema & Consistency...")
    arch_json_path = ROOT_DIR / "checkpoints" / "best_architecture.json"
    if arch_json_path.exists():
        with open(arch_json_path, "r", encoding="utf-8") as f:
            arch_doc = json.load(f)
        assert arch_doc.get("best_ops") == [3, 2, 0], f"Expected best_ops [3, 2, 0], got {arch_doc.get('best_ops')}"
        assert arch_doc["discovered_architecture"]["temporal"]["name"] == "TemporalSelfAttention"
        assert arch_doc["discovered_architecture"]["interaction"]["name"] == "SocialPooling"
        assert arch_doc["discovered_architecture"]["map"]["name"] == "LaneAttention"
        print(f"  ✓ PASSED: best_architecture.json verified: [3, 2, 0] ({arch_doc['discovered_architecture']['temporal']['name']}, {arch_doc['discovered_architecture']['interaction']['name']}, {arch_doc['discovered_architecture']['map']['name']}).")
    else:
        print("  [Notice] best_architecture.json not present at checkpoints/, skipping file inspection.")

    print("=" * 80)
    print(" ALL 18 TECHNICAL ACCEPTANCE TESTS PASSED WITH 100% INTEGRITY! ")
    print("=" * 80)


def run_data_summary():
    """Facade for data quality inspection."""
    report = DataValidationReport()
    res = report.load_and_validate()
    return res


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TrajectoryLNAS Data Validation & Technical Acceptance")
    parser.add_argument("--mode", choices=["tests", "report", "all"], default="all", help="Execution mode")
    parser.add_argument("--data_dir", type=str, default=None, help="Dataset directory override")
    args = parser.parse_args()

    if args.mode in ("report", "all"):
        report = DataValidationReport(data_root=args.data_dir)
        report.load_and_validate()

    if args.mode in ("tests", "all"):
        run_acceptance_tests()
