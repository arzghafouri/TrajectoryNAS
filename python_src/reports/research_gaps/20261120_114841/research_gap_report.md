# TrajectoryLNAS Research Gap Evaluation Report

- **Generated (UTC)**: `2026-11-20T11:48:41Z`
- **Output Directory**: `/app/applet/reports/research_gaps/20261120_114841`
- **Requested Experiment Suite**: `audit`
- **Dataset**: `nuscenes` | **Seeds**: `[42]`
- **Execution Environment**: PyTorch `Not installed` | CUDA Available: `False` | Device: `x86_64`

## 1. Executive Summary

| Component | Status | Scientific Classification | Summary |
| --- | --- | --- | --- |
| Data Integrity & Reliability Audit | `completed` | Executed & Measured | Real nuScenes sequence files not found under 'data/nuscenes/v1.0-trainval'. Run 'python python_src/data/preprocessing.py --dataroot <path_to_nuscenes>' before running training/search experiments. |
| Gap A: Factorized Three-Branch NAS (A0-A7) | `not_run` | Implemented (Not Executed) | Not selected in this invocation |
| Gap B: Hardware-Aware NAS Profiling | `not_run` | Implemented (Not Executed) | Not selected in this invocation |
| Gap C: Scene-Adaptive Evaluation | `not_run` | Implemented (Not Executed) | Not selected in this invocation |
| Warmup & Temperature Annealing (T0-T3) | `not_run` | Implemented (Not Executed) | Not selected in this invocation |

## 2. Dataset and Data Integrity Audit

- **Audit Status**: `completed` | **Safe for Comparable Experiments**: `False`
- **Configured Dataroot**: `data/nuscenes`
- **Resolved Sequence Dir**: `data/nuscenes/v1.0-trainval` (Ready: `False`)
- **Horizon & Sampling Grid**: past=`4` steps (`2.0s`), future=`12` steps (`6.0s`), hz=`2.0`, K=`5` (verified output K=`None`)
- **DARTS K Observation**: Observed in python_src/core/darts_algorithm.py (lines 237-242): DARTSAlgorithm.validate() passes k=6 to min_ade_k/min_fde_k while ModelConfig.num_modes=5. Inside evaluations/metrics.py (_top_k_predictions), k_eval=min(k, shape[1]) clamps k to 5, so numerical results reflect 5 modes. This experiment runner derives K dynamically from model.num_modes and pred_trajs.shape[1] across minADE, minFDE, and MissRate.
- **Scene-Level Split Separation**: `NOT_VERIFIED_MISSING_FILES` (train scenes=`None`, val scenes=`None`, shared=`None`)
- **Warning**: Real nuScenes sequence files not found under 'data/nuscenes/v1.0-trainval'. Run 'python python_src/data/preprocessing.py --dataroot <path_to_nuscenes>' before running training/search experiments.

## 3. Factorized NAS Ablation Results (Research Gap A)

_Factorized ablation was not executed in this run (`not_run`)._

## 4. Hardware Efficiency Results (Research Gap B)

_Hardware profiling was not executed in this run (`not_run`)._

## 5. Scene Complexity Analysis (Research Gap C)

_Scene complexity evaluation was not executed in this run (`not_run`)._

## 6. Warmup and Temperature Annealing Ablation

_Warmup & temperature ablation was not executed in this run (`not_run`)._

## 7. Literature Comparison Table (Protocol-Aware Audit)

> **Protocol Disclaimer**: Published literature values below come from heterogeneous datasets, horizons, prediction mode counts (K), and metric definitions. Do NOT treat cross-protocol numbers as direct numerical rankings. TrajectoryLNAS figures in this report reflect only measurements verified in this execution.

| Model | Architecture / NAS Paradigm | Temporal Modeling | Interaction Modeling | HD-Map Modeling | NAS Strategy | Hardware Objective | Scene-Adaptive | Dataset & Evaluation Protocol | Published / Newly Measured Metrics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| VectorNet (CVPR 2020) | Manual hierarchical graph network (polyline subgraph + global interaction graph) | 1D Polyline Node MLP + MaxPool | Global self-attention interaction graph over agent/lane nodes | Vectorized HD-map polyline subgraphs | None (hand-crafted architecture) | None | Static fixed architecture | Argoverse 1 (past 2s @10Hz, future 3s, K=1/K=6) & internal benchmarks; NOT directly comparable to nuScenes 6s K=5 | Argoverse val: minADE@6=0.79m (paper Table 1/leaderboard variants); incompatible dataset/horizon/K vs nuScenes 6s |
| LaneGCN (ECCV 2020) | Manual multi-scale 1D CNN + Lane Graph / Actor-Map fusion network | Multi-scale 1D ResNet + FPN (ActorNet) | Actor-to-Actor (A2A) spatial attention after map fusion | 4-type directed lane graph convolution (LaneGCN) + A2L/L2A/L2L attention | None (hand-crafted architecture) | None | Static fixed architecture | Argoverse 1 (2s->3s @10Hz, K=6); incompatible dataset/horizon/K vs nuScenes 6s | Argoverse test: minADE@6=0.87m, minFDE@6=1.36m, MR@2m=16.3% (ECCV 2020) |
| HiVT (CVPR 2022) | Manual translation/rotation-invariant hierarchical vectorized Transformer | Temporal Transformer encoder over local relative displacements | Local Transformer + Global inter-agent Transformer interaction | Local agent-lane cross-attention (vectorized lane segments) | None (hand-crafted architecture) | Explicit lightweight edge parameter budget (~0.66M-2.5M params) without NAS FLOPs penalty | Static fixed architecture (local-global hierarchy) | Argoverse 1 (2s->3s @10Hz, K=6); incompatible dataset/horizon/K vs nuScenes 6s | Argoverse test (HiVT-64/128): minADE@6=0.77-0.80m, minFDE@6=1.17-1.27m (CVPR 2022) |
| TrajectoryNAS (Sensors 2024) | End-to-end 3D LiDAR detection, tracking & trajectory forecasting NAS | Searched spatiotemporal backbone blocks | Implicit BEV multi-object spatial features | Implicit BEV feature encoding (raw LiDAR + optional HD map raster/BEV) | Hardware-constrained architecture search over perception+prediction blocks | Latency/Compute constrained (22 ms reported on RTX A4000) | Static searched architecture | nuScenes Mini/Trainval end-to-end APf detection+forecasting (K=1/K=5); incompatible metric definition (mAPf vs minADE@K) | nuScenes K=5: vehicle mAPf=36.2, ped mAPf=32.5; RTX A4000 latency=22ms (Sharifi et al., Sensors 2024) |
| TrajGNAS (Literature GNN-NAS) | Differentiable/evolutionary NAS over spatiotemporal GNN trajectory encoders | Searched temporal / recurrent GNN message-passing blocks | Searched graph topology / spatial aggregation edges | Scene/map context graph nodes when applicable (benchmark dependent) | NAS over graph neural architecture space | Parameter/FLOPs awareness depending on target deployment configuration | Static searched graph architecture (or topology weights per sample) | Traffic/urban trajectory benchmarks; exact split, horizon, K, and codebase not present in this workspace | Published exact protocol/source not present in repository; numerical values omitted to avoid unverified comparison |
| TrajectoryLNAS (This Repo) | Factorized 3-branch differentiable DARTS SuperNet -> discrete standalone FinalTrajectoryModel | Searched from {GRUEncoder, LSTMEncoder, TemporalConvEncoder, TemporalSelfAttention} | Searched from {GCNLayer, GATLayer, SocialPooling, PairwiseAttention} (+ static obstacles) | Searched from {LaneAttention, LaneGraphEncoder, MapFeatureFusion} over PolylineEncoder | Bi-level first-order DARTS with architecture warmup + exponential temperature annealing | Expected SuperNet GFLOPs penalty (lambda_flops * E[GFLOPs]) in validation alpha loss | Static globally discovered architecture (Gap C investigates scene-conditioned prototype) | nuScenes vectorized sequences (past 2.0s [4 steps] @2Hz, future 6.0s [12 steps] @2Hz, K=5 modes from ModelConfig) | Pending trained TrajectoryLNAS checkpoint evaluation on extracted nuScenes validation set |

## 8. Figures and Charts

_No charts generated (requires Matplotlib and at least one completed numerical measurement series)._

## 9. Failed or Incomplete Experiments

- None.

## 10. Limitations and Threats to Validity

- **No Future Neighbor Ground Truth**: `neighbors_future` is absent from preprocessed sequences; physical Collision Rate cannot be evaluated reliably.
- **No Explicit Lane-Change Annotations**: Gap C intersection and turning slices rely on `map_lane_types` (`lane_connector`/`ped_crossing`) and ego trajectory curvature proxies.
- **Search-to-Eval Gap**: DARTS SuperNet validation loss reflects continuous mixed-op weights, whereas paper claims must be grounded in discrete standalone `FinalTrajectoryModel` metrics.
- **Protocol Differences Across Literature**: Argoverse 3s K=6 and LiDAR APf benchmarks are not numerically comparable to nuScenes 6s K=5.

## 11. Conclusions Supported by Actual Measurements

- Only claims backed by `completed` runs in Sections 2-6 above are empirically verified; `pending` or `not_run` hypotheses remain **unverified** until executed on real GPU hardware with extracted nuScenes sequences.

## 12. Recommended Next Experiments

1. Verify dataset sequences exist (`python python_src/data/preprocessing.py --dataroot <path_to_nuscenes>`) and run `--experiment audit`.
2. Train reference TrajectoryLNAS checkpoint (`python python_src/main.py`) and run `--experiment hardware` & `--experiment scene`.
3. Execute multi-seed (`--seeds 42 123 456`) runs for `--experiment factorized` and `--experiment warmup_temperature` on GPU.
4. Run `--experiment hardware --run-lambda-search` and `--experiment scene --run-scene-prototype` for full empirical Pareto & adaptive curves.
