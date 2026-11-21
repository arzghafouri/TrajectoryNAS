r"""
Data Configuration
==================
Configuration for dataset extraction, preprocessing, and splits.
"""

import os
from dataclasses import dataclass
from configs.base_config import BaseConfig


@dataclass
class DataConfig(BaseConfig):
    """Configuration for dataset extraction, preprocessing, and splits.

    Serves as the Single Source of Truth for all dataset paths across TrajectoryLNAS.
    Changing dataroot or output_dir here (or via NUSCENES_DATAROOT / NUSCENES_OUTPUT_DIR)
    is automatically reflected across data loaders, search, training, and evaluation.
    """
    dataroot: str = os.environ.get(
        "NUSCENES_DATAROOT",
        os.environ.get("NUSCENES_ROOT", r"F:\Program\Dataset\nuScenes" if os.name == "nt" else "data/nuscenes")
    )                                               # Root directory path of raw nuScenes tables and expansion maps
    output_dir: str = os.environ.get(
        "NUSCENES_OUTPUT_DIR",
        ""
    )                                               # Destination directory for processed sequence files (.jsonl)
    target_count: int | None = None                 # Target sequence extraction limit (None = all available sequences)
    category: str | list[str] | None = None         # Specific agent category filter (None = all valid agent types)
    mode: str = "auto"                              # Extraction mode: 'auto' (detect), 'real' (strict), or 'synthetic'
    seed: int = 42                                  # Reproducibility seed for deterministic dataset shuffling
    max_train_scenes: int | None = None             # Small train subset for extraction review; set None for all scenes
    max_val_scenes: int | None = None               # Small validation subset for extraction review; set None for all scenes
    resume_extraction: bool = True                  # Continue from the last completed-scene checkpoint when available


    def __post_init__(self):
        # Normalize dataroot for cross-platform robustness
        if self.dataroot:
            self.dataroot = os.path.normpath(self.dataroot)

        # If output_dir is not explicitly specified, derive it from dataroot
        if not self.output_dir:
            self.output_dir = os.path.normpath(os.path.join(self.dataroot, "v1.0-trainval"))
        else:
            self.output_dir = os.path.normpath(self.output_dir)

    @property
    def train_sequences_path(self) -> str:
        jsonl = os.path.join(self.output_dir, "train_sequences.jsonl")
        json_f = os.path.join(self.output_dir, "train_sequences.json")
        return jsonl if os.path.exists(jsonl) else json_f

    @property
    def val_sequences_path(self) -> str:
        jsonl = os.path.join(self.output_dir, "val_sequences.jsonl")
        json_f = os.path.join(self.output_dir, "val_sequences.json")
        return jsonl if os.path.exists(jsonl) else json_f

    def is_dataset_ready(self) -> bool:
        has_tr = os.path.exists(os.path.join(self.output_dir, "train_sequences.jsonl")) or os.path.exists(os.path.join(self.output_dir, "train_sequences.json"))
        has_va = os.path.exists(os.path.join(self.output_dir, "val_sequences.jsonl")) or os.path.exists(os.path.join(self.output_dir, "val_sequences.json"))
        return has_tr and has_va
