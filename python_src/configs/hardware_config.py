r"""
Hardware Configuration
======================
Describes the target device and its resource constraints.
Execution settings belong to SearchConfig and TrainConfig.
"""

from dataclasses import dataclass
from configs.base_config import BaseConfig


@dataclass
class HardwareConfig(BaseConfig):
    """Hardware target profile and resource constraints."""
    device_type: str = "cuda"                               # Target compute device ('cuda' for GPU or 'cpu')
    gpu_name: str = "NVIDIA GeForce RTX 3050 Laptop GPU"    # Edge GPU hardware target identifier
    vram_gb: float = 4.0                                    # Dedicated GPU video memory in GB (4.0GB physical constraint)
    system_ram_gb: float = 32.0                             # Host system physical RAM capacity in GB
