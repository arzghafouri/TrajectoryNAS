"""
DataLoaders Utility Manager
===========================
Factory function for PyTorch DataLoaders (nuScenes and Synthetic).
"""

from torch.utils.data import DataLoader


def get_dataloaders(train_path=None, val_path=None, batch_size=32, dataset_type="nuscenes", sample_percent=None, allow_synthetic_fallback=False, hard_sample_mining=None, **kwargs):
    """
    Creates PyTorch DataLoaders via data.nuscenes_dataset (nuScenes or Synthetic) with optional Hard Sample Mining.
    Strictly enforces real data loading; silent fallback to synthetic data is disabled by default.
    """
    resolved_train = kwargs.pop("train_dir", train_path)
    resolved_val = kwargs.pop("val_dir", val_path)
    resolved_ds = kwargs.pop("dataset_name", dataset_type)
    kwargs.pop("train_path", None)
    kwargs.pop("val_path", None)
    kwargs.pop("dataset_type", None)

    try:
        from data.nuscenes_dataset import get_trajectory_dataloaders
        return get_trajectory_dataloaders(
            dataset_name=resolved_ds,
            train_dir=resolved_train,
            val_dir=resolved_val,
            batch_size=batch_size,
            sample_percent=sample_percent,
            hard_sample_mining=hard_sample_mining,
            **kwargs
        )
    except Exception as e:
        if allow_synthetic_fallback:
            print(f"[DataLoader Notice] Explicit synthetic fallback allowed for testing: {e}")
            from data.synthetic_dataset import SyntheticTrajectoryDataset
            from configs.model_config import ModelConfig
            model_cfg = kwargs.get("model_config") or ModelConfig()
            train_dataset = SyntheticTrajectoryDataset(
                num_samples=1200, num_agents=model_cfg.max_agents, model_config=model_cfg
            )
            val_dataset = SyntheticTrajectoryDataset(
                num_samples=300, num_agents=model_cfg.max_agents, model_config=model_cfg
            )

            hsm = float(hard_sample_mining or kwargs.get("hard_sample_sensitivity", 0.0) or 0.0)
            train_sampler = None
            if hsm > 0.0:
                from data.hard_sample_mining import HardSampleMiningManager
                manager = HardSampleMiningManager(num_samples=len(train_dataset), sensitivity=hsm)
                train_sampler = manager.get_sampler(replacement=True)

            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=(train_sampler is None),
                sampler=train_sampler
            )
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
            return train_loader, val_loader
        else:
            raise RuntimeError(
                f"[DataLoader Error] Failed to load real '{dataset_type}' dataset ({e}).\n"
                f"Silent synthetic data fallback is strictly disabled to prevent invalid research results.\n"
                f"Please ensure real dataset files exist or pass 'allow_synthetic_fallback=True' explicitly for smoke tests only."
            ) from e
