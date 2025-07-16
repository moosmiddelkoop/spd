"""SPD decomposition for Computation in Superposition model."""

import argparse
from pathlib import Path
from typing import override

import torch
import wandb
from torch.utils.data import DataLoader, Dataset

from spd.configs import Config
from spd.experiments.cis.models import CISModel
from spd.experiments.cis.plotting import (
    create_raw_weights_heatmap,
    create_stacked_weight_plot,
)
from spd.log import logger
from spd.run_spd import optimize
from spd.utils.general_utils import get_device, load_config
from spd.utils.run_utils import get_output_dir


class CISDataset(Dataset[torch.Tensor]):
    """Simple dataset for CIS model that returns single tensors."""

    def __init__(self, data: torch.Tensor):
        self.data = data

    def __len__(self) -> int:
        return len(self.data)

    @override
    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.data[idx]


def _create_data_loader(
    model: CISModel, batch_size: int, feature_sparsity: float, num_samples: int = 10000
) -> DataLoader[torch.Tensor]:
    """Create a simple data loader for the CIS model.

    Args:
        model: The CIS model to create data for
        batch_size: Batch size for the data loader
        feature_sparsity: Probability that each feature is zero
        num_samples: Number of samples to generate (default: 10000)
    """
    n_features = model.config.n_features

    # Generate example data more efficiently
    data = torch.zeros(num_samples, n_features)
    for i in range(num_samples):
        # Generate sparse features
        active_mask = torch.rand(n_features) > feature_sparsity
        values = torch.rand(n_features) * 2 - 1  # Uniform [-1, 1]
        data[i] = values * active_mask

    dataset = CISDataset(data)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def run_cis_decomposition(config_path: Path) -> None:
    """Run SPD decomposition on a CIS model."""
    config = load_config(config_path, Config)

    # Get device for computation
    device = get_device()

    # Check that pretrained_model_path is provided
    if config.pretrained_model_path is None:
        raise ValueError("pretrained_model_path must be provided in config")

    # Load the pretrained model
    model, model_config_dict = CISModel.from_pretrained(config.pretrained_model_path)
    model.to(device)

    train_config = model_config_dict

    # Create data loaders
    train_loader = _create_data_loader(
        model, config.batch_size, train_config.get("feature_sparsity", 0.9)
    )
    eval_loader = _create_data_loader(
        model, config.batch_size, train_config.get("feature_sparsity", 0.9)
    )

    # Get output directory
    out_dir = get_output_dir()

    # Initialize wandb
    if config.wandb_project:
        # Generate simple run name from model path
        model_path_str = str(config.pretrained_model_path)
        run_name = f"cis_decomposition_{model_path_str.split('/')[-1] if '/' in model_path_str else model_path_str}"
        wandb.init(
            project=config.wandb_project,
            name=run_name,
            config=config.model_dump(),
            tags=["cis", "spd"],
        )

    # Run SPD optimization
    optimize(
        target_model=model,
        config=config,
        device=device,
        train_loader=train_loader,
        eval_loader=eval_loader,
        n_eval_steps=10,
        out_dir=out_dir,
    )

    # Create basic visualizations
    logger.info("Creating visualizations...")

    # Create stacked weight plot
    create_stacked_weight_plot(
        model,
        out_dir / "stacked_weights.png",
        sort_by_importance=True,
    )

    # Create raw weight heatmaps
    create_raw_weights_heatmap(model, out_dir / "w1_heatmap.png", layer="W1")
    create_raw_weights_heatmap(model, out_dir / "w2_heatmap.png", layer="W2")

    # Log basic visualizations to wandb
    if config.wandb_project:
        wandb.log(
            {
                "stacked_weights": wandb.Image(str(out_dir / "stacked_weights.png")),
                "w1_heatmap": wandb.Image(str(out_dir / "w1_heatmap.png")),
                "w2_heatmap": wandb.Image(str(out_dir / "w2_heatmap.png")),
            }
        )

    logger.info(f"Decomposition complete. Results saved to {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SPD decomposition on CIS model")
    parser.add_argument("config_path", type=Path, help="Path to config file")
    args = parser.parse_args()

    run_cis_decomposition(args.config_path)
