#!/usr/bin/env python3
"""SPD decomposition for modular addition experiments."""

import json
from pathlib import Path
from typing import Any, override

import fire
import torch
import wandb
from torch.utils.data import Dataset

from spd.configs import Config, ModularAdditionTaskConfig
from spd.experiments.modular_addition.models import load_pretrained_modular_addition_model
from spd.experiments.modular_addition.vendored.transformers import Config as GrokkingConfig
from spd.experiments.modular_addition.vendored.transformers import gen_train_test
from spd.log import logger
from spd.run_spd import optimize
from spd.utils.data_utils import DatasetGeneratedDataLoader
from spd.utils.general_utils import get_device, load_config, set_seed
from spd.utils.run_utils import get_output_dir, save_file
from spd.utils.wandb_utils import init_wandb


class ModularAdditionDataset(Dataset[torch.Tensor]):
    """Dataset for modular addition data."""
    
    def __init__(self, data_tuples: list[tuple[int, int, int]], device: str):
        self.data = torch.tensor(data_tuples, dtype=torch.long, device=device)

    @override
    def __len__(self) -> int:
        return 2**31
        
    def generate_batch(self, batch_size: int) -> torch.Tensor:
        """Generate a batch by randomly sampling from the dataset."""
        if batch_size >= len(self.data):
            # Return full dataset if batch size is larger
            return self.data
        else:
            # Randomly sample batch_size examples
            indices = torch.randperm(len(self.data))[:batch_size]
            return self.data[indices]


def save_target_model_info(
    save_to_wandb: bool,
    out_dir: Path,
    model: torch.nn.Module,
    model_config_dict: dict[str, Any],
) -> None:
    save_file(model.state_dict(), out_dir / "modular_addition_model.pth")
    save_file(model_config_dict, out_dir / "modular_addition_config.yaml")

    if save_to_wandb:
        wandb.save(str(out_dir / "modular_addition_model.pth"), base_path=out_dir, policy="now")
        wandb.save(str(out_dir / "modular_addition_config.yaml"), base_path=out_dir, policy="now")


def main(
    config_path_or_obj: Path | str | Config,
    evals_id: str | None = None,
    sweep_id: str | None = None,
    sweep_params_json: str | None = None,
) -> None:
    sweep_params = (
        None if sweep_params_json is None else json.loads(sweep_params_json.removeprefix("json:"))
    )

    device = get_device()
    logger.info(f"Using device: {device}")

    config = load_config(config_path_or_obj, config_model=Config)

    if config.wandb_project:
        tags = ["modular_addition"]
        if evals_id:
            tags.append(evals_id)
        if sweep_id:
            tags.append(sweep_id)
        config = init_wandb(config, config.wandb_project, tags=tags)

    # Get output directory (automatically uses wandb run ID if available)
    out_dir = get_output_dir()

    set_seed(config.seed)
    logger.info(config)

    # Validate task config
    assert isinstance(config.task_config, ModularAdditionTaskConfig), (
        "Task config must be ModularAdditionTaskConfig for modular addition decomposition."
    )

    # Load pretrained modular addition model
    checkpoint_path = Path(__file__).parent / "full_run_data.pth"
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found at {checkpoint_path}. "
            "Run download_checkpoint.py first to download the pretrained model."
        )
    
    target_model, model_config_dict = load_pretrained_modular_addition_model(
        checkpoint_path=checkpoint_path,
        epoch=40000,  # Use grokked model from epoch 40k
        device=device
    )

    if config.wandb_project:
        assert wandb.run, "wandb.run must be initialized before training"
        if config.wandb_run_name:
            wandb.run.name = config.wandb_run_name

    save_file(config.model_dump(mode="json"), out_dir / "final_config.yaml")
    if sweep_params:
        save_file(sweep_params, out_dir / "sweep_params.yaml")
    if config.wandb_project:
        wandb.save(str(out_dir / "final_config.yaml"), base_path=out_dir, policy="now")
        if sweep_params:
            wandb.save(str(out_dir / "sweep_params.yaml"), base_path=out_dir, policy="now")

    save_target_model_info(
        save_to_wandb=config.wandb_project is not None,
        out_dir=out_dir,
        model=target_model,
        model_config_dict=model_config_dict,
    )

    # Create dataset using the same config as the pretrained model
    grokking_config = GrokkingConfig(**model_config_dict)
    train_data, test_data = gen_train_test(grokking_config)
    
    # Create datasets
    train_dataset = ModularAdditionDataset(train_data, device)
    eval_dataset = ModularAdditionDataset(test_data, device)
    
    # Create data loaders
    train_loader = DatasetGeneratedDataLoader(train_dataset, batch_size=config.batch_size, shuffle=False)
    eval_loader = DatasetGeneratedDataLoader(eval_dataset, batch_size=config.batch_size, shuffle=False)

    optimize(
        target_model=target_model,
        config=config,
        device=device,
        train_loader=train_loader,
        eval_loader=eval_loader,
        n_eval_steps=config.n_eval_steps,
        out_dir=out_dir,
        tied_weights=None,
        evals_id=evals_id,
    )

    if config.wandb_project:
        wandb.finish()


if __name__ == "__main__":
    fire.Fire(main)