"""Computation in Superposition model training from the Toy Models of Superposition paper."""

from pathlib import Path
from typing import Literal, Self

import numpy as np
import torch
import wandb
from pydantic import BaseModel, ConfigDict, PositiveInt, model_validator
from tqdm import tqdm, trange

from spd.experiments.cis.models import CISModel, CISModelConfig
from spd.experiments.cis.plotting import (
    create_raw_weights_heatmap,
    create_stacked_weight_plot,
)
from spd.log import logger
from spd.utils.general_utils import set_seed
from spd.utils.run_utils import get_output_dir, save_file


class CISDataset:
    """Dataset for Computation in Superposition experiment."""

    def __init__(
        self,
        n_features: int,
        feature_sparsity: float,
        importance_decay: float,
        device: str,
        value_range: tuple[float, float],
        batch_size: int,
    ):
        # Input validation
        if n_features <= 0:
            raise ValueError("n_features must be positive")
        if not (0.0 <= feature_sparsity <= 1.0):
            raise ValueError("feature_sparsity must be between 0 and 1")
        if importance_decay <= 0:
            raise ValueError("importance_decay must be positive")
        if len(value_range) != 2 or value_range[0] >= value_range[1]:
            raise ValueError("value_range must be a tuple (min, max) with min < max")

        self.n_features = n_features
        self.feature_sparsity = feature_sparsity
        self.importance_decay = importance_decay
        self.device = device
        self.value_range = value_range
        # Create feature importance weights: I_i = importance_decay^i
        self.feature_importance = torch.tensor(
            [importance_decay**i for i in range(n_features)], device=device, dtype=torch.float32
        )
        self.importance = self.feature_importance.unsqueeze(0).expand(batch_size, -1)

    def generate_batch(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate a batch of data.

        Returns:
            x: Input vectors (sparse)
            y: Target outputs (abs(x))
        """
        # Generate sparse features: probability of being non-zero is (1 - feature_sparsity)
        active_mask = (
            torch.rand(batch_size, self.n_features, device=self.device) > self.feature_sparsity
        )
        values = (
            torch.rand(batch_size, self.n_features, device=self.device)
            * (self.value_range[1] - self.value_range[0])
            + self.value_range[0]
        )
        x = values * active_mask

        # Target is element-wise absolute value
        y = torch.abs(x)

        return x, y


class CISTrainConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    wandb_project: str | None = None
    cis_model_config: CISModelConfig
    feature_sparsity: float
    importance_decay: float
    batch_size: PositiveInt
    steps: PositiveInt
    seed: int
    lr: float
    lr_schedule: Literal["linear", "cosine", "constant"]
    # Dataset parameters
    value_range: tuple[float, float]
    # Training parameters
    print_freq: PositiveInt
    weight_decay: float

    @model_validator(mode="after")
    def validate_model(self) -> Self:
        if not (0.0 <= self.feature_sparsity <= 1.0):
            raise ValueError("feature_sparsity must be between 0 and 1")
        if self.importance_decay <= 0:
            raise ValueError("importance_decay must be positive")
        if self.importance_decay >= 1.0:
            raise ValueError("importance_decay should be less than 1.0 for proper decay")
        if self.weight_decay < 0:
            raise ValueError("weight_decay must be non-negative")
        if self.lr <= 0:
            raise ValueError("learning rate must be positive")
        if self.value_range[0] >= self.value_range[1]:
            raise ValueError("value_range must have min < max")
        if self.print_freq <= 0:
            raise ValueError("print_freq must be positive")
        return self


def linear_lr(step: int, steps: int) -> float:
    """Linear learning rate decay from 1.0 to 0.0."""
    return 1 - (step / steps)


def constant_lr(*_: int) -> float:
    """Constant learning rate of 1.0."""
    return 1.0


def cosine_decay_lr(step: int, steps: int) -> float:
    """Cosine decay learning rate schedule."""
    return np.cos(0.5 * np.pi * step / (steps - 1))


def train(
    model: CISModel,
    dataset: CISDataset,
    log_wandb: bool,
    steps: int,
    batch_size: int,
    print_freq: int,
    lr: float,
    lr_schedule: Literal["linear", "cosine", "constant"],
    weight_decay: float,
) -> None:
    """Train the CIS model."""

    assert lr_schedule in ["linear", "cosine", "constant"], f"Invalid lr_schedule: {lr_schedule}"
    lr_schedule_fn = {
        "linear": linear_lr,
        "cosine": cosine_decay_lr,
        "constant": constant_lr,
    }[lr_schedule]

    opt = torch.optim.AdamW(list(model.parameters()), lr=lr, weight_decay=weight_decay)

    with trange(steps, ncols=0) as t:
        for step in t:
            step_lr = lr * lr_schedule_fn(step, steps)
            for group in opt.param_groups:
                group["lr"] = step_lr

            opt.zero_grad(set_to_none=True)

            # Generate batch
            x, y = dataset.generate_batch(batch_size)

            # Forward pass
            y_pred = model(x)

            # Weighted MSE loss: L = Σ I_i * (y_i - y'_i)^2
            error = dataset.importance * (y - y_pred) ** 2
            loss = error.mean()

            # Backward pass
            loss.backward()
            opt.step()

            if step % print_freq == 0 or (step + 1 == steps):
                tqdm.write(f"Step {step} Loss: {loss.item()}")
                t.set_postfix(
                    loss=loss.item(),
                    lr=step_lr,
                )
                if log_wandb:
                    wandb.log(
                        {"loss": loss.item(), "lr": step_lr, "weight_decay": weight_decay},
                        step=step,
                    )


def generate_plots(model: CISModel, out_dir: Path) -> list[Path]:
    """Generate and save plots after training."""
    logger.info("Generating plots...")

    # Generate stacked weight plot
    stacked_plot_path = out_dir / "stacked_weights.png"
    create_stacked_weight_plot(
        model=model,
        filepath=stacked_plot_path,
        sort_by_importance=True,
    )
    logger.info(f"Saved stacked weight plot to {stacked_plot_path}")

    # Generate raw weights heatmaps
    w1_heatmap_path = out_dir / "raw_weights_W1.png"
    create_raw_weights_heatmap(
        model=model,
        filepath=w1_heatmap_path,
        layer="W1",
    )
    logger.info(f"Saved W1 heatmap to {w1_heatmap_path}")

    w2_heatmap_path = out_dir / "raw_weights_W2.png"
    create_raw_weights_heatmap(
        model=model,
        filepath=w2_heatmap_path,
        layer="W2",
    )
    logger.info(f"Saved W2 heatmap to {w2_heatmap_path}")

    # Return list of plot paths for WandB upload
    return [
        stacked_plot_path,
        w1_heatmap_path,
        w2_heatmap_path,
    ]


def run_train(config: CISTrainConfig, device: str) -> None:
    """Run training for the CIS model."""
    model = CISModel(config=config.cis_model_config)
    model.to(device)

    dataset = CISDataset(
        n_features=config.cis_model_config.n_features,
        feature_sparsity=config.feature_sparsity,
        importance_decay=config.importance_decay,
        batch_size=config.batch_size,
        device=device,
        value_range=config.value_range,
    )

    model_cfg = config.cis_model_config
    run_name = (
        f"cis_n-features{model_cfg.n_features}_n-hidden{model_cfg.n_hidden}_"
        f"sparsity{config.feature_sparsity}_decay{config.importance_decay}_"
        f"wd{config.weight_decay}_seed{config.seed}"
    )

    out_dir = get_output_dir()

    if config.wandb_project:
        tags = [f"cis_{model_cfg.n_features}-{model_cfg.n_hidden}"]
        wandb.init(project=config.wandb_project, name=run_name, tags=tags)

    # Save config
    config_path = out_dir / "cis_train_config.yaml"
    save_file(config.model_dump(mode="json"), config_path)
    if config.wandb_project:
        wandb.save(str(config_path), base_path=out_dir, policy="now")
    logger.info(f"Saved config to {config_path}")

    train(
        model,
        dataset=dataset,
        log_wandb=config.wandb_project is not None,
        steps=config.steps,
        batch_size=config.batch_size,
        print_freq=config.print_freq,
        lr=config.lr,
        lr_schedule=config.lr_schedule,
        weight_decay=config.weight_decay,
    )

    model_path = out_dir / "cis.pth"
    save_file(model.state_dict(), model_path)
    if config.wandb_project:
        wandb.save(str(model_path), base_path=out_dir, policy="now")
    logger.info(f"Saved model to {model_path}")

    # Generate plots after training
    plot_paths = generate_plots(model, out_dir)

    # Upload plots to WandB as media
    if config.wandb_project:
        # Log plots as images in WandB
        plot_names = [
            "stacked_weights",
            "raw_weights_W1",
            "raw_weights_W2",
        ]

        plot_images = {}
        for plot_path, name in zip(plot_paths, plot_names, strict=False):
            plot_images[name] = wandb.Image(str(plot_path))
            wandb.save(str(plot_path), base_path=out_dir, policy="now")

        # Log all plots as media
        wandb.log(plot_images)
        logger.info("Uploaded plots to WandB as media")


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # CIS model with parameters from the paper: n=100 features, m=40 hidden neurons
    config = CISTrainConfig(
        wandb_project="spd",
        cis_model_config=CISModelConfig(
            n_features=100,
            n_hidden=40,
            device=device,
        ),
        feature_sparsity=0.95,  # High sparsity to observe superposition
        importance_decay=0.8,  # I_i = 0.8^i as in the paper
        batch_size=16384,
        steps=100000,
        seed=0,
        lr=1e-3,
        lr_schedule="constant",
        value_range=(-1.0, 1.0),
        print_freq=100,
        weight_decay=0.1,
    )
    # Weight decay wasn't mentioned in the paper but was needed for this implementation to semi-work

    set_seed(config.seed)
    run_train(config, device)
