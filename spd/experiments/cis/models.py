from pathlib import Path
from typing import Any, Self, override

import torch
import wandb
import yaml
from jaxtyping import Float
from pydantic import BaseModel, ConfigDict, PositiveInt, model_validator
from torch import Tensor, nn
from torch.nn import functional as F
from wandb.apis.public import Run

from spd.spd_types import WANDB_PATH_PREFIX, ModelPath
from spd.utils.run_utils import check_run_exists
from spd.utils.wandb_utils import (
    download_wandb_file,
    fetch_latest_wandb_checkpoint,
    fetch_wandb_run_dir,
)


class CISModelPaths(BaseModel):
    """Paths to output files from a CISModel training run."""

    cis_train_config: Path
    checkpoint: Path


class CISModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    n_features: PositiveInt
    n_hidden: PositiveInt
    device: str

    @model_validator(mode="after")
    def validate_device(self) -> Self:
        valid_devices = ["cpu", "cuda", "mps"]
        if self.device not in valid_devices and not self.device.startswith("cuda:"):
            raise ValueError(f"device must be one of {valid_devices} or 'cuda:N'")
        return self


class CISModel(nn.Module):
    """Computation in Superposition model: x -> ReLU(W1*x) -> ReLU(W2*h + b)"""

    def __init__(self, config: CISModelConfig):
        super().__init__()
        self.config = config

        # W1: input to hidden (no bias)
        self.W1 = nn.Linear(config.n_features, config.n_hidden, bias=False)

        # W2: hidden to output (with bias)
        self.W2 = nn.Linear(config.n_hidden, config.n_features, bias=True)

    @override
    def forward(
        self, x: Float[Tensor, "... n_features"], **_: Any
    ) -> Float[Tensor, "... n_features"]:
        # h = ReLU(W1 * x)
        h = F.relu(self.W1(x))

        # y' = ReLU(W2 * h + b)
        y_prime = F.relu(self.W2(h))

        return y_prime

    @staticmethod
    def _download_wandb_files(wandb_project_run_id: str) -> CISModelPaths:
        """Download the relevant files from a wandb run."""
        api = wandb.Api()
        run: Run = api.run(wandb_project_run_id)
        run_dir = fetch_wandb_run_dir(run.id)

        cis_model_config_path = download_wandb_file(run, run_dir, "cis_train_config.yaml")

        checkpoint = fetch_latest_wandb_checkpoint(run)
        checkpoint_path = download_wandb_file(run, run_dir, checkpoint.name)
        return CISModelPaths(cis_train_config=cis_model_config_path, checkpoint=checkpoint_path)

    @classmethod
    def from_pretrained(cls, path: ModelPath) -> tuple["CISModel", dict[str, Any]]:
        """Fetch a pretrained model from wandb or a local path to a checkpoint.

        Args:
            path: The path to local checkpoint or wandb project. If a wandb project, format must be
                `wandb:<entity>/<project>/<run_id>` or `wandb:<entity>/<project>/runs/<run_id>`.
                If `api.entity` is set (e.g. via setting WANDB_ENTITY in .env), <entity> can be
                omitted, and if `api.project` is set, <project> can be omitted. If local path,
                assumes that `cis_train_config.yaml` is in the same directory as the checkpoint.

        Returns:
            model: The pretrained CISModel
            cis_model_config_dict: The config dict used to train the model
        """
        if isinstance(path, str) and path.startswith(WANDB_PATH_PREFIX):
            # Check if run exists in shared filesystem first
            run_dir = check_run_exists(path)
            if run_dir:
                # Use local files from shared filesystem
                paths = CISModelPaths(
                    cis_train_config=run_dir / "cis_train_config.yaml",
                    checkpoint=run_dir / "cis.pth",
                )
            else:
                # Download from wandb
                wandb_path = path.removeprefix(WANDB_PATH_PREFIX)
                paths = cls._download_wandb_files(wandb_path)
        else:
            # `path` should be a local path to a checkpoint
            paths = CISModelPaths(
                cis_train_config=Path(path).parent / "cis_train_config.yaml",
                checkpoint=Path(path),
            )

        # Validate paths exist
        if not paths.cis_train_config.exists():
            raise FileNotFoundError(f"Config file not found: {paths.cis_train_config}")
        if not paths.checkpoint.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {paths.checkpoint}")

        # Load and validate config
        try:
            with open(paths.cis_train_config) as f:
                cis_train_config_dict = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ValueError(f"Failed to parse config file {paths.cis_train_config}: {e}") from e
        except Exception as e:
            raise OSError(f"Failed to read config file {paths.cis_train_config}: {e}") from e

        if not isinstance(cis_train_config_dict, dict):
            raise ValueError("Config file must contain a dictionary")

        if "cis_model_config" not in cis_train_config_dict:
            raise ValueError("Config file missing required 'cis_model_config' key")

        # Create model
        try:
            cis_config = CISModelConfig(**cis_train_config_dict["cis_model_config"])
            cis = cls(config=cis_config)
        except Exception as e:
            raise ValueError(f"Failed to create model config: {e}") from e

        # Load checkpoint
        try:
            params = torch.load(paths.checkpoint, weights_only=True, map_location="cpu")
        except Exception as e:
            raise OSError(f"Failed to load checkpoint {paths.checkpoint}: {e}") from e

        # Load state dict with validation
        try:
            cis.load_state_dict(params)
        except Exception as e:
            raise ValueError(
                f"Failed to load model weights - checkpoint may be incompatible: {e}"
            ) from e

        return cis, cis_train_config_dict
