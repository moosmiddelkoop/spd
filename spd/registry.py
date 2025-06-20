"""Registry of all experiments."""

from dataclasses import dataclass
from pathlib import Path


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment.

    Attributes:
        wandb_project: WandB project name for the experiment
        decomp_script: Path to the decomposition script
        config_path: Path to the configuration YAML file
    """

    wandb_project: str
    decomp_script: Path
    config_path: Path
    expected_runtime: str


EXPERIMENT_REGISTRY: dict[str, ExperimentConfig] = {
    "tms_5-2": ExperimentConfig(
        wandb_project="spd-gf-tms",
        decomp_script=Path("spd/experiments/tms/tms_decomposition.py"),
        config_path=Path("spd/experiments/tms/tms_5-2_config.yaml"),
        expected_runtime="4m",
    ),
    "tms_5-2-id": ExperimentConfig(
        wandb_project="spd-gf-tms",
        decomp_script=Path("spd/experiments/tms/tms_decomposition.py"),
        config_path=Path("spd/experiments/tms/tms_5-2-id_config.yaml"),
        expected_runtime="4m",
    ),
    "tms_40-10": ExperimentConfig(
        wandb_project="spd-gf-tms",
        decomp_script=Path("spd/experiments/tms/tms_decomposition.py"),
        config_path=Path("spd/experiments/tms/tms_40-10_config.yaml"),
        expected_runtime="5m",
    ),
    "tms_40-10-id": ExperimentConfig(
        wandb_project="spd-gf-tms",
        decomp_script=Path("spd/experiments/tms/tms_decomposition.py"),
        config_path=Path("spd/experiments/tms/tms_40-10-id_config.yaml"),
        expected_runtime="5m",
    ),
    "resid_mlp1": ExperimentConfig(
        wandb_project="spd-gf-resid-mlp",
        decomp_script=Path("spd/experiments/resid_mlp/resid_mlp_decomposition.py"),
        config_path=Path("spd/experiments/resid_mlp/resid_mlp1_config.yaml"),
        expected_runtime="3m",
    ),
    "resid_mlp2": ExperimentConfig(
        wandb_project="spd-gf-resid-mlp",
        decomp_script=Path("spd/experiments/resid_mlp/resid_mlp_decomposition.py"),
        config_path=Path("spd/experiments/resid_mlp/resid_mlp2_config.yaml"),
        expected_runtime="11m",
    ),
    "resid_mlp3": ExperimentConfig(
        wandb_project="spd-gf-resid-mlp",
        decomp_script=Path("spd/experiments/resid_mlp/resid_mlp_decomposition.py"),
        config_path=Path("spd/experiments/resid_mlp/resid_mlp3_config.yaml"),
        expected_runtime="1h",
    ),
    "ss_emb": ExperimentConfig(
        wandb_project="spd-gf-lm",
        decomp_script=Path("spd/experiments/lm/lm_decomposition.py"),
        config_path=Path("spd/experiments/lm/ss_emb_config.yaml"),
        expected_runtime="1h",
    ),
}
