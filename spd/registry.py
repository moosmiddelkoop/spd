"""Registry of all experiments."""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment.

    Attributes:
        decomp_script: Path to the decomposition script
        config_path: Path to the configuration YAML file
        expected_runtime: Expected runtime of the experiment. Used for SLURM job names.
    """

    experiment_type: Literal["tms", "resid_mlp", "lm"]
    decomp_script: Path
    config_path: Path
    expected_runtime: str


EXPERIMENT_REGISTRY: dict[str, ExperimentConfig] = {
    "tms_5-2": ExperimentConfig(
        experiment_type="tms",
        decomp_script=Path("spd/experiments/tms/tms_decomposition.py"),
        config_path=Path("spd/experiments/tms/tms_5-2_config.yaml"),
        expected_runtime="4m",
    ),
    "tms_5-2-id": ExperimentConfig(
        experiment_type="tms",
        decomp_script=Path("spd/experiments/tms/tms_decomposition.py"),
        config_path=Path("spd/experiments/tms/tms_5-2-id_config.yaml"),
        expected_runtime="4m",
    ),
    "tms_40-10": ExperimentConfig(
        experiment_type="tms",
        decomp_script=Path("spd/experiments/tms/tms_decomposition.py"),
        config_path=Path("spd/experiments/tms/tms_40-10_config.yaml"),
        expected_runtime="5m",
    ),
    "tms_40-10-id": ExperimentConfig(
        experiment_type="tms",
        decomp_script=Path("spd/experiments/tms/tms_decomposition.py"),
        config_path=Path("spd/experiments/tms/tms_40-10-id_config.yaml"),
        expected_runtime="5m",
    ),
    "resid_mlp1": ExperimentConfig(
        experiment_type="resid_mlp",
        decomp_script=Path("spd/experiments/resid_mlp/resid_mlp_decomposition.py"),
        config_path=Path("spd/experiments/resid_mlp/resid_mlp1_config.yaml"),
        expected_runtime="3m",
    ),
    "resid_mlp2": ExperimentConfig(
        experiment_type="resid_mlp",
        decomp_script=Path("spd/experiments/resid_mlp/resid_mlp_decomposition.py"),
        config_path=Path("spd/experiments/resid_mlp/resid_mlp2_config.yaml"),
        expected_runtime="11m",
    ),
    "resid_mlp3": ExperimentConfig(
        experiment_type="resid_mlp",
        decomp_script=Path("spd/experiments/resid_mlp/resid_mlp_decomposition.py"),
        config_path=Path("spd/experiments/resid_mlp/resid_mlp3_config.yaml"),
        expected_runtime="1h",
    ),
    # "ss_emb": ExperimentConfig(
    #     experiment_type="lm",
    #     decomp_script=Path("spd/experiments/lm/lm_decomposition.py"),
    #     config_path=Path("spd/experiments/lm/ss_emb_config.yaml"),
    #     expected_runtime="1h",
    # ),
    "gemma_mlp_up_14": ExperimentConfig(
        experiment_type="lm",
        decomp_script=Path("spd/experiments/lm/lm_decomposition.py"),
        config_path=Path("spd/experiments/lm/gemma_config.yaml"),
        expected_runtime="10h", # roughly
    ),
}
