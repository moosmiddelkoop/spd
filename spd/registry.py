"""Registry of all experiments."""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from spd.utils.target_solutions import DenseColumnsPattern, IdentityPattern, TargetSolution


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment.

    Attributes:
        decomp_script: Path to the decomposition script
        config_path: Path to the configuration YAML file
        expected_runtime: Expected runtime of the experiment in minutes. Used for SLURM job names.
    """

    experiment_type: Literal["tms", "resid_mlp", "lm"]
    decomp_script: Path
    config_path: Path
    expected_runtime: int


EXPERIMENT_REGISTRY: dict[str, ExperimentConfig] = {
    "tms_5-2": ExperimentConfig(
        experiment_type="tms",
        decomp_script=Path("spd/experiments/tms/tms_decomposition.py"),
        config_path=Path("spd/experiments/tms/tms_5-2_config.yaml"),
        expected_runtime=4,
    ),
    "tms_5-2-id": ExperimentConfig(
        experiment_type="tms",
        decomp_script=Path("spd/experiments/tms/tms_decomposition.py"),
        config_path=Path("spd/experiments/tms/tms_5-2-id_config.yaml"),
        expected_runtime=4,
    ),
    "tms_40-10": ExperimentConfig(
        experiment_type="tms",
        decomp_script=Path("spd/experiments/tms/tms_decomposition.py"),
        config_path=Path("spd/experiments/tms/tms_40-10_config.yaml"),
        expected_runtime=5,
    ),
    "tms_40-10-id": ExperimentConfig(
        experiment_type="tms",
        decomp_script=Path("spd/experiments/tms/tms_decomposition.py"),
        config_path=Path("spd/experiments/tms/tms_40-10-id_config.yaml"),
        expected_runtime=5,
    ),
    "resid_mlp1": ExperimentConfig(
        experiment_type="resid_mlp",
        decomp_script=Path("spd/experiments/resid_mlp/resid_mlp_decomposition.py"),
        config_path=Path("spd/experiments/resid_mlp/resid_mlp1_config.yaml"),
        expected_runtime=3,
    ),
    "resid_mlp2": ExperimentConfig(
        experiment_type="resid_mlp",
        decomp_script=Path("spd/experiments/resid_mlp/resid_mlp_decomposition.py"),
        config_path=Path("spd/experiments/resid_mlp/resid_mlp2_config.yaml"),
        expected_runtime=11,
    ),
    "resid_mlp3": ExperimentConfig(
        experiment_type="resid_mlp",
        decomp_script=Path("spd/experiments/resid_mlp/resid_mlp_decomposition.py"),
        config_path=Path("spd/experiments/resid_mlp/resid_mlp3_config.yaml"),
        expected_runtime=60,
    ),
    # "ss_emb": ExperimentConfig(
    #     experiment_type="lm",
    #     decomp_script=Path("spd/experiments/lm/lm_decomposition.py"),
    #     config_path=Path("spd/experiments/lm/ss_emb_config.yaml"),
    #     expected_runtime=60,
    # ),
}


SOLUTION_REGISTRY = {
    "tms_5-2": TargetSolution(
        {"linear1": IdentityPattern(n_features=5), "linear2": IdentityPattern(n_features=5)}
    ),
    "tms_5-2-id": TargetSolution(
        {
            "linear1": IdentityPattern(n_features=5),
            "linear2": IdentityPattern(n_features=5),
            "hidden_layers.0": DenseColumnsPattern(k=2),
        }
    ),
    "tms_40-10": TargetSolution(
        {"linear1": IdentityPattern(n_features=40), "linear2": IdentityPattern(n_features=40)}
    ),
    "tms_40-10-id": TargetSolution(
        {
            "linear1": IdentityPattern(n_features=40),
            "linear2": IdentityPattern(n_features=40),
            "hidden_layers.0": DenseColumnsPattern(k=10),
        }
    ),
    "resid_mlp1": TargetSolution(
        {
            "layers.0.mlp_in": IdentityPattern(n_features=100),
            "layers.0.mlp_out": DenseColumnsPattern(k=50),
        }
    ),
    "resid_mlp2": TargetSolution(
        {
            "layers.0.mlp_in": IdentityPattern(n_features=100),
            "layers.0.mlp_out": DenseColumnsPattern(k=25),
            "layers.1.mlp_in": IdentityPattern(n_features=100),
            "layers.1.mlp_out": DenseColumnsPattern(k=25),
        }
    ),
    "resid_mlp3": TargetSolution(
        {
            "layers.0.mlp_in": IdentityPattern(n_features=102),
            "layers.0.mlp_out": DenseColumnsPattern(k=17),
            "layers.1.mlp_in": IdentityPattern(n_features=102),
            "layers.1.mlp_out": DenseColumnsPattern(k=17),
            "layers.2.mlp_in": IdentityPattern(n_features=102),
            "layers.2.mlp_out": DenseColumnsPattern(k=17),
        }
    ),
}
