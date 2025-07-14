"""Registry of all experiments."""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from spd.utils.target_ci_solutions import DenseCIPattern, IdentityCIPattern, TargetCISolution


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment.

    Attributes:
        decomp_script: Path to the decomposition script
        config_path: Path to the configuration YAML file
        expected_runtime: Expected runtime of the experiment in minutes. Used for SLURM job names.
        target_solution: Optional target solution for evaluating SPD convergence.
    """

    experiment_type: Literal["tms", "resid_mlp", "lm"]
    decomp_script: Path
    config_path: Path
    expected_runtime: int
    target_solution: TargetCISolution | None = None


EXPERIMENT_REGISTRY: dict[str, ExperimentConfig] = {
    "tms_5-2": ExperimentConfig(
        experiment_type="tms",
        decomp_script=Path("spd/experiments/tms/tms_decomposition.py"),
        config_path=Path("spd/experiments/tms/tms_5-2_config.yaml"),
        expected_runtime=4,
        target_solution=TargetCISolution(
            {"linear1": IdentityCIPattern(n_features=5), "linear2": IdentityCIPattern(n_features=5)}
        ),
    ),
    "tms_5-2-id": ExperimentConfig(
        experiment_type="tms",
        decomp_script=Path("spd/experiments/tms/tms_decomposition.py"),
        config_path=Path("spd/experiments/tms/tms_5-2-id_config.yaml"),
        expected_runtime=4,
        target_solution=TargetCISolution(
            {
                "linear1": IdentityCIPattern(n_features=5),
                "linear2": IdentityCIPattern(n_features=5),
                "hidden_layers.0": DenseCIPattern(k=2),
            }
        ),
    ),
    "tms_40-10": ExperimentConfig(
        experiment_type="tms",
        decomp_script=Path("spd/experiments/tms/tms_decomposition.py"),
        config_path=Path("spd/experiments/tms/tms_40-10_config.yaml"),
        expected_runtime=5,
        target_solution=TargetCISolution(
            {"linear1": IdentityCIPattern(n_features=40), "linear2": IdentityCIPattern(n_features=40)}
        ),
    ),
    "tms_40-10-id": ExperimentConfig(
        experiment_type="tms",
        decomp_script=Path("spd/experiments/tms/tms_decomposition.py"),
        config_path=Path("spd/experiments/tms/tms_40-10-id_config.yaml"),
        expected_runtime=5,
        target_solution=TargetCISolution(
            {
                "linear1": IdentityCIPattern(n_features=40),
                "linear2": IdentityCIPattern(n_features=40),
                "hidden_layers.0": DenseCIPattern(k=10),
            }
        ),
    ),
    "resid_mlp1": ExperimentConfig(
        experiment_type="resid_mlp",
        decomp_script=Path("spd/experiments/resid_mlp/resid_mlp_decomposition.py"),
        config_path=Path("spd/experiments/resid_mlp/resid_mlp1_config.yaml"),
        expected_runtime=3,
        target_solution=TargetCISolution(
            {
                "layers.0.mlp_in": IdentityCIPattern(n_features=100),
                "layers.0.mlp_out": DenseCIPattern(k=50),
            }
        ),
    ),
    "resid_mlp2": ExperimentConfig(
        experiment_type="resid_mlp",
        decomp_script=Path("spd/experiments/resid_mlp/resid_mlp_decomposition.py"),
        config_path=Path("spd/experiments/resid_mlp/resid_mlp2_config.yaml"),
        expected_runtime=11,
        target_solution=TargetCISolution(
            {
                "layers.*.mlp_in": IdentityCIPattern(n_features=100),
                "layers.*.mlp_out": DenseCIPattern(k=25),
            },
            expected_matches=4
        ),
    ),
    "resid_mlp3": ExperimentConfig(
        experiment_type="resid_mlp",
        decomp_script=Path("spd/experiments/resid_mlp/resid_mlp_decomposition.py"),
        config_path=Path("spd/experiments/resid_mlp/resid_mlp3_config.yaml"),
        expected_runtime=60,
        target_solution=TargetCISolution(
            {
                "layers.*.mlp_in": IdentityCIPattern(n_features=102),
                "layers.*.mlp_out": DenseCIPattern(k=17),
            },
            expected_matches=6
        ),
    ),
    # "ss_emb": ExperimentConfig(
    #     experiment_type="lm",
    #     decomp_script=Path("spd/experiments/lm/lm_decomposition.py"),
    #     config_path=Path("spd/experiments/lm/ss_emb_config.yaml"),
    #     expected_runtime=60,
    # ),
}


def has_ci_solution(experiment_id: str) -> bool:
    """Check if an experiment has a target CI solution defined."""
    return (
        experiment_id in EXPERIMENT_REGISTRY 
        and EXPERIMENT_REGISTRY[experiment_id].target_solution is not None
    )
