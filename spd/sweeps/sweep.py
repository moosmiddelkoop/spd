"""Python sweep script for running SPD experiments on a SLURM cluster.

This script creates a wandb sweep and deploys multiple SLURM agents to run it.
This script is an entrypoint of the spd package, and thus can be called with `spd-sweep` once
the package is installed.

Usage:
    spd-sweep <experiment> <num_agents> [--job_suffix <suffix>] [--cpu]

Examples:
    spd-sweep tms_5-2 5                    # Run TMS 5-2 sweep with 5 GPU agents
    spd-sweep resid_mlp1 3 --cpu           # Run ResidMLP1 sweep with 3 CPU agents
    spd-sweep ss_emb 2 --job_suffix test   # Run with job suffix

Before running, update the spd/sweeps/sweep_params.yaml file with the desired parameters.
"""

import tempfile
from pathlib import Path
from typing import Any

import wandb
import yaml

from spd.git_utils import create_git_snapshot
from spd.registry import EXPERIMENT_REGISTRY
from spd.settings import REPO_ROOT
from spd.slurm_utils import create_slurm_script, submit_slurm_jobs

WANDB_ORG = "goodfire"
WANDB_PROJECT = "oli-spd"


def get_sweep_configuration(
    decomp_script: Path, base_config_path: Path, sweep_params_path: Path
) -> dict[str, Any]:
    """Create a sweep configuration dictionary for the given experiment.

    Args:
        decomp_script: Path to the decomposition script
        base_config_path: Path to the configuration YAML file

    Returns:
        Dictionary containing the sweep configuration
    """
    # Load sweep parameters from YAML file
    with open(sweep_params_path) as f:
        sweep_params = yaml.safe_load(f)

    # Build the full sweep configuration
    sweep_config = {
        "program": str(decomp_script),
        "command": ["${env}", "${interpreter}", "${program}", str(base_config_path)],
    }

    # Inject the loaded parameters into the configuration
    sweep_config.update(sweep_params)

    return sweep_config


def main(
    experiment: str,
    n_agents: int,
    sweep_params_path: Path = REPO_ROOT / "spd/sweeps/sweep_params.yaml",
    job_suffix: str = "",
    cpu: bool = False,
) -> None:
    """Create a wandb sweep and deploy SLURM agents to run it.

    Args:
        experiment: Experiment name. See spd/registry.py for available experiments.
            Currently: tms_5-2, tms_5-2-id, tms_40-10, tms_40-10-id, resid_mlp1, resid_mlp2,
            resid_mlp3, ss_emb, gemma
        n_agents: Number of SLURM agents to deploy for the sweep (must be positive)
        job_suffix: Optional suffix to add to SLURM job names for identification
        cpu: Use CPU instead of GPU (default: False, uses GPU)

    """
    # Validate arguments
    if n_agents <= 0:
        raise ValueError("Please supply a positive integer for agents.")

    # Get experiment configuration from registry
    config = EXPERIMENT_REGISTRY[experiment]
    decomp_script = REPO_ROOT / config.decomp_script
    config_path = REPO_ROOT / config.config_path
    job_suffix = job_suffix or config.expected_runtime

    print(f"Using sweep config for {experiment}:")
    print(f"  Decomposition script: {decomp_script}")
    print(f"  Config file: {config_path}")

    # Create sweep configuration dictionary
    sweep_config_dict = get_sweep_configuration(decomp_script, config_path, sweep_params_path)

    # Create the sweep using wandb API
    sweep_id = wandb.sweep(sweep=sweep_config_dict, project=WANDB_PROJECT)

    # Construct the full agent ID for the sweep
    wandb_url = f"https://wandb.ai/{WANDB_ORG}/{WANDB_PROJECT}/sweeps/{sweep_id}"

    print(f"Deploying {n_agents} agents for experiment {experiment}...")

    # Create single git snapshot for all agents
    snapshot_branch = create_git_snapshot(branch_name_prefix="sweep")
    print(f"Using git snapshot: {snapshot_branch}")

    job_name = f"spd-sweep-{job_suffix}" if job_suffix else "spd-sweep"
    agent_id = f"{WANDB_ORG}/{WANDB_PROJECT}/{sweep_id}"
    command = f"export WANDB_DISABLE_SERVICE=true && wandb agent {agent_id}"

    # Use a temporary directory for the agent script
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        run_agent_script = temp_path / "run_agent.sh"

        create_slurm_script(
            script_path=run_agent_script,
            job_name=job_name,
            command=command,
            cpu=cpu,
            snapshot_branch=snapshot_branch,
        )

        # Submit the job n times to create n parallel agents
        script_paths = [run_agent_script] * n_agents
        job_ids = submit_slurm_jobs(script_paths)

        print(f"Job IDs: {', '.join(job_ids)}")
        print("\nView logs in: ./slurm_logs/.../slurm-<job_id>.out")
        print(f"Sweep URL: {wandb_url}")
        # Temporary directory and script file are automatically cleaned up here


if __name__ == "__main__":
    main(
        experiment="ss_attn",
        n_agents=4,
        sweep_params_path=REPO_ROOT / "spd/sweeps/sweep_params_ml.yaml",
        job_suffix="oli-4h",
    )
