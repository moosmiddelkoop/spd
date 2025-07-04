"""Python sweep script for running SPD experiments on a SLURM cluster.

This script creates a wandb sweep and deploys multiple SLURM agents to run it.
This script is an entrypoint of the spd package, and thus can be called with `spd-sweep` once
the package is installed.

Usage:
    spd-sweep <experiment> <num_agents> [--job_suffix <suffix>] [--cpu] [--sweep_params_file <file>]

Examples:
    spd-sweep tms_5-2 5                                        # Run TMS 5-2 sweep with 5 GPU agents
    spd-sweep resid_mlp1 3 --cpu                               # Run ResidMLP1 sweep with 3 CPU agents
    spd-sweep ss_emb 2 --job_suffix test                       # Run with job suffix
    spd-sweep tms_5-2 4 --sweep_params_file my_params.yaml     # Use custom sweep params file

Before running, update the default sweep parameters YAML file or
create a new sweep.yaml and pass it with --sweep_params_file.
"""

import tempfile
from pathlib import Path
from typing import Any

import fire
import wandb
import yaml

from spd.git_utils import create_git_snapshot
from spd.registry import EXPERIMENT_REGISTRY
from spd.settings import REPO_ROOT
from spd.slurm_utils import create_slurm_array_script, format_runtime_str, submit_slurm_array


def resolve_sweep_params_path(sweep_params_file: str) -> Path:
    """Resolve the full path to the sweep parameters file.

    Args:
        sweep_params_file: Filename or path to sweep params file
            - "my_sweep" -> <repo_root>/spd/sweeps/my_sweep.yaml
            - "my_sweep.yaml" -> <repo_root>/spd/sweeps/my_sweep.yaml
            - "experiments/sweep.yaml" -> <repo_root>/experiments/sweep.yaml

    Returns:
        Full resolved path to the sweep params file
    """
    # Add .yaml extension if not present
    if not sweep_params_file.endswith((".yaml", ".yml")):
        sweep_params_file = f"{sweep_params_file}.yaml"

    # Handle sweep params path
    if "/" not in sweep_params_file:
        return REPO_ROOT / "spd/sweeps" / sweep_params_file
    else:
        return REPO_ROOT / sweep_params_file


def get_sweep_configuration(
    decomp_script: Path,
    config_path: Path,
    experiment_name: str,
    sweep_params_file: str = "sweep_params.yaml",
) -> dict[str, Any]:
    """Create a sweep configuration dictionary for the given experiment.

    Args:
        decomp_script: Path to the decomposition script
        config_path: Path to the configuration YAML file
        experiment_name: Name of the experiment for the sweep
        sweep_params_file: Sweep parameters YAML file (default: sweep_params.yaml)

    Returns:
        Dictionary containing the sweep configuration
    """
    sweep_params_path = resolve_sweep_params_path(sweep_params_file)

    with open(sweep_params_path) as f:
        sweep_params = yaml.safe_load(f)

    # Build the full sweep configuration
    sweep_config = {
        "name": experiment_name,
        "program": str(decomp_script),
        "command": ["${env}", "${interpreter}", "${program}", str(config_path)],
    }

    # Inject the loaded parameters into the configuration
    sweep_config.update(sweep_params)

    return sweep_config


def main(
    experiment: str,
    n_agents: int,
    job_suffix: str = "",
    cpu: bool = False,
    sweep_params_file: str = "sweep_params.yaml",
) -> None:
    """Create a wandb sweep and deploy SLURM agents to run it.

    Args:
        experiment: Experiment name. See spd/registry.py for available experiments.
            Currently: tms_5-2, tms_5-2-id, tms_40-10, tms_40-10-id, resid_mlp1, resid_mlp2,
            resid_mlp3, ss_emb
        n_agents: Number of SLURM agents to deploy for the sweep (must be positive)
        job_suffix: Optional suffix to add to SLURM job names for identification
        cpu: Use CPU instead of GPU (default: False, uses GPU)
        sweep_params_file: Sweep parameters YAML file (default: sweep_params.yaml)

    """
    if n_agents <= 0:
        raise ValueError("Please supply a positive integer for agents.")

    config = EXPERIMENT_REGISTRY[experiment]
    decomp_script = REPO_ROOT / config.decomp_script
    config_path = REPO_ROOT / config.config_path

    # Resolve the full path for sweep params file
    sweep_params_full_path = resolve_sweep_params_path(sweep_params_file)

    print(f"Using sweep config for {experiment}:")
    print(f"  Decomposition script: {decomp_script}")
    print(f"  Config file: {config_path}")
    print(f"  Sweep params file: {sweep_params_full_path}")

    sweep_config_dict = get_sweep_configuration(
        decomp_script=decomp_script,
        config_path=config_path,
        experiment_name=experiment,
        sweep_params_file=sweep_params_file,
    )

    sweep_id = wandb.sweep(sweep=sweep_config_dict, project="spd")

    api = wandb.Api()
    org_name = api.settings["entity"]
    project_name = api.settings["project"]

    # Construct the full agent ID for the sweep
    wandb_url = f"https://wandb.ai/{org_name}/{project_name}/sweeps/{sweep_id}"

    print(f"Deploying {n_agents} agents for experiment {experiment}...")

    # Create single git snapshot for all agents
    snapshot_branch = create_git_snapshot(branch_name_prefix="sweep")
    print(f"Using git snapshot: {snapshot_branch}")

    # Build job name with expected runtime or custom suffix
    runtime_str = format_runtime_str(config.expected_runtime)
    job_name = f"spd-sweep-{job_suffix}" if job_suffix else f"spd-sweep-{runtime_str}"

    agent_id = f"{org_name}/{project_name}/{sweep_id}"
    print(f"Agent ID: {agent_id}")
    command = f"wandb agent {agent_id}"

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        run_agent_script = temp_path / "run_agent.sh"

        # Create list of identical commands for each agent
        commands = [command] * n_agents

        create_slurm_array_script(
            script_path=run_agent_script,
            job_name=job_name,
            commands=commands,
            cpu=cpu,
            snapshot_branch=snapshot_branch,
        )

        job_id = submit_slurm_array(run_agent_script)

        print(f"Job Array ID: {job_id}")
        print(f"\nView logs in: ~/slurm_logs/slurm-{job_id}_*.out")
        print(f"Sweep URL: {wandb_url}")


def cli():
    """Command line interface for the sweep script."""
    fire.Fire(main)


if __name__ == "__main__":
    cli()
