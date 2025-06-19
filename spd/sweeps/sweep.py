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

import subprocess
import textwrap
from pathlib import Path
from typing import Any

import fire
import wandb
import yaml

from spd.registry import EXPERIMENT_REGISTRY
from spd.settings import REPO_ROOT


def get_sweep_configuration(decomp_script: Path, config_path: Path) -> dict[str, Any]:
    """Create a sweep configuration dictionary for the given experiment.

    Args:
        decomp_script: Path to the decomposition script
        config_path: Path to the configuration YAML file

    Returns:
        Dictionary containing the sweep configuration
    """
    # Load sweep parameters from YAML file
    sweep_params_path = REPO_ROOT / "spd/sweeps/sweep_params.yaml"
    with open(sweep_params_path) as f:
        sweep_params = yaml.safe_load(f)

    # Build the full sweep configuration
    sweep_config = {
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
) -> None:
    """Create a wandb sweep and deploy SLURM agents to run it.

    Args:
        experiment: Experiment name. See spd/registry.py for available experiments.
            Currently: tms_5-2, tms_5-2-id, tms_40-10, tms_40-10-id, resid_mlp1, resid_mlp2,
            resid_mlp3, ss_emb
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

    print(f"Using sweep config for {experiment}:")
    print(f"  Decomposition script: {decomp_script}")
    print(f"  Config file: {config_path}")
    print(f"  Project: {config.wandb_project}")

    # Create sweep configuration dictionary
    sweep_config_dict = get_sweep_configuration(
        decomp_script=decomp_script, config_path=config_path
    )

    # Create the sweep using wandb API
    sweep_id = wandb.sweep(sweep=sweep_config_dict, project=config.wandb_project)

    api = wandb.Api()
    org_name = api.settings["entity"]
    project_name = api.settings["project"]

    # Construct the full agent ID for the sweep
    wandb_url = f"https://wandb.ai/{org_name}/{project_name}/sweeps/{sweep_id}"

    print(f"Deploying {n_agents} agents for experiment {experiment}...")

    # Set up SLURM job configuration
    gpu_config = "#SBATCH --gres=gpu:0" if cpu else "#SBATCH --gres=gpu:1"
    job_name = f"spd-sweep-{job_suffix}" if job_suffix else "spd-sweep"

    # Ensure SLURM logs directory exists
    slurm_logs_dir = Path.home() / "slurm_logs"
    slurm_logs_dir.mkdir(exist_ok=True)

    agent_id = f"{org_name}/{project_name}/{sweep_id}"

    # Create the run_agent.sh script
    run_agent_script = Path.home() / "run_agent.sh"
    script_content = textwrap.dedent(f"""
        #!/bin/bash
        #SBATCH --nodes=1
        {gpu_config}
        #SBATCH --time=24:00:00
        #SBATCH --job-name={job_name}
        #SBATCH --partition=all
        #SBATCH --output={Path.home()}/slurm_logs/slurm-%j.out

        # Change to the SPD repository directory
        cd {REPO_ROOT}

        # This is the actual command that runs in each SLURM job
        wandb agent {agent_id}
    """).strip()

    with open(run_agent_script, "w") as f:
        f.write(script_content)

    # Make script executable
    run_agent_script.chmod(0o755)

    # Submit the job n times to create n parallel agents
    job_ids = []
    for _ in range(n_agents):
        result = subprocess.run(
            ["sbatch", str(run_agent_script)], capture_output=True, text=True, check=True
        )
        # Extract job ID from sbatch output (format: "Submitted batch job 12345")
        job_id = result.stdout.strip().split()[-1]
        job_ids.append(job_id)

    print(f"Job IDs: {', '.join(job_ids)}")
    print("View logs in: ~/slurm_logs/slurm-<job_id>.out")
    print(f"\nSweep URL: {wandb_url}")


def cli():
    """Command line interface for the sweep script."""
    fire.Fire(main)


if __name__ == "__main__":
    cli()
