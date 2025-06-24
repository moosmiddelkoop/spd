import subprocess
from datetime import datetime
from pathlib import Path
import textwrap
from time import sleep
from typing import Any

import wandb
import yaml

from spd.registry import EXPERIMENT_REGISTRY
from spd.settings import REPO_ROOT
from spd.slurm_utils import create_slurm_script


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
            resid_mlp3, gemma_mlp_up_14
        n_agents: Number of SLURM agents to deploy for the sweep (must be positive)
        job_suffix: Optional suffix to add to SLURM job names for identification
        cpu: Use CPU instead of GPU (default: False, uses GPU)

    """
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

    sweep_config_dict = get_sweep_configuration(
        decomp_script=decomp_script, config_path=config_path
    )

    print("Starting sweep...")
    sweep_id = wandb.sweep(sweep=sweep_config_dict, project="spd")

    api = wandb.Api()
    org_name = api.settings["entity"]
    project_name = api.settings["project"]

    wandb_url = f"https://wandb.ai/{org_name}/{project_name}/sweeps/{sweep_id}"

    job_name = f"spd-sweep-{job_suffix}" if job_suffix else "spd-sweep"
    agent_id = f"{org_name}/{project_name}/{sweep_id}"
    command = f"export WANDB_DISABLE_SERVICE=true && wandb agent {agent_id}"  # stops it from trying to connect to the service started by wandb.sweep above

    experiment_dir = (
        REPO_ROOT / "out" / f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_{experiment}"
    )
    experiment_dir.mkdir(exist_ok=True, parents=True)

    print("Writing SLURM script...")
    script_text = create_slurm_script(
        output_dir=experiment_dir,
        job_name=job_name,
        command=command,
        cpu=cpu,
    )
    script_path = experiment_dir / "run_agent.sh"
    with open(script_path, "w") as f:
        f.write(script_text)

    script_path.chmod(0o755)  # Make script executable

    print(f"Deploying {n_agents} agents for experiment {experiment}...")

    for _ in range(n_agents):
        result = subprocess.run(
            ["sbatch", str(script_path)], capture_output=True, text=True, check=True
        )
        # format: "Submitted batch job 12345"
        job_id = result.stdout.strip().split()[-1]
        print(f"Submitted job {job_id}")

    print()
    print("View logs in: ~/slurm_logs/slurm-<job_id>.out")
    print(f"Sweep URL: {wandb_url}")


if __name__ == "__main__":
    main(
        experiment="gemma_mlp_up_14",
        n_agents=8,
        job_suffix="test",
    )

    # THIS DOES WORK WHEN RUN SEPARATELY
    # for _ in range(2):
    #     result = subprocess.run(
    #         ["sbatch", "out/2025-06-24_20-59-51_gemma_mlp_up_14/run_agent.sh"],
    #         capture_output=True,
    #         text=True,
    #         check=True,
    #     )
    #     # format: "Submitted batch job 12345"
    #     job_id = result.stdout.strip().split()[-1]
    #     print(f"Submitted job {job_id}")
