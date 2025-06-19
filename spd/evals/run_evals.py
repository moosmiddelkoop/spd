"""Run SPD evaluations by deploying experiments as individual SLURM jobs.

This script runs SPD experiments from the registry as individual SLURM jobs,
allowing for parallel evaluation of multiple experiments.

Usage:
    spd-evals                                                  # Run all experiments
    spd-evals --experiments tms_5-2,resid_mlp3,ss_emb          # Run specific experiments
"""

from pathlib import Path

import fire

from spd.registry import EXPERIMENT_REGISTRY
from spd.settings import REPO_ROOT
from spd.slurm_utils import create_slurm_script, print_job_summary, submit_slurm_jobs


def main(experiments: str | None = None) -> None:
    """Deploy SPD experiments as individual SLURM jobs.

    Args:
        experiments: Comma-separated list of experiment names to run. If None, runs all
            experiments from the registry. Available experiments: tms_5-2, tms_5-2-id,
            tms_40-10, tms_40-10-id, resid_mlp1, resid_mlp2, resid_mlp3, ss_emb
    """
    # If no experiments specified, run all experiments from registry
    if experiments is None:
        experiments_list = list(EXPERIMENT_REGISTRY.keys())
    else:
        # Split comma-separated experiments
        experiments_list = [exp.strip() for exp in experiments.split(",")]

    # Validate experiment names
    invalid_experiments = [exp for exp in experiments_list if exp not in EXPERIMENT_REGISTRY]
    if invalid_experiments:
        available = ", ".join(EXPERIMENT_REGISTRY.keys())
        raise ValueError(
            f"Invalid experiments: {invalid_experiments}. Available experiments: {available}"
        )

    print(f"Deploying {len(experiments_list)} experiments as individual SLURM jobs...")

    script_paths = []
    job_info_list = []

    # Create SLURM scripts for all experiments
    for experiment in experiments_list:
        config = EXPERIMENT_REGISTRY[experiment]
        decomp_script = REPO_ROOT / config.decomp_script
        config_path = REPO_ROOT / config.config_path

        # Use expected_runtime in the job name for easier tracking
        job_name = f"spd-eval-{config.expected_runtime}"
        command = f"python {decomp_script} {config_path}"

        print(f"Preparing {experiment}:")
        print(f"  Script: {decomp_script}")
        print(f"  Config: {config_path}")
        print(f"  Job name: {job_name}")

        # Create experiment-specific run script
        run_script = Path.home() / f"run_eval_{experiment}.sh"
        create_slurm_script(
            script_path=run_script,
            job_name=job_name,
            command=command,
            cpu=False,
        )

        script_paths.append(run_script)
        print()

    # Submit all jobs
    job_ids = submit_slurm_jobs(script_paths)

    # Create job info list for summary
    for experiment, job_id in zip(experiments_list, job_ids, strict=False):
        job_info_list.append(f"{experiment}:{job_id}")

    # Print summary
    print_job_summary(job_info_list)


def cli():
    """Command line interface for the evaluation script."""
    fire.Fire(main)


if __name__ == "__main__":
    cli()
