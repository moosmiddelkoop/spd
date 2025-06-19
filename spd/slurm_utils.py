"""Shared utilities for SLURM job management."""

import subprocess
import textwrap
from pathlib import Path

from spd.settings import REPO_ROOT


def setup_slurm_logs_directory() -> Path:
    """Ensure SLURM logs directory exists and return its path.

    Returns:
        Path to the slurm_logs directory
    """
    slurm_logs_dir = Path.home() / "slurm_logs"
    slurm_logs_dir.mkdir(exist_ok=True)
    return slurm_logs_dir


def get_gpu_config(cpu: bool) -> str:
    """Get SLURM GPU configuration string.

    Args:
        cpu: If True, use CPU only (gpu:0), otherwise use GPU (gpu:1)

    Returns:
        SLURM configuration string for GPU/CPU allocation
    """
    return "#SBATCH --gres=gpu:0" if cpu else "#SBATCH --gres=gpu:1"


def create_slurm_script(
    script_path: Path,
    job_name: str,
    command: str,
    cpu: bool = False,
    time_limit: str = "24:00:00",
) -> None:
    """Create a SLURM batch script.

    Args:
        script_path: Path where the script should be written
        job_name: Name for the SLURM job
        command: Command to execute in the job
        cpu: If True, use CPU only, otherwise use GPU
        time_limit: Time limit for the job (default: 24:00:00)
    """
    gpu_config = get_gpu_config(cpu)
    slurm_logs_dir = setup_slurm_logs_directory()

    script_content = textwrap.dedent(f"""
        #!/bin/bash
        #SBATCH --nodes=1
        {gpu_config}
        #SBATCH --time={time_limit}
        #SBATCH --job-name={job_name}
        #SBATCH --partition=all
        #SBATCH --output={slurm_logs_dir}/slurm-%j.out

        # Change to the SPD repository directory
        cd {REPO_ROOT}

        # Execute the command
        {command}
    """).strip()

    with open(script_path, "w") as f:
        f.write(script_content)

    # Make script executable
    script_path.chmod(0o755)


def submit_slurm_jobs(script_paths: list[Path]) -> list[str]:
    """Submit multiple SLURM jobs and return their job IDs.

    Args:
        script_paths: List of paths to SLURM batch scripts

    Returns:
        List of job IDs from submitted jobs
    """
    job_ids = []

    for script_path in script_paths:
        result = subprocess.run(
            ["sbatch", str(script_path)], capture_output=True, text=True, check=True
        )
        # Extract job ID from sbatch output (format: "Submitted batch job 12345")
        job_id = result.stdout.strip().split()[-1]
        job_ids.append(job_id)

    return job_ids


def print_job_summary(job_info_list: list[str]) -> None:
    """Print summary of submitted jobs.

    Args:
        job_info_list: List of job information strings (can be just job IDs
                      or formatted as "experiment:job_id")
    """
    print("=" * 50)
    print("DEPLOYMENT SUMMARY")
    print("=" * 50)
    print(f"Deployed {len(job_info_list)} jobs:")

    for job_info in job_info_list:
        if ":" in job_info:
            experiment, job_id = job_info.split(":", 1)
            print(f"  {experiment}: {job_id}")
        else:
            print(f"  Job ID: {job_info}")

    print("\nView logs in: ~/slurm_logs/slurm-<job_id>.out")
