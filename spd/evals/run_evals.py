"""Run SPD evaluations by deploying experiments as individual SLURM jobs.

This script runs SPD experiments from the registry as individual SLURM jobs,
allowing for parallel evaluation of multiple experiments.

Usage:
    spd-evals                                                  # Run all experiments
    spd-evals --experiments tms_5-2,resid_mlp3,ss_emb          # Run specific experiments
"""

import tempfile
from datetime import datetime
from pathlib import Path

import fire
import wandb_workspaces.reports.v2 as wr

from spd.git_utils import create_git_snapshot
from spd.registry import EXPERIMENT_REGISTRY
from spd.settings import REPO_ROOT
from spd.slurm_utils import create_slurm_script, print_job_summary, submit_slurm_jobs


def generate_evals_id() -> str:
    """Generate a unique evaluation ID based on timestamp."""
    return f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def create_wandb_report(evals_id: str, experiments_list: list[str]) -> str:
    """Create a W&B report for the evaluation batch."""
    report = wr.Report(
        project="spd",
        title=f"SPD Evaluation Report - {evals_id}",
        description=f"Evaluations on: {', '.join(experiments_list)}",
    )

    unique_experiment_types = set(
        EXPERIMENT_REGISTRY[exp_name].experiment_type for exp_name in experiments_list
    )

    # Create separate panel grids for each experiment type
    for exp_type in unique_experiment_types:
        # Use experiment type tag for filtering
        combined_filter = (
            f'(Metric("tags") in ["evals_id:{evals_id}"]) and (Metric("tags") in ["{exp_type}"])'
        )

        # Create runset for this experiment type
        runset = wr.Runset(
            name=f"{exp_type} Runs",
            filters=combined_filter,
        )

        # Create panel grid with causal importance media and loss plots
        panel_grid = wr.PanelGrid(
            runsets=[runset],
            panels=[
                wr.MediaBrowser(
                    media_keys=["causal_importances_upper_leaky"],
                    layout=wr.Layout(x=-6, y=0, w=24, h=12),  # Full width at top
                ),
                wr.LinePlot(
                    x="Step",
                    y=["loss/stochastic_recon_layerwise", "loss/stochastic_recon"],
                    log_y=True,
                    layout=wr.Layout(x=-6, y=12, w=10, h=6),  # First row, first plot
                ),
                wr.LinePlot(
                    x="Step",
                    y=["loss/faithfulness"],
                    log_y=True,
                    layout=wr.Layout(x=4, y=12, w=10, h=6),  # First row, second plot
                ),
                wr.LinePlot(
                    x="Step",
                    y=["loss/importance_minimality"],
                    layout=wr.Layout(x=14, y=12, w=10, h=6),  # First row, third plot
                ),
                wr.LinePlot(
                    x="Step",
                    y=["misc/masked_kl_loss_vs_target"],
                    layout=wr.Layout(x=-6, y=18, w=10, h=6),  # Second row, first plot
                ),
                wr.LinePlot(
                    x="Step",
                    y=["misc/unmasked_kl_loss_vs_target"],
                    layout=wr.Layout(x=4, y=18, w=10, h=6),  # Second row, second plot
                ),
            ],
        )

        # Add title block and panel grid
        report.blocks.append(wr.H2(text=exp_type))
        report.blocks.append(panel_grid)

    # Save the report and return URL
    report.save()
    return report.url


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

    # Create single git snapshot for all experiments
    snapshot_branch = create_git_snapshot(branch_name_prefix="eval")
    print(f"Made git snapshot on branch: {snapshot_branch}")

    # Generate unique evaluation ID
    evals_id = generate_evals_id()
    print(f"Evaluation ID: {evals_id}")

    report_url = create_wandb_report(evals_id, experiments_list)

    # Use a temporary directory for script files
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        script_paths = []
        job_info_list = []

        print(f"Deploying {len(experiments_list)} experiments as individual SLURM jobs...")
        # Create SLURM scripts for all experiments
        for experiment in experiments_list:
            config = EXPERIMENT_REGISTRY[experiment]
            decomp_script = REPO_ROOT / config.decomp_script
            config_path = REPO_ROOT / config.config_path

            # Use expected_runtime in the job name for easier tracking
            job_name = f"spd-eval-{config.expected_runtime}"
            command = f"python {decomp_script} {config_path} --evals_id {evals_id}"

            # Create experiment-specific run script in temp directory
            run_script = temp_path / f"run_eval_{experiment}_{evals_id}.sh"
            create_slurm_script(
                script_path=run_script,
                job_name=job_name,
                command=command,
                cpu=False,
                snapshot_branch=snapshot_branch,
            )

            script_paths.append(run_script)

        job_ids = submit_slurm_jobs(script_paths)

        # Create job info list for summary
        for experiment, job_id in zip(experiments_list, job_ids, strict=False):
            job_info_list.append(f"{experiment}:{job_id}")

        print_job_summary(job_info_list)

        print(f"View the report at: {report_url}")
        # Temporary directory and all script files are automatically cleaned up here


def cli():
    """Command line interface for the evaluation script."""
    fire.Fire(main)


if __name__ == "__main__":
    cli()
