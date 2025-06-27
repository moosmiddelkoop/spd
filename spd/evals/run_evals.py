"""Run SPD evaluations by deploying experiments as a SLURM job array.

This script is an entrypoint of the spd package, and thus can be called with `spd-evals` once
the package is installed.

Usage:
    spd-evals                                                  # Run all experiments
    spd-evals --experiments tms_5-2,resid_mlp3,ss_emb          # Run specific experiments
    spd-evals tms_5-2,resid_mlp1 --job-suffix my_suffix        # Add a suffix to the job name
"""

import tempfile
from datetime import datetime
from pathlib import Path

import fire
import wandb_workspaces.reports.v2 as wr

from spd.git_utils import create_git_snapshot
from spd.registry import EXPERIMENT_REGISTRY
from spd.settings import REPO_ROOT
from spd.slurm_utils import (
    create_slurm_array_script,
    format_runtime_str,
    print_job_summary,
    submit_slurm_array,
)


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


def main(experiments: str | None = None, job_suffix: str | None = None) -> None:
    """Deploy SPD experiments as a SLURM job array.

    Args:
        experiments: Comma-separated list of experiment names to run. If None, runs all
            experiments from the registry. Available experiments: tms_5-2, tms_5-2-id,
            tms_40-10, tms_40-10-id, resid_mlp1, resid_mlp2, resid_mlp3, ss_emb
        job_suffix: Optional suffix to add to the job name
    """
    if experiments is None:
        experiments_list = list(EXPERIMENT_REGISTRY.keys())
    else:
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

    evals_id = generate_evals_id()
    print(f"Evaluation ID: {evals_id}")

    report_url = create_wandb_report(evals_id, experiments_list)

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        commands: list[str] = []
        experiment_names: list[str] = []

        # Create commands for all experiments
        for experiment in experiments_list:
            config = EXPERIMENT_REGISTRY[experiment]
            decomp_script = REPO_ROOT / config.decomp_script
            config_path = REPO_ROOT / config.config_path

            command = f"python {decomp_script} {config_path} --evals_id {evals_id}"
            commands.append(command)
            experiment_names.append(experiment)

        # Determine the longest expected runtime for job naming
        max_runtime = max(EXPERIMENT_REGISTRY[exp].expected_runtime for exp in experiments_list)
        runtime_str = format_runtime_str(max_runtime)

        job_name = f"spd-evals-{runtime_str}" if job_suffix is None else f"spd-evals-{job_suffix}"

        array_script = temp_path / f"run_eval_array_{evals_id}.sh"
        create_slurm_array_script(
            script_path=array_script,
            job_name=job_name,
            commands=commands,
            cpu=False,
            snapshot_branch=snapshot_branch,
        )

        array_job_id = submit_slurm_array(array_script)

        job_info_list = [f"Array Job ID: {array_job_id}"]
        for i, experiment in enumerate(experiment_names, 1):
            job_info_list.append(f"  Task {i}: {experiment}")

        print_job_summary(job_info_list)
        print(f"Array job submitted with ID: {array_job_id}")
        print(f"Individual task logs will be in: ~/slurm_logs/slurm-{array_job_id}_<task_id>.out")
        print(f"View the report at: {report_url}")


def cli():
    """Command line interface for the evaluation script."""
    fire.Fire(main)


if __name__ == "__main__":
    cli()
