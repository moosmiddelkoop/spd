"""Grid search script for running SPD experiments on a SLURM cluster.

This script creates a grid search from sweep parameters and deploys multiple SLURM jobs to run it.
Uses wandb workspace views instead of wandb sweeps.

Usage:
    spd-sweep <experiment> [--n_agents <num>] [--job_suffix <suffix>] [--cpu] [--sweep_params_file <file>]

Examples:
    spd-sweep tms_5-2                                        # Run TMS 5-2 sweep
    spd-sweep resid_mlp1 --n_agents 3 --cpu                  # Run ResidMLP1 with max 3 CPU agents
    spd-sweep ss_emb --job_suffix test                       # Run with job suffix
    spd-sweep tms_5-2 --sweep_params_file my_params.yaml     # Use custom sweep params file
"""

import copy
import itertools
import json
import tempfile
from pathlib import Path
from typing import Any

import fire
import wandb_workspaces.workspaces as ws
import yaml

from spd.configs import Config
from spd.git_utils import create_git_snapshot
from spd.registry import EXPERIMENT_REGISTRY
from spd.settings import REPO_ROOT
from spd.slurm_utils import create_slurm_array_script, format_runtime_str, submit_slurm_array
from spd.utils import generate_sweep_id, load_config

# TODO: Work out why reports don't seem to inherit from the workspace template
WORKSPACE_TEMPLATES = {"tms": "https://wandb.ai/goodfire/spd?nw=css034maye"}


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


def generate_grid_combinations(parameters: dict[str, Any]) -> list[dict[str, Any]]:
    """Generate all combinations for a grid search from parameter specifications.

    Args:
        parameters: Dictionary mapping parameter names to their specifications.
                   Each specification should have a 'values' key with a list of values,
                   or a 'parameters' key for nested parameters.

    Returns:
        List of dictionaries, each representing one combination of parameters.
    """
    # Flatten nested parameters first
    flattened_params = {}

    def flatten_params(params: dict[str, Any], prefix: str = "") -> None:
        """Recursively flatten nested parameters."""
        for key, value in params.items():
            full_key = f"{prefix}.{key}" if prefix else key

            if isinstance(value, dict):
                if "values" in value:
                    # This is a parameter specification
                    flattened_params[full_key] = value["values"]
                elif "parameters" in value:
                    # This is a nested parameter group
                    flatten_params(value["parameters"], full_key)
                else:
                    # This might be a direct nested structure
                    flatten_params(value, full_key)

    flatten_params(parameters)

    # Extract parameter names and their values
    param_names = list(flattened_params.keys())
    param_values = [flattened_params[name] for name in param_names]

    # Generate all combinations
    combinations = []
    for values in itertools.product(*param_values):
        combination = dict(zip(param_names, values, strict=True))
        combinations.append(combination)

    return combinations


def apply_nested_updates(base_dict: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    """Apply nested updates to a dictionary.

    Args:
        base_dict: The base dictionary to update
        updates: Dictionary with potentially dotted keys (e.g., 'task_config.max_seq_len')

    Returns:
        Updated dictionary with nested values applied
    """
    result = copy.deepcopy(base_dict)

    for key, value in updates.items():
        if "." in key:
            # Handle nested keys
            keys = key.split(".")
            current = result

            # Navigate to the parent of the final key
            for k in keys[:-1]:
                if k not in current:
                    current[k] = {}
                current = current[k]

            # Set the final value
            current[keys[-1]] = value
        else:
            # Simple key
            result[key] = value

    return result


def create_workspace_view(sweep_id: str, experiment_name: str) -> str:
    """Create a wandb workspace view for the sweep.

    Args:
        sweep_id: The unique identifier for this sweep
        experiment_name: Name of the experiment being run

    Returns:
        URL to the workspace view
    """

    # TODO: Change workspace template
    # Load the template workspace
    workspace = ws.Workspace.from_url(WORKSPACE_TEMPLATES["tms"])

    # Update the workspace name to include sweep info
    workspace.name = f"Sweep - {experiment_name} - {sweep_id}"

    # Add filter for the sweep ID
    workspace.runset_settings.filters = [ws.Tags("tags").isin([f"{sweep_id}"])]

    # Save as a new view
    workspace.save_as_new_view()

    return workspace.url


def main(
    experiment: str,
    n_agents: int,
    job_suffix: str = "",
    cpu: bool = False,
    sweep_params_file: str = "sweep_params.yaml",
) -> None:
    """Create a sweep and deploy SLURM jobs to run it.

    Args:
        experiment: Experiment name. See spd/registry.py for available experiments.
        n_agents: Maximum number of SLURM agents to run concurrently
        job_suffix: Optional suffix to add to SLURM job names for identification
        cpu: Use CPU instead of GPU (default: False, uses GPU)
        sweep_params_file: Sweep parameters YAML file (default: sweep_params.yaml)
    """
    config_entry = EXPERIMENT_REGISTRY[experiment]
    decomp_script = REPO_ROOT / config_entry.decomp_script
    config_path = REPO_ROOT / config_entry.config_path

    # Resolve the full path for sweep params file
    sweep_params_full_path = resolve_sweep_params_path(sweep_params_file)

    print(f"Using sweep config for {experiment}:")
    print(f"  Decomposition script: {decomp_script}")
    print(f"  Config file: {config_path}")
    print(f"  Sweep params file: {sweep_params_full_path}")

    # Load sweep parameters
    with open(sweep_params_full_path) as f:
        sweep_params = yaml.safe_load(f)

    # Extract parameters (expecting format like wandb sweeps)
    if "parameters" not in sweep_params:
        raise ValueError("Sweep params file must contain a 'parameters' key")

    parameters = sweep_params["parameters"]

    # Generate all parameter combinations
    combinations = generate_grid_combinations(parameters)
    print(f"\nGenerated {len(combinations)} parameter combinations for grid search")

    # Generate sweep ID
    sweep_id = generate_sweep_id()
    print(f"Sweep ID: {sweep_id}")

    # Create workspace view
    workspace_url = create_workspace_view(sweep_id, experiment)

    # Load base config
    base_config = load_config(config_path, Config)

    # Create commands for each combination
    commands = []
    for i, param_combo in enumerate(combinations):
        # Create config dict with nested overrides
        base_config_dict = base_config.model_dump(mode="json")
        config_dict_with_overrides = apply_nested_updates(base_config_dict, param_combo)

        # Create config from the updated dict
        config_with_overrides = Config(**config_dict_with_overrides)

        # Convert to JSON string with prefix to prevent Fire parsing with ast.literal_eval
        # (https://github.com/google/python-fire/issues/332)
        config_json = f"json:{json.dumps(config_with_overrides.model_dump(mode='json'))}"

        # Create sweep params JSON (the parameter specification, not the specific values)
        sweep_params_json = json.dumps(parameters)

        # Build command
        command = (
            f"python {decomp_script} '{config_json}' "
            f"--sweep_id {sweep_id} "
            f"--sweep_params '{sweep_params_json}'"
        )
        commands.append(command)

        # Print first two combinations as examples
        if i < 2:
            print(f"Example combination {i + 1}: {param_combo}")

    # Create git snapshot
    snapshot_branch = create_git_snapshot(branch_name_prefix="sweep")
    print(f"\nUsing git snapshot: {snapshot_branch}")

    # Build job name
    runtime_str = format_runtime_str(config_entry.expected_runtime)
    job_name = f"spd-sweep-{job_suffix}" if job_suffix else f"spd-sweep-{runtime_str}"

    # Submit to SLURM
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        array_script = temp_path / f"run_grid_{sweep_id}.sh"

        create_slurm_array_script(
            script_path=array_script,
            job_name=job_name,
            commands=commands,
            cpu=cpu,
            snapshot_branch=snapshot_branch,
            max_concurrent_tasks=n_agents,
        )

        job_id = submit_slurm_array(array_script)

        print(f"\nJob Array ID: {job_id}")
        print(f"Total tasks: {len(commands)}")
        print(f"Max concurrent tasks: {n_agents}")
        print(f"\nView logs in: ~/slurm_logs/slurm-{job_id}_*.out")
        print(f"Workspace URL: {workspace_url}")


def cli():
    """Command line interface for the grid search script."""
    fire.Fire(main)


if __name__ == "__main__":
    cli()
