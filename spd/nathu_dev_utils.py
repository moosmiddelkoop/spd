"""Simple utilities for pulling data from wandb runs and sweeps."""

import json

import pandas as pd
import torch
import wandb
from tqdm import tqdm

from spd.utils.wandb_utils import download_wandb_file, fetch_wandb_run_dir


def get_experiment_df(
    sweep_run_id: str, experiment_name: str, project: str, download_ci_masks: bool = False
) -> pd.DataFrame:
    """Get all runs matching a specific sweep_run_id AND experiment_name.

    Uses the same filtering logic as spd/scripts/run.py where runs are tagged
    with both the sweep_run_id and experiment name.

    Args:
        sweep_run_id: The run ID tag (e.g. "run_20240115_143022")
        experiment_name: The experiment name tag (e.g. "tms_5-2")
        project: The wandb project path
        download_ci_masks: If True, download final causal importance arrays for each run

    Returns:
        DataFrame with columns: wandb_run_id, run_name, state, config.*, summary.*
        If download_ci_masks=True, adds a 'ci_masks' column with downloaded arrays
    """
    api = wandb.Api()

    # Use the same filtering pattern as run.py
    # Runs must have BOTH tags
    filters = {"tags": {"$all": [sweep_run_id, experiment_name]}}

    runs = api.runs(project, filters=filters)

    # Convert to list to get length for progress bar
    runs_list = list(runs)

    data = []
    for run in tqdm(runs_list, desc=f"Processing {experiment_name} runs"):
        row = {
            "wandb_run_id": run.id,
            "run_name": run.name,
            "state": run.state,
            "tags": run.tags,
        }

        # Add all config values
        if run.config:
            for key, value in run.config.items():
                row[f"config.{key}"] = value

        # Add all summary values
        if run.summary:
            for key, value in run.summary.items():
                row[f"summary.{key}"] = value

        # Download CI masks if requested
        if download_ci_masks:
            try:
                ci_masks = download_final_causal_importance_arrays(run.id, project)
                row["ci_masks"] = ci_masks
            except Exception as e:
                print(f"Failed to download CI masks for run {run.id}: {e}")
                row["ci_masks"] = None

        data.append(row)

    return pd.DataFrame(data)


def download_final_causal_importance_arrays(
    wandb_run_id: str, project: str, final_step: int | None = None
) -> dict[str, torch.Tensor]:
    """Download the final causal importance arrays from a wandb run.

    Args:
        wandb_run_id: The wandb run ID (e.g. "yp8e47in")
        project: The wandb project path
        final_step: The final step number (if None, will find the highest step)

    Returns:
        Dictionary mapping layer names to torch tensors
        e.g. {"layers.0.mlp_in": tensor(...), "layers.1.mlp_out": tensor(...)}
    """
    api = wandb.Api()
    run = api.run(f"{project}/{wandb_run_id}")

    # Use existing wandb_utils function to get cache dir
    wandb_run_dir = fetch_wandb_run_dir(wandb_run_id)

    # Find all array files (now .pt files)
    array_files = [
        f for f in run.files() if f.name.startswith("arrays/") and f.name.endswith(".pt")
    ]

    # If final_step not specified, find the highest step
    if final_step is None:
        steps = []
        for f in array_files:
            if "step_" in f.name:
                step_str = f.name.split("step_")[-1].replace(".pt", "")
                try:
                    steps.append(int(step_str))
                except ValueError:
                    continue
        final_step = max(steps) if steps else None

    # Download arrays for the final step
    arrays = {}
    for file in array_files:
        if final_step is not None and f"step_{final_step}.pt" not in file.name:
            continue

        # Extract layer name from path like "arrays/layers.0.mlp_in/raw_ci_upper_leaky/step_25000.pt"
        parts = file.name.split("/")
        if len(parts) >= 2:
            layer_name = parts[1]  # e.g. "layers.0.mlp_in"

            # Download using existing utility
            file_path = download_wandb_file(run, wandb_run_dir, file.name)

            # Load the tensor
            tensor = torch.load(file_path, map_location="cpu")
            arrays[layer_name] = tensor

    return arrays


def get_varying_columns(df, prefix="config", blacklist=["config.seed"]):
    """Get columns starting with prefix that have multiple unique values."""

    def has_variation(series):
        try:
            return series.nunique(dropna=False) > 1
        except TypeError:
            try:
                json_values = series.apply(lambda x: json.dumps(x, sort_keys=True))
                return json_values.nunique(dropna=False) > 1
            except (TypeError, ValueError):
                return False

    return [
        c
        for c in df.columns
        if c.startswith(prefix) and has_variation(df[c]) and c not in blacklist
    ]


# Define aggregation function for non-numeric columns
def safe_first(series):
    """Take first value if all are equal, else return NaN."""
    try:
        if series.nunique(dropna=False) == 1:
            return series.iloc[0]
        return pd.NA
    except TypeError:
        # For non-hashable types
        try:
            json_values = series.apply(
                lambda x: json.dumps(x, sort_keys=True) if x is not None else None
            )
            if json_values.nunique() == 1:
                return series.iloc[0]
            return pd.NA
        except:
            # If we can't serialize, check if all values are the same object
            first_val = series.iloc[0]
            if all(v is first_val for v in series):
                return first_val
            return pd.NA


def aggregate_over(
    df: pd.DataFrame, aggregate_cols: str | list[str], method: str = "mean"
) -> pd.DataFrame:
    """
    Aggregate over specified columns (e.g., seeds) while grouping by other varying columns.

    WARNING: Only considers columns starting with 'config' when determining grouping columns.

    Args:
        df: DataFrame with experiment results
        aggregate_cols: Column(s) to aggregate over (e.g., 'config.seed')
        method: Aggregation for numeric columns ('mean', 'median', 'std', 'min', 'max')

    Returns:
        DataFrame aggregated over specified columns

    Example:
        >>> df_avg = aggregate_over(df, "config.seed")  # Average over seeds
    """
    if isinstance(aggregate_cols, str):
        aggregate_cols = [aggregate_cols]

    # Find all varying columns except the ones we're aggregating over
    all_varying = get_varying_columns(df)
    group_cols = [col for col in all_varying if col not in aggregate_cols]

    # Get all columns to aggregate (everything except group_cols and aggregate_cols)
    cols_to_agg = [col for col in df.columns if col not in group_cols and col not in aggregate_cols]

    # Split into numeric and non-numeric
    numeric_cols = [
        col for col in cols_to_agg if col in df.select_dtypes(include=["number"]).columns
    ]
    non_numeric_cols = [col for col in cols_to_agg if col not in numeric_cols]

    # Build aggregation dictionary
    agg_dict = {}

    # Add numeric columns with specified method
    for col in numeric_cols:
        agg_dict[col] = method

    # Add non-numeric columns with safe_first
    for col in non_numeric_cols:
        agg_dict[col] = safe_first

    if not group_cols:
        raise ValueError(
            f"No columns left to group by after excluding {aggregate_cols}. "
            "All varying columns are being aggregated over."
        )

    # Perform aggregation
    return df.groupby(group_cols).agg(agg_dict).reset_index()


def optimize_over(
    df: pd.DataFrame, metric: str, params: list[str], minimize: bool = True, verbose: bool = False
) -> pd.DataFrame:
    """
    Find optimal hyperparameters for each unique combination of other varying config columns.

    WARNING: Only considers columns starting with 'config' when determining grouping columns.

    Args:
        df: DataFrame with experiment results
        metric: Column to optimize (e.g., 'loss')
        params: Columns to optimize over (e.g., ['config.lr', 'config.wd'])
        minimize: If True, find minimum; if False, find maximum
        verbose: If True, print optimal parameters for each group

    Returns:
        DataFrame with optimal rows and 'optimal_hparams' column

    Example:
        >>> df_best = optimize_over(df, "loss", ["config.learning_rate"], verbose=True)
    """
    # Validate inputs
    if metric not in df.columns:
        raise ValueError(f"Metric column '{metric}' not found in DataFrame")

    missing_params = [p for p in params if p not in df.columns]
    if missing_params:
        raise ValueError(f"Parameter columns not found: {missing_params}")

    # Get grouping columns (all varying columns except those being optimized)
    all_varying = get_varying_columns(df)
    group_cols = [col for col in all_varying if col not in params]

    def get_optimal_rows(group_df):
        # Find the best row in this group
        best_idx = group_df[metric].idxmin() if minimize else group_df[metric].idxmax()
        best_row = group_df.loc[[best_idx]]  # Keep as DataFrame

        # Add optimal parameters as a dictionary
        optimal_params = {param: best_row[param].iloc[0] for param in params}
        best_row = best_row.copy()
        best_row["optimal_hparams"] = [optimal_params]

        if verbose:
            # Print group conditions and optimal parameters
            if group_cols:
                # Get the actual values for the grouping columns from the best row
                group_values = {col: best_row[col].iloc[0] for col in group_cols}
                group_desc = ", ".join([f"{col}={val}" for col, val in group_values.items()])
            else:
                group_desc = "all data"

            param_desc = ", ".join([f"{p.split('.')[-1]}={v}" for p, v in optimal_params.items()])
            metric_val = best_row[metric].iloc[0]
            print(f"When {group_desc}: optimal params are {param_desc} ({metric}={metric_val:.4f})")

        return best_row

    if not group_cols:
        # If no grouping columns, treat entire DataFrame as one group
        return get_optimal_rows(df)
    # Apply optimization to each group (empty group_cols means one group)
    result_df = (
        df.groupby(group_cols).apply(get_optimal_rows, include_groups=True).reset_index(drop=True)
    )

    return result_df
