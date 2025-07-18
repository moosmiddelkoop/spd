"""Simple utilities for pulling data from wandb runs and sweeps."""

import json
import re

import matplotlib.pyplot as plt
import pandas as pd
import wandb
from tqdm import tqdm


def get_experiment_df(
    sweep_run_id, experiment_name: str, project: str, download_ci_masks: bool = False
) -> pd.DataFrame:
    """Get all runs matching specific sweep_run_id(s) AND experiment_name.

    Args:
        sweep_run_id: Single run ID or list of run IDs
        experiment_name: The experiment name tag
        project: The wandb project path
        download_ci_masks: If True, download final causal importance arrays

    Returns:
        DataFrame with columns: wandb_run_id, run_name, state, config.*, summary.*
    """
    api = wandb.Api()

    # Handle single or multiple run IDs
    run_ids = [sweep_run_id] if isinstance(sweep_run_id, str) else sweep_run_id

    all_data = []

    for run_id in run_ids:
        filters = {"tags": {"$all": [run_id, experiment_name]}}
        runs = list(api.runs(project, filters=filters))

        for run in tqdm(runs, desc=f"{experiment_name} - {run_id}"):
            row = {
                "wandb_run_id": run.id,
                "run_name": run.name,
                "state": run.state,
                "tags": run.tags,
                "sweep_run_id": run_id,  # Track which sweep this came from
            }

            # Add config values
            if run.config:
                for k, v in run.config.items():
                    row[f"config.{k}"] = v

            # Add summary values
            if run.summary:
                for k, v in run.summary.items():
                    row[f"summary.{k}"] = v

            all_data.append(row)

    return pd.DataFrame(all_data)


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
    return df.groupby(group_cols, dropna=False).agg(agg_dict).reset_index()


def optimize_over_old(
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
        df.groupby(group_cols, dropna=False)
        .apply(get_optimal_rows, include_groups=True)
        .reset_index(drop=True)
    )

    return result_df


def optimize_over(
    df: pd.DataFrame,
    metric: str,
    params: list[str],
    minimize: bool = True,
    verbose: bool = False,
    aggregate_cols: str | list[str] | None = None,
) -> pd.DataFrame:
    """
    Find optimal hyperparameters for each unique combination of other varying config columns.

    If aggregate_cols is provided, aggregates over those columns (e.g., seeds) before
    optimization, then returns the unaggregated data with optimal parameters.

    WARNING: Only considers columns starting with 'config' when determining grouping columns.

    Args:
        df: DataFrame with experiment results
        metric: Column to optimize (e.g., 'loss')
        params: Columns to optimize over (e.g., ['config.lr', 'config.wd'])
        minimize: If True, find minimum; if False, find maximum
        verbose: If True, print optimal parameters for each group
        aggregate_cols: If provided, aggregate over these columns before optimizing
                       (e.g., 'config.seed' to average over seeds)

    Returns:
        DataFrame with optimal rows and 'optimal_hparams' column.
        If aggregate_cols is provided, returns all original rows matching optimal params.

    Example:
        >>> # Find best params averaging over seeds, return all seed runs
        >>> df_best = optimize_over(df, "loss", ["config.lr"], aggregate_cols="config.seed")
    """
    # Store original df if we're aggregating
    original_df = df.copy() if aggregate_cols else None

    # Aggregate if requested
    if aggregate_cols:
        if verbose:
            print(f"Aggregating over {aggregate_cols} before optimization...")
        working_df = aggregate_over(df, aggregate_cols)
    else:
        working_df = df

    # Validate inputs
    if metric not in working_df.columns:
        raise ValueError(f"Metric column '{metric}' not found in DataFrame")

    missing_params = [p for p in params if p not in working_df.columns]
    if missing_params:
        raise ValueError(f"Parameter columns not found: {missing_params}")

    # Get grouping columns (all varying columns except those being optimized)
    all_varying = get_varying_columns(working_df)
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
        opt_df = get_optimal_rows(working_df)
    else:
        # Apply optimization to each group
        opt_df = (
            working_df.groupby(group_cols, dropna=False)
            .apply(get_optimal_rows, include_groups=True)
            .reset_index(drop=True)
        )

    # If we aggregated, return the unaggregated data with optimal parameters
    if aggregate_cols and original_df is not None:
        # Get columns that define optimal configurations
        if isinstance(aggregate_cols, str):
            aggregate_cols = [aggregate_cols]

        # All varying columns from original data except those we aggregated over
        varying_cols = get_varying_columns(original_df, blacklist=aggregate_cols)

        # Columns that define each group (excluding the params we optimized)
        group_cols = [col for col in varying_cols if col not in params]

        # Merge columns = group columns + optimized params
        merge_cols = group_cols + params

        # Merge original data with optimal configurations
        result_df = original_df.merge(
            opt_df[merge_cols + ["optimal_hparams"]], on=merge_cols, how="inner"
        )

        if verbose:
            print(f"\nReturning {len(result_df)} unaggregated runs with optimal parameters")
            print(f"(from {len(original_df)} total runs)")

        return result_df
    else:
        return opt_df


def prettify_column_name(col_name: str) -> str:
    """Convert column names like 'summary.loss/faithfulness' to 'Faithfulness Loss'"""
    try:
        # Remove common prefixes
        name = re.sub(r"^(summary\.|config\.|metrics\.)", "", col_name)

        # Handle target solution error with tolerance patterns
        if "target_solution_error/total" in name:
            if name == "target_solution_error/total":
                return "Target Solution Error\n(tolerance=0.1)"
            elif "total_0p2" in name:
                return "Target Solution Error\n(tolerance=0.2)"
            # Add more tolerance patterns as needed

        # Handle other special patterns
        replacements = {
            "loss/": "",
            "_": " ",
            "/": " ",
            "stochastic recon": "Stochastic Reconstruction",
            "target solution error": "Target Solution Error",
        }

        for old, new in replacements.items():
            name = name.replace(old, new)

        # Title case
        name = name.title()

        # Add "Loss" suffix if it was in the original but got removed
        if "loss/" in col_name.lower() and "Loss" not in name:
            name += " Loss"

        return name
    except Exception as e:
        print(f"Error prettifying column name '{col_name}': {e}")
        return col_name


def format_group_label(group_col: str, group_value) -> str:
    """Format group label as 'param_name = value'"""
    # Extract just the parameter name (remove config. prefix)
    param_name = group_col.split(".")[-1] if "." in group_col else group_col

    # Format the value (handle floats nicely)
    if isinstance(group_value, float):
        # Use fewer decimal places if it's a round number
        if group_value == int(group_value):
            value_str = str(int(group_value))
        else:
            value_str = f"{group_value:.3g}"
    else:
        value_str = str(group_value)

    return f"{param_name} = {value_str}"


def plot_scatter_grid(
    df, x_col, y_cols, group_col="config.sigmoid_type", log_y_cols=None, figsize=None, title=""
):
    """
    Create scatter plot grid with automatic prettification and formatted legend labels.

    Args:
        df: DataFrame with data to plot
        x_col: Column for x-axis
        y_cols: List of columns for y-axes
        group_col: Column to group by (creates different colors)
        log_y_cols: Columns to use log scale for (auto-detected if None)
        figsize: Figure size (auto-calculated if None)
        title: Overall figure title
    """
    n_plots = len(y_cols)
    if figsize is None:
        figsize = (6 * n_plots, 6)

    if log_y_cols is None:
        # Auto-detect: use log scale for anything with "loss" in the name
        log_y_cols = [col for col in y_cols if "loss" in col.lower()]

    fig, axes = plt.subplots(1, n_plots, figsize=figsize)
    if n_plots == 1:
        axes = [axes]

    x_label = prettify_column_name(x_col)

    for ax, y_col in zip(axes, y_cols, strict=False):
        # Plot each group
        for group in sorted(df[group_col].unique()):  # Sort for consistent ordering
            group_df = df[df[group_col] == group]
            label = format_group_label(group_col, group)
            ax.scatter(
                group_df[x_col], group_df[y_col], label=label, alpha=0.7, s=50
            )  # s=50 for slightly larger points

        # Labels and formatting
        ax.set_xlabel(x_label)
        ax.set_ylabel(prettify_column_name(y_col))
        ax.set_title(f"{prettify_column_name(y_col)} vs {x_label}")

        if y_col in log_y_cols:
            ax.set_yscale("log")
            ax.set_ylabel(ax.get_ylabel() + " (log scale)")

        ax.legend()

    if title:
        fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.show()


def plot_metrics_by_group(
    df,
    y_cols,
    group_col="group_name",
    suptitle=None,
    group_order=None,
):
    """Create bar plots for each metric grouped by group_name."""

    fig, axs = plt.subplots(
        nrows=1,
        ncols=len(y_cols),
        figsize=(4 * len(y_cols), 5),
    )

    # Get groups in specified order or sorted
    if group_order is not None:
        groups = group_order
    else:
        groups = sorted(df[group_col].unique())

    colors = [f"C{i}" for i in range(len(groups))]

    # First pass: collect data and find y-limits for target solution error plots
    target_solution_ylims = [float("inf"), float("-inf")]
    all_plot_data = []

    for i, (y_col, y_label) in enumerate(y_cols.items()):
        # Calculate stats for each group
        x_pos = []
        means = []
        stds = []

        for j, group in enumerate(groups):
            group_data = df[df[group_col] == group][y_col]
            means.append(group_data.mean())
            stds.append(group_data.std())
            x_pos.append(j)

        all_plot_data.append((x_pos, means, stds, y_col, y_label))

        # Track min/max for target solution error plots
        if "target_solution_error" in y_col:
            # Calculate the range including error bars
            lower_bounds = [m - s for m, s in zip(means, stds, strict=False)]
            upper_bounds = [m + s for m, s in zip(means, stds, strict=False)]
            target_solution_ylims[0] = min(target_solution_ylims[0], min(lower_bounds))
            target_solution_ylims[1] = max(target_solution_ylims[1], max(upper_bounds))

    # Second pass: create the plots
    for i, (x_pos, means, stds, y_col, y_label) in enumerate(all_plot_data):
        ax = axs[i]

        # Create bars
        bars = ax.bar(x_pos, means, yerr=stds, capsize=5, color=colors, alpha=0.8, width=0.6)

        # Formatting
        ax.set_title(y_label, fontsize=16)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(groups, rotation=0, ha="center")
        # Remove top and right spines
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Apply different scales based on column type
        if "loss" in y_col:
            # Log scale for loss columns
            ax.set_yscale("log")
        elif "target_solution_error" in y_col:
            # Integer scale for target solution error columns
            ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
            # Set shared y-limits for target solution error plots
            ax.set_ylim(target_solution_ylims[0] * 0.9, target_solution_ylims[1] * 1.1)

    # Add legend below the plots
    fig.legend(bars, groups, loc="lower center", bbox_to_anchor=(0.5, -0.1), ncol=len(groups))

    # Add suptitle if provided
    if suptitle:
        fig.suptitle(suptitle, y=1.02, fontsize=20, fontweight="bold")

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.4)  # Increase horizontal spacing
    return fig, axs
