import itertools
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D
from tqdm import tqdm

from spd.clustering.merge import merge_iteration


@dataclass
class SweepConfig:
    """Configuration for hyperparameter sweep."""
    activation_thresholds: list[float]
    check_thresholds: list[float] 
    alphas: list[float]
    rank_cost_funcs: dict[str, Callable[[float], float]]
    iters: int = 100


@dataclass 
class SweepResult:
    """Results from a single hyperparameter configuration."""
    activation_threshold: float
    check_threshold: float
    alpha: float
    rank_cost_name: str
    non_diag_costs_min: list[float]
    non_diag_costs_max: list[float] 
    max_considered_cost: list[float]
    selected_pair_cost: list[float]
    total_iterations: int
    final_k_groups: int


def format_value(val: Any) -> str:
    """Format value to max 3 digits precision."""
    if isinstance(val, float):
        return f"{val:.3g}"
    return str(val)


def format_range(values: list[Any]) -> str:
    """Format range of values."""
    if len(values) == 1:
        return format_value(values[0])
    elif len(values) == 2:
        return f"[{format_value(values[0])}, {format_value(values[-1])}]"
    else:
        return f"[{format_value(values[0])}...{format_value(values[-1])}]"


def get_unique_param_values(results: list[SweepResult]) -> dict[str, list[Any]]:
    """Extract unique parameter values from results."""
    all_params: list[str] = ['activation_threshold', 'check_threshold', 'alpha', 'rank_cost_name']
    return {param: sorted(list(set(getattr(r, param) for r in results))) for param in all_params}


def filter_results_by_params(results: list[SweepResult], fixed_params: dict[str, Any]) -> list[SweepResult]:
    """Filter results by fixed parameter values."""
    filtered_results: list[SweepResult] = results
    for param, value in fixed_params.items():
        filtered_results = [r for r in filtered_results if getattr(r, param) == value]
    return filtered_results


def validate_plot_params(lines_by: str, rows_by: str, cols_by: str, fixed_params: dict[str, Any]) -> None:
    """Validate that all required parameters are fixed for 3D plotting."""
    all_params: list[str] = ['activation_threshold', 'check_threshold', 'alpha', 'rank_cost_name']
    used_params: set[str] = {lines_by, rows_by, cols_by}
    unused_params: list[str] = [p for p in all_params if p not in used_params]
    
    missing_fixed: list[str] = [p for p in unused_params if p not in fixed_params]
    if missing_fixed:
        raise ValueError(f"Must fix all unused parameters. Missing: {missing_fixed}")


def create_colormap(line_values: list[Any]) -> tuple[Any, Any]:
    """Create colormap for line parameter."""
    if isinstance(line_values[0], (int, float)):
        norm = LogNorm(vmin=min(line_values), vmax=max(line_values))
        cmap = cm.viridis
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        return norm, {'cmap': cmap, 'sm': sm}
    else:
        colors: np.ndarray = cm.viridis(np.linspace(0, 1, len(line_values)))
        color_dict: dict[Any, Any] = {val: colors[i] for i, val in enumerate(line_values)}
        return None, {'color_dict': color_dict}


def create_suptitle(lines_by: str, rows_by: str, cols_by: str, 
                   param_values: dict[str, list[Any]], fixed_params: dict[str, Any]) -> str:
    """Create informative suptitle showing parameter organization."""
    title_parts: list[str] = []
    title_parts.append(f"lines_by: {lines_by} ∈ {format_range(param_values[lines_by])}")
    title_parts.append(f"rows_by: {rows_by} ∈ {format_range(param_values[rows_by])}")
    title_parts.append(f"cols_by: {cols_by} ∈ {format_range(param_values[cols_by])}")
    
    if fixed_params:
        fixed_str: str = ", ".join([f"{k}={format_value(v)}" for k, v in fixed_params.items()])
        title_parts.append(f"fixed values: {fixed_str}")
    
    return '\n'.join(title_parts)


def process_values(values: np.ndarray, normalize_to_zero: bool, log_delta: bool) -> np.ndarray:
    """Process metric values with normalization and log transformation."""
    if normalize_to_zero and len(values) > 0:
        values = values - values[0]
        if log_delta and len(values) > 1:
            values = np.sign(values) * np.log10(np.abs(values) + 1e-10)
    return values


def get_iterations(n_values: int, log_iterations: bool) -> np.ndarray:
    """Get iteration values, optionally with log scaling."""
    iterations: np.ndarray = np.array(range(n_values))
    if log_iterations:
        iterations = iterations + 1  # Start from 1 for log scale
    return iterations


def setup_axis(ax: plt.Axes, row_idx: int, col_idx: int, n_rows: int, n_cols: int,
               row_val: Any, col_val: Any, rows_by: str, cols_by: str, 
               metric: str, normalize_to_zero: bool, log_delta: bool, log_iterations: bool) -> None:
    """Set up axis labels and scales."""
    if row_idx == 0:  # Only first row gets titles
        ax.set_title(f"{cols_by}\n{format_value(col_val)}")
    
    if log_iterations:
        ax.set_xscale('log')
    
    if row_idx == n_rows - 1:  # Only bottom row gets x-labels
        xlabel: str = "log(Iteration)" if log_iterations else "Iteration"
        ax.set_xlabel(xlabel)
    
    if col_idx == 0:  # Only leftmost column gets y-label
        ylabel: str = metric.split('_')[-1].title()
        if normalize_to_zero:
            ylabel = f"log|Δ{ylabel}|" if log_delta else f"Δ{ylabel}"
        ax.set_ylabel(ylabel)
    
    if col_idx == n_cols - 1:  # Rightmost column gets row value
        ax.yaxis.set_label_position("right")
        ax.set_ylabel(f"{rows_by}\n={format_value(row_val)}", rotation=270, labelpad=25)
    
    ax.grid(True, alpha=0.3)


def add_colorbar_or_legend(fig: plt.Figure, axes: np.ndarray, line_values: list[Any], 
                          lines_by: str, colormap_info: dict[str, Any]) -> None:
    """Add colorbar for numeric parameters or legend for categorical."""
    if isinstance(line_values[0], (int, float)):
        cbar = fig.colorbar(colormap_info['sm'], ax=axes, orientation='horizontal', 
                           fraction=0.05, pad=-0.25)
        cbar.set_label(lines_by)
    else:
        color_dict: dict[Any, Any] = colormap_info['color_dict']
        legend_elements: list[Line2D] = [
            Line2D([0], [0], color=color_dict[val], lw=2, label=f"{lines_by}={format_value(val)}") 
            for val in line_values
        ]
        fig.legend(handles=legend_elements, loc='lower center', 
                  bbox_to_anchor=(0.5, -0.1), ncol=len(line_values))


def run_hyperparameter_sweep(raw_activations: torch.Tensor, sweep_config: SweepConfig) -> list[SweepResult]:
    """Run hyperparameter sweep across all parameter combinations."""
    param_combinations: list[tuple] = list(itertools.product(
        sweep_config.activation_thresholds,
        sweep_config.check_thresholds,
        sweep_config.alphas,
        sweep_config.rank_cost_funcs.items()
    ))
    
    print(f"{len(param_combinations) = }")
    
    results: list[SweepResult] = []
    
    for i, (act_thresh, check_thresh, alpha, (rank_name, rank_func)) in tqdm(
        enumerate(param_combinations), total=len(param_combinations)
    ):
        try:
            coact_bool: torch.Tensor = raw_activations > act_thresh
            coact: torch.Tensor = coact_bool.float().T @ coact_bool.float()
            
            result_dict: dict[str, Any] = merge_iteration(
                coact=coact,
                activation_mask=coact_bool,
                check_threshold=check_thresh,
                alpha=alpha,
                rank_cost=rank_func,
                iters=sweep_config.iters,
                plot_every=None,
                plot_final=False,
                component_labels=None,
            )
            
            results.append(SweepResult(
                activation_threshold=act_thresh,
                check_threshold=check_thresh, 
                alpha=alpha,
                rank_cost_name=rank_name,
                non_diag_costs_min=result_dict['non_diag_costs_min'],
                non_diag_costs_max=result_dict['non_diag_costs_max'],
                max_considered_cost=result_dict['max_considered_cost'],
                selected_pair_cost=result_dict['selected_pair_cost'],
                total_iterations=result_dict['total_iterations'],
                final_k_groups=result_dict['final_k_groups'],
            ))
        except Exception as e:
            print(f"Failed: {e}")
    
    print(f"{len(results) = }")
    return results


def plot_evolution_histories(
    results: list[SweepResult], 
    fixed_params: dict[str, Any],
    metric: str = "non_diag_costs_min",
    lines_by: str = "alpha",
    rows_by: str = "activation_threshold", 
    cols_by: str = "check_threshold",
    figsize: tuple[int, int] = (15, 10),
    normalize_to_zero: bool = True,
    log_delta: bool = True,
    log_iterations: bool = False,
) -> None:
    """Plot evolution histories with 3D parameter organization."""
    
    validate_plot_params(lines_by, rows_by, cols_by, fixed_params)
    
    filtered_results: list[SweepResult] = filter_results_by_params(results, fixed_params)
    if not filtered_results:
        raise ValueError(f"No results match fixed parameters: {fixed_params}")
    
    param_values: dict[str, list[Any]] = get_unique_param_values(filtered_results)
    row_values: list[Any] = param_values[rows_by]
    col_values: list[Any] = param_values[cols_by]
    line_values: list[Any] = param_values[lines_by]
    
    n_rows: int = len(row_values)
    n_cols: int = len(col_values)
    
    fig: plt.Figure
    axes: np.ndarray
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False, sharex=True)
    
    norm, colormap_info = create_colormap(line_values)
    
    suptitle: str = create_suptitle(lines_by, rows_by, cols_by, param_values, fixed_params)
    fig.suptitle(suptitle, fontsize=12)
    
    for row_idx, row_val in enumerate(row_values):
        for col_idx, col_val in enumerate(col_values):
            ax: plt.Axes = axes[row_idx, col_idx]
            
            subset_results: list[SweepResult] = [
                r for r in filtered_results 
                if getattr(r, rows_by) == row_val and getattr(r, cols_by) == col_val
            ]
            
            for line_val in line_values:
                line_results: list[SweepResult] = [
                    r for r in subset_results if getattr(r, lines_by) == line_val
                ]
                
                if line_results:
                    result: SweepResult = line_results[0]
                    values: np.ndarray = np.array(getattr(result, metric))
                    values = process_values(values, normalize_to_zero, log_delta)
                    iterations: np.ndarray = get_iterations(len(values), log_iterations)
                    
                    if isinstance(line_values[0], (int, float)):
                        color = colormap_info['cmap'](norm(line_val))
                    else:
                        color = colormap_info['color_dict'][line_val]
                    
                    ax.plot(iterations, values, color=color, alpha=0.8, linewidth=2)
            
            setup_axis(ax, row_idx, col_idx, n_rows, n_cols, row_val, col_val, 
                      rows_by, cols_by, metric, normalize_to_zero, log_delta, log_iterations)
    
    add_colorbar_or_legend(fig, axes, line_values, lines_by, colormap_info)
    plt.tight_layout()
    plt.show()


def setup_heatmap_axes() -> tuple[plt.Figure, list[plt.Axes]]:
    """Set up figure and axes for heatmaps with proper handling of edge cases."""
    def _handle_axes_array(axes_obj: Any, n_items: int, n_rows: int, n_cols: int) -> list[plt.Axes]:
        if n_items == 1:
            return [axes_obj]
        elif n_rows == 1 and n_cols > 1:
            return list(axes_obj)
        elif n_rows > 1:
            return axes_obj.flatten()
        else:
            return [axes_obj]
    
    return _handle_axes_array


def create_heatmap_data(results: list[SweepResult], rank_cost_name: str, 
                       unique_act_thresh: list[float], unique_check_thresh: list[float],
                       statistic_func: Callable[[SweepResult], float]) -> np.ndarray:
    """Create heatmap data by averaging statistic across alpha values."""
    rank_results: list[SweepResult] = [r for r in results if r.rank_cost_name == rank_cost_name]
    heatmap_data: np.ndarray = np.full((len(unique_act_thresh), len(unique_check_thresh)), np.nan)
    
    for act_idx, act_thresh in enumerate(unique_act_thresh):
        for check_idx, check_thresh in enumerate(unique_check_thresh):
            matching_results: list[SweepResult] = [
                r for r in rank_results 
                if r.activation_threshold == act_thresh and r.check_threshold == check_thresh
            ]
            if matching_results:
                stat_values: list[float] = [statistic_func(r) for r in matching_results]
                heatmap_data[act_idx, check_idx] = np.mean(stat_values)
    
    return heatmap_data


def create_heatmaps(
    results: list[SweepResult],
    statistic_func: Callable[[SweepResult], float],
    statistic_name: str,
    figsize: tuple[int, int] = (20, 15),
) -> None:
    """Create heatmaps showing statistics across hyperparameter combinations."""
    
    unique_act_thresh: list[float] = sorted(list(set(r.activation_threshold for r in results)))
    unique_check_thresh: list[float] = sorted(list(set(r.check_threshold for r in results))) 
    unique_alpha: list[float] = sorted(list(set(r.alpha for r in results)))
    unique_rank_cost: list[str] = sorted(list(set(r.rank_cost_name for r in results)))
    
    n_rank_costs: int = len(unique_rank_cost)
    n_cols: int = min(2, n_rank_costs)
    n_rows: int = (n_rank_costs + n_cols - 1) // n_cols
    
    fig: plt.Figure
    axes_obj: Any
    fig, axes_obj = plt.subplots(n_rows, n_cols, figsize=figsize)
    
    handle_axes = setup_heatmap_axes()
    axes: list[plt.Axes] = handle_axes(axes_obj, n_rank_costs, n_rows, n_cols)
    
    for rank_idx, rank_cost_name in enumerate(unique_rank_cost):
        if rank_idx >= len(axes):
            break
            
        ax: plt.Axes = axes[rank_idx]
        heatmap_data: np.ndarray = create_heatmap_data(
            results, rank_cost_name, unique_act_thresh, unique_check_thresh, statistic_func
        )
        
        im = ax.imshow(heatmap_data, aspect='auto', cmap='viridis')
        ax.set_xticks(range(len(unique_check_thresh)))
        ax.set_xticklabels([f"{x:.3g}" for x in unique_check_thresh])
        ax.set_yticks(range(len(unique_act_thresh)))
        ax.set_yticklabels([f"{x:.3g}" for x in unique_act_thresh])
        ax.set_xlabel("Check Threshold")
        ax.set_ylabel("Activation Threshold")
        
        title: str = f"{statistic_name}\\n{rank_cost_name}"
        if len(unique_alpha) > 1:
            alpha_range: str = (f"[{unique_alpha[0]:.3g}...{unique_alpha[-1]:.3g}]" 
                              if len(unique_alpha) > 2 
                              else f"[{unique_alpha[0]:.3g}, {unique_alpha[-1]:.3g}]")
            title += f"\\n(avg over α∈{alpha_range})"
        ax.set_title(title)
        
        plt.colorbar(im, ax=ax)
    
    # Hide empty subplots
    for i in range(n_rank_costs, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.show()


# Simple stopping condition examples using lambdas
def cost_ratio_condition(ratio: float, metric: str) -> Callable[[dict[str, Any]], bool]:
    """Create stopping condition for cost ratio."""
    return lambda stats: (len(stats[metric]) >= 2 and 
                         stats[metric][-1] >= stats[metric][0] * ratio)

def iteration_condition(max_iters: int) -> Callable[[dict[str, Any]], bool]:
    """Create stopping condition for max iterations."""
    return lambda stats: stats['iteration'] >= max_iters