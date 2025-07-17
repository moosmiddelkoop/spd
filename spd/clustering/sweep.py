
from dataclasses import dataclass
from typing import Any, Callable
import itertools

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import torch
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

def run_hyperparameter_sweep(
    raw_activations: torch.Tensor,
    sweep_config: SweepConfig,
) -> list[SweepResult]:
    """Run hyperparameter sweep across all parameter combinations."""
    
    param_combinations = list(itertools.product(
        sweep_config.activation_thresholds,
        sweep_config.check_thresholds,
        sweep_config.alphas,
        sweep_config.rank_cost_funcs.items()
    ))
    
    print(f"{len(param_combinations) = }")
    
    results = []
    
    for i, (act_thresh, check_thresh, alpha, (rank_name, rank_func)) in tqdm(enumerate(param_combinations), total=len(param_combinations)):
        # tqdm.write(f"{i+1}/{len(param_combinations)}: {act_thresh = }, {check_thresh = }, {alpha = }, {rank_name = }")
        
        try:
            # Apply activation threshold
            coact_bool = raw_activations > act_thresh
            coact = coact_bool.float().T @ coact_bool.float()
            
            # Run merge iteration
            result_dict = merge_iteration(
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
    metric: str = "non_diag_costs_min",
    lines_by: str = "alpha",
    rows_by: str = "activation_threshold", 
    cols_by: str = "check_threshold",
    figsize: tuple[int, int] = (15, 10),
    normalize_to_zero: bool = True,
    log_delta: bool = True,
) -> None:
    """Plot evolution histories with 3D parameter organization."""
    
    def format_value(val):
        """Format value to max 3 digits precision."""
        if isinstance(val, float):
            return f"{val:.3g}"
        return str(val)
    
    def format_range(values):
        """Format range of values."""
        if len(values) == 1:
            return format_value(values[0])
        elif len(values) == 2:
            return f"[{format_value(values[0])}, {format_value(values[-1])}]"
        else:
            return f"[{format_value(values[0])}...{format_value(values[-1])}]"
    
    # Get unique values for each parameter
    all_params = ['activation_threshold', 'check_threshold', 'alpha', 'rank_cost_name']
    param_values = {param: sorted(list(set(getattr(r, param) for r in results))) for param in all_params}
    
    # Get unique values for row/col organization
    row_values = sorted(list(set(getattr(r, rows_by) for r in results)))
    col_values = sorted(list(set(getattr(r, cols_by) for r in results)))
    line_values = sorted(list(set(getattr(r, lines_by) for r in results)))
    
    # Create subplot grid
    n_rows = len(row_values)
    n_cols = len(col_values)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False, sharex=True)
    
    # Create colormap for line parameter
    if isinstance(line_values[0], (int, float)):
        # Numeric colormap
        norm = plt.Normalize(vmin=min(line_values), vmax=max(line_values))
        cmap = cm.viridis
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
    else:
        # Categorical colormap
        colors = cm.viridis(np.linspace(0, 1, len(line_values)))
        color_dict = {val: colors[i] for i, val in enumerate(line_values)}
    
    # Identify parameters that vary vs are fixed
    used_params = {lines_by, rows_by, cols_by}
    other_params = [p for p in all_params if p not in used_params]
    varied_other = {k: v for k, v in param_values.items() if k in other_params and len(v) > 1}
    fixed_other = {k: v[0] for k, v in param_values.items() if k in other_params and len(v) == 1}
    
    # Create suptitle with ranges
    title_parts = []
    if fixed_other:
        fixed_str = ", ".join([f"{k}={format_value(v)}" for k, v in fixed_other.items()])
        title_parts.append(f"Fixed: {fixed_str}")
    if varied_other:
        varied_str = ", ".join([f"{k}∈{format_range(v)}" for k, v in varied_other.items()])
        title_parts.append(f"Varied: {varied_str}")
    
    # Add rank cost names on separate line if it's not one of the main dimensions
    if 'rank_cost_name' not in used_params:
        rank_names = param_values['rank_cost_name']
        if len(rank_names) > 1:
            title_parts.append(f"Rank costs: {', '.join(rank_names)}")
    
    if title_parts:
        fig.suptitle('\n'.join(title_parts), fontsize=12)
    
    # Fill subplots
    for row_idx, row_val in enumerate(row_values):
        for col_idx, col_val in enumerate(col_values):
            ax = axes[row_idx, col_idx]
            
            # Get results for this row/col combination
            subset_results = [
                r for r in results 
                if getattr(r, rows_by) == row_val and getattr(r, cols_by) == col_val
            ]
            
            # Plot lines for each value of the line parameter
            for line_val in line_values:
                line_results = [r for r in subset_results if getattr(r, lines_by) == line_val]
                
                if line_results:
                    # Should be exactly one result, but take first if multiple
                    result = line_results[0]
                    values = np.array(getattr(result, metric))
                    
                    # Normalize to start at 0 if requested
                    if normalize_to_zero and len(values) > 0:
                        values = values - values[0]
                        if log_delta and len(values) > 1:
                            # Log of absolute delta, preserving sign
                            values = np.sign(values) * np.log10(np.abs(values) + 1e-10)
                    
                    # Get color
                    if isinstance(line_values[0], (int, float)):
                        color = cmap(norm(line_val))
                    else:
                        color = color_dict[line_val]
                    
                    ax.plot(range(len(values)), values, color=color, 
                           alpha=0.8, linewidth=2)
            
            # Set subplot title and labels
            if row_idx == 0:  # Only first row gets titles
                ax.set_title(f"{cols_by}\n{format_value(col_val)}")
            
            if row_idx == n_rows - 1:  # Only bottom row gets x-labels
                ax.set_xlabel("Iteration")
            
            if col_idx == 0:  # Only leftmost column gets y-label
                # Terse y-label
                ylabel = metric.split('_')[-1].title()  # Just the last part
                if normalize_to_zero:
                    if log_delta:
                        ylabel = f"log|Δ{ylabel}|"
                    else:
                        ylabel = f"Δ{ylabel}"
                ax.set_ylabel(ylabel)
            
            if col_idx == n_cols - 1:  # Rightmost column gets row value
                ax.yaxis.set_label_position("right")
                ax.set_ylabel(f"{rows_by}={format_value(row_val)}", rotation=270, labelpad=15)
            
            ax.grid(True, alpha=0.3)
    
    # Add colorbar for line parameter
    if isinstance(line_values[0], (int, float)):
        cbar = fig.colorbar(sm, ax=axes, orientation='horizontal', fraction=0.05, pad=0.08)
        cbar.set_label(f"{lines_by}")
    else:
        # Create custom legend for categorical
        from matplotlib.lines import Line2D
        legend_elements = [Line2D([0], [0], color=color_dict[val], lw=2, label=f"{lines_by}={format_value(val)}") 
                          for val in line_values]
        fig.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, -0.05), ncol=len(line_values))
    
    plt.tight_layout()
    plt.show()

def create_heatmaps(
    results: list[SweepResult],
    statistic_func: Callable[[SweepResult], float],
    statistic_name: str,
    figsize: tuple[int, int] = (20, 15),
) -> None:
    """Create heatmaps showing statistics across hyperparameter combinations."""
    
    # Get unique values for each parameter
    unique_act_thresh = sorted(list(set(r.activation_threshold for r in results)))
    unique_check_thresh = sorted(list(set(r.check_threshold for r in results))) 
    unique_alpha = sorted(list(set(r.alpha for r in results)))
    unique_rank_cost = sorted(list(set(r.rank_cost_name for r in results)))
    
    # Create separate heatmaps for different rank cost functions
    n_rank_costs = len(unique_rank_cost)
    n_cols = min(2, n_rank_costs)
    n_rows = (n_rank_costs + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_rank_costs == 1:
        axes = [axes]
    elif n_rows == 1 and n_cols > 1:
        axes = list(axes)
    elif n_rows > 1:
        axes = axes.flatten()
    else:
        axes = [axes]
    
    for rank_idx, rank_cost_name in enumerate(unique_rank_cost):
        if rank_idx >= len(axes):
            break
            
        ax = axes[rank_idx]
        
        # Filter results for this rank cost function
        rank_results = [r for r in results if r.rank_cost_name == rank_cost_name]
        
        # Create heatmap averaging over alpha values (or pick first alpha for each combination)
        heatmap_data = np.full((len(unique_act_thresh), len(unique_check_thresh)), np.nan)
        
        # For each (activation_threshold, check_threshold) combination, 
        # take the mean across all alpha values for this rank cost
        for act_idx, act_thresh in enumerate(unique_act_thresh):
            for check_idx, check_thresh in enumerate(unique_check_thresh):
                matching_results = [
                    r for r in rank_results 
                    if r.activation_threshold == act_thresh and r.check_threshold == check_thresh
                ]
                if matching_results:
                    # Average the statistic across all alpha values
                    stat_values = [statistic_func(r) for r in matching_results]
                    heatmap_data[act_idx, check_idx] = np.mean(stat_values)
        
        im = ax.imshow(heatmap_data, aspect='auto', cmap='viridis')
        ax.set_xticks(range(len(unique_check_thresh)))
        ax.set_xticklabels([f"{x:.3g}" for x in unique_check_thresh])
        ax.set_yticks(range(len(unique_act_thresh)))
        ax.set_yticklabels([f"{x:.3g}" for x in unique_act_thresh])
        ax.set_xlabel("Check Threshold")
        ax.set_ylabel("Activation Threshold")
        title = f"{statistic_name}\\n{rank_cost_name}"
        if len(unique_alpha) > 1:
            alpha_range = f"[{unique_alpha[0]:.3g}...{unique_alpha[-1]:.3g}]" if len(unique_alpha) > 2 else f"[{unique_alpha[0]:.3g}, {unique_alpha[-1]:.3g}]"
            title += f"\\n(avg over α∈{alpha_range})"
        ax.set_title(title)
        
        # Add colorbar
        plt.colorbar(im, ax=ax)
    
    # Hide empty subplots
    for i in range(n_rank_costs, len(axes)):
        if i < len(axes):
            axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.show()

# Stopping condition functions
def cost_ratio_stopping_condition(ratio: float = 2.0, metric: str = "non_diag_costs_min"):
    """Create stopping condition for when cost reaches ratio times original."""
    def condition(stats: dict[str, Any]) -> bool:
        costs = stats[metric]
        if len(costs) < 2:
            return False
        return costs[-1] >= costs[0] * ratio
    return condition

def iterations_stopping_condition(max_iters: int):
    """Create stopping condition for max iterations."""
    def condition(stats: dict[str, Any]) -> bool:
        return stats['iteration'] >= max_iters
    return condition