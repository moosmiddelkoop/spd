
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
    group_by: str = "alpha",
    figsize: tuple[int, int] = (15, 10),
    max_plots_per_figure: int = 16,
) -> None:
    """Plot evolution histories grouped by a hyperparameter."""
    
    # Group results by the specified parameter
    grouped_results: dict[Any, list[SweepResult]] = {}
    for result in results:
        key = getattr(result, group_by)
        if key not in grouped_results:
            grouped_results[key] = []
        grouped_results[key].append(result)
    
    # Create subplots
    n_groups = len(grouped_results)
    n_cols = min(4, n_groups)
    n_rows = (n_groups + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_groups == 1:
        axes = [axes]
    elif n_rows == 1 and n_cols > 1:
        axes = list(axes)
    elif n_rows > 1:
        axes = axes.flatten()
    else:
        axes = [axes]
    
    # Color map for different hyperparameter values
    colors = cm.viridis(np.linspace(0, 1, max(4, max(len(group) for group in grouped_results.values()))))
    
    for i, (group_key, group_results) in enumerate(grouped_results.items()):
        if i >= len(axes):
            break
            
        ax = axes[i]
        
        for j, result in enumerate(group_results[:max_plots_per_figure]):
            values = getattr(result, metric)
            iterations = range(len(values))
            
            # Create label with other hyperparameters
            other_params = {k: getattr(result, k) for k in ['activation_threshold', 'check_threshold', 'alpha', 'rank_cost_name']}
            other_params.pop(group_by)  # Remove the grouping parameter
            
            label = ", ".join([f"{k}={v}" for k, v in other_params.items()])
            
            ax.plot(iterations, values, color=colors[j % len(colors)], 
                   label=label, alpha=0.7, linewidth=1.5)
        
        ax.set_title(f"{group_by} = {group_key}")
        ax.set_xlabel("Iteration")
        ax.set_ylabel(metric.replace("_", " ").title())
        ax.grid(True, alpha=0.3)
        
        # Only show legend for first few plots to avoid clutter
        if i < 4 and len(group_results) <= 8:
            ax.legend(fontsize=8)
    
    # Hide empty subplots
    for i in range(n_groups, len(axes)):
        axes[i].set_visible(False)
    
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
        
        # Create heatmap data structure
        # We'll create multiple smaller heatmaps for different alpha values
        n_alphas = len(unique_alpha)
        
        if n_alphas <= 4:
            # Single heatmap with alpha as color intensity variation
            heatmap_data = np.full((len(unique_act_thresh), len(unique_check_thresh)), np.nan)
            
            for result in rank_results:
                act_idx = unique_act_thresh.index(result.activation_threshold)
                check_idx = unique_check_thresh.index(result.check_threshold)
                stat_value = statistic_func(result)
                heatmap_data[act_idx, check_idx] = stat_value
            
            im = ax.imshow(heatmap_data, aspect='auto', cmap='viridis')
            ax.set_xticks(range(len(unique_check_thresh)))
            ax.set_xticklabels([f"{x:.2f}" for x in unique_check_thresh])
            ax.set_yticks(range(len(unique_act_thresh)))
            ax.set_yticklabels([f"{x:.3f}" for x in unique_act_thresh])
            ax.set_xlabel("Check Threshold")
            ax.set_ylabel("Activation Threshold")
            ax.set_title(f"{statistic_name}\\n{rank_cost_name}")
            
            # Add colorbar
            plt.colorbar(im, ax=ax)
            
        else:
            # Multiple smaller heatmaps for each alpha
            ax.text(0.5, 0.5, f"Too many alphas ({n_alphas})\\nfor single heatmap", 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f"{rank_cost_name}")
    
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