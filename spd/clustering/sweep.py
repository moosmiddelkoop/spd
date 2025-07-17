
import matplotlib.pyplot as plt
import torch
from muutils.dbg import dbg_auto

from spd.clustering.activations import component_activations, process_activations
from spd.clustering.merge import compute_merge_costs, merge_iteration
from spd.clustering.merge_matrix import GroupMerge
from spd.experiments.resid_mlp.resid_mlp_dataset import ResidualMLPDataset
from spd.models.component_model import ComponentModel
from spd.utils.data_utils import DatasetGeneratedDataLoader

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Tuple, Union
import numpy as np
import itertools
from concurrent.futures import ProcessPoolExecutor
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import seaborn as sns

@dataclass
class SweepConfig:
    """Configuration for hyperparameter sweep."""
    # Core parameters to sweep
    activation_thresholds: list[float]
    check_thresholds: list[float] 
    alphas: list[float]
    rank_cost_funcs: list[Callable[[float], float]]
    
    # Fixed parameters
    iters: int = 100
    plot_every: int|None = None  # Disable plotting during sweep
    
    # Sweep control
    max_workers: int = 4
    
    def __post_init__(self):
        """Add names for rank cost functions for plotting."""
        if not hasattr(self, 'rank_cost_names'):
            self.rank_cost_names = [f"rank_cost_{i}" for i in range(len(self.rank_cost_funcs))]

@dataclass 
class SweepResult:
    """Results from a single hyperparameter configuration."""
    # Hyperparameters
    activation_threshold: float
    check_threshold: float
    alpha: float
    rank_cost_name: str
    
    # Results
    non_diag_costs_min: list[float]
    non_diag_costs_max: list[float] 
    max_considered_cost: list[float]
    selected_pair_cost: list[float]
    total_iterations: int
    final_k_groups: int
    
    @property
    def param_str(self) -> str:
        """String representation of hyperparameters."""
        return f"thresh={self.activation_threshold:.3f}, check={self.check_threshold:.2f}, α={self.alpha:.2f}, rank={self.rank_cost_name}"

def run_single_sweep(
    coact_orig: torch.Tensor,
    activation_mask_orig: torch.Tensor, 
    activation_threshold: float,
    check_threshold: float,
    alpha: float,
    rank_cost_func: Callable[[float], float],
    rank_cost_name: str,
    iters: int = 100,
) -> SweepResult:
    """Run merge_iteration for a single hyperparameter configuration."""
    
    # Apply activation threshold
    coact_bool = coact_orig > activation_threshold
    
    # Run merge iteration without plotting
    results = merge_iteration(
        coact=coact_bool.float().T @ coact_bool.float(),
        activation_mask=coact_bool,
        check_threshold=check_threshold,
        alpha=alpha,
        rank_cost=rank_cost_func,
        iters=iters,
        plot_every=None,  # No plotting during sweep
        component_labels=None,
        plot_final=False,
    )
    
    return SweepResult(
        activation_threshold=activation_threshold,
        check_threshold=check_threshold, 
        alpha=alpha,
        rank_cost_name=rank_cost_name,
        non_diag_costs_min=results['non_diag_costs_min'],
        non_diag_costs_max=results['non_diag_costs_max'],
        max_considered_cost=results['max_considered_cost'],
        selected_pair_cost=results['selected_pair_cost'],
        total_iterations=results['total_iterations'],
        final_k_groups=results['final_k_groups'],
    )

def run_hyperparameter_sweep(
    activation_mask: torch.Tensor,
    sweep_config: SweepConfig,
) -> list[SweepResult]:
    """Run hyperparameter sweep across all parameter combinations."""
    
    # Generate all parameter combinations
    param_combinations = list(itertools.product(
        sweep_config.activation_thresholds,
        sweep_config.check_thresholds,
        sweep_config.alphas,
        sweep_config.rank_cost_funcs,
    ))
    
    print(f"Running sweep with {len(param_combinations)} parameter combinations...")
    
    results = []
    
    # Run sweep (sequential for now to avoid complexity with torch/cuda in multiprocessing)
    for i, (act_thresh, check_thresh, alpha, rank_func) in enumerate(param_combinations):
        print(f"Running {i+1}/{len(param_combinations)}: "
              f"act_thresh={act_thresh}, check_thresh={check_thresh}, alpha={alpha}, rank={rank_name}")
        
        try:
            result = run_single_sweep(
                coact_orig=coact,
                activation_mask_orig=activation_mask,
                activation_threshold=act_thresh,
                check_threshold=check_thresh, 
                alpha=alpha,
                rank_cost_func=rank_func,
                iters=sweep_config.iters,
            )
            results.append(result)
        except Exception as e:
            print(f"Failed for params {act_thresh}, {check_thresh}, {alpha}, {rank_name}: {e}")
            continue
    
    print(f"Completed sweep with {len(results)} successful runs")
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
    elif n_rows == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
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

def compute_iterations_until_cost_ratio(
    result: SweepResult, 
    cost_ratio_threshold: float = 2.0,
    metric: str = "non_diag_costs_min"
) -> int:
    """Compute iterations until cost reaches ratio times the original cost."""
    values = getattr(result, metric)
    if len(values) == 0:
        return 0
    
    original_cost = values[0]
    target_cost = original_cost * cost_ratio_threshold
    
    for i, cost in enumerate(values):
        if cost >= target_cost:
            return i
    
    return len(values)  # Never reached the threshold

def create_heatmaps(
    results: list[SweepResult],
    statistic_func: Callable[[SweepResult], float] = lambda r: compute_iterations_until_cost_ratio(r, 2.0),
    statistic_name: str = "Iterations until 2x cost",
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
    elif n_rows == 1:
        axes = axes if n_cols > 1 else [axes]
    else:
        axes = axes.flatten()
    
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