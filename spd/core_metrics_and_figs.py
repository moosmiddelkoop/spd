"""Core metrics and figures for SPD experiments.

This file contains the default metrics and visualizations that are logged during SPD optimization.
These are separate from user-defined metrics/figures to allow for easier comparison and extension.
"""
# pyright: reportMissingImports=false

import torch
import wandb
from jaxtyping import Float, Int
from matplotlib import pyplot as plt
from torch import Tensor
from torch.utils.data import DataLoader

from spd.configs import Config
from spd.losses import calc_ce_losses
from spd.models.component_model import ComponentModel
from spd.models.components import EmbeddingComponent, Gate, GateMLP, LinearComponent
from spd.plotting import (
    create_embed_ci_sample_table,
    get_single_feature_causal_importances,
    plot_ci_histograms,
    plot_mean_component_activation_counts,
    plot_single_feature_causal_importances,
    plot_UV_matrices,
)
from spd.registry import EXPERIMENT_REGISTRY, has_ci_solution
from spd.utils.component_utils import calc_ci_l_zero, component_activation_statistics
from spd.utils.general_utils import calc_kl_divergence_lm
from spd.utils.target_ci_solutions import compute_target_metrics

try:
    from spd.user_metrics_and_figs import compute_user_metrics, create_user_figures
except ImportError:
    compute_user_metrics = None
    create_user_figures = None


def create_metrics(
    model: ComponentModel,
    components: dict[str, LinearComponent | EmbeddingComponent],
    gates: dict[str, Gate | GateMLP],
    causal_importances: dict[str, Float[Tensor, "... C"]],
    target_out: Float[Tensor, "... d_model_out"],
    batch: Tensor,
    device: str,
    config: Config,
    step: int,
    evals_id: str | None = None,
    targets: Tensor | None = None,
) -> dict[str, float | int | wandb.Table]:
    """Create metrics for logging."""
    metrics: dict[str, float | int | wandb.Table] = {"misc/step": step}

    masked_component_out = model.forward_with_components(
        batch, components=components, masks=causal_importances
    )
    unmasked_component_out = model.forward_with_components(batch, components=components, masks=None)

    if config.task_config.task_name == "lm":
        metrics["misc/unmasked_kl_loss_vs_target"] = calc_kl_divergence_lm(
            pred=unmasked_component_out, target=target_out
        ).item()
        metrics["misc/masked_kl_loss_vs_target"] = calc_kl_divergence_lm(
            pred=masked_component_out, target=target_out
        ).item()

    if config.log_ce_losses:
        ce_losses = calc_ce_losses(
            model=model,
            batch=batch,
            components=components,
            masks=causal_importances,
            unmasked_component_logits=unmasked_component_out,
            masked_component_logits=masked_component_out,
            target_logits=target_out,
        )
        metrics.update(ce_losses)

    for key in ["transformer.wte", "model.embed_tokens"]:
        if key in causal_importances:
            embed_ci_table = create_embed_ci_sample_table(causal_importances, key)
            metrics["misc/embed_ci_sample"] = embed_ci_table
            break

    # Causal importance L0
    ci_l_zero = calc_ci_l_zero(causal_importances=causal_importances)
    for layer_name, layer_ci_l_zero in ci_l_zero.items():
        metrics[f"{layer_name}/ci_l0"] = layer_ci_l_zero

    # Canonical solution metrics
    if evals_id is not None and config.task_config.task_name in ["tms", "residual_mlp"]:
        if has_ci_solution(evals_id):
            # Get causal importance arrays using single active features
            ci_arrays, _ = get_single_feature_causal_importances(
                model=model,
                components=components,
                gates=gates,
                batch_shape=batch.shape,
                device=device,
                input_magnitude=0.75,
                sigmoid_type=config.sigmoid_type,
            )

            target_solution = EXPERIMENT_REGISTRY[evals_id].target_solution
            assert target_solution is not None  # Guaranteed by has_ci_solution check
            target_metrics = compute_target_metrics(
                causal_importances=ci_arrays,
                target_solution=target_solution,
            )
            metrics.update(target_metrics)

    if compute_user_metrics is not None:
        user_metrics = compute_user_metrics(
            model=model,
            components=components,
            gates=gates,
            causal_importances=causal_importances,
            unmasked_component_out=unmasked_component_out,
            masked_component_out=masked_component_out,
            target_out=target_out,
            batch=batch,
            device=device,
            config=config,
            step=step,
        )
        metrics.update(user_metrics)

    return metrics


def create_figures(
    model: ComponentModel,
    components: dict[str, LinearComponent | EmbeddingComponent],
    gates: dict[str, Gate | GateMLP],
    causal_importances: dict[str, Float[Tensor, "... C"]],
    target_out: Float[Tensor, "... d_model_out"],
    batch: Int[Tensor, "... d_model_in"] | Float[Tensor, "... d_model_in"],
    device: str | torch.device,
    config: Config,
    step: int,
    eval_loader: DataLoader[Int[Tensor, "..."]]
    | DataLoader[tuple[Float[Tensor, "..."], Float[Tensor, "..."]]],
    n_eval_steps: int,
    evals_id: str | None = None,
) -> dict[str, plt.Figure]:
    """Create figures for logging.

    Args:
        model: The ComponentModel
        components: Dictionary of components
        gates: Dictionary of gates
        causal_importances: Current causal importances
        target_out: Output of target model
        batch: Current batch tensor
        device: Current device (cuda/cpu)
        config: The full configuration object
        step: Current training step
        eval_loader: Evaluation loader
        n_eval_steps: Number of evaluation steps
        evals_id: Optional experiment ID for target-aware permutation

    Returns:
        Dictionary of figures
    """
    fig_dict = {}

    # Core plots for all experiments
    ci_histogram_figs = plot_ci_histograms(causal_importances=causal_importances)
    fig_dict.update(ci_histogram_figs)

    mean_component_activation_counts = component_activation_statistics(
        model=model, dataloader=eval_loader, n_steps=n_eval_steps, device=str(device)
    )[1]
    fig_dict["mean_component_activation_counts"] = plot_mean_component_activation_counts(
        mean_component_activation_counts=mean_component_activation_counts,
    )

    # TMS and ResidMLP experiments get causal importance value plots and UV matrix plots
    if config.task_config.task_name in ["tms", "residual_mlp"]:
        figures, all_perm_indices = plot_single_feature_causal_importances(
            model=model,
            components=components,
            gates=gates,
            batch_shape=batch.shape,
            device=device,
            input_magnitude=0.75,
            sigmoid_type=config.sigmoid_type,
            experiment_id=evals_id,
        )
        fig_dict.update(figures)

        fig_dict["UV_matrices"] = plot_UV_matrices(
            components=components, all_perm_indices=all_perm_indices
        )

    if create_user_figures is not None:
        user_figures = create_user_figures(
            model=model,
            components=components,
            gates=gates,
            causal_importances=causal_importances,
            target_out=target_out,
            batch=batch,
            device=device,
            config=config,
            step=step,
            eval_loader=eval_loader,
        )
        fig_dict.update(user_figures)
    return fig_dict
