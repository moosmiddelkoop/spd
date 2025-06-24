"""Run SPD on a model."""

from pathlib import Path
from typing import Protocol, cast

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from einops import reduce
from jaxtyping import Bool, Float, Int
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from spd.configs import Config
from spd.log import logger
from spd.losses import calc_ce_losses, calculate_losses
from spd.models.component_model import ComponentModel, init_Vs_and_Us_
from spd.models.component_utils import (
    calc_causal_importances,
    calc_ci_l_zero,
    component_activation_statistics,
)
from spd.models.components import EmbeddingComponent, Gate, GateMLP, LinearComponent
from spd.models.sigmoids import SigmoidTypes
from spd.plotting import (
    create_embed_ci_sample_table,
    plot_ci_histograms,
    plot_mean_component_activation_counts,
)
from spd.utils import (
    calc_kl_divergence_lm,
    extract_batch_data,
    get_lr_schedule_fn,
    get_lr_with_warmup,
)

CAUSAL_IMPORTANCE_ALIVE_THRESHOLD = 0.1


def get_common_run_name_suffix(config: Config) -> str:
    """Generate a run suffix based on Config that is common to all experiments."""
    run_suffix = ""
    run_suffix += f"nmasks{config.n_mask_samples}_"
    if config.stochastic_recon_coeff is not None:
        run_suffix += f"stochrecon{config.stochastic_recon_coeff:.2e}_"
    if config.stochastic_recon_layerwise_coeff is not None:
        run_suffix += f"stochreconlayer{config.stochastic_recon_layerwise_coeff:.2e}_"
    if config.schatten_coeff is not None:
        run_suffix += f"schatten{config.schatten_coeff:.2e}_"
    if config.embedding_recon_coeff is not None:
        run_suffix += f"embedrecon{config.embedding_recon_coeff:.2e}_"
    run_suffix += f"p{config.pnorm:.2e}_"
    run_suffix += f"impmin{config.importance_minimality_coeff:.2e}_"
    run_suffix += f"C{config.C}_"
    run_suffix += f"sd{config.seed}_"
    run_suffix += f"lr{config.lr:.2e}_"
    run_suffix += f"bs{config.batch_size}_"
    return run_suffix


class PlotResultsFn(Protocol):
    def __call__(
        self,
        model: ComponentModel,
        components: dict[str, LinearComponent | EmbeddingComponent],
        gates: dict[str, Gate | GateMLP],
        batch_shape: tuple[int, ...],
        device: str | torch.device,
        sigmoid_type: SigmoidTypes,
    ) -> dict[str, plt.Figure]: ...


def optimize(
    target_model: nn.Module,
    config: Config,
    device: str,
    train_loader: DataLoader[Int[Tensor, "..."]]
    | DataLoader[tuple[Float[Tensor, "..."], Float[Tensor, "..."]]],
    # eval_loader: DataLoader[Int[Tensor, "..."]]
    # | DataLoader[tuple[Float[Tensor, "..."], Float[Tensor, "..."]]],
    n_eval_steps: int,
    out_dir: Path | None,
    plot_results_fn: PlotResultsFn | None = None,
    tied_weights: list[tuple[str, str]] | None = None,
) -> None:
    """Run the optimization loop for LM decomposition."""

    model = ComponentModel(
        base_model=target_model,
        target_module_patterns=config.target_module_patterns,
        C=config.C,
        n_ci_mlp_neurons=config.n_ci_mlp_neurons,
        pretrained_model_output_attr=config.pretrained_model_output_attr,
    )

    for param in target_model.parameters():
        param.requires_grad = False
    logger.info("Target model parameters frozen.")

    # We used "-" instead of "." as module names can't have "." in them
    gates: dict[str, Gate | GateMLP] = {
        k.removeprefix("gates.").replace("-", "."): cast(Gate | GateMLP, v)
        for k, v in model.gates.items()
    }
    components: dict[str, LinearComponent | EmbeddingComponent] = {
        k.removeprefix("components.").replace("-", "."): cast(
            LinearComponent | EmbeddingComponent, v
        )
        for k, v in model.components.items()
    }

    model.to(device)
    init_Vs_and_Us_(model=model, components=components)

    if tied_weights is not None:
        # Tie component weights. Assume that the first element is a transpose of the second element
        # NOTE: Tying weights will make your training nondeterministic
        for src_name, tgt_name in tied_weights:
            components[tgt_name].U.data = components[src_name].V.data.T
            components[tgt_name].V.data = components[src_name].U.data.T

    component_params: list[torch.nn.Parameter] = []
    gate_params: list[torch.nn.Parameter] = []
    for name, component in components.items():
        component_params.extend(list(component.parameters()))
        gate_params.extend(list(gates[name].parameters()))

    assert len(component_params) > 0, "No parameters found in components to optimize"

    parameters = component_params + gate_params
    optimizer = optim.AdamW(parameters, lr=config.lr, weight_decay=0)

    lr_schedule_fn = get_lr_schedule_fn(config.lr_schedule, config.lr_exponential_halflife)
    logger.info(f"Base LR scheduler created: {config.lr_schedule}")

    n_params = sum(model.model.get_parameter(n + ".weight").numel() for n in components)

    data_iter = iter(train_loader)

    alive_components_M: dict[str, Bool[Tensor, " C"]] = {
        layer_name: torch.zeros(config.C, device=device).bool() for layer_name in components
    }

    def get_lr(step: int) -> float:
        return get_lr_with_warmup(
            step=step,
            steps=config.steps,
            lr=config.lr,
            lr_schedule_fn=lr_schedule_fn,
            lr_warmup_pct=config.lr_warmup_pct,
        )

    # Iterate one extra step for final logging/plotting/saving
    for step in tqdm(range(config.steps + 1), ncols=0):
        step_lr = get_lr(step)

        for group in optimizer.param_groups:
            group["lr"] = step_lr

        optimizer.zero_grad()

        try:
            batch_item = next(data_iter)
            batch = extract_batch_data(batch_item)
        except StopIteration:
            logger.warning("Dataloader exhausted, resetting iterator.")
            data_iter = iter(train_loader)
            batch_item = next(data_iter)
            batch = extract_batch_data(batch_item)
        batch = batch.to(device)

        target_out, pre_weight_acts = model.forward_with_pre_forward_cache_hooks(
            batch, module_names=list(components.keys())
        )
        Vs = {module_name: components[module_name].V for module_name in components}

        ci_lower_leaky_BxM, ci_upper_leaky_BxM = calc_causal_importances(
            pre_weight_acts=pre_weight_acts,
            Vs=Vs,
            gates=gates,
            sigmoid_type=config.sigmoid_type,
            detach_inputs=False,
        )

        for layer_name, ci_BxM in ci_upper_leaky_BxM.items():
            alive_this_step_BxM = ci_BxM > CAUSAL_IMPORTANCE_ALIVE_THRESHOLD
            alive_this_step_M = reduce(alive_this_step_BxM, "... m -> m", torch.any)
            alive_components_M[layer_name] = alive_components_M[layer_name] | alive_this_step_M

        total_loss, loss_terms = calculate_losses(
            model=model,
            batch=batch,
            config=config,
            components=components,
            ci_lower_leaky=ci_lower_leaky_BxM,
            ci_upper_leaky=ci_upper_leaky_BxM,
            target_out=target_out,
            device=device,
            n_params=n_params,
        )

        with torch.inference_mode():
            # --- Logging --- #
            if step % config.print_freq == 0:
                log_data: dict[str, float | wandb.Table] = {
                    "loss/total": total_loss.item(),
                    **loss_terms,
                    "lr": step_lr,
                }

                tqdm.write(f"--- Step {step} ---")
                tqdm.write(f"LR: {step_lr:.6f}")
                tqdm.write(f"Total Loss: {total_loss.item():.7f}")
                for name, value in loss_terms.items():
                    tqdm.write(f"{name}: {value:.7f}")

                if step > 0:
                    for layer_name, layer_alive_components_M in alive_components_M.items():
                        log_data[f"metric/n_alive_01/{layer_name}"] = (
                            layer_alive_components_M.sum().item()
                        )
                        alive_components_M[layer_name] = torch.zeros(config.C, device=device).bool()

                # Calculate component logits and KL losses
                masked_component_logits = model.forward_with_components(
                    batch, components=components, masks=ci_lower_leaky_BxM
                )
                unmasked_component_logits = model.forward_with_components(
                    batch, components=components, masks=None
                )

                target_logits = model(batch)

                log_data["misc/unmasked_kl_loss_vs_target"] = calc_kl_divergence_lm(
                    pred=unmasked_component_logits, target=target_logits
                ).item()
                log_data["misc/masked_kl_loss_vs_target"] = calc_kl_divergence_lm(
                    pred=masked_component_logits, target=target_logits
                ).item()

                if config.log_ce_losses:
                    ce_losses = calc_ce_losses(
                        model=model,
                        batch=batch,
                        components=components,
                        masks=ci_lower_leaky_BxM,
                        unmasked_component_logits=unmasked_component_logits,
                        masked_component_logits=masked_component_logits,
                        target_logits=target_logits,
                    )
                    log_data.update(ce_losses)

                embed_ci_table = create_embed_ci_sample_table(ci_lower_leaky_BxM)
                if embed_ci_table is not None:
                    log_data["misc/embed_ci_sample"] = embed_ci_table

                if config.wandb_project:
                    ci_l_zero = calc_ci_l_zero(causal_importances=ci_lower_leaky_BxM)
                    for layer_name, layer_ci_l_zero in ci_l_zero.items():
                        log_data[f"metric/ci_l0_{layer_name}"] = layer_ci_l_zero
                    wandb.log(log_data, step=step)

            # --- Plotting --- #
            if (
                config.image_freq is not None
                and step % config.image_freq == 0
                and (step > 0 or config.image_on_first_step)
            ):
                logger.info(f"Step {step}: Generating plots...")
                fig_dict = {}
                if plot_results_fn is not None:
                    fig_dict = plot_results_fn(
                        model=model,
                        components=components,
                        gates=gates,
                        batch_shape=batch.shape,
                        device=device,
                        sigmoid_type=config.sigmoid_type,
                    )

                ci_histogram_figs = plot_ci_histograms(causal_importances=ci_lower_leaky_BxM)
                fig_dict.update(ci_histogram_figs)

                mean_component_activation_counts = component_activation_statistics(
                    model=model,
                    # dataloader=eval_loader,
                    data_iter=data_iter,
                    n_steps=n_eval_steps,
                    device=device,
                    sigmoid_type=config.sigmoid_type,
                )[1]
                assert mean_component_activation_counts is not None
                fig_dict["mean_component_activation_counts"] = (
                    plot_mean_component_activation_counts(
                        mean_component_activation_counts=mean_component_activation_counts,
                    )
                )

                if config.wandb_project:
                    wandb.log(
                        {k: wandb.Image(v) for k, v in fig_dict.items()},
                        step=step,
                    )
                    if out_dir is not None:
                        for k, v in fig_dict.items():
                            v.savefig(out_dir / f"{k}_{step}.png")
                            tqdm.write(f"Saved plot to {out_dir / f'{k}_{step}.png'}")

        # --- Saving Checkpoint --- #
        if (
            (config.save_freq is not None and step % config.save_freq == 0 and step > 0)
            or step == config.steps
        ) and out_dir is not None:
            torch.save(model.state_dict(), out_dir / f"model_{step}.pth")
            logger.info(f"Saved model, optimizer, and out_dir to {out_dir}")
            if config.wandb_project:
                wandb.save(str(out_dir / f"model_{step}.pth"), base_path=str(out_dir), policy="now")
                wandb.save(
                    str(out_dir / f"optimizer_{step}.pth"), base_path=str(out_dir), policy="now"
                )

        # --- Backward Pass & Optimize --- #
        # Skip gradient step if we are at the last step (last step just for plotting and logging)
        if step != config.steps:
            total_loss.backward(retain_graph=True)

            if step % config.print_freq == 0 and config.wandb_project:
                with torch.no_grad():
                    grad_norm: Float[Tensor, ""] = torch.zeros((), device=device)
                    for param in model.parameters():
                        if param.grad is not None:
                            grad_norm += param.grad.data.flatten().pow(2).sum()
                    grad_norm_val = grad_norm.sqrt().item()
                    wandb.log({"grad_norm": grad_norm_val}, step=step)

            norm_torch = torch.nn.utils.clip_grad_norm_(parameters, max_norm=1.0)
            wandb.log({"grad_norm_torch": norm_torch}, step=step)

            optimizer.step()

    logger.info("Finished training loop.")
