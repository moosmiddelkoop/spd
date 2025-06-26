"""Run SPD on a model."""

import time
from collections import defaultdict
from collections.abc import Iterator, Mapping
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Protocol, cast

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
from spd.models.component_utils import calc_causal_importances, calc_ci_l_zero
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

CI_ALIVE_THRESHOLD = 0.1


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


def eval(
    model: ComponentModel,
    components: dict[str, LinearComponent | EmbeddingComponent],
    gates: dict[str, Gate | GateMLP],
    device: str | torch.device,
    sigmoid_type: SigmoidTypes,
    data_iter: Iterator[dict[str, Any]],
    n_eval_microbatches: int,
    log_ce_losses: bool,
    Vs: Mapping[str, Tensor],
) -> dict[str, float | wandb.Table]:
    metrics = defaultdict[str, list[float]](list)

    all_causal_importance_BxM = defaultdict[str, list[Tensor]](list)

    for _ in range(n_eval_microbatches):
        batch = extract_batch_data(next(data_iter)).to(device)

        target_logits, pre_weight_acts = model.forward_with_pre_forward_cache_hooks(
            batch, module_names=list(components.keys())
        )

        # no upper-leaky needed bc no loss here
        ci_BxM, _ = calc_causal_importances(
            pre_weight_acts=pre_weight_acts,
            Vs=Vs,
            gates=gates,
            sigmoid_type=sigmoid_type,
            detach_inputs=False,
        )

        for k, v in ci_BxM.items():
            all_causal_importance_BxM[k].append(v)

        unmasked_component_logits = model.forward_with_components(
            batch,
            components=components,
            masks=None,
        )
        masked_component_logits = model.forward_with_components(
            batch,
            components=components,
            masks=ci_BxM,
        )

        metrics["misc/unmasked_kl_loss_vs_target"].append(
            calc_kl_divergence_lm(pred=unmasked_component_logits, target=target_logits).item()
        )
        metrics["misc/masked_kl_loss_vs_target"].append(
            calc_kl_divergence_lm(pred=masked_component_logits, target=target_logits).item()
        )

        if log_ce_losses:
            ce_losses = calc_ce_losses(
                model=model,
                batch=batch,
                components=components,
                masks=ci_BxM,
                unmasked_component_logits=unmasked_component_logits,
                masked_component_logits=masked_component_logits,
                target_logits=target_logits,
            )
            for name, value in ce_losses.items():
                metrics[name].append(value)

    log_data: dict[str, float | wandb.Table] = {k: sum(v) / len(v) for k, v in metrics.items()}

    # cat along batch bin
    stacked_ci_BxM = {k: torch.cat(v) for k, v in all_causal_importance_BxM.items()}

    embed_ci_table = create_embed_ci_sample_table(stacked_ci_BxM)
    if embed_ci_table is not None:
        log_data["misc/embed_ci_sample"] = embed_ci_table

    ci_l_zero = calc_ci_l_zero(stacked_ci_BxM)
    for layer_name, layer_ci_l_zero in ci_l_zero.items():
        log_data[f"metric/ci_l0_{layer_name}"] = layer_ci_l_zero

    return log_data


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
    # plot_results_fn: PlotResultsFn | None = None,
    tied_weights: list[tuple[str, str]] | None = None,
) -> None:
    """Run the optimization loop for LM decomposition."""

    model = ComponentModel(
        base_model=target_model,
        target_module_patterns=config.target_module_patterns,
        C=config.C,
        n_ci_mlp_neurons=config.n_ci_mlp_neurons,
        init_central=config.gate_init_central,
        pretrained_model_output_attr=config.pretrained_model_output_attr,
        dtype=torch.float32,  # torch.bfloat16 if config.autocast_bfloat16 else torch.float32,
    )
    model.to(device)

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

    Vs = {module_name: components[module_name].V for module_name in components}

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

    def autocast_ctx():
        if config.autocast_bfloat16:
            return torch.autocast(str(device), dtype=torch.bfloat16)
        return nullcontext()

    loop_start_time = time.perf_counter()

    for step in tqdm(range(config.steps + 1), ncols=0):
        step_lr = get_lr(step)

        for group in optimizer.param_groups:
            group["lr"] = step_lr

        optimizer.zero_grad()

        total_loss = 0.0
        loss_terms = defaultdict[str, float](float)

        for _microbatch_idx in range(config.gradient_accumulation_steps):
            with autocast_ctx():
                try:
                    batch_item = next(data_iter)
                except StopIteration:
                    logger.warning("Dataloader exhausted, resetting iterator.")
                    data_iter = iter(train_loader)
                    batch_item = next(data_iter)

                batch = extract_batch_data(batch_item).to(device)

                target_out, pre_weight_acts = model.forward_with_pre_forward_cache_hooks(
                    batch, module_names=list(components.keys())
                )

                ci_lower_leaky_BxM, ci_upper_leaky_BxM = calc_causal_importances(
                    pre_weight_acts=pre_weight_acts,
                    Vs=Vs,
                    gates=gates,
                    sigmoid_type=config.sigmoid_type,
                    detach_inputs=False,
                )

                for layer_name, ci_BxM in ci_upper_leaky_BxM.items():
                    alive_this_step_BxM = ci_BxM > CI_ALIVE_THRESHOLD
                    alive_this_step_M = reduce(alive_this_step_BxM, "... m -> m", torch.any)
                    alive_components_M[layer_name] |= alive_this_step_M

                microbatch_total_loss, microbatch_loss_terms = calculate_losses(
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

                scaled_loss = microbatch_total_loss / config.gradient_accumulation_steps

            # IMPORTANT: backward is outside of the loop
            scaled_loss.backward()
            # optimizer step is down after eval so we eval on the same exact model params as loss above

            # Bookkeeping
            total_loss += scaled_loss.item()
            for name, value in microbatch_loss_terms.items():
                loss_terms[name] += value / config.gradient_accumulation_steps

        if step % config.print_freq == 0:
            tqdm.write(f"--- Step {step} ---")
            tqdm.write(f"LR: {step_lr:.6f}")
            tqdm.write(f"Total Loss: {total_loss:.7f}")
            for name, value in loss_terms.items():
                tqdm.write(f"{name}: {value:.7f}")

            n_tokens = batch.numel() * config.gradient_accumulation_steps * step  # pyright: ignore [reportPossiblyUnboundVariable]
            tqdm.write(f"tokens/sec: {n_tokens / (time.perf_counter() - loop_start_time):.2f}")

            if config.wandb_project:
                log_data: dict[str, float | wandb.Table] = {
                    "loss/total": total_loss,
                    **loss_terms,
                    "lr": step_lr,
                    "tps": n_tokens / (time.perf_counter() - loop_start_time),
                    "n_tokens": n_tokens,
                }

                if step > 0:
                    # TODO: investigate whether this is wrong due to only running every n steps,
                    # i.e. off by a constant factor
                    for layer_name, layer_alive_components_M in alive_components_M.items():
                        log_data[f"metrics/n_alive_01/{layer_name}"] = (
                            layer_alive_components_M.sum().item()
                        )
                        alive_components_M[layer_name] = torch.zeros(config.C, device=device).bool()

                with torch.inference_mode():
                    eval_results = eval(
                        model=model,
                        components=components,
                        gates=gates,
                        data_iter=data_iter,
                        device=device,
                        sigmoid_type=config.sigmoid_type,
                        n_eval_microbatches=10,  # TODO configure
                        log_ce_losses=config.log_ce_losses,
                        Vs=Vs,
                    )

                log_data.update(eval_results)

                wandb.log(log_data, step=step)

        if (
            config.image_freq is not None
            and step % config.image_freq == 0
            and (step > 0 or config.image_on_first_step)
        ):
            logger.info(f"Step {step}: Generating plots...")

            n_tokens = {module_name.replace("-", "."): 0 for module_name in components}

            component_activation_counts = {
                module_name.replace("-", "."): torch.zeros(model.C, device=device)
                for module_name in components
            }

            for _ in range(n_eval_steps):
                batch = extract_batch_data(next(data_iter)).to(device)
                _, pre_weight_acts = model.forward_with_pre_forward_cache_hooks(
                    batch, module_names=list(components.keys())
                )
                causal_importances, _ = calc_causal_importances(
                    pre_weight_acts=pre_weight_acts,
                    Vs=Vs,
                    gates=gates,
                    sigmoid_type=config.sigmoid_type,
                    detach_inputs=False,
                )

                for module_name, ci in causal_importances.items():
                    # mask (batch, pos, C) or (batch, C)
                    n_tokens[module_name] += ci.shape[:-1].numel()

                    n_active_components_C = reduce(ci > CI_ALIVE_THRESHOLD, "... C -> C", torch.sum)
                    component_activation_counts[module_name] += n_active_components_C

            mean_component_activation_counts_plot = plot_mean_component_activation_counts(
                mean_component_activation_counts={
                    module_name: component_activation_counts[module_name] / n_tokens[module_name]
                    for module_name in components
                }
            )

            # a little gross but it's fine - just use the last loop's causal_importances
            ci_histogram_figs = plot_ci_histograms(causal_importances)  # pyright: ignore [reportPossiblyUnboundVariable]

            figs_dict = {
                **ci_histogram_figs,
                "mean_component_activation_counts": mean_component_activation_counts_plot,
            }
            fig_imgs_dict = {k: wandb.Image(v) for k, v in figs_dict.items()}

            wandb.log(fig_imgs_dict, step=step)

            if out_dir is not None:
                for k, v in figs_dict.items():
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

        # Skip gradient step if we are at the last step (last step just for plotting and logging)
        if step != config.steps:
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
