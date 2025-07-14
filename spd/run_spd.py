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
from jaxtyping import Float
from torch import Tensor
from torch.utils.data import DataLoader
from torch.profiler import (
    profile,
    record_function,
    ProfilerActivity,
    tensorboard_trace_handler,
)
from tqdm import tqdm

from spd.configs import Config
from spd.constants import CI_ALIVE_THRESHOLD
from spd.log import logger
from spd.losses import calc_ce_losses, calc_faithfulness_fvu, calculate_losses
from spd.models.component_model import ComponentModel, init_Vs_and_Us_
from spd.models.component_utils import calc_causal_importances, calc_ci_l_zero
from spd.models.components import EmbeddingComponent, Gate, GateMLP, LinearComponent
from spd.models.sigmoids import SigmoidTypes
from spd.plotting import plot_ci_histograms, plot_mean_component_activation_counts
from spd.utils import extract_batch_data, get_lr_schedule_fn, get_lr_with_warmup
from spd.wandb_utils import WandbSections


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
    device: str,
    sigmoid_type: SigmoidTypes,
    eval_iter: Iterator[Tensor],
    n_eval_microbatches: int,
    log_ce_losses: bool,
    Vs: Mapping[str, Tensor],
) -> dict[str, float | wandb.Table]:
    metrics = defaultdict[str, list[float]](list)

    all_causal_importance_BxM = defaultdict[str, list[Tensor]](list)

    for _ in range(n_eval_microbatches):
        batch = next(eval_iter).to(device)

        target_logits, pre_weight_acts = model.forward_with_pre_forward_cache_hooks(
            batch, module_names=list(components.keys())
        )

        # no upper-leaky needed bc no loss here
        ci_BxM, _, _ = calc_causal_importances(
            pre_weight_acts=pre_weight_acts,
            Vs=Vs,
            gates=gates,
            sigmoid_type=sigmoid_type,
            detach_inputs=False,
        )

        for k, v in ci_BxM.items():
            all_causal_importance_BxM[k].append(v)

        if log_ce_losses:
            ce_losses = calc_ce_losses(
                model=model,
                batch_BS=batch,
                components=components,
                ci_masks_BxM=ci_BxM,
                target_logits_BSV=target_logits,
            )
            for name, value in ce_losses.items():
                metrics[name].append(value)

    log_data: dict[str, float | wandb.Table] = {k: sum(v) / len(v) for k, v in metrics.items()}

    # cat along batch bin
    stacked_ci_BxM = {k: torch.cat(v) for k, v in all_causal_importance_BxM.items()}

    # embed_ci_table = create_embed_ci_sample_table(stacked_ci_BxM)
    # if embed_ci_table is not None:
    #     log_data["misc/embed_ci_sample"] = embed_ci_table

    ci_l_zero = calc_ci_l_zero(stacked_ci_BxM, cutoff=0.1)
    if len(ci_l_zero) == 1:
        log_data[f"{WandbSections.METRICS.value}/ci_l0_01"] = list(ci_l_zero.values())[0]
    else:
        for layer_name, layer_ci_l_zero in ci_l_zero.items():
            log_data[f"{WandbSections.METRICS.value}/ci_l0_01_{layer_name}"] = layer_ci_l_zero

    ci_l_zero = calc_ci_l_zero(stacked_ci_BxM, cutoff=0.01)
    if len(ci_l_zero) == 1:
        log_data[f"{WandbSections.METRICS.value}/ci_l0_001"] = list(ci_l_zero.values())[0]
    else:
        for layer_name, layer_ci_l_zero in ci_l_zero.items():
            log_data[f"{WandbSections.METRICS.value}/ci_l0_001_{layer_name}"] = layer_ci_l_zero

    ci_l_zero = calc_ci_l_zero(stacked_ci_BxM, cutoff=0.5)
    if len(ci_l_zero) == 1:
        log_data[f"{WandbSections.METRICS.value}/ci_l0_05"] = list(ci_l_zero.values())[0]
    else:
        for layer_name, layer_ci_l_zero in ci_l_zero.items():
            log_data[f"{WandbSections.METRICS.value}/ci_l0_05_{layer_name}"] = layer_ci_l_zero

    log_data[f"{WandbSections.METRICS.value}/fvu"] = calc_faithfulness_fvu(
        components=components,
        target_model=model.model,
        device=device,
    ).item()

    return log_data


def dl_loop_wrapper(
    dl: DataLoader[dict[str, Any] | tuple[torch.Tensor, ...] | torch.Tensor], name: str
) -> Iterator[Tensor]:
    iterator = iter(dl)
    while True:
        try:
            batch_item = next(iterator)
        except StopIteration:
            logger.warning(f"Dataloader '{name}' exhausted, resetting iterator.")
            iterator = iter(dl)
            batch_item = next(iterator)
        yield extract_batch_data(batch_item)


def get_target_module_mean_input_norms(
    model: ComponentModel,
    target_module_patterns: list[str],
    train_iter: Iterator[Tensor],
    device: str,
    n_tokens: int = 100_000,
) -> dict[str, float]:
    target_module_input_norms = defaultdict[str, list[float]](list)
    n_tokens_seen = 0
    pbar = tqdm(total=n_tokens, desc="Computing target module mean input norms")
    while n_tokens_seen < n_tokens:
        batch = next(train_iter).to(device)
        n_tokens_seen += batch.numel()
        pbar.update(batch.numel())
        _, pre_weight_acts = model.forward_with_pre_forward_cache_hooks(
            batch, module_names=target_module_patterns
        )
        for name, act in pre_weight_acts.items():
            assert act.ndim == 3, "expected 3D tensor (b, s, d)"
            target_module_input_norms[name].append(act.norm(dim=-1).mean().item())
    pbar.close()
    return {name: sum(norms) / len(norms) for name, norms in target_module_input_norms.items()}


class AliveTracker:
    def __init__(
        self,
        module_names: list[str],
        C: int,
        device: torch.device,
        n_batches_until_dead: int,
        threshold: float,
    ):
        self.module_names = module_names
        self.batches_since_fired_C = {
            module_name: torch.zeros(C, dtype=torch.int64, device=device)
            for module_name in module_names
        }
        self.n_batches_until_dead = n_batches_until_dead
        self.threshold = threshold

    @torch.no_grad()
    def watch_batch(self, importance_vals_dict_BxC: dict[str, Tensor]) -> None:
        assert set(importance_vals_dict_BxC.keys()) == set(self.module_names), (
            "importance_vals_BxC must have the same keys as module_names"
        )
        for module_name, importance_vals_BxC in importance_vals_dict_BxC.items():
            component_is_alive_C = reduce(
                importance_vals_BxC > self.threshold, "... C -> C", torch.any
            )
            batches_since_fired_C = torch.where(
                component_is_alive_C, 0, self.batches_since_fired_C[module_name] + 1
            )
            self.batches_since_fired_C[module_name] = batches_since_fired_C

    @torch.no_grad()
    def n_alive(self) -> dict[str, Tensor]:
        return {
            module_name: (self.batches_since_fired_C[module_name] < self.n_batches_until_dead).sum()
            for module_name in self.module_names
        }


# Profiler schedule:
# - wait: initial steps to skip before profiling starts (e.g., for warm-up)
# - warmup: steps where profiler collects data but discards it (to reduce overhead)
# - active: steps where profiler records events
# - repeat: number of times to repeat the wait-warmup-active cycle

# Trace handler: saves profiling results for TensorBoard
from datetime import datetime

logdir = f"./pt_logs/{datetime.now().strftime('%Y%m%d_%H%M%S')}/"
Path(logdir).mkdir(parents=True, exist_ok=True)
trace_handler = tensorboard_trace_handler(logdir)

# record_function = lambda *args, **kwargs: nullcontext()


def optimize(
    target_model: nn.Module,
    config: Config,
    device: str,
    train_loader: DataLoader[dict[str, Any] | tuple[torch.Tensor, ...] | torch.Tensor],
    eval_loader: DataLoader[dict[str, Any] | tuple[torch.Tensor, ...] | torch.Tensor],
    n_eval_steps: int,
    out_dir: Path | None,
    plot_results_fn: PlotResultsFn | None = None,
    tied_weights: list[tuple[str, str]] | None = None,
) -> None:
    """Run the optimization loop for LM decomposition."""
    train_iter = dl_loop_wrapper(train_loader, "train")
    eval_iter = dl_loop_wrapper(eval_loader, "eval")

    model = ComponentModel(
        base_model=target_model,
        target_module_patterns=config.target_module_patterns,
        C=config.C,
        type_=config.gate_type,
        n_ci_mlp_neurons=config.n_ci_mlp_neurons,
        init_central=config.gate_init_central,
        pretrained_model_output_attr=config.pretrained_model_output_attr,
        dtype=torch.float32,
    )
    model.to(device)

    alive_tracker_01 = AliveTracker(
        module_names=config.target_module_patterns,
        C=config.C,
        device=torch.device(device),
        n_batches_until_dead=(
            config.n_tokens_till_dead // (config.batch_size * config.task_config.max_seq_len)  # pyright: ignore [reportAttributeAccessIssue]
        ),
        threshold=0.1,
    )

    alive_tracker_001 = AliveTracker(
        module_names=config.target_module_patterns,
        C=config.C,
        device=torch.device(device),
        n_batches_until_dead=(
            config.n_tokens_till_dead // (config.batch_size * config.task_config.max_seq_len)  # pyright: ignore [reportAttributeAccessIssue]
        ),
        threshold=0.01,
    )

    alive_tracker_05 = AliveTracker(
        module_names=config.target_module_patterns,
        C=config.C,
        device=torch.device(device),
        n_batches_until_dead=(
            config.n_tokens_till_dead // (config.batch_size * config.task_config.max_seq_len)  # pyright: ignore [reportAttributeAccessIssue]
        ),
        threshold=0.5,
    )

    target_module_mean_input_norms = get_target_module_mean_input_norms(
        model=model,
        target_module_patterns=config.target_module_patterns,
        train_iter=train_iter,
        device=device,
    )
    model.init_gates_from_mean_input_norms_(target_module_mean_input_norms)

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

    total_loss = torch.tensor(0.0, device=device)
    loss_terms = defaultdict[str, Tensor](lambda: torch.tensor(0.0, device=device))

    for step in tqdm(range(config.steps + 1), ncols=0):
        step_lr = get_lr(step)

        for group in optimizer.param_groups:
            group["lr"] = step_lr

        optimizer.zero_grad()

        for _microbatch_idx in range(config.gradient_accumulation_steps):
            with autocast_ctx():
                batch = next(train_iter).to(device)

                target_out, pre_weight_acts = model.forward_with_pre_forward_cache_hooks(
                    batch, module_names=list(components.keys())
                )

                ci_lower_leaky_BxM, ci_upper_leaky_BxM, _ = calc_causal_importances(
                    pre_weight_acts=pre_weight_acts,
                    Vs=Vs,
                    gates=gates,
                    sigmoid_type=config.sigmoid_type,
                    detach_inputs=False,
                )

                alive_tracker_01.watch_batch(ci_upper_leaky_BxM)
                alive_tracker_001.watch_batch(ci_upper_leaky_BxM)
                alive_tracker_05.watch_batch(ci_upper_leaky_BxM)

                microbatch_total_loss, microbatch_loss_terms = calculate_losses(
                    model=model,
                    batch=batch,
                    config=config,
                    components=components,
                    ci_lower_leaky=ci_lower_leaky_BxM,
                    ci_upper_leaky=ci_upper_leaky_BxM,
                    target_out=target_out,
                    device=device,
                    training_pct=step / config.steps,
                    n_params=n_params,
                )

                scaled_loss = microbatch_total_loss / config.gradient_accumulation_steps

            # IMPORTANT: backward is outside of the loop
            scaled_loss.backward()
            # optimizer step is down after eval so we eval on the same exact model params as loss above

            # Bookkeeping
            total_loss += microbatch_total_loss.detach()
            for name, value in microbatch_loss_terms.items():
                loss_terms[name] += value.detach()

        if step % config.print_freq == 0:
            avg_loss = total_loss / config.gradient_accumulation_steps / config.print_freq
            avg_loss_terms = {
                k: v / config.gradient_accumulation_steps / config.print_freq
                for k, v in loss_terms.items()
            }
            total_loss = torch.tensor(0.0, device=device)
            loss_terms.clear()

            tqdm.write(f"--- Step {step} ---")
            tqdm.write(f"LR: {step_lr:.6f}")
            tqdm.write(f"Total Loss: {avg_loss:.7f}")
            for name, value in avg_loss_terms.items():
                tqdm.write(f"{name}: {value:.7f}")

            n_tokens = batch.numel() * config.gradient_accumulation_steps * step  # pyright: ignore [reportPossiblyUnboundVariable]
            tqdm.write(f"tokens/sec: {n_tokens / (time.perf_counter() - loop_start_time):.2f}")

            if config.wandb_project:
                log_data: dict[str, float | wandb.Table] = {
                    f"{WandbSections.LOSS.value}/total": float(avg_loss.item()),
                    **{k: float(v.item()) for k, v in avg_loss_terms.items()},
                    f"{WandbSections.TRAIN.value}/lr": step_lr,
                    f"{WandbSections.TRAIN.value}/tps": n_tokens
                    / (time.perf_counter() - loop_start_time),
                    f"{WandbSections.TRAIN.value}/n_tokens": n_tokens,
                }

                if step > 0:
                    for module_name, n_alive in alive_tracker_01.n_alive().items():
                        log_data[f"metrics/n_alive_01/{module_name}"] = n_alive.item()
                    for module_name, n_alive in alive_tracker_001.n_alive().items():
                        log_data[f"metrics/n_alive_001/{module_name}"] = n_alive.item()
                    for module_name, n_alive in alive_tracker_05.n_alive().items():
                        log_data[f"metrics/n_alive_05/{module_name}"] = n_alive.item()

                with torch.inference_mode():
                    eval_results = eval(
                        model=model,
                        components=components,
                        gates=gates,
                        eval_iter=eval_iter,
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
            and config.wandb_project is not None
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
                batch = next(eval_iter).to(device)
                _, pre_weight_acts = model.forward_with_pre_forward_cache_hooks(
                    batch, module_names=list(components.keys())
                )
                causal_importances, _, _ = calc_causal_importances(
                    pre_weight_acts=pre_weight_acts,
                    Vs=Vs,
                    gates=gates,
                    sigmoid_type=config.sigmoid_type,
                    detach_inputs=False,
                )

                for module_name, ci in causal_importances.items():
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
                **{f"{WandbSections.MISC.value}/{k}": v for k, v in ci_histogram_figs.items()},
                f"{WandbSections.MISC.value}/mean_component_activation_counts": mean_component_activation_counts_plot,
            }
            if plot_results_fn is not None:
                figs_dict.update(
                    plot_results_fn(
                        model=model,
                        components=components,
                        gates=gates,
                        batch_shape=batch.shape,  # pyright: ignore [reportPossiblyUnboundVariable]
                        device=device,
                        sigmoid_type=config.sigmoid_type,
                    )
                )

            fig_imgs_dict = {k: wandb.Image(v) for k, v in figs_dict.items()}

            wandb.log(fig_imgs_dict, step=step)

            if out_dir is not None:
                for k, v in figs_dict.items():
                    name = k.split("/")[-1]
                    v.savefig(out_dir / f"{name}_{step}.png")
                    tqdm.write(f"Saved plot to {out_dir / f'{name}_{step}.png'}")

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
                    grad_norm_val = grad_norm.sqrt()
                    wandb.log(
                        {f"{WandbSections.TRAIN.value}/grad_norm": grad_norm_val.item()}, step=step
                    )

            torch.nn.utils.clip_grad_norm_(parameters, max_norm=0.015)
            optimizer.step()
        # prof.step()
    logger.info("Finished training loop.")
