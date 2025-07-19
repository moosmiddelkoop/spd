"""Run SPD on a model."""

import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from jaxtyping import Bool, Float, Int
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from spd.configs import Config
from spd.core_metrics_and_figs import create_figures, create_metrics
from spd.log import logger
from spd.losses import calculate_losses
from spd.models.component_model import ComponentModel
from spd.utils.general_utils import (
    extract_batch_data,
    get_lr_schedule_fn,
    get_lr_with_warmup,
)
from spd.utils.run_utils import save_file


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


def optimize(
    target_model: nn.Module,
    config: Config,
    device: str,
    train_loader: DataLoader[Int[Tensor, "..."]]
    | DataLoader[tuple[Float[Tensor, "..."], Float[Tensor, "..."]]],
    eval_loader: DataLoader[Int[Tensor, "..."]]
    | DataLoader[tuple[Float[Tensor, "..."], Float[Tensor, "..."]]],
    n_eval_steps: int,
    out_dir: Path | None,
    tied_weights: list[tuple[str, str]] | None = None,
) -> None:
    """Run the optimization loop for LM decomposition."""

    logger.info(f"Output directory: {out_dir}")
    metrics_file = out_dir / "metrics.jsonl" if out_dir is not None else None

    model = ComponentModel(
        target_model=target_model,
        target_module_patterns=config.target_module_patterns,
        C=config.C,
        gate_type=config.gate_type,
        gate_hidden_dims=config.gate_hidden_dims,
        pretrained_model_output_attr=config.pretrained_model_output_attr,
    )
    model.freeze_target_model()

    for name, param in model.named_parameters():
        if param.requires_grad:
            logger.info(f"Parameter {name} is trainable.")
        else:
            logger.info(f"Parameter {name} is frozen.")
    logger.info("Model parameters un-frozen.")

    if tied_weights is not None:
        # Tie component weights. Assume that the first element is a transpose of the second element
        # NOTE: Tying weights will make your training nondeterministic
        for src_name, tgt_name in tied_weights:
            tgt = model.components_or_modules[tgt_name].components
            src = model.components_or_modules[src_name].components
            tgt.U.data = src.V.data.T
            tgt.V.data = src.U.data.T

    component_params: list[torch.nn.Parameter] = []
    gate_params: list[torch.nn.Parameter] = []
    for name, component in model.components.items():
        component_params.extend(list(component.parameters()))
        gate_params.extend(list(model.gates[name].parameters()))

    assert len(component_params) > 0, "No parameters found in components to optimize"

    optimizer = optim.AdamW(component_params + gate_params, lr=config.lr, weight_decay=0)

    lr_schedule_fn = get_lr_schedule_fn(config.lr_schedule, config.lr_exponential_halflife)
    logger.info(f"Base LR scheduler created: {config.lr_schedule}")

    n_params = sum(
        component.original.weight.numel() for component in model.components_or_modules.values()
    )

    data_iter = iter(train_loader)

    # TODO(oli): replace with AliveTracker class
    alive_components: dict[str, Bool[Tensor, " C"]] = {
        layer_name: torch.zeros(config.C, device=device).bool() for layer_name in model.components
    }

    # Iterate one extra step for final logging/plotting/saving
    for step in tqdm(range(config.steps + 1), ncols=0):
        step_lr = get_lr_with_warmup(
            step=step,
            steps=config.steps,
            lr=config.lr,
            lr_schedule_fn=lr_schedule_fn,
            lr_warmup_pct=config.lr_warmup_pct,
        )
        for group in optimizer.param_groups:
            group["lr"] = step_lr

        log_data: dict[str, int | float | wandb.Table] = {"misc/step": step, "misc/lr": step_lr}

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
            batch, module_names=model.target_module_paths
        )

        causal_importances, causal_importances_upper_leaky = model.calc_causal_importances(
            pre_weight_acts=pre_weight_acts,
            detach_inputs=False,
            sigmoid_type=config.sigmoid_type,
        )

        for layer_name, ci in causal_importances.items():
            alive_components[layer_name] = alive_components[layer_name] | (ci > 0.1).any(dim=(0, 1))

        total_loss, loss_terms = calculate_losses(
            model=model,
            batch=batch,
            config=config,
            causal_importances=causal_importances,
            causal_importances_upper_leaky=causal_importances_upper_leaky,
            target_out=target_out,
            device=device,
            n_params=n_params,
        )

        log_data["loss/total"] = total_loss.item()
        log_data.update(loss_terms)

        with torch.inference_mode():
            # --- Logging --- #
            if step % config.print_freq == 0:
                tqdm.write(f"--- Step {step} ---")
                tqdm.write(f"LR: {step_lr:.6f}")
                tqdm.write(f"Total Loss: {log_data['loss/total']:.7f}")
                for name, value in loss_terms.items():
                    tqdm.write(f"{name}: {value:.7f}")

                if step > 0:
                    for layer_name, layer_alive_components in alive_components.items():
                        log_data[f"{layer_name}/n_alive_01"] = layer_alive_components.sum().item()
                        alive_components[layer_name] = torch.zeros(config.C, device=device).bool()

                metrics = create_metrics(
                    model=model,
                    causal_importances=causal_importances,
                    target_out=target_out,
                    batch=batch,
                    device=device,
                    config=config,
                    step=step,
                )
                log_data.update(metrics)

                if metrics_file is not None:
                    # Filter out non-JSON-serializable objects (like wandb.Table) for file logging
                    file_metrics = {
                        k: v for k, v in log_data.items() if not isinstance(v, wandb.Table)
                    }
                    with open(metrics_file, "a") as f:
                        f.write(json.dumps(file_metrics) + "\n")

                if config.wandb_project:
                    wandb.log(log_data, step=step)

            # --- Plotting --- #
            if (
                config.image_freq is not None
                and step % config.image_freq == 0
                and (step > 0 or config.image_on_first_step)
            ):
                logger.info(f"Step {step}: Generating plots...")

                fig_dict = create_figures(
                    model=model,
                    causal_importances=causal_importances,
                    target_out=target_out,
                    batch=batch,
                    device=device,
                    config=config,
                    step=step,
                    eval_loader=eval_loader,
                    n_eval_steps=n_eval_steps,
                )

                if config.wandb_project:
                    wandb.log(
                        {k: wandb.Image(v) for k, v in fig_dict.items()},
                        step=step,
                    )
                    if out_dir is not None:
                        fig_dir = out_dir / "figures"
                        for k, v in fig_dict.items():
                            save_file(v, fig_dir / f"{k}_{step}.png")
                            tqdm.write(f"Saved plot to {fig_dir / f'{k}_{step}.png'}")

        # --- Saving Checkpoint --- #
        if (
            (config.save_freq is not None and step % config.save_freq == 0 and step > 0)
            or step == config.steps
        ) and out_dir is not None:
            save_file(model.state_dict(), out_dir / f"model_{step}.pth")
            logger.info(f"Saved model, optimizer, and out_dir to {out_dir}")
            if config.wandb_project:
                wandb.save(str(out_dir / f"model_{step}.pth"), base_path=str(out_dir), policy="now")

        # --- Backward Pass & Optimize --- #
        # Skip gradient step if we are at the last step (last step just for plotting and logging)
        if step != config.steps:
            total_loss.backward(retain_graph=True)
            if config.wandb_project:
                grad_norm: Float[Tensor, ""] = torch.zeros((), device=device)
                for param in component_params + gate_params:
                    if param.grad is not None:
                        grad_norm += param.grad.data.flatten().pow(2).sum()
                if step % config.print_freq == 0:
                    wandb.log({"misc/grad_norm": grad_norm.sqrt().item()}, step=step)
            optimizer.step()

    logger.info("Finished training loop.")
