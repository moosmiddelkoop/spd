# %%
from typing import Literal, override

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float, Int
from torch import Tensor

from spd.configs import Config
from spd.constants import CI_ALIVE_THRESHOLD
from spd.models.component_model import ComponentModel
from spd.models.component_utils import calc_stochastic_masks
from spd.models.components import EmbeddingComponent, LinearComponent
from spd.utils import calc_kl_divergence_lm
from spd.wandb_utils import WandbSections


def interpolate(start: float, end: float, t: float) -> float:
    return start * (1 - t) + end * t


def calc_embedding_recon_loss(
    model: ComponentModel,
    batch: Int[Tensor, "..."],
    component: EmbeddingComponent,
    masks: list[dict[str, Float[Tensor, "... C"]]],
    embed_module_name: str,
    unembed: bool = False,
) -> Float[Tensor, ""]:
    """
    recon loss that directly compares the outputs of the (optionally masked)
    ``EmbeddingComponent``(s) to the outputs of the original ``nn.Embedding`` modules.

    If ``unembed`` is ``True``, both the masked embedding output and the target embedding
    output are unembedded using the ``lm_head`` module, and the KL divergence is used as the loss.

    If ``unembed`` is ``False``, the loss is the MSE between the masked embedding output
    and the target embedding output is used as the loss.
    """

    # --- original embedding output --------------------------------------------------------- #
    orig_module = model.model.get_submodule(embed_module_name)
    assert isinstance(orig_module, nn.Embedding), (
        f"Module {embed_module_name} expected to be nn.Embedding, got {type(orig_module)}"
    )
    target_out: Float[Tensor, "... d_emb"] = orig_module(batch)

    # --- masked embedding output ----------------------------------------------------------- #
    loss = torch.tensor(0.0, device=component.V.device)
    for mask_info in masks:
        component.mask = mask_info[embed_module_name]

        masked_out: Float[Tensor, "... d_emb"] = component(batch)
        component.mask = None

        if unembed:
            assert hasattr(model.model, "lm_head"), "Only supports unembedding named lm_head"
            target_out_unembed = model.model.lm_head(target_out)
            masked_out_unembed = model.model.lm_head(masked_out)
            loss += calc_kl_divergence_lm(pred=masked_out_unembed, target=target_out_unembed)
        else:
            loss += ((masked_out - target_out) ** 2).sum(dim=-1).mean()

    loss /= len(masks)

    return loss


def calc_schatten_loss(
    ci_upper_leaky: dict[str, Float[Tensor, "... C"]],
    pnorm: float,
    components: dict[str, LinearComponent | EmbeddingComponent],
    device: str,
) -> Float[Tensor, ""]:
    """Calculate the Schatten loss on the active components.

    The Schatten loss is calculated as:
        L = Σ_{components} mean(ci_upper_leaky^pnorm · (||V||_2^2 + ||U||_2^2))

    where:
        - ci_upper_leaky are the upper leaky relu causal importances for each component
        - pnorm is the power to raise the mask to
        - V and U are the component matrices
        - ||·||_2 is the L2 norm

    Args:
        ci_upper_leaky: Dictionary of upper leaky relu causal importances for each layer.
        pnorm: The pnorm to use for the importance minimality loss. Must be positive.
        components: Dictionary of components for each layer.
        device: The device to compute the loss on.

    Returns:
        The Schatten loss as a scalar tensor.
    """

    total_loss = torch.tensor(0.0, device=device)
    for component_name, component in components.items():
        V_norms = component.V.square().sum(dim=-2)
        U_norms = component.U.square().sum(dim=-1)
        schatten_norms = V_norms + U_norms
        loss = einops.einsum(
            ci_upper_leaky[component_name] ** pnorm, schatten_norms, "... C, C -> ..."
        )
        total_loss += loss.mean()
    return total_loss


def calc_importance_minimality_loss(
    ci_upper_leaky: dict[str, Tensor],
    pnorm: float,
    eps: float = 1e-12,
) -> Float[Tensor, ""]:
    """Calculate the importance minimality loss on the upper leaky relu causal importances.

    Args:
        ci_upper_leaky: Dictionary of causal importances upper leaky relu for each layer.
        pnorm: The pnorm to use for the importance minimality loss. Must be positive.

    Returns:
        The importance minimality loss on the upper leaky relu causal importances.
    """
    total_loss = torch.zeros_like(next(iter(ci_upper_leaky.values())))

    for layer_ci_upper_leaky in ci_upper_leaky.values():
        # Note, the paper uses an absolute value but our layer_ci_upper_leaky is already > 0
        total_loss = total_loss + (layer_ci_upper_leaky + eps) ** pnorm

    # Sum over the C dimension and mean over the other dimensions
    return total_loss.sum(dim=-1).mean()


def calc_masked_recon_layerwise_loss(
    model: ComponentModel,
    batch: Int[Tensor, "..."],
    device: str,
    components: dict[str, LinearComponent | EmbeddingComponent],
    masks: list[dict[str, Float[Tensor, "... C"]]],
    target_out: Float[Tensor, "... d_model_out"],
    loss_type: Literal["mse", "kl"] = "kl",
) -> Float[Tensor, ""]:
    """Calculate the recon loss when augmenting the model one (masked) component at a time."""
    assert loss_type in ["mse", "kl"], f"Invalid loss type: {loss_type}"
    total_loss = torch.tensor(0.0, device=device)
    for mask_info in masks:
        for component_name, component in components.items():
            modified_out = model.forward_with_components(
                batch,
                components={component_name: component},
                masks={component_name: mask_info[component_name]},
            )
            if loss_type == "mse":
                loss = ((modified_out - target_out) ** 2).mean()
            else:
                loss = calc_kl_divergence_lm(pred=modified_out, target=target_out)
            total_loss += loss
    n_modified_components = len(masks[0])
    return total_loss / (n_modified_components * len(masks))


def calc_masked_recon_loss(
    model: ComponentModel,
    batch: Float[Tensor, "... d_in"],
    components: dict[str, LinearComponent | EmbeddingComponent],
    masks: dict[str, Float[Tensor, "... C"]],
    target_out: Float[Tensor, "... d_mdoel_out"],
    loss_type: Literal["mse", "kl"] = "mse",
) -> Float[Tensor, ""]:
    """Calculate the MSE over all masks."""
    # Do a forward pass with all components
    out = model.forward_with_components(batch, components=components, masks=masks)
    assert loss_type in ["mse", "kl"], f"Invalid loss type: {loss_type}"
    if loss_type == "mse":
        loss = ((out - target_out) ** 2).mean()
    else:
        loss = calc_kl_divergence_lm(pred=out, target=target_out)

    return loss


def _calc_tensors_mse(
    params1: dict[str, Float[Tensor, "d_in d_out"]],
    params2: dict[str, Float[Tensor, "d_in d_out"]],
    n_params: int,
    device: str,
) -> Float[Tensor, ""]:
    """Calculate the MSE between params1 and params2, summing over the d_in and d_out dimensions.

    Normalizes by the number of parameters in the model.

    Args:
        params1: The first set of parameters
        params2: The second set of parameters
        n_params: The number of parameters in the model
        device: The device to use for calculations
    """
    faithfulness_loss = torch.tensor(0.0, device=device)
    for name in params1:
        faithfulness_loss = faithfulness_loss + ((params2[name] - params1[name]) ** 2).sum()
    return faithfulness_loss / n_params


def calc_faithfulness_loss(
    components: dict[str, LinearComponent | EmbeddingComponent],
    target_model: nn.Module,
    n_params: int,
    device: str,
) -> Float[Tensor, ""]:
    """Calculate the MSE loss between component parameters (V@U + bias) and target parameters."""
    target_params: dict[str, Float[Tensor, "d_in d_out"]] = {}
    component_params: dict[str, Float[Tensor, "d_in d_out"]] = {}

    for comp_name, component in components.items():
        component_params[comp_name] = component.weight
        submodule = target_model.get_submodule(comp_name)
        assert isinstance(submodule, nn.Linear | nn.Embedding)
        target_params[comp_name] = submodule.weight
        assert component_params[comp_name].shape == target_params[comp_name].shape

    faithfulness_loss = _calc_tensors_mse(
        params1=component_params,
        params2=target_params,
        n_params=n_params,
        device=device,
    )
    return faithfulness_loss


def calc_ablation_loss(
    model: ComponentModel,
    batch: Float[Tensor, "... d_in"],
    components: dict[str, LinearComponent | EmbeddingComponent],
    masks: dict[str, Float[Tensor, "... C"]],
    target_out: Float[Tensor, "... d_mdoel_out"],
    loss_type: Literal["mse", "kl"] = "mse",
) -> Float[Tensor, ""]:
    """Calculate the MSE over all masks."""
    # Do a forward pass with all components
    out = model.forward_plus_components(batch, components=components, masks=masks)
    if loss_type == "mse":
        loss = ((out - target_out) ** 2).mean()
    elif loss_type == "kl":
        loss = calc_kl_divergence_lm(pred=out, target=target_out)
    return loss


def calc_stochastic_ablation_masks(
    causal_importances: dict[str, Float[Tensor, "... C"]],
    n_mask_samples: int,
) -> list[dict[str, Float[Tensor, "... C"]]]:
    """Calculate n_mask_samples stochastic masks with the formula `-(1 - ci) * rand_unif(0,1)`.
    Args:
        causal_importances: The masks to use for the random masks.
        n_mask_samples: The number of random masks to calculate.
    Return:
        A list of n_mask_samples dictionaries, each containing the stochastic masks for each layer.
    """
    stochastic_masks = []
    for _ in range(n_mask_samples):
        stochastic_masks.append(
            {layer: (ci - 1.0) * torch.rand_like(ci) for layer, ci in causal_importances.items()}
        )
    return stochastic_masks


def calc_layerwise_ablation_loss(
    model: ComponentModel,
    batch: Int[Tensor, "..."],
    device: str,
    components: dict[str, LinearComponent | EmbeddingComponent],
    masks: list[dict[str, Float[Tensor, "... C"]]],
    target_out: Float[Tensor, "... d_model_out"],
    loss_type: Literal["mse", "kl"] = "kl",
) -> Float[Tensor, ""]:
    """Calculate the recon loss when augmenting the model one (masked) component at a time."""
    total_loss = torch.tensor(0.0, device=device)
    for mask_info in masks:
        for component_name, component in components.items():
            modified_out = model.forward_plus_components(
                batch,
                components={component_name: component},
                masks={component_name: mask_info[component_name]},
            )
            if loss_type == "mse":
                loss = ((modified_out - target_out) ** 2).mean()
            elif loss_type == "kl":
                loss = calc_kl_divergence_lm(pred=modified_out, target=target_out)
            else:
                raise ValueError(f"Invalid loss type: {loss_type}")
            total_loss += loss
    n_modified_components = len(masks[0])
    return total_loss / (n_modified_components * len(masks))


def calc_ce_losses(
    model: ComponentModel,
    batch_BS: Tensor,
    components: dict[str, LinearComponent | EmbeddingComponent],
    ci_masks_BxM: dict[str, Tensor],
    target_logits_BSV: Tensor,
) -> dict[str, float]:
    """Calculate cross-entropy losses for various masking scenarios.

    Args:
        model: The component model
        batch: Input batch
        components: Dictionary of components
        masks: Dictionary of masks for components
        unmasked_component_logits: Logits from unmasked components
        masked_component_logits: Logits from masked components
        target_logits: Target model logits

    Returns:
        Dictionary containing CE losses for different scenarios
    """
    # Flatten logits and batch for CE calculation
    assert batch_BS.ndim == 2, "expected 2 dimensions (batch, seq_len)"

    # make sure labels don't "wrap around": you **can't** predict the first token.
    masked_batch_BS = batch_BS.clone()
    masked_batch_BS[:, 0] = -100  # F.cross_entropy ignores -99
    flat_masked_batch_Bs = masked_batch_BS.flatten()

    def ce_vs_labels(logits_BSV: Tensor) -> Float[Tensor, ""]:
        flat_logits_BsV = einops.rearrange(logits_BSV, "b seq_len vocab -> (b seq_len) vocab")
        return F.cross_entropy(flat_logits_BsV[:-1], flat_masked_batch_Bs[1:], ignore_index=-100)

    # CE When...
    # every component is used
    unmasked_component_logits_BSV = model.forward_with_components(
        batch_BS, components=components, masks=None
    )
    # we use the causal importances as a mask
    masked_component_logits_BSV = model.forward_with_components(
        batch_BS, components=components, masks=ci_masks_BxM
    )
    # we use the rounded causal importances as a mask
    rounded_ci_masks_01_BxM = {k: (v > 0.1).to(v) for k, v in ci_masks_BxM.items()}
    snap_masked_component_logits_01_BSV = model.forward_with_components(
        batch_BS, components=components, masks=rounded_ci_masks_01_BxM
    )
    rounded_ci_masks_001_BxM = {k: (v > 0.01).to(v) for k, v in ci_masks_BxM.items()}
    snap_masked_component_logits_001_BSV = model.forward_with_components(
        batch_BS, components=components, masks=rounded_ci_masks_001_BxM
    )
    rounded_ci_masks_05_BxM = {k: (v > 0.5).to(v) for k, v in ci_masks_BxM.items()}
    snap_masked_component_logits_05_BSV = model.forward_with_components(
        batch_BS, components=components, masks=rounded_ci_masks_05_BxM
    )
    # every component is fully masked out (all-zero masks)
    zero_masked_component_logits_BSV = model.forward_with_components(
        batch_BS,
        components=components,
        masks={k: torch.zeros_like(v) for k, v in ci_masks_BxM.items()},
    )
    # every component is completely randomly masked
    rand_masked_component_logits_BSV = model.forward_with_components(
        batch_BS,
        components=components,
        masks={k: torch.rand_like(v) for k, v in ci_masks_BxM.items()},
    )

    # CE vs true labels
    ce_unmasked = ce_vs_labels(unmasked_component_logits_BSV)
    ce_masked = ce_vs_labels(masked_component_logits_BSV)
    ce_snap_masked_01 = ce_vs_labels(snap_masked_component_logits_01_BSV)
    ce_snap_masked_001 = ce_vs_labels(snap_masked_component_logits_001_BSV)
    ce_snap_masked_05 = ce_vs_labels(snap_masked_component_logits_05_BSV)
    ce_rand_masked = ce_vs_labels(rand_masked_component_logits_BSV)
    # bounds:
    ce_zero_masked = ce_vs_labels(zero_masked_component_logits_BSV)
    ce_target = ce_vs_labels(target_logits_BSV)

    # CE unrecovered: how much worse is some CE compared to zero-ablation?
    ce_unrecovered_unmasked = (ce_unmasked - ce_target) / (ce_zero_masked - ce_target)
    ce_unrecovered_masked = (ce_masked - ce_target) / (ce_zero_masked - ce_target)
    ce_unrecovered_snap_masked_01 = (ce_snap_masked_01 - ce_target) / (ce_zero_masked - ce_target)
    ce_unrecovered_snap_masked_001 = (ce_snap_masked_001 - ce_target) / (ce_zero_masked - ce_target)
    ce_unrecovered_snap_masked_05 = (ce_snap_masked_05 - ce_target) / (ce_zero_masked - ce_target)
    ce_unrecovered_rand_masked = (ce_rand_masked - ce_target) / (ce_zero_masked - ce_target)

    # KL
    unmasked_kl_vs_target = calc_kl_divergence_lm(unmasked_component_logits_BSV, target_logits_BSV)
    masked_kl_vs_target = calc_kl_divergence_lm(masked_component_logits_BSV, target_logits_BSV)
    snap_masked_kl_vs_target_01 = calc_kl_divergence_lm(
        snap_masked_component_logits_01_BSV, target_logits_BSV
    )
    snap_masked_kl_vs_target_001 = calc_kl_divergence_lm(
        snap_masked_component_logits_001_BSV, target_logits_BSV
    )
    snap_masked_kl_vs_target_05 = calc_kl_divergence_lm(
        snap_masked_component_logits_05_BSV, target_logits_BSV
    )
    rand_masked_kl_vs_target = calc_kl_divergence_lm(
        rand_masked_component_logits_BSV, target_logits_BSV
    )

    ce_losses = {
        # CE unrecovered (between target and zero-masked)
        f"{WandbSections.CE_UNRECOVERED.value}/unmasked": ce_unrecovered_unmasked.item(),
        f"{WandbSections.CE_UNRECOVERED.value}/masked": ce_unrecovered_masked.item(),
        f"{WandbSections.CE_UNRECOVERED.value}/snap_masked_01": ce_unrecovered_snap_masked_01.item(),
        f"{WandbSections.CE_UNRECOVERED.value}/snap_masked_001": ce_unrecovered_snap_masked_001.item(),
        f"{WandbSections.CE_UNRECOVERED.value}/snap_masked_05": ce_unrecovered_snap_masked_05.item(),
        f"{WandbSections.CE_UNRECOVERED.value}/rand_masked": ce_unrecovered_rand_masked.item(),
        # KL
        f"{WandbSections.MISC.value}/unmasked_kl_loss_vs_target": unmasked_kl_vs_target.item(),
        f"{WandbSections.MISC.value}/masked_kl_loss_vs_target": masked_kl_vs_target.item(),
        f"{WandbSections.MISC.value}/snap_masked_kl_loss_vs_target_01": snap_masked_kl_vs_target_01.item(),
        f"{WandbSections.MISC.value}/snap_masked_kl_loss_vs_target_001": snap_masked_kl_vs_target_001.item(),
        f"{WandbSections.MISC.value}/snap_masked_kl_loss_vs_target_05": snap_masked_kl_vs_target_05.item(),
        f"{WandbSections.MISC.value}/rand_masked_kl_loss_vs_target": rand_masked_kl_vs_target.item(),
        # bounds:
        f"{WandbSections.MISC.value}/z_ce/target_ce_loss_vs_labels": ce_target.item(),
        f"{WandbSections.MISC.value}/z_ce/zero_masked_ce_loss_vs_labels": ce_zero_masked.item(),
        # raw CE
        f"{WandbSections.MISC.value}/z_ce/unmasked_ce_loss_vs_labels": ce_unmasked.item(),
        f"{WandbSections.MISC.value}/z_ce/masked_ce_loss_vs_labels": ce_masked.item(),
        f"{WandbSections.MISC.value}/z_ce/snap_masked_ce_loss_vs_labels_01": ce_snap_masked_01.item(),
        f"{WandbSections.MISC.value}/z_ce/snap_masked_ce_loss_vs_labels_001": ce_snap_masked_001.item(),
        f"{WandbSections.MISC.value}/z_ce/snap_masked_ce_loss_vs_labels_05": ce_snap_masked_05.item(),
        f"{WandbSections.MISC.value}/z_ce/rand_masked_ce_loss_vs_labels": ce_rand_masked.item(),
    }

    return ce_losses


def _get_pnorm(
    pnorm: float | Literal["anneal-2-1"] | Literal["anneal-2-0.5"], training_pct: float
) -> float:
    if pnorm == "anneal-2-1":
        return interpolate(2.0, 1.0, training_pct)
    elif pnorm == "anneal-2-0.5":
        return interpolate(2.0, 0.5, training_pct)
    else:
        return pnorm
    raise ValueError(f"Invalid pnorm: {pnorm}")


def calculate_losses(
    model: ComponentModel,
    batch: Int[Tensor, "..."],
    config: Config,
    components: dict[str, LinearComponent | EmbeddingComponent],
    ci_lower_leaky: dict[str, Float[Tensor, "... C"]],
    ci_upper_leaky: dict[str, Float[Tensor, "... C"]],
    target_out: Tensor,
    device: str,
    training_pct: float,
    n_params: int,
) -> tuple[Float[Tensor, ""], dict[str, Tensor]]:
    """Calculate all losses and return total loss and individual loss terms.

    Args:
        model: The component model
        batch: Input batch
        config: Configuration object with loss coefficients
        components: Dictionary of component modules
        causal_importances: Causal importance masks
        causal_importances_upper_leaky: Upper leaky causal importances for regularization
        target_out: Target model output
        device: Device to run computations on
        n_params: Total number of parameters in the model

    Returns:
        Tuple of (total_loss, loss_terms_dict)
    """
    total_loss = torch.tensor(0.0, device=device)
    loss_terms: dict[str, Tensor] = {}

    # Faithfulness loss
    if config.faithfulness_coeff is not None:
        faithfulness_loss = calc_faithfulness_loss(
            components=components, target_model=model.model, n_params=n_params, device=device
        )
        total_loss += config.faithfulness_coeff * faithfulness_loss
        loss_terms[f"{WandbSections.LOSS.value}/faithfulness"] = faithfulness_loss.detach()

    # Reconstruction loss
    if config.recon_coeff is not None:
        recon_loss = calc_masked_recon_loss(
            model=model,
            batch=batch,
            components=components,
            masks=ci_lower_leaky,
            target_out=target_out,
            loss_type=config.output_loss_type,
        )
        total_loss += config.recon_coeff * recon_loss
        loss_terms[f"{WandbSections.LOSS.value}/recon"] = recon_loss.detach()

    # Stochastic reconstruction loss
    if config.stochastic_recon_coeff is not None:
        stochastic_masks = calc_stochastic_masks(
            causal_importances=ci_lower_leaky,
            n_mask_samples=config.n_mask_samples,
            sample_type=config.sample_type,
            min_prob=config.min_prob,
        )
        stochastic_recon_loss = torch.tensor(0.0, device=target_out.device)
        for i in range(len(stochastic_masks)):
            stochastic_recon_loss += calc_masked_recon_loss(
                model=model,
                batch=batch,
                components=components,
                masks=stochastic_masks[i],
                target_out=target_out,
                loss_type=config.output_loss_type,
            )
        stochastic_recon_loss = stochastic_recon_loss / len(stochastic_masks)
        total_loss += config.stochastic_recon_coeff * stochastic_recon_loss
        loss_terms[f"{WandbSections.LOSS.value}/stochastic_recon"] = stochastic_recon_loss.detach()

    # Reconstruction layerwise loss
    if config.recon_layerwise_coeff is not None:
        recon_layerwise_loss = calc_masked_recon_layerwise_loss(
            model=model,
            batch=batch,
            device=device,
            components=components,
            masks=[ci_lower_leaky],
            target_out=target_out,
            loss_type=config.output_loss_type,
        )
        total_loss += config.recon_layerwise_coeff * recon_layerwise_loss
        loss_terms[f"{WandbSections.LOSS.value}/recon_layerwise"] = recon_layerwise_loss.detach()

    # Stochastic reconstruction layerwise loss
    if config.stochastic_recon_layerwise_coeff is not None:
        layerwise_stochastic_masks = calc_stochastic_masks(
            causal_importances=ci_lower_leaky,
            n_mask_samples=config.n_mask_samples,
            sample_type=config.sample_type,
            min_prob=config.min_prob,
        )
        stochastic_recon_layerwise_loss = calc_masked_recon_layerwise_loss(
            model=model,
            batch=batch,
            device=device,
            components=components,
            masks=layerwise_stochastic_masks,
            target_out=target_out,
            loss_type=config.output_loss_type,
        )
        total_loss += config.stochastic_recon_layerwise_coeff * stochastic_recon_layerwise_loss
        loss_terms[f"{WandbSections.LOSS.value}/stochastic_recon_layerwise"] = (
            stochastic_recon_layerwise_loss.detach()
        )

    # Importance minimality loss
    pnorm = _get_pnorm(config.pnorm, training_pct)

    loss_terms[f"{WandbSections.MISC.value}/pnorm"] = torch.tensor(pnorm, device=device)

    importance_minimality_loss = calc_importance_minimality_loss(ci_upper_leaky, pnorm)

    importance_minimality_coeff = (
        (
            training_pct
            / config.importance_minimality_warmup_pct
            * config.importance_minimality_coeff
        )
        if (
            config.importance_minimality_warmup_pct is not None
            and training_pct < config.importance_minimality_warmup_pct
        )
        else config.importance_minimality_coeff
    )

    loss_terms[f"{WandbSections.TRAIN.value}/importance_minimality_coeff"] = torch.tensor(
        importance_minimality_coeff, device=device
    )

    total_loss += importance_minimality_coeff * importance_minimality_loss
    loss_terms[f"{WandbSections.LOSS.value}/importance_minimality"] = (
        importance_minimality_loss.detach()
    )


    return total_loss, loss_terms


def _calc_tensors_fvu(
    true_params: dict[str, Tensor],
    pred_params: dict[str, Tensor],
    device: str,
) -> Float[Tensor, ""]:
    """
    Fraction of variance unexplained (FVU) between two parameter dictionaries.

    FVU = Σ (y_pred - y_true)^2  /  Σ (y_true - mean(y_true))^2

    Here ``y_true``  is every element of ``params1``,
    ``y_pred``       is the corresponding element of ``params2``,
    and the mean is computed *per-tensor* in ``params1``.

    Args:
        params1: baseline / “true" parameters
        params2: comparison / “predicted" parameters
        device : torch device to hold the running totals

    Returns:
        A scalar Tensor on ``device``: 0 → perfect match, 1 → predicts per-tensor mean,
        >1 → worse than predicting the mean.
    """
    sse = torch.tensor(0.0, device=device)  # Σ squared errors
    tss = torch.tensor(0.0, device=device)  # Σ squared deviations from per-tensor mean

    for name, true_param in true_params.items():
        pred_param = pred_params[name]

        # Sum of squared errors for this tensor
        sse += ((pred_param - true_param) ** 2).sum()

        # Total sum of squares around the tensor's own mean
        mean_true_param = true_param.mean()
        tss += ((true_param - mean_true_param) ** 2).sum()

    # If tss == 0 (all-constant tensor), define FVU = 0 to avoid NaN/inf.
    if tss == 0:
        return torch.tensor(0.0, device=device)

    return sse / tss


def calc_faithfulness_fvu(
    components: dict[str, LinearComponent | EmbeddingComponent],
    target_model: nn.Module,
    device: str,
) -> Float[Tensor, ""]:
    """Calculate the MSE loss between component parameters (V@U + bias) and target parameters."""
    target_params: dict[str, Float[Tensor, "d_in d_out"]] = {}
    component_params: dict[str, Float[Tensor, "d_in d_out"]] = {}

    for comp_name, component in components.items():
        component_params[comp_name] = component.weight
        submodule = target_model.get_submodule(comp_name)
        assert isinstance(submodule, nn.Linear | nn.Embedding)
        target_params[comp_name] = submodule.weight
        assert component_params[comp_name].shape == target_params[comp_name].shape

    faithfulness_loss = _calc_tensors_fvu(
        true_params=target_params,
        pred_params=component_params,
        device=device,
    )

    return faithfulness_loss
