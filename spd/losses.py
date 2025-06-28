from typing import Literal

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float, Int
from torch import Tensor

from spd.configs import Config
from spd.models.component_model import ComponentModel
from spd.models.component_utils import calc_stochastic_masks
from spd.models.components import EmbeddingComponent, LinearComponent
from spd.utils import calc_kl_divergence_lm


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
    ci_upper_leaky: dict[str, Float[Tensor, "... C"]], pnorm: float
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
        total_loss = total_loss + layer_ci_upper_leaky**pnorm

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
    unmasked_ce = ce_vs_labels(unmasked_component_logits_BSV)
    masked_ce = ce_vs_labels(masked_component_logits_BSV)
    rand_masked_ce = ce_vs_labels(rand_masked_component_logits_BSV)
    # bounds:
    zero_masked_ce = ce_vs_labels(zero_masked_component_logits_BSV)
    target_ce = ce_vs_labels(target_logits_BSV)

    # CE unrecovered: how much worse is some CE compared to zero-ablation?
    ce_unrecovered_masked = (masked_ce - target_ce) / (zero_masked_ce - target_ce)
    ce_unrecovered_unmasked = (unmasked_ce - target_ce) / (zero_masked_ce - target_ce)
    ce_unrecovered_rand_masked = (rand_masked_ce - target_ce) / (zero_masked_ce - target_ce)

    # KL
    unmasked_kl_vs_target = calc_kl_divergence_lm( unmasked_component_logits_BSV, target_logits_BSV)
    masked_kl_vs_target = calc_kl_divergence_lm(masked_component_logits_BSV, target_logits_BSV)

    ce_losses = {
        # bounds:
        "ce_loss/target_ce_loss_vs_labels": target_ce.item(),
        "ce_loss/zero_masked_ce_loss_vs_labels": zero_masked_ce.item(),
        # raw CE
        "ce_loss/unmasked_ce_loss_vs_labels": unmasked_ce.item(),
        "ce_loss/masked_ce_loss_vs_labels": masked_ce.item(),
        "ce_loss/rand_masked_ce_loss_vs_labels": rand_masked_ce.item(),
        # CE unrecovered (between target and zero-masked)
        "ce_unrecovered/unmasked": ce_unrecovered_unmasked.item(),
        "ce_unrecovered/masked": ce_unrecovered_masked.item(),
        "ce_unrecovered/rand_masked": ce_unrecovered_rand_masked.item(),
        # KL
        "misc/unmasked_kl_loss_vs_target": unmasked_kl_vs_target.item(),
        "misc/masked_kl_loss_vs_target": masked_kl_vs_target.item(),
    }

    return ce_losses


def calculate_losses(
    model: ComponentModel,
    batch: Int[Tensor, "..."],
    config: Config,
    components: dict[str, LinearComponent | EmbeddingComponent],
    ci_lower_leaky: dict[str, Float[Tensor, "... C"]],
    ci_upper_leaky: dict[str, Float[Tensor, "... C"]],
    target_out: Tensor,
    device: str,
    n_params: int,
) -> tuple[Float[Tensor, ""], dict[str, float]]:
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
    loss_terms: dict[str, float] = {}

    # Faithfulness loss
    faithfulness_loss = calc_faithfulness_loss(
        components=components, target_model=model.model, n_params=n_params, device=device
    )
    total_loss += config.faithfulness_coeff * faithfulness_loss
    loss_terms["loss/faithfulness"] = faithfulness_loss.item()

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
        loss_terms["loss/recon"] = recon_loss.item()

    # Stochastic reconstruction loss
    if config.stochastic_recon_coeff is not None:
        stochastic_masks = calc_stochastic_masks(
            causal_importances=ci_lower_leaky, n_mask_samples=config.n_mask_samples
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
        loss_terms["loss/stochastic_recon"] = stochastic_recon_loss.item()

    # # Reconstruction layerwise loss
    # if config.recon_layerwise_coeff is not None:
    #     recon_layerwise_loss = calc_masked_recon_layerwise_loss(
    #         model=model,
    #         batch=batch,
    #         device=device,
    #         components=components,
    #         masks=[ci_lower_leaky],
    #         target_out=target_out,
    #         loss_type=config.output_loss_type,
    #     )
    #     total_loss += config.recon_layerwise_coeff * recon_layerwise_loss
    #     loss_terms["loss/recon_layerwise"] = recon_layerwise_loss.item()

    # # Stochastic reconstruction layerwise loss
    # if config.stochastic_recon_layerwise_coeff is not None:
    #     layerwise_stochastic_masks = calc_stochastic_masks(
    #         causal_importances=ci_lower_leaky, n_mask_samples=config.n_mask_samples
    #     )
    #     stochastic_recon_layerwise_loss = calc_masked_recon_layerwise_loss(
    #         model=model,
    #         batch=batch,
    #         device=device,
    #         components=components,
    #         masks=layerwise_stochastic_masks,
    #         target_out=target_out,
    #         loss_type=config.output_loss_type,
    #     )
    #     total_loss += config.stochastic_recon_layerwise_coeff * stochastic_recon_layerwise_loss
    #     loss_terms["loss/stochastic_recon_layerwise"] = stochastic_recon_layerwise_loss.item()

    # Importance minimality loss
    importance_minimality_loss = calc_importance_minimality_loss(
        ci_upper_leaky=ci_upper_leaky, pnorm=config.pnorm
    )
    total_loss += config.importance_minimality_coeff * importance_minimality_loss
    loss_terms["loss/importance_minimality"] = importance_minimality_loss.item()

    # Schatten loss
    if config.schatten_coeff is not None:
        schatten_loss = calc_schatten_loss(
            ci_upper_leaky=ci_upper_leaky,
            pnorm=config.pnorm,
            components=components,
            device=device,
        )
        total_loss += config.schatten_coeff * schatten_loss
        loss_terms["loss/schatten"] = schatten_loss.item()

    # # Output reconstruction loss
    # if config.out_recon_coeff is not None:
    #     masks_all_ones = {k: torch.ones_like(v) for k, v in ci_lower_leaky.items()}
    #     out_recon_loss = calc_masked_recon_loss(
    #         model=model,
    #         batch=batch,
    #         components=components,
    #         masks=masks_all_ones,
    #         target_out=target_out,
    #         loss_type=config.output_loss_type,
    #     )
    #     total_loss += config.out_recon_coeff * out_recon_loss
    #     loss_terms["loss/output_recon"] = out_recon_loss.item()

    # # Embedding reconstruction loss
    # if config.embedding_recon_coeff is not None:
    #     stochastic_masks = calc_stochastic_masks(
    #         causal_importances=ci_lower_leaky, n_mask_samples=config.n_mask_samples
    #     )
    #     assert len(components) == 1, "Only one embedding component is supported"
    #     component = list(components.values())[0]
    #     assert isinstance(component, EmbeddingComponent)
    #     embedding_recon_loss = calc_embedding_recon_loss(
    #         model=model,
    #         batch=batch,
    #         component=component,
    #         masks=stochastic_masks,
    #         embed_module_name=next(iter(components.keys())),
    #         unembed=config.is_embed_unembed_recon,
    #     )
    #     total_loss += config.embedding_recon_coeff * embedding_recon_loss
    #     loss_terms["loss/embedding_recon"] = embedding_recon_loss.item()

    # # stochastic layerwise ablation loss
    # if config.layerwise_random_ablation_coeff is not None:
    #     layerwise_ablation_masks = calc_stochastic_ablation_masks(
    #         causal_importances=ci_lower_leaky, n_mask_samples=config.n_mask_samples
    #     )
    #     layerwise_random_ablation_loss = calc_layerwise_ablation_loss(
    #         model=model,
    #         batch=batch,
    #         device=device,
    #         components=components,
    #         masks=layerwise_ablation_masks,
    #         target_out=target_out,
    #         loss_type=config.output_loss_type,
    #     )
    #     total_loss += config.layerwise_random_ablation_coeff * layerwise_random_ablation_loss
    #     loss_terms["loss/layerwise_random_ablation_loss"] = (
    #         layerwise_random_ablation_loss.item()
    #     )

    # stochastic ablation loss
    if config.stochastic_ablation_coeff is not None:
        stochastic_ablation_masks = calc_stochastic_ablation_masks(
            causal_importances=ci_lower_leaky, n_mask_samples=config.n_mask_samples
        )
        random_ablation_loss = torch.tensor(0.0, device=target_out.device)
        for i in range(len(stochastic_ablation_masks)):
            random_ablation_loss = calc_ablation_loss(
                model=model,
                batch=batch,
                components=components,
                masks=stochastic_ablation_masks[i],
                target_out=target_out,
                loss_type=config.output_loss_type,
            )
        random_ablation_loss = random_ablation_loss / len(stochastic_ablation_masks)
        total_loss += config.stochastic_ablation_coeff * random_ablation_loss
        loss_terms["loss/random_ablation_loss"] = random_ablation_loss.item()

    return total_loss, loss_terms
