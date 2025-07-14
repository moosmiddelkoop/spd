import torch
from jaxtyping import Float, Int
from torch import Tensor
from torch.utils.data import DataLoader

from spd.configs import BernoulliSampleConfig, UniformSampleConfig
from spd.models.component_model import ComponentModel
from spd.utils.general_utils import extract_batch_data


def sample_uniform_to_1(min: Tensor) -> Tensor:
    return min + (1 - min) * torch.rand_like(min)


class BernoulliSTE(torch.autograd.Function):
    @override
    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        sigma: Tensor,
        stochastic: bool,
    ) -> Tensor:
        ctx.save_for_backward(sigma)
        z = torch.bernoulli(sigma) if stochastic else (sigma >= 0.5).to(sigma.dtype)

        return z

    @override
    @staticmethod
    def backward(  # pyright: ignore [reportIncompatibleMethodOverride]
        ctx: torch.autograd.function.FunctionCtx,
        grad_outputs: Tensor,
    ) -> tuple[Tensor, None]:
        return grad_outputs.clone(), None


def bernoulli_ste(x: Tensor, min: float) -> Tensor:
    input = x * (1 - min) + min
    return BernoulliSTE.apply(input, True)  # pyright: ignore [reportReturnType]


def calc_stochastic_masks(
    causal_importances: dict[str, Float[Tensor, "... C"]],
    n_mask_samples: int,
    sample_config: UniformSampleConfig | BernoulliSampleConfig,
) -> list[dict[str, Float[Tensor, "... C"]]]:
    """Calculate n_mask_samples stochastic masks with the formula `ci + (1 - ci) * rand_unif(0,1)`.

    Args:
        causal_importances: The causal importances to use for the stochastic masks.
        n_mask_samples: The number of stochastic masks to calculate.

    Return:
        A list of n_mask_samples dictionaries, each containing the stochastic masks for each layer.
    """
    sample = (
        sample_uniform_to_1
        if sample_config.sample_type == "uniform"
        else partial(bernoulli_ste, min=sample_config.min)
    )
    stochastic_masks = []
    for _ in range(n_mask_samples):
        stochastic_masks.append({layer: sample(ci) for layer, ci in causal_importances.items()})
    return stochastic_masks


def calc_ci_l_zero(
    causal_importances: dict[str, Float[Tensor, "... C"]],
    cutoff: float = 1e-2,
) -> dict[str, float]:
    """Calculate the L0 loss on the causal importances, summed over the C dimension."""
    ci_l_zero = {}
    for layer_name, ci in causal_importances.items():
        mean_dims = tuple(range(ci.ndim - 1))
        ci_l_zero[layer_name] = (ci > cutoff).float().mean(dim=mean_dims).sum().item()
    return ci_l_zero


def component_activation_statistics(
    model: ComponentModel,
    dataloader: DataLoader[Int[Tensor, "..."]]
    | DataLoader[tuple[Float[Tensor, "..."], Float[Tensor, "..."]]],
    n_steps: int,
    device: str,
    threshold: float = 0.1,
) -> tuple[dict[str, float], dict[str, Float[Tensor, " C"]]]:
    """Get the number and strength of the masks over the full dataset."""
    n_tokens = {module_name: 0 for module_name in model.replaced_components}
    total_n_active_components = {module_name: 0 for module_name in model.replaced_components}
    component_activation_counts = {
        module_name: torch.zeros(model.C, device=device)
        for module_name in model.replaced_components
    }
    data_iter = iter(dataloader)
    for _ in range(n_steps):
        # --- Get Batch --- #
        batch = extract_batch_data(next(data_iter))
        batch = batch.to(device)

        _, pre_weight_acts = model.forward_with_pre_forward_cache_hooks(
            batch, module_names=list(model.replaced_components.keys())
        )

        causal_importances, _ = model.calc_causal_importances(pre_weight_acts, detach_inputs=False)
        for module_name, ci in causal_importances.items():
            # mask (batch, pos, C) or (batch, C)
            n_tokens[module_name] += ci.shape[:-1].numel()

            # Count the number of components that are active above the threshold
            active_components = ci > threshold
            total_n_active_components[module_name] += int(active_components.sum().item())

            sum_dims = tuple(range(ci.ndim - 1))
            component_activation_counts[module_name] += active_components.sum(dim=sum_dims)

    # Show the mean number of components
    mean_n_active_components_per_token: dict[str, float] = {
        module_name: (total_n_active_components[module_name] / n_tokens[module_name])
        for module_name in model.replaced_components
    }
    mean_component_activation_counts: dict[str, Float[Tensor, " C"]] = {
        module_name: component_activation_counts[module_name] / n_tokens[module_name]
        for module_name in model.replaced_components
    }

    return mean_n_active_components_per_token, mean_component_activation_counts
