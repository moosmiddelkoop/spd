from collections.abc import Callable, Mapping
from functools import partial
from typing import cast, override

import einops
import torch
from jaxtyping import Float, Int
from torch import Tensor
from torch.utils.data import DataLoader

from spd.configs import SampleConfig
from spd.models.component_model import ComponentModel
from spd.models.components import EmbeddingComponent, GateMLP, LinearComponent, VectorGateMLP
from spd.models.sigmoids import SIGMOID_TYPES, SigmoidTypes
from spd.utils.general_utils import extract_batch_data


def sample_uniform_to_1(x: Tensor) -> Tensor:
    return x + (1 - x) * torch.rand_like(x)


class BernoulliSTE(torch.autograd.Function):
    @override
    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        sigma: Tensor,
    ) -> Tensor:
        ctx.save_for_backward(sigma)
        return torch.bernoulli(sigma)

    @override
    @staticmethod
    def backward(  # pyright: ignore [reportIncompatibleMethodOverride]
        ctx: torch.autograd.function.FunctionCtx,
        grad_outputs: Tensor,
    ) -> tuple[Tensor]:
        return (grad_outputs.clone(),)


def rescaled_bernoulli_ste(x: Tensor) -> Tensor:
    input = x * (1 - 0.5) + 0.5
    return BernoulliSTE.apply(input)  # pyright: ignore [reportReturnType]


def rescaled_binary_concrete_ste(
    prob: Tensor,
    temp: float,
    eps: float = 1e-20,
) -> Tensor:
    prob = prob * (1 - 0.5) + 0.5  # rescale to [0.5, 1]
    prob = torch.clamp(prob, max=1 - 1e-6)

    logit = torch.log(prob / (1 - prob))
    u = torch.rand_like(logit)
    logistic = torch.log(u + eps) - torch.log1p(-u + eps)  # logistic noise ~ log(u) - log(1-u)
    y_soft = torch.sigmoid((logit + logistic) / temp)

    # hard threshold in forward pass, preserve grad of y_soft
    y_hard = (y_soft > 0.5).to(y_soft.dtype)
    return y_hard + (y_soft - y_soft.detach())

    # if training:
    # else:
    #     y_soft = prob

    # if not hard:
    #     return y_soft


def rescaled_binary_hard_concrete_ste(
    prob: Tensor,
    temp: float,
    eps: float = 1e-20,
    bounds: tuple[float, float] = (-0.1, 1.1),
) -> Tensor:
    prob = prob * (1 - 0.5) + 0.5  # rescale to [0.5, 1]
    prob = torch.clamp(prob, max=1 - 1e-6)

    logit = torch.log(prob / (1 - prob))
    u = torch.rand_like(logit)
    logistic = torch.log(u + eps) - torch.log1p(-u + eps)  # logistic noise ~ log(u) - log(1-u)
    y = torch.sigmoid((logit + logistic) / temp)

    low, high = bounds
    y = low + (high - low) * y
    return y.clamp(0, 1)


def get_sample_fn(sample_config: SampleConfig) -> Callable[[Tensor], Tensor]:
    if sample_config.sample_type == "uniform":
        return sample_uniform_to_1
    elif sample_config.sample_type == "bernoulli_ste":
        return rescaled_bernoulli_ste
    elif sample_config.sample_type == "concrete_ste":
        return partial(rescaled_binary_concrete_ste, temp=sample_config.temp)
    elif sample_config.sample_type == "hard_concrete":
        return partial(
            rescaled_binary_hard_concrete_ste,
            temp=sample_config.temp,
            bounds=sample_config.bounds,
        )
    raise ValueError(f"Invalid sample type: {sample_config.sample_type}")  # pyright: ignore [reportUnreachable]


def calc_stochastic_masks(
    causal_importances: dict[str, Float[Tensor, "... C"]],
    n_mask_samples: int,
    sample_config: SampleConfig,
) -> list[dict[str, Float[Tensor, "... C"]]]:
    """Calculate n_mask_samples stochastic masks with the formula `ci + (1 - ci) * rand_unif(0,1)`.

    Args:
        causal_importances: The causal importances to use for the stochastic masks.
        n_mask_samples: The number of stochastic masks to calculate.

    Return:
        A list of n_mask_samples dictionaries, each containing the stochastic masks for each layer.
    """
    sample = get_sample_fn(sample_config)
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
    # We used "-" instead of "." as module names can't have "." in them
    gates: dict[str, GateMLP | VectorGateMLP] = {
        k.removeprefix("gates.").replace("-", "."): cast(GateMLP | VectorGateMLP, v)
        for k, v in model.gates.items()
    }
    components: dict[str, LinearComponent | EmbeddingComponent] = {
        k.removeprefix("components.").replace("-", "."): cast(
            LinearComponent | EmbeddingComponent, v
        )
        for k, v in model.components.items()
    }

    n_tokens = {module_name.replace("-", "."): 0 for module_name in components}
    total_n_active_components = {module_name.replace("-", "."): 0 for module_name in components}
    component_activation_counts = {
        module_name.replace("-", "."): torch.zeros(model.C, device=device)
        for module_name in components
    }
    data_iter = iter(dataloader)
    for _ in range(n_steps):
        # --- Get Batch --- #
        batch = extract_batch_data(next(data_iter))
        batch = batch.to(device)

        _, pre_weight_acts = model.forward_with_pre_forward_cache_hooks(
            batch, module_names=list(components.keys())
        )
        Vs = {module_name: v.V for module_name, v in components.items()}

        causal_importances, _ = calc_causal_importances(
            pre_weight_acts=pre_weight_acts,
            Vs=Vs,
            gates=gates,
            detach_inputs=False,
        )
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
        for module_name in components
    }
    mean_component_activation_counts: dict[str, Float[Tensor, " C"]] = {
        module_name: component_activation_counts[module_name] / n_tokens[module_name]
        for module_name in components
    }

    return mean_n_active_components_per_token, mean_component_activation_counts


def calc_causal_importances(
    pre_weight_acts: dict[str, Float[Tensor, "... d_in"] | Int[Tensor, "... pos"]],
    Vs: Mapping[str, Float[Tensor, "d_in C"]],
    gates: Mapping[str, GateMLP | VectorGateMLP],
    detach_inputs: bool = False,
    sigmoid_type: SigmoidTypes = "leaky_hard",
) -> tuple[dict[str, Float[Tensor, "... C"]], dict[str, Float[Tensor, "... C"]]]:
    """Calculate component activations and causal importances in one pass to save memory.

    Args:
        pre_weight_acts: The activations before each layer in the target model.
        Vs: The V matrix at each layer.
        gates: The gates to use for the mask.
        detach_inputs: Whether to detach the inputs to the gates.
        sigmoid_type: Type of sigmoid to use.

    Returns:
        Tuple of (causal_importances, causal_importances_upper_leaky) dictionaries for each layer.
    """
    causal_importances = {}
    causal_importances_upper_leaky = {}

    for param_name in pre_weight_acts:
        acts = pre_weight_acts[param_name]
        gate = gates[param_name]

        if isinstance(gate, GateMLP):
            # need to get the inner activation for GateMLP
            if not acts.dtype.is_floating_point:
                # Embedding layer
                inner_acts = Vs[param_name][acts]
            else:
                # Linear layer
                inner_acts = einops.einsum(acts, Vs[param_name], "... d_in, d_in C -> ... C")
            gate_input = inner_acts
        else:
            gate_input = acts

        if detach_inputs:
            gate_input = gate_input.detach()

        gate_output = gate(gate_input)

        if sigmoid_type == "leaky_hard":
            causal_importances[param_name] = SIGMOID_TYPES["lower_leaky_hard"](gate_output)
            causal_importances_upper_leaky[param_name] = SIGMOID_TYPES["upper_leaky_hard"](
                gate_output
            )
        else:
            # For other sigmoid types, use the same function for both
            sigmoid_fn = SIGMOID_TYPES[sigmoid_type]
            causal_importances[param_name] = sigmoid_fn(gate_output)
            # Use absolute value to ensure upper_leaky values are non-negative for importance minimality loss
            causal_importances_upper_leaky[param_name] = sigmoid_fn(gate_output).abs()

    return causal_importances, causal_importances_upper_leaky
