from typing import override

import einops
import torch
from jaxtyping import Float
from torch import Tensor, nn
from torch.nn import functional as F

from spd.module_utils import init_param_


class Gate(nn.Module):
    """A gate that maps a single input to a single output."""

    def __init__(self, C: int, init_central: bool, dtype: torch.dtype):
        super().__init__()
        self.weight = nn.Parameter(torch.empty((C,), dtype=dtype))
        self.bias = nn.Parameter(torch.zeros((C,), dtype=dtype))
        if init_central:
            with torch.no_grad():
                self.bias.add_(0.5)

        fan_val = 1  # Since each weight gets applied independently
        init_param_(self.weight, fan_val=fan_val, nonlinearity="linear")

    @override
    def forward(self, x: Float[Tensor, "... C"]) -> Float[Tensor, "... C"]:
        return x * self.weight + self.bias


class GateMLP(nn.Module):
    """A gate with a hidden layer that maps a single input to a single output."""

    def __init__(self, C: int, n_ci_mlp_neurons: int, init_central: bool, dtype: torch.dtype):
        super().__init__()
        self.n_ci_mlp_neurons = n_ci_mlp_neurons

        self.mlp_in = nn.Parameter(torch.empty((C, n_ci_mlp_neurons), dtype=dtype))
        self.in_bias = nn.Parameter(torch.zeros((C, n_ci_mlp_neurons), dtype=dtype))
        self.mlp_out = nn.Parameter(torch.empty((C, n_ci_mlp_neurons), dtype=dtype))
        self.out_bias = nn.Parameter(torch.zeros((C,), dtype=dtype))
        if init_central:
            with torch.no_grad():
                self.out_bias.add_(0.5)

        init_param_(self.mlp_in, fan_val=1, nonlinearity="relu")
        init_param_(self.mlp_out, fan_val=n_ci_mlp_neurons, nonlinearity="linear")

    @override
    def forward(self, x: Float[Tensor, "... C"]) -> Float[Tensor, "... C"]:
        hidden = (
            einops.einsum(
                x,
                self.mlp_in,
                "... C, C n_ci_mlp_neurons -> ... C n_ci_mlp_neurons",
            )
            + self.in_bias
        )
        hidden = F.gelu(hidden)

        out = (
            einops.einsum(
                hidden,
                self.mlp_out,
                "... C n_ci_mlp_neurons, C n_ci_mlp_neurons -> ... C",
            )
            + self.out_bias
        )
        return out


class ResidGateMLP(nn.Module):
    """A gate with a hidden layer that maps a single input to a single output."""

    def __init__(
        self,
        C: int,
        d_in: int,
        n_ci_mlp_neurons: int,
        init_central: bool,
        dtype: torch.dtype,
    ):
        super().__init__()
        self.n_ci_mlp_neurons = n_ci_mlp_neurons

        self.mlp_in_CIH = nn.Parameter(torch.empty((C, d_in, n_ci_mlp_neurons), dtype=dtype))
        self.in_bias_CH = nn.Parameter(torch.zeros((C, n_ci_mlp_neurons), dtype=dtype))
        self.mlp_out_CH = nn.Parameter(torch.empty((C, n_ci_mlp_neurons), dtype=dtype))
        self.out_bias_C = nn.Parameter(torch.zeros((C,), dtype=dtype))
        if init_central:
            with torch.no_grad():
                self.out_bias_C.add_(0.5)

        init_param_(self.mlp_in_CIH, fan_val=d_in, nonlinearity="relu")
        init_param_(self.mlp_out_CH, fan_val=n_ci_mlp_neurons, nonlinearity="linear")

    @override
    def forward(self, x_BxD: Tensor) -> Tensor:
        hidden_CH = (
            einops.einsum(
                x_BxD,
                self.mlp_in_CIH,
                "... d_in, C d_in n_ci_mlp_neurons -> ... C n_ci_mlp_neurons",
            )
            + self.in_bias_CH
        )
        hidden_CH = F.gelu(hidden_CH)

        out_BxC = (
            einops.einsum(
                hidden_CH,
                self.mlp_out_CH,
                "... C n_ci_mlp_neurons, C n_ci_mlp_neurons -> ... C",
            )
            + self.out_bias_C
        )
        return out_BxC
    
    @torch.no_grad()
    def init_weights_from_mean_input_norm_(self, input_mean_norm: float):
        """Initialize the weights of the gate from the mean of the input."""
        self.mlp_in_CIH.div_(input_mean_norm)


class LinearComponent(nn.Module):
    """A linear transformation made from V and U matrices for SPD.

    The weight matrix W is decomposed as W = U^T @ V^T, where V and U are learned parameters.
    """

    def __init__(self, d_in: int, d_out: int, C: int, bias: Tensor | None, dtype: torch.dtype):
        super().__init__()
        self.C = C

        self.V = nn.Parameter(torch.empty(d_in, C, dtype=dtype))
        self.U = nn.Parameter(torch.empty(C, d_out, dtype=dtype))
        self.bias = bias

        init_param_(self.V, fan_val=d_out, nonlinearity="linear")
        init_param_(self.U, fan_val=C, nonlinearity="linear")

        self.mask: Float[Tensor, "... C"] | None = None  # Gets set on sparse forward passes

    @property
    def weight(self) -> Float[Tensor, "d_out d_in"]:
        """U^T @ V^T"""
        return einops.einsum(self.V, self.U, "d_in C, C d_out -> d_out d_in")

    @override
    def forward(self, x: Float[Tensor, "... d_in"]) -> Float[Tensor, "... d_out"]:
        """Forward pass through V and U matrices.

        Args:
            x: Input tensor
            mask: Tensor which masks parameter components. May be boolean or float.
        Returns:
            output: The summed output across all components
        """
        component_acts = einops.einsum(x, self.V, "... d_in, d_in C -> ... C")

        if self.mask is not None:
            component_acts *= self.mask

        out = einops.einsum(component_acts, self.U, "... C, C d_out -> ... d_out")

        if self.bias is not None:
            out += self.bias

        return out


class EmbeddingComponent(nn.Module):
    """An efficient embedding component for SPD that avoids one-hot encoding."""

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        C: int,
        dtype: torch.dtype,
    ):
        super().__init__()
        self.C: int = C

        self.V: nn.Parameter = nn.Parameter(torch.empty(vocab_size, C, dtype=dtype))
        self.U: nn.Parameter = nn.Parameter(torch.empty(C, embedding_dim, dtype=dtype))

        init_param_(self.V, fan_val=embedding_dim, nonlinearity="linear")
        init_param_(self.U, fan_val=C, nonlinearity="linear")

        # For masked forward passes
        self.mask: Float[Tensor, "batch pos C"] | None = None

    @property
    def weight(self) -> Float[Tensor, "vocab_size embedding_dim"]:
        """V @ U"""
        return einops.einsum(
            self.V, self.U, "vocab_size C, ... C embedding_dim -> vocab_size embedding_dim"
        )

    @override
    def forward(self, x: Float[Tensor, "batch pos"]) -> Float[Tensor, "batch pos embedding_dim"]:
        """Forward through the embedding component using nn.Embedding for efficient lookup

        NOTE: Unlike a LinearComponent, here we alter the mask with an instance attribute rather
        than passing it in the forward pass. This is just because we only use this component in the
        newer lm_decomposition.py setup which does monkey-patching of the modules rather than using
        a SPDModel object.

        Args:
            x: Input tensor of token indices
        """
        # From https://github.com/pytorch/pytorch/blob/main/torch/_decomp/decompositions.py#L1211
        component_acts = self.V[x]  # (batch pos C)

        if self.mask is not None:
            component_acts *= self.mask

        out = einops.einsum(
            component_acts, self.U, "batch pos C, ... C embedding_dim -> batch pos embedding_dim"
        )
        return out
