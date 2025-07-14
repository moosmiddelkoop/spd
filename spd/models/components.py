from collections.abc import Iterable
from typing import Literal, cast, override

import einops
import torch
from jaxtyping import Float
from torch import Tensor, nn
from torch.nn import functional as F

from spd.utils.module_utils import init_param_

GateType = Literal["mlp", "vector_mlp"]


class GateMLP(nn.Module):
    """A gate with a hidden layer that maps a single input to a single output."""

    def __init__(self, C: int, hidden_dims: list[int]):
        super().__init__()

        self.hidden_dims = hidden_dims
        dim_pairs = list(zip([1] + hidden_dims, hidden_dims + [1], strict=True))

        self.weights_CDiDo: nn.ParameterList = nn.ParameterList(
            [nn.Parameter(torch.empty((C, in_dim, out_dim))) for in_dim, out_dim in dim_pairs]
        )

        self.biases_CDo: nn.ParameterList = nn.ParameterList(
            [nn.Parameter(torch.zeros((C, out_dim))) for _in_dim, out_dim in dim_pairs]
        )

        for weight_CDiDo, (in_dim, _out_dim) in zip(
            cast(Iterable[nn.Parameter], self.weights_CDiDo), dim_pairs, strict=True
        ):
            init_param_(weight_CDiDo, fan_val=in_dim, nonlinearity="relu")

    def forward(self, x_BxC: Tensor) -> Tensor:
        hidden_BxCDi = einops.rearrange(x_BxC, "... C -> ... C 1")
        for weight_CDiDo, bias_CDo in zip(self.weights_CDiDo, self.biases_CDo, strict=True):
            hidden_BxCDo = (
                einops.einsum(
                    hidden_BxCDi, weight_CDiDo, "... C d_in, C in_dim out_dim -> ... C out_dim"
                )
                + bias_CDo
            )
            hidden_BxCDi = F.gelu(hidden_BxCDo)

        assert hidden_BxCDi.shape[-1] == 1, "Last dimension should be 1 after the final layer"
        hidden_BxC = hidden_BxCDi[..., 0]

        return hidden_BxC


class VectorGateMLP(nn.Module):
    """An MLP based gate that maps a vector valued input (residual stream,
    MLP hidden activations, etc.) to a single output."""

    def __init__(self, C: int, input_dim: int, hidden_dims: list[int]):
        super().__init__()

        self.hidden_dims = hidden_dims
        dim_pairs = list(zip([input_dim] + hidden_dims, hidden_dims + [1], strict=True))

        self.weights_CDiDo: nn.ParameterList = nn.ParameterList(
            [nn.Parameter(torch.empty((C, in_dim, out_dim))) for in_dim, out_dim in dim_pairs]
        )

        self.biases_CDo: nn.ParameterList = nn.ParameterList(
            [nn.Parameter(torch.zeros((C, out_dim))) for _in_dim, out_dim in dim_pairs]
        )

        for weight_CDiDo, (in_dim, _out_dim) in zip(
            cast(Iterable[nn.Parameter], self.weights_CDiDo), dim_pairs, strict=True
        ):
            init_param_(weight_CDiDo, fan_val=in_dim, nonlinearity="relu")

    def forward(self, x_BxD: Tensor) -> Tensor:
        hidden_BxCDi = einops.rearrange(
            x_BxD, "... d_in -> ... C d_in", C=1
        )  # this C=1 will broadcast out to actual C size, but no need to expand out yet
        for weight_CDiDo, bias_CDo in zip(self.weights_CDiDo, self.biases_CDo, strict=True):
            hidden_BxCDo = (
                einops.einsum(
                    hidden_BxCDi, weight_CDiDo, "... C d_in, C in_dim out_dim -> ... C out_dim"
                )
                + bias_CDo
            )
            hidden_BxCDi = F.gelu(hidden_BxCDo)

        assert hidden_BxCDi.shape[-1] == 1, "Last dimension should be 1 after the final layer"
        hidden_BxC = hidden_BxCDi[..., 0]

        return hidden_BxC


class LinearComponent(nn.Module):
    """A linear transformation made from V and U matrices for SPD.

    The weight matrix W is decomposed as W = U^T @ V^T, where V and U are learned parameters.
    """

    def __init__(self, d_in: int, d_out: int, C: int, bias: Tensor | None):
        super().__init__()
        self.C = C

        self.V = nn.Parameter(torch.empty(d_in, C))
        self.U = nn.Parameter(torch.empty(C, d_out))
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
    ):
        super().__init__()
        self.C: int = C

        self.V: nn.Parameter = nn.Parameter(torch.empty(vocab_size, C))
        self.U: nn.Parameter = nn.Parameter(torch.empty(C, embedding_dim))

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
