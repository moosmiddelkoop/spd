from abc import ABC, abstractmethod
from typing import Literal, override

import einops
import torch
from jaxtyping import Float
from torch import Tensor, nn

from spd.utils.module_utils import init_param_

GateType = Literal["mlp", "vector_mlp"]


class ParallelLinear(nn.Module):
    """C parallel linear layers"""

    def __init__(self, C: int, input_dim: int, output_dim: int, nonlinearity: str):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.W = nn.Parameter(torch.empty(C, input_dim, output_dim))
        self.b = nn.Parameter(torch.zeros(C, output_dim))
        init_param_(self.W, fan_val=input_dim, nonlinearity=nonlinearity)

    @override
    def forward(self, x: Float[Tensor, "... C d_in"]) -> Float[Tensor, "... C d_out"]:
        return einops.einsum(x, self.W, "... C d_in, C d_in d_out -> ... C d_out") + self.b


class GateMLP(nn.Module):
    """A gate with a hidden layer that maps a scalar input to a scalar output."""

    def __init__(self, C: int, hidden_dims: list[int]):
        super().__init__()

        self.hidden_dims = hidden_dims

        self.layers = nn.Sequential()
        for i in range(len(hidden_dims)):
            input_dim = 1 if i == 0 else hidden_dims[i - 1]
            output_dim = hidden_dims[i]
            self.layers.append(ParallelLinear(C, input_dim, output_dim, nonlinearity="relu"))
            self.layers.append(nn.GELU())
        self.layers.append(ParallelLinear(C, hidden_dims[-1], 1, nonlinearity="linear"))

    @override
    def forward(self, x: Float[Tensor, "... C"]) -> Float[Tensor, "... C"]:
        x = einops.rearrange(x, "... C -> ... C 1")
        x = self.layers(x)
        assert x.shape[-1] == 1, "Last dimension should be 1 after the final layer"
        return x[..., 0]


class VectorGateMLP(nn.Module):
    """An MLP based gate that maps a vector valued input to a single output."""

    def __init__(self, C: int, input_dim: int, hidden_dims: list[int]):
        super().__init__()

        self.hidden_dims = hidden_dims

        self.layers = nn.Sequential()
        for i in range(len(hidden_dims)):
            input_dim = input_dim if i == 0 else hidden_dims[i - 1]
            output_dim = hidden_dims[i]
            self.layers.append(ParallelLinear(C, input_dim, output_dim, nonlinearity="relu"))
            self.layers.append(nn.GELU())

        self.layers.append(ParallelLinear(C, hidden_dims[-1], 1, nonlinearity="linear"))

    @override
    def forward(self, x: Float[Tensor, "... d_in"]) -> Float[Tensor, "... C"]:
        # this 1 will broadcast out to actual C size, but no need to expand out yet
        x = self.layers(einops.rearrange(x, "... d_in -> ... 1 d_in"))
        assert x.shape[-1] == 1, "Last dimension should be 1 after the final layer"
        return x[..., 0]


class Component(ABC, nn.Module):
    def __init__(self, C: int, rows: int, cols: int):
        super().__init__()
        self.C = C
        self.V = nn.Parameter(torch.empty(rows, C))
        self.U = nn.Parameter(torch.empty(C, cols))

    @property
    def weight(self) -> Float[Tensor, "rows cols"]:
        """V @ U"""
        return einops.einsum(self.V, self.U, "rows C, C cols -> rows cols")

    def init_from_target_weight(self, target_weight: Tensor) -> None:
        """Initialize the V and U matrices.
        1. Normalize every component to 1.
        2. Take inner product with original model
        3. This gives you roughly how much overlap there is with the target model.
        4. Scale the Us by this value (we can choose either matrix)
        """

        V = self.V
        U = self.U

        # Make V and U have unit norm in the d_in and d_out dimensions
        V.data[:] = torch.randn_like(V.data)
        U.data[:] = torch.randn_like(U.data)
        V.data[:] = V.data / V.data.norm(dim=-2, keepdim=True)
        U.data[:] = U.data / U.data.norm(dim=-1, keepdim=True)

        # Calculate inner products
        inner = einops.einsum(U, target_weight, "C d_out, d_out d_in -> C d_in")
        C_norms = einops.einsum(inner, V, "C d_in, d_in C -> C")

        # Scale U by the inner product.
        U.data[:] = U.data * C_norms.unsqueeze(-1)

    @override
    @abstractmethod
    def forward(self, x: Tensor, mask: Tensor | None) -> Tensor:
        """Forward pass through the component."""
        raise NotImplementedError()


class LinearComponent(Component):
    """A linear transformation made from V and U matrices for SPD.

    The weight matrix W is decomposed as W = U^T @ V^T, where V and U are learned parameters.
    """

    def __init__(
        self,
        C: int,
        d_in: int,
        d_out: int,
        bias: Tensor | None = None,
    ):
        super().__init__(C, rows=d_out, cols=d_in)  # NOTE: linear weights are (d_out, d_in)
        self.d_in = d_in
        self.d_out = d_out
        self.bias = bias

    @override
    def forward(
        self, x: Float[Tensor, "... d_in"], mask: Tensor | None = None
    ) -> Float[Tensor, "... d_out"]:
        """Forward pass through V and U matrices.

        Args:
            x: Input tensor
            mask: Tensor which masks parameter components. May be boolean or float.
        Returns:
            output: The summed output across all components
        """
        component_acts = einops.einsum(x, self.V, "... d_in, d_in C -> ... C")

        if mask is not None:
            component_acts *= mask

        out = einops.einsum(component_acts, self.U, "... C, C d_out -> ... d_out")

        if self.bias is not None:
            out += self.bias

        return out


class EmbeddingComponent(Component):
    """An efficient embedding component for SPD that avoids one-hot encoding."""

    def __init__(
        self,
        C: int,
        vocab_size: int,
        embedding_dim: int,
    ):
        super().__init__(C, rows=vocab_size, cols=embedding_dim)
        self.vocab_size: int = vocab_size
        self.embedding_dim: int = embedding_dim

    @override
    def forward(
        self, x: Float[Tensor, "batch pos"], mask: Tensor | None
    ) -> Float[Tensor, "batch pos embedding_dim"]:
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

        if mask is not None:
            component_acts *= mask

        out = einops.einsum(
            component_acts, self.U, "batch pos C, ... C embedding_dim -> batch pos embedding_dim"
        )
        return out


# TODO(oli) make this the only public class here
class ReplacedComponent(nn.Module):
    def __init__(
        self,
        original: nn.Linear | nn.Embedding,
        replacement: LinearComponent | EmbeddingComponent,
    ):
        super().__init__()
        assert isinstance(original, nn.Linear) == isinstance(replacement, LinearComponent)
        self.original = original
        self.replacement = replacement

        self.forward_mode: Literal["original"] | Literal["replacement"] | None = None
        self.mask: Tensor | None = None

    @override
    def forward(self, x: Tensor) -> Tensor:
        if self.forward_mode is None:
            raise ValueError("Forward mode not set")

        if self.forward_mode == "original":
            assert self.mask is None, "Mask should not be present in original mode"
            return self.original(x)
        elif self.forward_mode == "replacement":
            # mask *can* but doesn't *need to* be present here
            return self.replacement(x, self.mask)

        raise ValueError(f"Invalid forward mode: {self.forward_mode}")
