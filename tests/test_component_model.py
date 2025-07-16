from typing import override

import pytest
import torch
from jaxtyping import Float
from torch import nn

from spd.models.component_model import ComponentModel
from spd.models.components import EmbeddingComponent, LinearComponent, ReplacedComponent


class SimpleTestModel(nn.Module):
    """Simple test model with Linear and Embedding layers for unit‑testing."""

    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(10, 5, bias=True)
        self.linear2 = nn.Linear(5, 3, bias=False)
        self.embedding = nn.Embedding(100, 8)
        self.other_layer = nn.ReLU()  # Non‑target layer (should never be wrapped)

    @override
    def forward(self, x: Float[torch.Tensor, "... 10"]):  # noqa: D401,E501
        return self.linear2(self.linear1(x))


@pytest.fixture(scope="function")
def component_model() -> ComponentModel:
    """Return a fresh ``ComponentModel`` for each test."""
    base_model = SimpleTestModel()
    return ComponentModel(
        base_model=base_model,
        target_module_patterns=["linear1", "linear2", "embedding"],
        C=4,
        gate_type="mlp",
        gate_hidden_dims=[4],
        pretrained_model_output_attr=None,
    )


def test_no_replacement_masks_means_original_mode(component_model: ComponentModel):
    cm = component_model

    # Initial state: nothing should be active
    assert all(comp.forward_mode is None for comp in cm.replaced_components.values())
    assert all(comp.mask is None for comp in cm.replaced_components.values())

    # No masks supplied: everything should stay in "original" mode
    with cm._replaced_modules({}):
        assert all(comp.forward_mode == "original" for comp in cm.replaced_components.values())
        assert all(comp.mask is None for comp in cm.replaced_components.values())

    # After the context the state must be fully reset
    assert all(comp.forward_mode is None for comp in cm.replaced_components.values())
    assert all(comp.mask is None for comp in cm.replaced_components.values())


def test_replaced_modules_sets_and_restores_masks(component_model: ComponentModel):
    cm = component_model
    full_masks = {
        name: torch.randn(1, cm.C, dtype=torch.float32) for name in cm.replaced_components
    }
    with cm._replaced_modules(full_masks):
        # All components should now be in replacement‑mode with the given masks
        for name, comp in cm.replaced_components.items():
            assert comp.forward_mode == "replacement"
            assert torch.equal(comp.mask, full_masks[name])  # pyright: ignore [reportArgumentType]

    # Back to pristine state
    assert all(comp.forward_mode is None for comp in cm.replaced_components.values())
    assert all(comp.mask is None for comp in cm.replaced_components.values())


def test_replaced_modules_sets_and_restores_masks_partial(component_model: ComponentModel):
    cm = component_model
    # Partial masking
    partial_masks = {"linear1": torch.ones(1, cm.C)}
    with cm._replaced_modules(partial_masks):
        assert cm.replaced_components["linear1"].forward_mode == "replacement"
        assert torch.equal(cm.replaced_components["linear1"].mask, partial_masks["linear1"])  # pyright: ignore [reportArgumentType]
        # Others fall back to original‑only mode with no masks
        assert cm.replaced_components["linear2"].forward_mode == "original"
        assert cm.replaced_components["linear2"].mask is None
        assert cm.replaced_components["embedding"].forward_mode == "original"

    # Back to pristine state
    assert all(comp.forward_mode is None for comp in cm.replaced_components.values())
    assert all(comp.mask is None for comp in cm.replaced_components.values())


def test_replaced_component_forward_linear_matches_modes():
    lin = nn.Linear(6, 4, bias=True)
    comp = LinearComponent(d_in=6, d_out=4, C=3, bias=lin.bias)
    rep = ReplacedComponent(original=lin, replacement=comp)

    x = torch.randn(5, 6)

    # --- Original path ---
    rep.forward_mode = "original"
    rep.mask = None
    out_orig = rep(x)
    expected_orig = lin(x)
    torch.testing.assert_close(out_orig, expected_orig, rtol=1e-4, atol=1e-5)

    # --- Replacement path (with mask) ---
    rep.forward_mode = "replacement"
    mask = torch.rand(5, 3)
    rep.mask = mask
    out_rep = rep(x)
    expected_rep = comp(x, mask)
    torch.testing.assert_close(out_rep, expected_rep, rtol=1e-4, atol=1e-5)


def test_replaced_component_forward_embedding_matches_modes():
    vocab_size = 50
    embedding_dim = 16
    C = 2

    emb = nn.Embedding(vocab_size, embedding_dim)
    comp = EmbeddingComponent(vocab_size=vocab_size, embedding_dim=embedding_dim, C=C)
    rep = ReplacedComponent(original=emb, replacement=comp)

    batch_size = 4
    seq_len = 7
    idx = torch.randint(0, vocab_size, (batch_size, seq_len))  # (batch pos)

    # --- Original path ---
    rep.forward_mode = "original"
    rep.mask = None
    out_orig = rep(idx)
    expected_orig = emb(idx)
    torch.testing.assert_close(out_orig, expected_orig, rtol=1e-4, atol=1e-5)

    # --- Replacement path (with mask) ---
    rep.forward_mode = "replacement"
    mask = torch.rand(batch_size, seq_len, C)  # (batch pos C)
    rep.mask = mask
    out_rep = rep(idx)
    expected_rep = comp.forward(idx, mask)
    torch.testing.assert_close(out_rep, expected_rep, rtol=1e-4, atol=1e-5)
