from typing import override

import pytest
import torch
from jaxtyping import Float
from torch import Tensor, nn

from spd.models.component_model import ComponentModel
from spd.models.components import ComponentsOrModule, EmbeddingComponents, LinearComponents


class SimpleTestModel(nn.Module):
    """Simple test model with Linear and Embedding layers for unit‑testing."""

    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(10, 5, bias=True)
        self.linear2 = nn.Linear(5, 3, bias=False)
        self.embedding = nn.Embedding(100, 8)
        self.other_layer = nn.ReLU()  # Non‑target layer (should never be wrapped)

    @override
    def forward(self, x: Float[Tensor, "... 10"]):  # noqa: D401,E501
        return self.linear2(self.linear1(x))


@pytest.fixture(scope="function")
def component_model() -> ComponentModel:
    """Return a fresh ``ComponentModel`` for each test."""
    target_model = SimpleTestModel()
    return ComponentModel(
        target_model=target_model,
        target_module_patterns=["linear1", "linear2", "embedding"],
        C=4,
        gate_type="mlp",
        gate_hidden_dims=[4],
        pretrained_model_output_attr=None,
    )


def test_no_replacement_masks_means_original_mode(component_model: ComponentModel):
    cm = component_model

    # Initial state: nothing should be active
    assert all(comp.forward_mode is None for comp in cm.components_or_modules.values())
    assert all(comp.mask is None for comp in cm.components_or_modules.values())

    # No masks supplied: everything should stay in "original" mode
    with cm._replaced_modules({}):
        assert all(comp.forward_mode == "original" for comp in cm.components_or_modules.values())
        assert all(comp.mask is None for comp in cm.components_or_modules.values())

    # After the context the state must be fully reset
    assert all(comp.forward_mode is None for comp in cm.components_or_modules.values())
    assert all(comp.mask is None for comp in cm.components_or_modules.values())


def test_replaced_modules_sets_and_restores_masks(component_model: ComponentModel):
    cm = component_model
    full_masks = {
        name: torch.randn(1, cm.C, dtype=torch.float32) for name in cm.components_or_modules
    }
    with cm._replaced_modules(full_masks):
        # All components should now be in replacement‑mode with the given masks
        for name, comp in cm.components_or_modules.items():
            assert comp.forward_mode == "components"
            assert torch.equal(comp.mask, full_masks[name])  # pyright: ignore [reportArgumentType]

    # Back to pristine state
    assert all(comp.forward_mode is None for comp in cm.components_or_modules.values())
    assert all(comp.mask is None for comp in cm.components_or_modules.values())


def test_replaced_modules_sets_and_restores_masks_partial(component_model: ComponentModel):
    cm = component_model
    # Partial masking
    partial_masks = {"linear1": torch.ones(1, cm.C)}
    with cm._replaced_modules(partial_masks):
        assert cm.components_or_modules["linear1"].forward_mode == "components"
        assert torch.equal(cm.components_or_modules["linear1"].mask, partial_masks["linear1"])  # pyright: ignore [reportArgumentType]
        # Others fall back to original‑only mode with no masks
        assert cm.components_or_modules["linear2"].forward_mode == "original"
        assert cm.components_or_modules["linear2"].mask is None
        assert cm.components_or_modules["embedding"].forward_mode == "original"

    # Back to pristine state
    assert all(comp.forward_mode is None for comp in cm.components_or_modules.values())
    assert all(comp.mask is None for comp in cm.components_or_modules.values())


def test_replaced_component_forward_linear_matches_modes():
    B = 5
    C = 3
    input_dim = 6
    output_dim = 4

    original = nn.Linear(input_dim, output_dim, bias=True)
    components = LinearComponents(d_in=input_dim, d_out=output_dim, C=3, bias=original.bias)
    components_or_module = ComponentsOrModule(original=original, components=components)

    x = torch.randn(B, input_dim)

    # --- Original path ---
    components_or_module.forward_mode = "original"
    components_or_module.mask = None
    out_orig = components_or_module(x)
    expected_orig = original(x)
    torch.testing.assert_close(out_orig, expected_orig, rtol=1e-4, atol=1e-5)

    # --- Replacement path (with mask) ---
    mask = torch.rand(B, C)
    components_or_module.forward_mode = "components"
    components_or_module.mask = mask
    out_rep = components_or_module(x)
    expected_rep = components(x, mask)
    torch.testing.assert_close(out_rep, expected_rep, rtol=1e-4, atol=1e-5)


def test_replaced_component_forward_embedding_matches_modes():
    vocab_size = 50
    embedding_dim = 16
    C = 2

    emb = nn.Embedding(vocab_size, embedding_dim)
    comp = EmbeddingComponents(vocab_size=vocab_size, embedding_dim=embedding_dim, C=C)
    rep = ComponentsOrModule(original=emb, components=comp)

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
    rep.forward_mode = "components"
    mask = torch.rand(batch_size, seq_len, C)  # (batch pos C)
    rep.mask = mask
    out_rep = rep(idx)
    expected_rep = comp.forward(idx, mask)
    torch.testing.assert_close(out_rep, expected_rep, rtol=1e-4, atol=1e-5)
