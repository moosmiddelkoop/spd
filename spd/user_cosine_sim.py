"""User-defined metrics and figures for SPD experiments.

This file provides custom metrics and visualizations that are logged during SPD optimization.
Users can modify this file to add their own metrics and figures without changing core framework code.
"""
# pyright: reportUnusedParameter=false

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float
from matplotlib import pyplot as plt
from torch import Tensor

from spd.models.components import EmbeddingComponent, LinearComponent


def calc_component_target_cosine_similarities(
    components: dict[str, LinearComponent | EmbeddingComponent],
    target_model: nn.Module,
    n_bins: int | None = None,
) -> dict[str, tuple[Float[Tensor, "..."], Float[Tensor, "..."]]]:
    """Calculate histogram of cosine similarities between subcomponents and target model parameters.
    For each component, we calculate the cosine similarity between each of the C subcomponents
    (formed by A[:, c] @ B[c, :]) and the corresponding target model weight matrix, then create
    a histogram of these values.
    Args:
        components: Dictionary mapping component names to components
        target_model: The target model containing the original parameters
        n_bins: Number of bins for the histogram. If None, uses C where C is the number of subcomponents.
    Returns:
        Dictionary mapping component names to (bin_centers, bin_counts) tuples
    """
    cosine_histograms = {}

    for comp_name, component in components.items():
        # Determine number of bins for this component
        component_n_bins = max(1, int(component.C)) if n_bins is None else n_bins

        # Get target weight matrix
        target_submodule = target_model.get_submodule(comp_name)
        assert isinstance(target_submodule, nn.Linear | nn.Embedding)
        target_weight = target_submodule.weight

        # For embedding components, we need to transpose to match the component format
        if isinstance(component, EmbeddingComponent):
            target_weight = (
                target_weight.T
            )  # (vocab_size, embedding_dim) -> (embedding_dim, vocab_size)

        # Calculate individual subcomponent weights: V[:, c] @ U[c, :] for each c
        V = component.V  # (d_in, C) or (vocab_size, C)
        U = component.U  # (C, d_out) or (C, embedding_dim)

        subcomp_weights = einops.einsum(V, U, "d_in C, C d_out -> C d_out d_in")  # (C, d_out, d_in)

        # Flatten for cosine similarity calculation
        subcomp_flat = subcomp_weights.flatten(start_dim=1)  # (C, d_out * d_in)
        target_flat = target_weight.flatten()  # (d_out * d_in)

        # Calculate cosine similarity
        cos_sims = F.cosine_similarity(subcomp_flat, target_flat, dim=1)

        # Move to CPU for histogram computation (torch.histogram doesn't support CUDA)
        cos_sims = cos_sims.cpu()

        # Create histogram with fixed range [-1, 1]
        bin_counts, bin_edges = torch.histogram(cos_sims, bins=component_n_bins, range=(-1.0, 1.0))

        # Calculate bin centers
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        cosine_histograms[comp_name] = (bin_centers, bin_counts)

    return cosine_histograms


def plot_cosine_similarity_histograms(
    cosine_histograms: dict[str, tuple[Float[Tensor, "..."], Float[Tensor, "..."]]],
) -> dict[str, plt.Figure]:
    """Plot histograms of cosine similarities between subcomponents and target model parameters.
    Args:
        cosine_histograms: Dictionary mapping component names to (bin_centers, bin_counts) tuples.
    Returns:
        Dictionary mapping layer names to histogram figures.
    """
    fig_dict = {}

    for layer_name, (bin_centers, bin_counts) in cosine_histograms.items():
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(bin_centers.cpu().numpy(), bin_counts.cpu().numpy(), width=1.0 / 100)
        ax.set_xlabel("Cosine Similarity")
        ax.set_ylabel("Count")
        ax.set_title(f"Cosine Similarity Distribution - {layer_name}")
        ax.set_xlim(-1, 1)
        ax.grid(True, alpha=0.3)

        fig_dict[f"cosine_sim_histogram_{layer_name}"] = fig

    return fig_dict
