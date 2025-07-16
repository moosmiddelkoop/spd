"""Plotting utilities for Computation in Superposition experiment."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from spd.experiments.cis.models import CISModel


def create_stacked_weight_plot(
    model: CISModel,
    filepath: Path,
    sort_by_importance: bool = True,
    weight_threshold: float = 0.1,
    degree_threshold: int = 10,
) -> None:
    """Create the stacked weight plot visualization.

    Args:
        model: The trained CIS model
        filepath: Path to save the plot
        sort_by_importance: Whether to sort neurons by importance of largest feature
        weight_threshold: Minimum weight magnitude to consider
        degree_threshold: Maximum degree to display for color mapping
    """

    # Get W1 weights (input to hidden)
    W1 = model.W1.weight.detach().cpu().numpy()  # Shape: (n_hidden, n_features)
    n_hidden, n_features = W1.shape

    # Get absolute weights for visualization
    abs_weights = np.abs(W1)
    abs_weights = np.where(abs_weights < weight_threshold, 0, abs_weights)

    # Count how many features each neuron responds to
    feature_degrees = np.sum(abs_weights > weight_threshold, axis=0)
    feature_degrees = np.where(
        feature_degrees < degree_threshold, feature_degrees, degree_threshold
    )
    represented_features = np.arange(n_features)[feature_degrees > 0]
    min_represented_feature_degree = np.min(feature_degrees[represented_features])
    feature_degrees = feature_degrees - min_represented_feature_degree
    normalized_feature_degrees = feature_degrees / (np.max(feature_degrees) + 1e-6)

    cmap = plt.colormaps["viridis"]

    # Sort neurons by importance of their largest feature if requested
    if sort_by_importance:
        # For each neuron, find the feature with largest absolute weight
        max_feature_per_neuron = np.argmax(abs_weights, axis=1)

        # Sort neurons by the importance (index) of their most important feature
        neuron_order = np.argsort(max_feature_per_neuron)
        abs_weights = abs_weights[neuron_order]

    # Create the plot
    _, ax = plt.subplots(figsize=(max(12, n_hidden * 0.3), 8))

    # For each neuron (column), create stacked bars
    for neuron_idx in range(n_hidden):
        weights = abs_weights[neuron_idx]

        # Sort features by weight magnitude for better visualization
        feature_order = np.argsort(weights)[::-1]  # Descending order
        main_weight = weights[feature_order[0]]

        # Create stacked bars
        bottom = 0
        for feature_idx in feature_order:
            weight = weights[feature_idx]
            if weight > 0 and feature_idx in represented_features:  # Only plot non-zero weights
                ax.bar(
                    neuron_idx,
                    weight / main_weight,
                    bottom=bottom / main_weight,
                    color=cmap(normalized_feature_degrees[feature_idx]),
                    width=0.8,
                    edgecolor="white",
                    linewidth=0.5,
                )
                bottom += weight

    # Customize the plot
    ax.set_xlabel("Hidden Neurons")
    ax.set_ylabel("Absolute Weight Magnitude")
    ax.set_title("Stacked Weight Plot: Feature Representation in Hidden Neurons")
    ax.set_xticks(range(n_hidden))
    ax.set_xticklabels([f"N{i}" for i in range(n_hidden)])

    # Add grid for better readability
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.close()


def create_raw_weights_heatmap(
    model: CISModel,
    filepath: Path,
    layer: str = "W1",
) -> None:
    """Create heatmap visualization of raw weights.

    Args:
        model: The trained CIS model
        filepath: Path to save the plot
        layer: Which layer to visualize ("W1" or "W2")
    """
    if layer == "W1":
        weights = model.W1.weight.detach().cpu().numpy()
        title = "W1 Weights (Input to Hidden)"
        xlabel = "Input Features"
        ylabel = "Hidden Neurons"
    elif layer == "W2":
        weights = model.W2.weight.detach().cpu().numpy()
        title = "W2 Weights (Hidden to Output)"
        xlabel = "Hidden Neurons"
        ylabel = "Output Features"
    else:
        raise ValueError("layer must be 'W1' or 'W2'")

    _, ax = plt.subplots(figsize=(12, 8))

    # Create heatmap
    im = ax.imshow(weights, cmap="RdBu_r", aspect="auto")

    # Add colorbar, where 0 is white and the max is red and the min is blue
    im.set_clim(-1, 1)
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Weight Value")

    # Customize the plot
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    # Set ticks
    ax.set_xticks(range(0, weights.shape[1], max(1, weights.shape[1] // 10)))
    ax.set_yticks(range(0, weights.shape[0], max(1, weights.shape[0] // 10)))

    plt.tight_layout()
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.close()
