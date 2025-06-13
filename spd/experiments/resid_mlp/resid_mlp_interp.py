import einops
import fire
import matplotlib.pyplot as plt
import numpy as np
import torch
from jaxtyping import Float
from torch import Tensor

from spd.experiments.resid_mlp.models import ResidualMLP
from spd.models.component_model import ComponentModel
from spd.models.components import LinearComponent
from spd.settings import REPO_ROOT
from spd.utils import set_seed


def feature_contribution_plot(
    ax: plt.Axes,
    relu_conns: Float[Tensor, "n_layers n_features d_mlp"],
    n_layers: int,
    n_features: int,
    d_mlp: int,
    pre_labelled_neurons: dict[int, list[int]] | None = None,
    legend: bool = True,
) -> dict[int, list[int]]:
    diag_relu_conns: Float[Tensor, "n_layers n_features d_mlp"] = relu_conns.cpu().detach()

    # Define colors for different layers
    assert n_layers in [1, 2, 3]
    layer_colors = ["blue", "red", "green"]  # Always use same colors regardless of n_layers

    distinct_colors = [
        "#E41A1C",  # red
        "#377EB8",  # blue
        "#4DAF4A",  # green
        "#984EA3",  # purple
        "#FF7F00",  # orange
        "#A65628",  # brown
        "#F781BF",  # pink
        "#1B9E77",  # teal
        "#D95F02",  # dark orange
        "#7570B3",  # slate blue
        "#66A61E",  # lime green
    ]

    # Add legend if there are two layers
    if n_layers == 2 and legend:
        # Create dummy scatter plots for legend
        ax.scatter([], [], c="blue", alpha=0.3, marker=".", label="First MLP")
        ax.scatter([], [], c="red", alpha=0.3, marker=".", label="Second MLP")
        ax.legend(loc="upper right")
    # Add legend if there are three layers
    if n_layers == 3 and legend:
        # Create dummy scatter plots for legend
        ax.scatter([], [], c="blue", alpha=0.3, marker=".", label="First MLP")
        ax.scatter([], [], c="red", alpha=0.3, marker=".", label="Second MLP")
        ax.scatter([], [], c="green", alpha=0.3, marker=".", label="Third MLP")
        ax.legend(loc="upper right")
    labelled_neurons: dict[int, list[int]] = {i: [] for i in range(n_features)}

    ax.axvline(-0.5, color="k", linestyle="--", alpha=0.3, lw=0.5)
    for i in range(n_features):
        # Split points by layer and plot separately
        for layer in range(n_layers):
            ax.scatter(
                [i] * d_mlp,
                diag_relu_conns[layer, i, :],
                alpha=0.3,
                marker=".",
                c=layer_colors[layer],
            )
        ax.axvline(i + 0.5, color="k", linestyle="--", alpha=0.3, lw=0.5)
        for layer in range(n_layers):
            for j in range(d_mlp):
                # Label the neuron if it's in the pre-labelled set or if no pre-labelled set is provided
                # and the neuron has a connection strength greater than 0.1
                if (
                    pre_labelled_neurons is not None
                    and layer * d_mlp + j in pre_labelled_neurons[i]
                ) or (pre_labelled_neurons is None and diag_relu_conns[layer, i, j].item() > 0.1):
                    color_idx = j % len(distinct_colors)
                    # Make the neuron label alternate between left and right (-0.1, 0.1)
                    # Add 0.05 or -0.05 to the x coordinate to shift the label left or right
                    ax.text(
                        i,
                        diag_relu_conns[layer, i, j].item(),
                        str(layer * d_mlp + j),
                        color=distinct_colors[color_idx],
                        ha="left" if (len(labelled_neurons[i]) + 1) % 2 == 0 else "right",
                    )
                    labelled_neurons[i].append(layer * d_mlp + j)
    ax.axhline(0, color="k", linestyle="--", alpha=0.3)
    ax.set_xlim(-0.5, n_features - 0.5)
    ax.set_xlabel("Features")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    return labelled_neurons


def compute_target_weight_neuron_contributions(
    target_model: ResidualMLP, n_features: int | None = None
) -> Float[Tensor, "n_layers n_features d_mlp"]:
    """Compute per-neuron contribution strengths for a *trained* ResidualMLP.

    The returned tensor has shape ``(n_layers, n_features, d_mlp)`` recording – for
    every hidden layer and every input feature – the *virtual* weight connecting
    that feature to each neuron after the ReLU (i.e. the product ``W_in * W_out``)
    as described in the original script. Only the first ``n_features`` are kept
    (or all features if ``n_features is None``).
    """

    n_features = target_model.config.n_features if n_features is None else n_features

    W_E: Float[Tensor, "n_features d_embed"] = target_model.W_E  # type: ignore
    assert torch.equal(W_E, target_model.W_U.T)

    # Stack mlp_in / mlp_out weights across layers so that einsums can broadcast
    W_in: Float[Tensor, "n_layers d_mlp d_embed"] = torch.stack(
        [layer.mlp_in.weight for layer in target_model.layers], dim=0
    )
    W_out: Float[Tensor, "n_layers d_embed d_mlp"] = torch.stack(
        [layer.mlp_out.weight for layer in target_model.layers], dim=0
    )

    # Compute connection strengths
    in_conns: Float[Tensor, "n_layers n_features d_mlp"] = einops.einsum(
        W_E,
        W_in,
        "n_features d_embed, n_layers d_mlp d_embed -> n_layers n_features d_mlp",
    )
    out_conns: Float[Tensor, "n_layers d_mlp n_features"] = einops.einsum(
        W_out,
        W_E,
        "n_layers d_embed d_mlp, n_features d_embed -> n_layers d_mlp n_features",
    )
    relu_conns: Float[Tensor, "n_layers n_features d_mlp"] = einops.einsum(
        in_conns,
        out_conns,
        "n_layers n_features d_mlp, n_layers d_mlp n_features -> n_layers n_features d_mlp",
    )

    # Truncate to the first *n_features* for visualisation
    return relu_conns[:, :n_features, :]


def compute_spd_weight_neuron_contributions(
    components: dict[str, LinearComponent],
    target_model: ResidualMLP,
    n_features: int | None = None,
) -> Float[Tensor, "n_layers n_features C d_mlp"]:
    """Compute per-neuron contribution strengths for the *SPD* factorisation.

    Returns a tensor of shape ``(n_layers, n_features, C, d_mlp)`` where *C* is
    the number of sub-components in the SPD decomposition.
    """

    n_layers: int = target_model.config.n_layers
    n_features = target_model.config.n_features if n_features is None else n_features

    W_E: Float[Tensor, "n_features d_embed"] = target_model.W_E  # type: ignore

    # Build the *virtual* input weight matrices (A @ B) for every layer
    W_in_spd: Float[Tensor, "n_layers d_embed C d_mlp"] = torch.stack(
        [
            einops.einsum(
                components[f"layers.{i}.mlp_in"].A,
                components[f"layers.{i}.mlp_in"].B,
                "d_embed C, C d_mlp -> d_embed C d_mlp",
            )
            for i in range(n_layers)
        ],
        dim=0,
    )

    # Output weights for every layer
    W_out_spd: Float[Tensor, "n_layers d_embed d_mlp"] = torch.stack(
        [components[f"layers.{i}.mlp_out"].weight for i in range(n_layers)],
        dim=0,
    )

    # Connection strengths
    in_conns_spd: Float[Tensor, "n_layers n_features C d_mlp"] = einops.einsum(
        W_E,
        W_in_spd,
        "n_features d_embed, n_layers d_embed C d_mlp -> n_layers n_features C d_mlp",
    )
    out_conns_spd: Float[Tensor, "n_layers d_mlp n_features"] = einops.einsum(
        W_out_spd,
        W_E,
        "n_layers d_embed d_mlp, n_features d_embed -> n_layers d_mlp n_features",
    )
    relu_conns_spd: Float[Tensor, "n_layers n_features C d_mlp"] = einops.einsum(
        in_conns_spd,
        out_conns_spd,
        "n_layers n_features C d_mlp, n_layers d_mlp n_features -> n_layers n_features C d_mlp",
    )

    return relu_conns_spd[:, :n_features, :, :]


def plot_spd_feature_contributions_truncated(
    components: dict[str, LinearComponent],
    target_model: ResidualMLP,
    n_features: int | None = 50,
):
    n_layers = target_model.config.n_layers
    n_features = target_model.config.n_features if n_features is None else n_features
    d_mlp = target_model.config.d_mlp

    # Assert that there are no biases
    assert not target_model.config.in_bias and not target_model.config.out_bias, (
        "Biases are not supported for these plots"
    )

    # --- Compute neuron contribution tensors ---
    relu_conns: Float[Tensor, "n_layers n_features d_mlp"] = (
        compute_target_weight_neuron_contributions(
            target_model=target_model,
            n_features=n_features,
        )
    )

    relu_conns_spd: Float[Tensor, "n_layers n_features C d_mlp"] = (
        compute_spd_weight_neuron_contributions(
            components=components,
            target_model=target_model,
            n_features=n_features,
        )
    )

    max_component_indices = []
    for i in range(n_layers):
        # For each feature, find the C component with the largest max value over d_mlp
        max_component_indices.append(relu_conns_spd[i].max(dim=-1).values.argmax(dim=-1))
    # For each feature, use the C values based on the max_component_indices
    max_component_contributions: Float[Tensor, "n_layers n_features d_mlp"] = torch.stack(
        [
            relu_conns_spd[i, torch.arange(n_features), max_component_indices[i], :]
            for i in range(n_layers)
        ],
        dim=0,
    )

    n_rows = 2
    fig1, axes1 = plt.subplots(n_rows, 1, figsize=(10, 7), constrained_layout=True)
    axes1 = np.atleast_1d(axes1)  # type: ignore

    labelled_neurons = feature_contribution_plot(
        ax=axes1[0],
        relu_conns=relu_conns,
        n_layers=n_layers,
        n_features=n_features,
        d_mlp=d_mlp,
        legend=True,
    )
    axes1[0].set_ylabel("Neuron contribution")
    axes1[0].set_xlabel(f"Input feature index (first {n_features} shown)")
    axes1[0].set_title("Target model")
    axes1[0].set_xticks(range(n_features))  # Ensure all xticks have labels

    feature_contribution_plot(
        ax=axes1[1],
        relu_conns=max_component_contributions,
        n_layers=n_layers,
        n_features=n_features,
        d_mlp=d_mlp,
        pre_labelled_neurons=labelled_neurons,
        legend=False,
    )
    axes1[1].set_ylabel("Neuron contribution")
    axes1[1].set_xlabel("Subcomponent index")
    axes1[1].set_title("Individual SPD subcomponents")
    axes1[1].set_xticks(range(n_features))

    # Set the same y-axis limits for both plots
    y_min = min(axes1[0].get_ylim()[0], axes1[1].get_ylim()[0])
    y_max = max(axes1[0].get_ylim()[1], axes1[1].get_ylim()[1])
    axes1[0].set_ylim(y_min, y_max)
    axes1[1].set_ylim(y_min, y_max)

    # Label the x axis with the subnets that have the largest neuron for each feature
    axes1[1].set_xticklabels(max_component_indices[0].tolist())  # Labels are the subnet indices

    return fig1


def plot_neuron_contribution_pairs(
    components: dict[str, LinearComponent],
    target_model: ResidualMLP,
    n_features: int | None = 50,
) -> plt.Figure:
    """Create a scatter plot comparing target model and SPD component neuron contributions.

    Each point represents a (component, input_feature, neuron) combination across all layers.
    X-axis: neuron contribution from the target model
    Y-axis: neuron contribution from the SPD component
    """
    n_layers = target_model.config.n_layers
    n_features = target_model.config.n_features if n_features is None else n_features
    d_mlp = target_model.config.d_mlp

    # Assert that there are no biases
    assert not target_model.config.in_bias and not target_model.config.out_bias, (
        "Biases are not supported for these plots"
    )

    # Compute neuron contribution tensors
    relu_conns: Float[Tensor, "n_layers n_features d_mlp"] = (
        compute_target_weight_neuron_contributions(
            target_model=target_model,
            n_features=n_features,
        )
    )

    relu_conns_spd: Float[Tensor, "n_layers n_features C d_mlp"] = (
        compute_spd_weight_neuron_contributions(
            components=components,
            target_model=target_model,
            n_features=n_features,
        )
    )

    # For each layer and feature, find the component with the largest max value over d_mlp
    max_component_indices = []
    for i in range(n_layers):
        # For each feature, find the C component with the largest max value over d_mlp
        max_component_indices.append(relu_conns_spd[i].max(dim=-1).values.argmax(dim=-1))

    # For each feature, use the C values based on the max_component_indices
    max_component_contributions: Float[Tensor, "n_layers n_features d_mlp"] = torch.stack(
        [
            relu_conns_spd[i, torch.arange(n_features), max_component_indices[i], :]
            for i in range(n_layers)
        ],
        dim=0,
    )

    # Define colors for different layers (same as in plot_spd_feature_contributions_truncated)
    assert n_layers in [1, 2, 3]
    layer_colors = ["blue", "red", "green"]  # Always use same colors regardless of n_layers

    # Create scatter plot
    fig, ax = plt.subplots(figsize=(8, 8))

    # Plot points separately for each layer with different colors
    for layer in range(n_layers):
        x_values = relu_conns[layer].flatten().cpu().detach().numpy()
        y_values = max_component_contributions[layer].flatten().cpu().detach().numpy()

        layer_label = {0: "First MLP", 1: "Second MLP", 2: "Third MLP"}.get(layer, f"Layer {layer}")

        ax.scatter(
            x_values,
            y_values,
            alpha=0.3,
            s=10,
            color=layer_colors[layer],
            edgecolors="none",
            label=layer_label if n_layers > 1 else None,
        )

    # Add y=x reference line
    lims = [
        np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
        np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
    ]
    ax.plot(lims, lims, "k--", alpha=0.2, zorder=0, label="y=x")

    # Labels and title
    ax.set_xlabel("Target Model Neuron Contribution", fontsize=12)
    ax.set_ylabel("SPD Component Neuron Contribution (Max Subcomponent)", fontsize=12)
    ax.set_title(
        f"{n_features} input features, {n_layers} layer{'s' if n_layers != 1 else ''}", fontsize=12
    )

    # Make axes equal and square
    ax.set_aspect("equal", adjustable="box")

    # Add grid for better readability
    ax.grid(True, alpha=0.3)

    # Remove top and right spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Add legend if there are multiple layers
    if n_layers > 1:
        ax.legend(loc="lower right")

    # Add some statistics to the plot
    # Calculate correlation for all points combined
    all_x = relu_conns.flatten().cpu().detach().numpy()
    all_y = max_component_contributions.flatten().cpu().detach().numpy()
    correlation = np.corrcoef(all_x, all_y)[0, 1]
    ax.text(
        0.05,
        0.95,
        f"Correlation: {correlation:.3f}",
        transform=ax.transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    return fig


def main():
    out_dir = REPO_ROOT / "spd/experiments/resid_mlp/out/figures/"
    out_dir.mkdir(parents=True, exist_ok=True)
    set_seed(0)
    device = "cpu" if torch.cuda.is_available() else "cpu"

    paths: list[str] = [
        "wandb:spd-resid-mlp/runs/ziro93xq",  # 1 layer
        "wandb:spd-resid-mlp/runs/wau744ht",  # 2 layer
        "wandb:spd-resid-mlp/runs/qqdugze1",  # 3 layer
    ]

    for path in paths:
        wandb_id = path.split("/")[-1]

        model = ComponentModel.from_pretrained(path)[0]
        model.to(device)

        target_model = model.model
        assert isinstance(target_model, ResidualMLP)
        n_layers = target_model.config.n_layers

        components: dict[str, LinearComponent] = {
            k.removeprefix("components.").replace("-", "."): v
            for k, v in model.components.items()
            if isinstance(v, LinearComponent)
        }  # type: ignore

        fig = plot_spd_feature_contributions_truncated(
            components=components,
            target_model=target_model,
            n_features=10,
        )
        fig.savefig(
            out_dir / f"resid_mlp_weights_{n_layers}layers_{wandb_id}.png",
            bbox_inches="tight",
            dpi=500,
        )
        print(f"Saved figure to {out_dir / f'resid_mlp_weights_{n_layers}layers_{wandb_id}.png'}")

        # Generate and save neuron contribution pairs plot
        fig_pairs = plot_neuron_contribution_pairs(
            components=components,
            target_model=target_model,
            n_features=None,  # Using same number of features as above
        )
        fig_pairs.savefig(
            out_dir / f"neuron_contribution_pairs_{n_layers}layers_{wandb_id}.png",
            bbox_inches="tight",
            dpi=500,
        )
        print(
            f"Saved figure to {out_dir / f'neuron_contribution_pairs_{n_layers}layers_{wandb_id}.png'}"
        )


if __name__ == "__main__":
    fire.Fire(main)
