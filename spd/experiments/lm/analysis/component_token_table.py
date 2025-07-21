"""Script to analyze which tokens activate each component in a trained language model."""

from collections import defaultdict
from pathlib import Path
from typing import ClassVar, cast

import torch
from pydantic import BaseModel, ConfigDict, Field, PositiveFloat, PositiveInt
from transformers import AutoTokenizer

from spd.data import DatasetConfig, create_data_loader
from spd.log import logger
from spd.models.component_model import ComponentModel
from spd.models.components import EmbeddingComponent, GateMLP, LinearComponent, VectorGateMLP
from spd.spd_types import ModelPath
from spd.utils.component_utils import calc_causal_importances
from spd.utils.general_utils import extract_batch_data, get_device


class ComponentTokenAnalysisConfig(BaseModel):
    """Configuration for component token activation analysis."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", frozen=True)

    model_path: ModelPath = Field(
        ...,
        description="Path to the trained ComponentModel checkpoint (local or wandb reference)",
    )
    dataset_name: str = Field(
        default="lennart-finke/SimpleStories",
        description="HuggingFace dataset identifier to analyze",
    )
    dataset_split: str = Field(
        default="train",
        description="Dataset split to use for analysis",
    )
    column_name: str = Field(
        default="story",
        description="Dataset column that contains the text to analyze",
    )
    causal_importance_threshold: PositiveFloat = Field(
        default=0.1,
        description="Minimum causal importance for a component to be considered active",
    )
    n_steps: PositiveInt = Field(
        default=100,
        description="Number of batches to process for analysis",
    )
    batch_size: PositiveInt = Field(
        default=8,
        description="Batch size for processing",
    )
    max_seq_len: PositiveInt = Field(
        default=512,
        description="Maximum sequence length for tokenization",
    )
    output_path: Path | None = Field(
        default=None,
        description="Path to save the markdown output (None to only print)",
    )


def analyze_component_activations(
    config: ComponentTokenAnalysisConfig,
) -> tuple[dict[str, dict[int, dict[int, int]]], dict[str, dict[int, dict[int, list[float]]]], int]:
    """Analyze which tokens activate each component.

    Args:
        config: Configuration for the analysis

    Returns:
        Tuple of:
        - Dictionary mapping module names to component IDs to token counts
        - Dictionary mapping module names to component IDs to token CI values
        - Total number of tokens processed
    """
    device = get_device()
    logger.info(f"Loading model from {config.model_path}")

    # Load the trained model and SPD config
    comp_model, spd_config, _ = ComponentModel.from_pretrained(config.model_path)
    comp_model = comp_model.to(device)
    comp_model.eval()

    # Note: tokenizer is loaded later for decoding

    # Create dataloader
    data_config = DatasetConfig(
        name=config.dataset_name,
        hf_tokenizer_path=spd_config.pretrained_model_name_hf,
        split=config.dataset_split,
        n_ctx=config.max_seq_len,
        is_tokenized=False,
        streaming=False,
        column_name=config.column_name,
    )

    dataloader, _ = create_data_loader(
        dataset_config=data_config,
        batch_size=config.batch_size,
        buffer_size=1000,
        global_seed=42,
        ddp_rank=0,
        ddp_world_size=1,
    )

    # Initialize token activation tracking
    # Structure: {module_name: {component_id: {token_id: count}}}
    component_token_activations: dict[str, dict[int, dict[int, int]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(int))
    )
    # Structure: {module_name: {component_id: {token_id: [ci_values]}}}
    component_token_ci_values: dict[str, dict[int, dict[int, list[float]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )

    # Extract gates and components with proper typing
    gates: dict[str, GateMLP | VectorGateMLP] = {
        k.removeprefix("gates.").replace("-", "."): cast(GateMLP | VectorGateMLP, v)
        for k, v in comp_model.gates.items()
    }
    components: dict[str, LinearComponent | EmbeddingComponent] = {
        k.removeprefix("components.").replace("-", "."): cast(
            LinearComponent | EmbeddingComponent, v
        )
        for k, v in comp_model.components.items()
    }

    logger.info(f"Analyzing activations over {config.n_steps} batches...")

    total_tokens_processed = 0
    data_iter = iter(dataloader)
    for _ in range(config.n_steps):
        batch = extract_batch_data(next(data_iter))
        batch = batch.to(device)

        # Count tokens in this batch
        total_tokens_processed += batch.numel()

        # Get activations before each component
        _, pre_weight_acts = comp_model.forward_with_pre_forward_cache_hooks(
            batch, module_names=list(components.keys())
        )

        Vs = {module_name: v.V for module_name, v in components.items()}

        with torch.no_grad():
            causal_importances, _ = calc_causal_importances(
                pre_weight_acts=pre_weight_acts,
                Vs=Vs,
                gates=gates,
                detach_inputs=True,
            )

        for module_name, ci in causal_importances.items():
            assert ci.ndim == 3, "CI must be 3D (batch, seq_len, C)"

            # Find active components
            active_mask = ci > config.causal_importance_threshold  # (batch, seq_len, C)

            # Get token IDs for this batch
            token_ids = batch  # (batch, seq_len)

            # For each component, track which tokens it activates on
            for component_idx in range(comp_model.C):
                # Get positions where this component is active
                component_active = active_mask[:, :, component_idx]  # (batch, seq_len)

                # Get the tokens at those positions
                active_tokens = token_ids[component_active]

                # Get the CI values at those positions
                active_ci_values = ci[:, :, component_idx][component_active]

                # Count occurrences and store CI values
                for token_id, ci_val in zip(
                    active_tokens.tolist(), active_ci_values.tolist(), strict=False
                ):
                    component_token_activations[module_name][component_idx][token_id] += 1
                    component_token_ci_values[module_name][component_idx][token_id].append(ci_val)

    return component_token_activations, component_token_ci_values, total_tokens_processed


def generate_token_table_markdown(
    component_token_activations: dict[str, dict[int, dict[int, int]]],
    component_token_ci_values: dict[str, dict[int, dict[int, list[float]]]],
    tokenizer: AutoTokenizer,
    total_tokens: int = 0,
) -> str:
    """Generate markdown tables showing top tokens for each component."""
    markdown_output = []

    # Add header with total token count
    markdown_output.append("# Component Token Analysis")
    markdown_output.append(f"*Analysis performed on {total_tokens:,} total tokens*\n")

    for module_name, components in sorted(component_token_activations.items()):
        markdown_output.append(f"## Module: {module_name}\n")
        markdown_output.append("| Component Idx | Tokens (mean_ci_val) [count] |")
        markdown_output.append("|---------------|-----------------------------------------|")

        for component_id, token_counts in sorted(components.items()):
            if not token_counts:
                continue

            # Create list of tokens with their mean CI values and counts
            token_ci_count_tuples = []
            for token_id, count in token_counts.items():
                try:
                    token_text = tokenizer.decode([token_id])  # pyright: ignore[reportAttributeAccessIssue]
                except Exception:
                    token_text = f"<token_{token_id}>"

                # Clean up the token text
                token_text = token_text.strip()
                if token_text:  # Only add non-empty tokens
                    # Calculate mean CI value for this token
                    ci_values = component_token_ci_values[module_name][component_id][token_id]
                    mean_ci = sum(ci_values) / len(ci_values) if ci_values else 0.0
                    token_ci_count_tuples.append((token_text, mean_ci, count))

            # Sort by count first (descending), then by mean CI value (descending)
            sorted_tokens = sorted(token_ci_count_tuples, key=lambda x: (x[2], x[1]), reverse=True)

            if not sorted_tokens:
                continue

            # Format tokens with CI values and counts
            formatted_tokens = []
            for token_text, mean_ci, count in sorted_tokens:
                formatted_tokens.append(f"{token_text} ({mean_ci:.2f}) [{count}]")

            # Join tokens with <> and escape pipe characters
            tokens_str = " <> ".join(formatted_tokens)
            tokens_str = tokens_str.replace("|", "\\|")

            markdown_output.append(f"| {component_id} | {tokens_str} |")

        markdown_output.append("")  # Empty line between modules

    return "\n".join(markdown_output)


def run_analysis(config: ComponentTokenAnalysisConfig) -> None:
    """Run the component token activation analysis with the given configuration."""
    logger.info("Starting component activation analysis...")

    activations, ci_values, total_tokens = analyze_component_activations(config)

    logger.info("Loading tokenizer...")
    _, spd_config, _ = ComponentModel.from_pretrained(config.model_path)
    tokenizer = AutoTokenizer.from_pretrained(spd_config.pretrained_model_name_hf)

    logger.info("Generating markdown table...")
    markdown_table = generate_token_table_markdown(
        component_token_activations=activations,
        component_token_ci_values=ci_values,
        tokenizer=tokenizer,
        total_tokens=total_tokens,
    )

    if config.output_path:
        config.output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(config.output_path, "w") as f:
            f.write(markdown_table)
        logger.info(f"Saved results to {config.output_path}")


if __name__ == "__main__":
    config = ComponentTokenAnalysisConfig(
        model_path="wandb:spd/runs/snq4ojcy",
        dataset_name="SimpleStories/SimpleStories",
        column_name="story",
        dataset_split="test",
        causal_importance_threshold=0.01,
        n_steps=5,
        batch_size=256,
        max_seq_len=512,
        output_path=Path(__file__).parent / "out" / "component_tokens_snq4ojcy.md",
    )

    run_analysis(config)
