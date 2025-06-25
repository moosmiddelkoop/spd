# %%
import gc
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer
from transformers.tokenization_utils import PreTrainedTokenizer

from research.cajal.tokens.sources import untokenized_text_hf
from research.experiments.parameter_decomposition.constants import CHECKPOINT_DIR
from research.experiments.parameter_decomposition.main import (
    ComponentModel,
    TrainConfig,
    ce,
    get_activation_scale,
    get_dataloader,
    get_model,
    sample_masks_BSM,
    upper_leaky_hard_sigmoid,
)


@dataclass
class ActivationsWithText:
    tokens_HS: torch.Tensor


@dataclass
class ComponentExample:
    tokens_S: torch.Tensor
    last_tok_importance: float


@dataclass
class ComponentSummary:
    module_name: str
    component_idx: int
    selected_examples: list[ComponentExample]
    density: float


class ComponentExaminer:
    def __init__(self, model: ComponentModel, tokens: untokenized_text_hf.UntokenisedTextHF, batch_size: int = 4):
        self.model = model
        self.tokens = tokens
        self.batch_size = batch_size

    @torch.no_grad()
    def gather_component_summaries_optimized(
        self,
        total_batches: int,
        importance_threshold: float = 0.1,
        separation_threshold_tokens: int = 5,
    ) -> list[ComponentSummary]:
        examples_by_component = defaultdict[tuple[str, int], list[ComponentExample]](list)
        num_firings_by_component = defaultdict[tuple[str, int], int](int)

        tokens_seen = 0

        pbar = tqdm(range(total_batches), desc="Gathering component examples")

        for _ in pbar:
            tokens_BS = self.tokens.get_token_batch(self.batch_size).to(self.model.device)
            B, S = tokens_BS.shape

            # Get importance values
            _, preact_cache_BSD, _ = self.model.forward_base_with_cache_BSV(tokens_BS)
            pre_sigmoid_importance_BSM = self.model.get_importance_values_BSM(preact_cache_BSD)

            # Process each module
            for module_name, pre_sigmoid_vals_BSM in pre_sigmoid_importance_BSM.items():
                assert pre_sigmoid_vals_BSM.shape == (B, S, self.model.m)
                importance_vals_BSM = upper_leaky_hard_sigmoid(pre_sigmoid_vals_BSM)

                # Find high importance positions
                for b_idx in range(B):
                    tokens_S = tokens_BS[b_idx]
                    importance_SM = importance_vals_BSM[b_idx]

                    # Get positions where any component exceeds threshold
                    high_importance_positions = (
                        (importance_SM > importance_threshold).any(dim=-1).nonzero(as_tuple=True)[0]
                    )

                    # For each position, find which components are active
                    for pos in high_importance_positions:
                        active_components = (importance_SM[pos] > importance_threshold).nonzero(as_tuple=True)[0]

                        for component_idx in active_components:
                            component_key = (module_name, int(component_idx.item()))

                            # Check separation from last example
                            if existing_examples := examples_by_component[component_key]:
                                last_example_len = len(existing_examples[-1].tokens_S)
                                if pos < last_example_len + separation_threshold_tokens:
                                    continue

                            example = ComponentExample(
                                tokens_S=tokens_S[: pos + 1],
                                last_tok_importance=importance_SM[pos, component_idx].item(),
                            )
                            examples_by_component[component_key].append(example)
                            num_firings_by_component[component_key] += 1

                pbar.set_postfix(n_components_with_examples=len(examples_by_component))

            tokens_seen += B * S

        # Create summaries
        summaries = []
        for (module_name, component_idx), examples in examples_by_component.items():
            summaries.append(
                ComponentSummary(
                    module_name=module_name,
                    component_idx=component_idx,
                    selected_examples=sorted(examples, key=lambda x: x.last_tok_importance, reverse=True),
                    density=num_firings_by_component[(module_name, component_idx)] / tokens_seen,
                )
            )

        return summaries

    @torch.no_grad()
    def gather_component_summaries(
        self,
        total_batches: int,
        importance_threshold: float = 0.1,
        separation_threshold_tokens: int = 5,
    ) -> list[ComponentSummary]:
        # Optimized version with GPU acceleration and vectorization
        examples_by_component = defaultdict[tuple[str, int], list[ComponentExample]](list)
        num_firings_by_component = defaultdict[tuple[str, int], int](int)

        # Pre-allocate buffers for better memory management
        last_position_by_component = {}

        tokens_seen = 0
        pbar = tqdm(range(total_batches), desc="Gathering component examples (optimized)")

        for _ in pbar:
            tokens_BS = self.tokens.get_token_batch(self.batch_size).to(self.model.device)
            B, S = tokens_BS.shape

            # Get importance values
            _, preact_cache_BSD = self.model.forward_base_with_preact_cache_BSV(tokens_BS)
            pre_sigmoid_importance_BSM = self.model.get_importance_values_BSM(preact_cache_BSD)

            # Process each module
            for module_name, pre_sigmoid_vals_BSM in pre_sigmoid_importance_BSM.items():
                assert pre_sigmoid_vals_BSM.shape == (B, S, self.model.m)
                importance_vals_BSM = upper_leaky_hard_sigmoid(pre_sigmoid_vals_BSM)

                # Vectorized approach: find all positions exceeding threshold at once
                high_importance_mask_BSM = importance_vals_BSM > importance_threshold

                # Get indices using torch.where - much faster than nested loops
                batch_indices, pos_indices, component_indices = torch.where(high_importance_mask_BSM)

                # Count firings efficiently
                component_counts = torch.bincount(component_indices, minlength=self.model.m)
                for comp_idx, count in enumerate(component_counts):
                    if count > 0:
                        key = (module_name, comp_idx)
                        num_firings_by_component[key] += int(count.item())

                # Process examples in batches to minimize Python overhead
                if len(batch_indices) > 0:
                    # Sort by batch, then position for cache-friendly access
                    sort_indices = torch.argsort(batch_indices * S + pos_indices)
                    batch_indices = batch_indices[sort_indices]
                    pos_indices = pos_indices[sort_indices]
                    component_indices = component_indices[sort_indices]

                    # Process each batch
                    for b_idx in range(B):
                        batch_mask = batch_indices == b_idx
                        if not batch_mask.any():
                            continue

                        batch_positions = pos_indices[batch_mask]
                        batch_components = component_indices[batch_mask]
                        tokens_S = tokens_BS[b_idx]

                        # Create position-component pairs
                        for i in range(len(batch_positions)):
                            pos = batch_positions[i].item()
                            comp_idx = batch_components[i].item()
                            component_key = (module_name, comp_idx)

                            # Check separation constraint
                            last_pos = last_position_by_component.get(component_key, -separation_threshold_tokens - 1)
                            if pos >= last_pos + separation_threshold_tokens:
                                importance = importance_vals_BSM[b_idx, pos, comp_idx].item()

                                example = ComponentExample(
                                    tokens_S=tokens_S[: pos + 1],
                                    last_tok_importance=importance,
                                )
                                examples_by_component[component_key].append(example)
                                last_position_by_component[component_key] = pos

                pbar.set_postfix(n_components_with_examples=len(examples_by_component))

            tokens_seen += B * S

        # Create summaries
        summaries = []
        for (module_name, component_idx), examples in examples_by_component.items():
            summaries.append(
                ComponentSummary(
                    module_name=module_name,
                    component_idx=component_idx,
                    selected_examples=sorted(examples, key=lambda x: x.last_tok_importance, reverse=True),
                    density=num_firings_by_component[(module_name, component_idx)] / tokens_seen,
                )
            )

        return summaries

    @torch.no_grad()
    def gather_component_summaries_ultra_fast(
        self,
        total_batches: int,
        importance_threshold: float = 0.1,
        separation_threshold_tokens: int = 5,
        max_examples_per_component: int = 1000,
    ) -> list[ComponentSummary]:
        """Ultra-fast version using advanced GPU operations and memory pre-allocation."""

        # Pre-allocate GPU tensors for tracking
        device = self.model.device
        max_components = self.model.m

        # Track last positions and counts on GPU
        component_last_positions = {}  # (module_name, comp_idx) -> last_position
        component_examples = defaultdict(list)  # (module_name, comp_idx) -> list of (tokens, importance)
        component_firing_counts = defaultdict(int)

        tokens_seen = 0
        pbar = tqdm(range(total_batches), desc="Gathering examples (ultra-fast)")

        # Process in larger batches if possible
        effective_batch_size = min(self.batch_size * 4, 32)  # Process more at once

        for batch_idx in pbar:
            # Get larger batch for better GPU utilization
            if batch_idx * self.batch_size < total_batches * self.batch_size:
                tokens_BS = self.tokens.get_token_batch(effective_batch_size).to(device)
                B, S = tokens_BS.shape

                # Get importance values
                with torch.cuda.amp.autocast():  # Use mixed precision for speed
                    _, preact_cache_BSD = self.model.forward_base_with_preact_cache_BSV(tokens_BS)
                    pre_sigmoid_importance_BSM = self.model.get_importance_values_BSM(preact_cache_BSD)

                # Process each module
                for module_name, pre_sigmoid_vals_BSM in pre_sigmoid_importance_BSM.items():
                    importance_vals_BSM = upper_leaky_hard_sigmoid(pre_sigmoid_vals_BSM)

                    # Find all high importance activations at once
                    high_mask = importance_vals_BSM > importance_threshold

                    if high_mask.any():
                        # Get indices efficiently
                        b_indices, s_indices, m_indices = torch.where(high_mask)
                        importances = importance_vals_BSM[b_indices, s_indices, m_indices]

                        # Group by component for efficient processing
                        unique_components = torch.unique(m_indices)

                        for comp_idx in unique_components:
                            comp_mask = m_indices == comp_idx
                            comp_b = b_indices[comp_mask]
                            comp_s = s_indices[comp_mask]
                            comp_imp = importances[comp_mask]

                            # Update firing count
                            key = (module_name, int(comp_idx.item()))
                            component_firing_counts[key] += len(comp_b)

                            # Get last position for this component
                            last_pos = component_last_positions.get(key, -separation_threshold_tokens - 1)

                            # Process examples for this component
                            for i in range(len(comp_b)):
                                b = int(comp_b[i].item())
                                s = int(comp_s[i].item())

                                # Check separation constraint
                                if (
                                    s >= last_pos + separation_threshold_tokens
                                    and len(component_examples[key]) < max_examples_per_component
                                ):
                                    # Only keep if under max examples
                                    example = ComponentExample(
                                        tokens_S=tokens_BS[b, : s + 1],
                                        last_tok_importance=int(comp_imp[i].item()),
                                    )
                                    component_examples[key].append(example)
                                    component_last_positions[key] = s

                pbar.set_postfix(n_components=len(component_examples))
                tokens_seen += B * S

        # Create summaries with top examples only
        summaries = []
        for (module_name, component_idx), examples in component_examples.items():
            # Keep only top examples by importance
            sorted_examples = sorted(examples, key=lambda x: x.last_tok_importance, reverse=True)
            top_examples = sorted_examples[:100]  # Keep top 100

            summaries.append(
                ComponentSummary(
                    module_name=module_name,
                    component_idx=component_idx,
                    selected_examples=top_examples,
                    density=component_firing_counts[(module_name, component_idx)] / tokens_seen,
                )
            )

        return summaries


def display_component_examples(
    summary: ComponentSummary,
    tokenizer: PreTrainedTokenizer,
    topk: int = 10,
    context_tokens: int = 10,
):
    """Display top examples for a component with compact visualization."""
    print(f"\n{'=' * 80}")
    print(f"Module: {summary.module_name}")
    print(f"Component: {summary.component_idx}")
    print(f"Density: {summary.density:.4%}")
    print(f"Total examples: {len(summary.selected_examples)}")
    print(f"{'=' * 80}\n")

    for i, example in enumerate(summary.selected_examples[:topk]):
        tokens_list = tokenizer.convert_ids_to_tokens(example.tokens_S.tolist())

        # Sanitize tokens for display
        sanitized_tokens = []
        for token in tokens_list:
            # Replace newlines with ↵ (U+21B5)
            token = token.replace("\n", "↵")
            token = token.replace("\r", "↵")
            # Replace tabs with → (U+2192)
            token = token.replace("\t", "→")
            # Replace spaces with · (U+00B7)
            token = token.replace(" ", "·")
            sanitized_tokens.append(token)

        target_idx = len(sanitized_tokens) - 1  # The last token is the target

        # Calculate the window around the target token
        start_idx = max(0, target_idx - context_tokens)
        end_idx = min(len(sanitized_tokens), target_idx + context_tokens + 1)

        # Extract tokens for display
        display_tokens = []
        if start_idx > 0:
            display_tokens.append("...")

        window_tokens = sanitized_tokens[start_idx:end_idx]
        display_tokens.extend(window_tokens)

        if end_idx < len(sanitized_tokens):
            display_tokens.append("...")

        # Find the target token in the display list and highlight it
        target_display_idx = target_idx - start_idx + (1 if start_idx > 0 else 0)
        if target_display_idx < len(display_tokens):
            display_tokens[target_display_idx] = f"\033[92m{display_tokens[target_display_idx]}\033[0m"

        print(f"{i + 1}. Importance: {example.last_tok_importance:.3f} | {''.join(display_tokens)}")


def double_leaky_hard_sigmoid(x: torch.Tensor, alpha: float = 0.01) -> torch.Tensor:
    return torch.clamp(x, min=0, max=1) + alpha * (torch.clamp(x, max=0)) + alpha * (torch.clamp(x, min=1) - 1)


def plot_multiple_gate_mlps(model: ComponentModel, module_name: str, n: int = 10, seed: int = 42):
    """Plot response curves for n randomly selected gate MLPs from a module."""
    # Set random seed for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Get the gate MLP for the specified module
    gate_mlp = model.dual_modules[module_name].gate
    x_range = (-15, 15)

    # Create input values - shape (B=1, S=100, M) where S is used as the linspace dimension
    S = 100
    x_values_S = torch.linspace(x_range[0], x_range[1], S).to(model.device)
    # Expand to (1, S, M) shape - broadcast the same values across all M components
    x_values_1SM = x_values_S.unsqueeze(0).unsqueeze(-1).expand(1, S, gate_mlp.mlp_in_MH.shape[0])

    with torch.no_grad():
        # Get outputs for all components at once
        output_1SM = gate_mlp.forwards_BSM(x_values_1SM)
        output_SM = output_1SM[0]  # Remove batch dimension

    WIDTH = 4

    # Create subplots
    cols = min(WIDTH, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 2 * rows))

    axes = [axes] if n == 1 else axes.flatten() if rows > 1 else axes

    # Convert x values to numpy once
    x_np = x_values_S.cpu().numpy()

    # Plot each component
    for i, comp_idx in enumerate(range(n)):
        ax = axes[i] if n > 1 else axes[0]

        # Get output for this component
        output_S = output_SM[:, comp_idx]

        # Plot the response curve
        y_np = output_S.cpu().numpy()
        ax.plot(x_np, y_np, linewidth=2, label="Pre-sigmoid")

        # Apply sigmoid to visualize actual gate output
        y_sigmoid = double_leaky_hard_sigmoid(output_S).cpu().numpy()
        ax.plot(x_np, y_sigmoid, linewidth=2, linestyle="--", alpha=0.7, label="After (double leaky) sigmoid")

        ax.set_xlim(x_range)
        ax.set_ylim(-0.5, 1.5)
        ax.set_xlabel("Input (projection onto component)")
        ax.set_ylabel("Gate output")
        ax.set_title(f"Component {comp_idx}")
        ax.grid(True, alpha=0.3)
        ax.legend()

    # Hide unused subplots
    for i in range(n, len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()
    plt.suptitle(f"Gate MLP Response Curves - {module_name}", y=1.02)

    return fig


def get_ce_unrecovered(
    batch_BS, model: ComponentModel, normalize_acts: Callable[[dict[str, torch.Tensor]], dict[str, torch.Tensor]]
):
    logits_target_BSV, preact_cache = model.forward_base_with_preact_cache_BSV(batch_BS)
    pre_sigmoid_importance_BSM = model.get_importance_values_BSM(normalize_acts(preact_cache))
    masks_BSM = sample_masks_BSM(pre_sigmoid_importance_BSM)
    logits_masked_BSV = model.forward_with_all_components_masked_BSV(batch_BS, masks_BSM)
    logits_masked_by_layer_BSV = model.forward_with_components_lw_masked_BSV(batch_BS, masks_BSM)
    logits_unmasked_BSV = model.forward_with_all_components_unmasked_BSV(batch_BS)
    ce_unmasked = ce(logits_unmasked_BSV, batch_BS)

    ce_masked = ce(logits_masked_BSV, batch_BS)

    ce_masked_lw = 0
    for logits_masked_lw_BSV in logits_masked_by_layer_BSV.values():
        ce_masked_lw += ce(logits_masked_lw_BSV, batch_BS).item()
    ce_masked_lw /= len(logits_masked_by_layer_BSV)
    # =====

    # bounding baselines:
    ce_base = ce(logits_target_BSV, batch_BS)
    logits_zero_masked_BSV = model.forward_with_all_components_ablated_BSV(batch_BS)
    ce_zero_masked = ce(logits_zero_masked_BSV, batch_BS)
    # =====

    # Cross-entropy unrecovered
    ce_unrecovered_masked = (ce_masked - ce_base) / (ce_zero_masked - ce_base)
    ce_unrecovered_masked_lw = (ce_masked_lw - ce_base) / (ce_zero_masked - ce_base)
    ce_unrecovered_unmasked = (ce_unmasked - ce_base) / (ce_zero_masked - ce_base)

    return (
        ce_unrecovered_masked,
        ce_unrecovered_masked_lw,
        ce_unrecovered_unmasked,
    )


def eval_ce_unrecovered(
    model: ComponentModel,
    dataloader: untokenized_text_hf.UntokenisedTextHF,
    normalize_acts: Callable[[dict[str, torch.Tensor]], dict[str, torch.Tensor]],
):
    res = defaultdict(list)
    for _ in tqdm(range(10)):
        batch_BS = dataloader.get_token_batch(2).to(model.device)
        (
            ce_unrecovered_masked,
            ce_unrecovered_masked_lw,
            ce_unrecovered_unmasked,
        ) = get_ce_unrecovered(batch_BS, model, normalize_acts)
        res["ce_unrecovered_masked"].append(ce_unrecovered_masked.item())
        res["ce_unrecovered_masked_lw"].append(ce_unrecovered_masked_lw.item())
        res["ce_unrecovered_unmasked"].append(ce_unrecovered_unmasked.item())
        del (
            batch_BS,
            ce_unrecovered_masked,
            ce_unrecovered_masked_lw,
            ce_unrecovered_unmasked,
        )

    print(f"""Cross-entropy unrecovered:
              masked: {np.mean(res["ce_unrecovered_masked"])}
              masked_lw: {np.mean(res["ce_unrecovered_masked_lw"])}
              unmasked: {np.mean(res["ce_unrecovered_unmasked"])}""")


# %%
if __name__ == "__main__":

    checkpoint_paths = (CHECKPOINT_DIR / "zi4tsgo7" / "i7vudzqk").glob("checkpoint_step_*.pt")
    last_checkpoint_path = sorted(checkpoint_paths, key=lambda x: int(x.stem.split("_")[-1]))[-1]

    print(f"Using checkpoint: {last_checkpoint_path}")
    batch_size: int = 16
    seq_len: int = 1024

    # Load model config and checkpoint
    checkpoint = torch.load(last_checkpoint_path, map_location="cuda")
    config = checkpoint["config"]

    tmp = config["gate_hidden_dims"]
    # %%

    config["gate_hidden_dim"] = tmp[0]
    del config["gate_hidden_dims"]
    # %%

    cfg = TrainConfig(**config)
    model = get_model(cfg, torch.device("cuda"))

    # Load trained weights
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # %%

    # Create data iterator
    print('getting tokenizer')
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    print('getting dataloader')
    dataloader = get_dataloader(cfg.model_name, seq_len, rank=0, world_size=1, seed=42)

    # %%

    activation_scale = get_activation_scale(dataloader, batch_size, model, torch.device("cuda"), n_tokens=10000)

    # %%
    def normalize_acts(preact_cache: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {
            module_name: preact_cache_BSD / activation_scale[module_name]
            for module_name, preact_cache_BSD in preact_cache.items()
        }


#     batch_BS = dataloader.get_token_batch(2).to(model.device)
#     import gc
#     gc.collect()
#     torch.cuda.empty_cache()

#     # %%

#     logits_target_BSV, preact_cache, out_cache = model.forward_base_with_cache_BSV(batch_BS)
#     # # %%
#     # logits_target_BSV[0, :20].softmax(-1).amax(dim=-1)
#     # # %%
#     # pre = preact_cache['model.layers.14.mlp.up_proj']
#     # pre.shape, (pre.norm(dim=-1)[0, 30:40] / activation_scale['model.layers.14.mlp.up_proj']).mean()
#     # %%

#     # %%
#     pre_sigmoid_importance_BSM = model.get_importance_values_BSM(normalize_acts(preact_cache))


#     # %%
#     # B = batch, S = seq_len, M = num_components
#     component_density_M = pre_sigmoid_importance_BSM['model.layers.14.mlp.up_proj'].gt(0).float().mean(dim=(0, 1))
#     plt.hist(component_density_M.cpu().detach().numpy(), bins=200, log=True)
#     plt.title('component density')
#     plt.show()

#     # %%

#     psi_BSM = pre_sigmoid_importance_BSM['model.layers.14.mlp.up_proj']
#     # psi_BSM[psi_BSM > 0]
#     plt.hist(psi_BSM.gt(0).float().mean(dim=(0, 1)).cpu().detach().numpy(), bins=200, log=True)
#     plt.show()
#     # %%
#     # %%
#     # %% 
#     masks_BSM = sample_masks_BSM(pre_sigmoid_importance_BSM)
#     # %%
#     plt.show()
    
#     plt.hist(masks_BSM['model.layers.14.mlp.up_proj'].detach().flatten().cpu().numpy(), bins=200)
#     plt.show()

#     # %%

#     mask = masks_BSM['model.layers.14.mlp.up_proj']
#     random_mask = torch.rand_like(mask)


#     mod_in = preact_cache['model.layers.14.mlp.up_proj']
#     mod_out = out_cache['model.layers.14.mlp.up_proj']

#     mod_out_base = model.dual_modules['model.layers.14.mlp.up_proj'].original.forward(mod_in)
#     mod_out_masked = model.dual_modules['model.layers.14.mlp.up_proj'].component.forward(mod_in, mask)
#     mod_out_zeroed = model.dual_modules['model.layers.14.mlp.up_proj'].component.forward(mod_in, mask * 0)
#     mod_out_random = model.dual_modules['model.layers.14.mlp.up_proj'].component.forward(mod_in, random_mask)
#     mod_out_unmasked = model.dual_modules['model.layers.14.mlp.up_proj'].component.forward_unmasked(mod_in)

#     def mse(a, b):
#         return (a - b).pow(2).mean()
    
#     # print(f"{mse(mod_out_base, mod_out)=}")
#     print("local mse")
#     print(f"zeroed:   {mse(mod_out_zeroed, mod_out):.4f}")
#     print(f"random:   {mse(mod_out_random, mod_out):.4f}")
#     print(f"masked:   {mse(mod_out_masked, mod_out):.4f}")
#     print(f"unmasked: {mse(mod_out_unmasked, mod_out):.4f}")


#     # %%

#     print('asdf')
#     logits_masked_BSV = model.forward_with_all_components_masked_BSV(batch_BS, masks_BSM)
#     print('asdfasdf')
#     # %%
#     logits_masked_by_layer_BSV = model.forward_with_components_lw_masked_BSV(batch_BS, masks_BSM)
#     logits_unmasked_BSV = model.forward_with_all_components_unmasked_BSV(batch_BS)
#     ce_unmasked = ce(logits_unmasked_BSV, batch_BS)

#     ce_masked = ce(logits_masked_BSV, batch_BS)

#     ce_masked_lw = 0
#     for logits_masked_lw_BSV in logits_masked_by_layer_BSV.values():
#         ce_masked_lw += ce(logits_masked_lw_BSV, batch_BS).item()
#     ce_masked_lw /= len(logits_masked_by_layer_BSV)
#     # =====

#     # bounding baselines:
#     ce_base = ce(logits_target_BSV, batch_BS)
#     logits_zero_masked_BSV = model.forward_with_all_components_ablated_BSV(batch_BS)
#     ce_zero_masked = ce(logits_zero_masked_BSV, batch_BS)
#     # =====

#     # Cross-entropy unrecovered
#     ce_unrecovered_masked = (ce_masked - ce_base) / (ce_zero_masked - ce_base)
#     ce_unrecovered_masked_lw = (ce_masked_lw - ce_base) / (ce_zero_masked - ce_base)
#     ce_unrecovered_unmasked = (ce_unmasked - ce_base) / (ce_zero_masked - ce_base)

#         return (
#             ce_unrecovered_masked,
#             ce_unrecovered_masked_lw,
#             ce_unrecovered_unmasked,
#         )

#     # %%

#     import gc
#     from collections import defaultdict

#     res = defaultdict(list)
#     for _ in tqdm(range(1)):
#         batch_BS = dataloader.get_token_batch(2).to(model.device)
#         (
#             ce_unrecovered_masked,
#             ce_unrecovered_masked_lw,
#             ce_unrecovered_unmasked,
#         ) = get_ce_unrecovered(batch_BS)
#         res["ce_unrecovered_masked"].append(ce_unrecovered_masked.item())
#         res["ce_unrecovered_masked_lw"].append(ce_unrecovered_masked_lw.item())
#         res["ce_unrecovered_unmasked"].append(ce_unrecovered_unmasked.item())
#         del (
#             batch_BS,
#             ce_unrecovered_masked,
#             ce_unrecovered_masked_lw,
#             ce_unrecovered_unmasked,
#         )
#         gc.collect()
#         torch.cuda.empty_cache()

#     print(f"""
# Cross-entropy unrecovered:
# masked: {np.mean(res["ce_unrecovered_masked"])}
# masked_lw: {np.mean(res["ce_unrecovered_masked_lw"])}
# unmasked: {np.mean(res["ce_unrecovered_unmasked"])}
# """)
# =======
    # %%
# >>>>>>> oli/parameter-decomposition

#     # %%
#     gc.collect()
#     torch.cuda.empty_cache()
#     # %%

#     examiner = ComponentExaminer(model, dataloader, batch_size=12)

# <<<<<<< oli/investigate
#     total_batches: int = 6
#     importance_threshold = 0.01
#     summaries = examiner.gather_component_summaries(total_batches, importance_threshold)
# =======
#     total_batches: int = 10  # Increase to get more diverse examples
#     importance_threshold = 0.2
#     summaries = examiner.gather_component_summaries(
#         total_batches,
#         importance_threshold,
#         separation_threshold_tokens=10,
#     )
# >>>>>>> oli/parameter-decomposition

#     # Display results
#     print(f"\nFound {len(summaries)} components with examples")
#     offset = 0
# <<<<<<< oli/investigate
#     # %%

#     # plot the density of the components
#     plt.hist([len(s.selected_examples) for s in summaries], bins=100, log=True)
#     plt.show()

#     # %%

#     offset += 10
# =======

#     # %%
# >>>>>>> oli/parameter-decomposition
#     for i in range(offset, offset + 10):
#         display_component_examples(summaries[i], tokenizer, topk=20)
#     offset += 10

#     # # %%
#     # assert len(cfg.target_module_names) == 1
#     # mod = cfg.target_module_names[0]

# <<<<<<< oli/investigate
#     importances = []
#     for i in range(20):
#         batch_BS = dataloader.get_token_batch(2).to(model.device)
#         _, preact_cache, _ = model.forward_base_with_preact_cache_BSV(batch_BS)
#         pre_sigmoid_importance_BSM = model.get_importance_values_BSM(normalize_acts(preact_cache))
#         importances.append(pre_sigmoid_importance_BSM['model.layers.14.mlp.down_proj'])
#         #%%
#     importances_t = torch.stack(importances)
#     importances_t.shape
#     # %%
#     importances_t = importances_t.clamp(min=0)
#     # %%
#     importances_t.gt(0).sum() / importances_t.numel()


# # if __name__ == "__main__":
#     # import fire
#     # fire.Fire(run_interpretability)
#     # run_interpretability()

# # %%
# =======
#     # if cfg.gate_type == "mlp":
#     #     plot_multiple_gate_mlps(model, mod, n=36)
#     #     plt.show()
# >>>>>>> oli/parameter-decomposition

