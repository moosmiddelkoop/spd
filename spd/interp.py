# %%
from transformers import AutoModelForCausalLM

import yaml
import html
import re
import gc
from collections import defaultdict
from collections.abc import Callable, Generator, Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer
from transformers.tokenization_utils import PreTrainedTokenizer

from spd.configs import Config, LMTaskConfig
from spd.data import DatasetConfig, create_data_loader
from spd.models.component_model import ComponentModel
from spd.models.component_utils import calc_causal_importances
from spd.models.sigmoids import SigmoidTypes

# from research.cajal.tokens.sources import untokenized_text_hf
# from research.experiments.parameter_decomposition.constants import CHECKPOINT_DIR
# ComponentModel,
# TrainConfig,
# ce,
# get_activation_scale,
# get_dataloader,
# get_model,
# sample_masks_BSM,
# upper_leaky_hard_sigmoid,


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
    def __init__(
        self,
        model: ComponentModel,
        tokens: Iterator[torch.Tensor],
        sigmoid_type: SigmoidTypes,
        device: torch.device,
    ):
        self.model = model
        self.tokens = tokens

        self.module_names = [m.replace("-", ".") for m in model.gates.keys()]
        print("self.module_names", self.module_names)

        self.Vs = {
            name.replace("-", "."): component.V for name, component in model.components.items()
        }
        print("self.Vs", list(self.Vs.keys()))

        self.gates = {name.replace("-", "."): gate for name, gate in self.model.gates.items()}
        print("self.gates", list(self.gates.keys()))

        self.sigmoid_type = sigmoid_type
        self.device = device

    def gather_component_summaries_fast(
        self,
        total_batches: int,
        importance_threshold: float,
    ) -> list[ComponentSummary]:
        component_examples: dict[tuple[str, int], list[ComponentExample]] = defaultdict(list)
        component_firing_counts: dict[tuple[str, int], int] = defaultdict(int)

        tokens_seen = 0

        pbar = tqdm(range(total_batches), desc="Gathering component summaries (fast)")

        for _ in pbar:
            tokens_BS = next(self.tokens).to(self.device)
            B, S = tokens_BS.shape
            _, preact_cache_BSD = self.model.forward_with_pre_forward_cache_hooks(
                tokens_BS, module_names=self.module_names
            )

            importance_BSM_by_module, _ = calc_causal_importances(
                pre_weight_acts=preact_cache_BSD,
                Vs=self.Vs,
                gates=self.gates,
                sigmoid_type=self.sigmoid_type,
                detach_inputs=False,
            )

            for module_name, imp_BSM in importance_BSM_by_module.items():
                mask = imp_BSM > importance_threshold  # shape (B, S, C)
                if not mask.any():
                    continue

                b_idx, s_idx, c_idx = torch.where(mask)
                imp_vals = imp_BSM[b_idx, s_idx, c_idx]

                # -- Accumulate firing counts (every hit counts, regardless of example quota)
                comps_unique, comps_counts = torch.unique(c_idx, return_counts=True)
                for comp, cnt in zip(comps_unique.tolist(), comps_counts.tolist()):
                    component_firing_counts[(module_name, comp)] += int(cnt)

                # -- Sort hits by descending importance so the top examples are kept first
                order = torch.argsort(imp_vals, descending=True)
                b_idx, s_idx, c_idx, imp_vals = (
                    b_idx[order],
                    s_idx[order],
                    c_idx[order],
                    imp_vals[order],
                )

                for b, s, c, imp in zip(
                    b_idx.tolist(),
                    s_idx.tolist(),
                    c_idx.tolist(),
                    imp_vals.tolist(),
                    strict=True,
                ):
                    key = (module_name, c)
                    ex_list = component_examples[key]

                    example = ComponentExample(
                        tokens_S=tokens_BS[b, : s + 1].detach().cpu(),
                        last_tok_importance=float(imp),
                    )
                    ex_list.append(example)

            tokens_seen += B * S
            pbar.set_postfix(n_components=len(component_examples))

        summaries: list[ComponentSummary] = []
        for (module_name, comp_idx), examples in component_examples.items():
            # Ensure deterministic ordering for repeatability
            examples.sort(key=lambda x: x.last_tok_importance, reverse=True)
            summaries.append(
                ComponentSummary(
                    module_name=module_name,
                    component_idx=comp_idx,
                    selected_examples=examples,
                    density=component_firing_counts[(module_name, comp_idx)] / tokens_seen,
                )
            )

        return summaries

        # @torch.no_grad()
        # def gather_component_summaries(
        #     self,
        #     total_batches: int,
        #     importance_threshold: float = 0.1,
        # ) -> list[ComponentSummary]:
        #     # Optimized version with GPU acceleration and vectorization
        #     examples_by_component = defaultdict[tuple[str, int], list[ComponentExample]](list)
        #     num_firings_by_component = defaultdict[tuple[str, int], int](int)

        #     # Pre-allocate buffers for better memory management
        #     last_position_by_component = {}

        #     tokens_seen = 0
        #     pbar = tqdm(range(total_batches), desc="Gathering component examples (optimized)")

        #     for _ in pbar:
        #         tokens_BS = next(self.tokens).to(self.device)
        #         B, S = tokens_BS.shape

        #         # Get importance values
        #         _, preact_cache_BSD = self.model.forward_with_pre_forward_cache_hooks(
        #             tokens_BS, module_names=self.module_names,
        #         )

        #         ci_BSM, _ = calc_causal_importances(
        #             pre_weight_acts=preact_cache_BSD,
        #             Vs=self.Vs,
        #             gates=self.gates,
        #             sigmoid_type=self.sigmoid_type,
        #             detach_inputs=False,
        #         )

        #         # Process each module
        #         for module_name, ci_BSM in ci_BSM.items():
        #             # assert ci_BSM.shape == (B, S, self.model.m)

        #             # Vectorized approach: find all positions exceeding threshold at once
        #             high_importance_mask_BSM = ci_BSM > importance_threshold

        #             # Get indices using torch.where - much faster than nested loops
        #             batch_indices, pos_indices, component_indices = torch.where(
        #                 high_importance_mask_BSM
        #             )

        #             # Count firings efficiently
        #             component_counts = torch.bincount(component_indices, minlength=self.model.C)
        #             for comp_idx, count in enumerate(component_counts):
        #                 if count > 0:
        #                     key = (module_name, comp_idx)
        #                     num_firings_by_component[key] += int(count.item())

        #             # Process examples in batches to minimize Python overhead
        #             if len(batch_indices) > 0:
        #                 # Sort by batch, then position for cache-friendly access
        #                 sort_indices = torch.argsort(batch_indices * S + pos_indices)
        #                 batch_indices = batch_indices[sort_indices]
        #                 pos_indices = pos_indices[sort_indices]
        #                 component_indices = component_indices[sort_indices]

        #                 # Process each batch
        #                 for b_idx in range(B):
        #                     batch_mask = batch_indices == b_idx
        #                     if not batch_mask.any():
        #                         continue

        #                     batch_positions = pos_indices[batch_mask]
        #                     batch_components = component_indices[batch_mask]
        #                     tokens_S = tokens_BS[b_idx]

        #                     # Create position-component pairs
        #                     for i in range(len(batch_positions)):
        #                         pos = int(batch_positions[i].item())
        #                         comp_idx = int(batch_components[i].item())
        #                         component_key = (module_name, comp_idx)

        #                         importance = ci_BSM[b_idx, pos, comp_idx].item()

        #                         example = ComponentExample(
        #                             tokens_S=tokens_S[: pos + 1],
        #                             last_tok_importance=importance,
        #                         )
        #                         examples_by_component[component_key].append(example)
        #                         last_position_by_component[component_key] = pos

        #             pbar.set_postfix(n_components_with_examples=len(examples_by_component))

        #         tokens_seen += B * S

        # Create summaries
        summaries = []
        for (module_name, component_idx), examples in examples_by_component.items():
            summaries.append(
                ComponentSummary(
                    module_name=module_name,
                    component_idx=component_idx,
                    selected_examples=sorted(
                        examples, key=lambda x: x.last_tok_importance, reverse=True
                    ),
                    density=num_firings_by_component[(module_name, component_idx)] / tokens_seen,
                )
            )

        return summaries

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
            tokens_BS = next(self.tokens).to(self.model.device)
            B, S = tokens_BS.shape

            # Get importance values
            _, preact_cache_BSD = self.model.forward_with_pre_forward_cache_hooks(
                tokens_BS, module_names=self.module_names
            )
            importance_BSM, _ = calc_causal_importances(
                pre_weight_acts=preact_cache_BSD,
                Vs=self.Vs,
                gates=self.gates,
                sigmoid_type=self.sigmoid_type,
                detach_inputs=False,
            )

            # Process each module
            for module_name, importance_vals_BSM in importance_BSM.items():
                assert importance_vals_BSM.shape == (B, S, self.model.C)

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
                        active_components = (importance_SM[pos] > importance_threshold).nonzero(
                            as_tuple=True
                        )[0]

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
                    selected_examples=sorted(
                        examples, key=lambda x: x.last_tok_importance, reverse=True
                    ),
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
        component_examples = defaultdict(
            list
        )  # (module_name, comp_idx) -> list of (tokens, importance)
        component_firing_counts = defaultdict(int)

        tokens_seen = 0
        pbar = tqdm(range(total_batches), desc="Gathering examples (ultra-fast)")

        # Process in larger batches if possible
        effective_batch_size = min(self.batch_size * 4, 32)  # Process more at once

        for batch_idx in pbar:
            # Get larger batch for better GPU utilization
            tokens_BS = next(self.tokens).to(self.device)
            B, S = tokens_BS.shape

            # Get importance values
            _, preact_cache_BSD = self.model.forward_with_pre_forward_cache_hooks(
                tokens_BS, module_names=self.module_names
            )

            importance_BSM, _ = calc_causal_importances(
                pre_weight_acts=preact_cache_BSD,
                Vs=self.Vs,
                gates=self.gates,
                sigmoid_type=self.sigmoid_type,
                detach_inputs=False,
            )

            # Process each module
            for module_name, importance_vals_BSM in importance_BSM.items():
                # Find all high importance activations at once
                high_mask = importance_vals_BSM > importance_threshold  # TODO: use threshold

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
                        last_pos = component_last_positions.get(
                            key, -separation_threshold_tokens - 1
                        )

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


# +=======================

# Inline CSS for the highlight – darker green for sufficient contrast
# _HIGHLIGHT_OPEN = '<span style="color:#1f7a1f;font-weight:bold;">'
_HIGHLIGHT_OPEN = '<span style="color:#fff;background-color:#1f7a1f;">'
_HIGHLIGHT_CLOSE = "</span>"

# Visible‑whitespace glyphs – identical to the OP’s choice
VISIBLE_SPACE = "·"  # U+00B7
VISIBLE_NL = "↵"  # U+21B5
VISIBLE_TAB = "→"  # U+2192


def _make_visible(text: str) -> str:
    """Replace control/whitespace characters with visible glyphs *only*."""
    return (
        text.replace("\t", VISIBLE_TAB)
        .replace("\r", VISIBLE_NL)
        .replace("\n", VISIBLE_NL)
        .replace(" ", VISIBLE_SPACE)
    )


def _apply_highlight(escaped_text: str, escaped_target: str) -> str:
    """Wrap **the last occurrence** of *escaped_target* in the highlight span.

    Both parameters **MUST** already be HTML-escaped.
    """
    pos = escaped_text.rfind(escaped_target)
    if pos == -1:
        # Fallback – shouldn’t normally happen, but fail gracefully
        return escaped_text
    return (
        escaped_text[:pos]
        + _HIGHLIGHT_OPEN
        + escaped_target
        + _HIGHLIGHT_CLOSE
        + escaped_text[pos + len(escaped_target) :]
    )


def render_component_summary_html(
    summary: "ComponentSummary",
    tokenizer: PreTrainedTokenizer,
    *,
    topk: int = 10,
    context_tokens: int = 20,
) -> str:
    """Return a fully escaped HTML `<section>` for *one* `ComponentSummary`."""

    rows: list[str] = []

    for ex in summary.selected_examples[:topk]:
        # --- reconstruct context string (right‑truncated) ---
        ids_slice = ex.tokens_S[-context_tokens:]
        ids_list = ids_slice.tolist()
        decoded = tokenizer.decode(ids_list, skip_special_tokens=True)

        # Last token (decoded) – needed later for the highlight search
        last_tok_str = tokenizer.decode([ids_list[-1]], skip_special_tokens=True)

        # Make control chars visible *before* escaping so glyphs are preserved
        visible_decoded = _make_visible(decoded)
        visible_last = _make_visible(last_tok_str)

        # Escape both for safe HTML insertion
        escaped_decoded = html.escape(visible_decoded, quote=False)
        escaped_last = html.escape(visible_last, quote=False)

        # Apply highlight (wrap last occurrence in <span>)
        highlighted_html = _apply_highlight(escaped_decoded, escaped_last)

        rows.append(
            f'<tr><td>{ex.last_tok_importance:.3f}</td><td style="white-space:nowrap;">{highlighted_html}</td></tr>'
        )

    return (
        f"<section>\n"
        f"  <h2>Module: {html.escape(summary.module_name)} / Component {summary.component_idx}</h2>\n"
        f"  <p>Density: {summary.density:.4%} &nbsp;|&nbsp; Total examples: {len(summary.selected_examples)}</p>\n"
        f"  <table>\n    <thead><tr><th>Importance</th><th>Context</th></tr></thead>\n    <tbody>\n      {'\n      '.join(rows)}\n    </tbody>\n  </table>\n"
        f"</section>\n"
    )


def write_component_summaries_html(
    summaries: Iterable[ComponentSummary],
    tokenizer: PreTrainedTokenizer,
    *,
    file_path: str | Path = "component_summaries.html",
    topk: int = 10,
    context_tokens: int = 20,
    page_title: str = "Component Example Visualisation",
) -> Path:
    """Compile *all* summaries into a single styled HTML file and return the path."""

    file_path = Path(file_path)

    sections = [
        render_component_summary_html(s, tokenizer, topk=topk, context_tokens=context_tokens)
        for s in summaries
    ]

    css = """
    body { font-family: system-ui, sans-serif; margin: 2rem; }
    h1   { margin-bottom: 0.5rem; }
    h2   { margin: 2rem 0 0.25rem; font-size: 1.1rem; }

    /* ─── Table styling ─────────────────────────────────────────────── */
    table {
        border-collapse: separate;
        border-spacing: 0;
        width: 100%;
        font-size: 0.9rem;
        border: 1px solid #e0e0e0;
        border-radius: 6px;
        overflow: hidden;      /* keep radius on overflow x */
    }

    thead th {
        background: #f3f4f6;
        font-weight: 600;
        border-bottom: 1px solid #e0e0e0;
        position: sticky;
        top: 0;                /* sticky header */
    }

    th, td {
        padding: 6px 8px;
        text-align: left;
        vertical-align: top;
    }

    /* allow the Context cell to wrap nicely */
    td:nth-child(3) { word-wrap: break-word; white-space: pre-wrap; }

    span { white-space: pre-wrap; }  /* keep highlighted token wrapping */
    """

    html_doc = (
        '<!DOCTYPE html>\n<html lang="en">\n<head>\n'
        f'  <meta charset="utf-8">\n  <title>{html.escape(page_title)}</title>\n'
        f"  <style>{css}</style>\n</head>\n<body>\n"
        f"  <h1>{html.escape(page_title)}</h1>\n  {''.join(sections)}\n</body>\n</html>\n"
    )

    file_path.write_text(html_doc, encoding="utf-8")
    return file_path

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available")

DEVICE = torch.device("cuda")


def write_summary_for_run(
    experiment_out_dir: Path,
    checkpoint_name: str,
    total_batches: int = 1_000,
    importance_threshold: float = 0.9,
    n_context_tokens: int = 20,
    topk_examples_to_render: int = 30,
):
    last_checkpoint_path = experiment_out_dir / checkpoint_name
    config_path = experiment_out_dir / "final_config.yaml"

    with open(config_path) as f:
        cfg = Config(**yaml.safe_load(f))

    llm = AutoModelForCausalLM.from_pretrained(
        cfg.pretrained_model_name_hf, torch_dtype=torch.bfloat16
    )

    model = ComponentModel(
        llm,
        cfg.target_module_patterns,
        cfg.C,
        cfg.n_ci_mlp_neurons,
        cfg.gate_init_central,
        cfg.pretrained_model_output_attr,
        dtype=torch.bfloat16,
    )

    cp = torch.load(last_checkpoint_path, map_location="cuda")

    model.load_state_dict(cp)
    model.eval()
    model.to(DEVICE)

    print("getting tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(cfg.pretrained_model_name_hf)

    tc = cfg.task_config
    assert isinstance(tc, LMTaskConfig)

    print("getting dataloader")

    train_data_config = DatasetConfig(
        name=tc.dataset_name,
        hf_tokenizer_path=cfg.pretrained_model_name_hf,
        split=tc.train_data_split,
        n_ctx=tc.max_seq_len,
        is_tokenized=False,
        streaming=True,
        column_name=tc.column_name,
    )

    train_loader, _ = create_data_loader(
        dataset_config=train_data_config,
        batch_size=cfg.microbatch_size(),
        buffer_size=tc.buffer_size,
        global_seed=cfg.seed,
        ddp_rank=0,
        ddp_world_size=1,
    )

    print("gathering component summaries")

    component_examiner = ComponentExaminer(
        model=model,
        tokens=(x["input_ids"] for x in train_loader),
        sigmoid_type=cfg.sigmoid_type,
        device=DEVICE,
    )

    component_summaries = component_examiner.gather_component_summaries_fast(
        total_batches=total_batches,
        importance_threshold=importance_threshold,
    )

    print(f"gathered {len(component_summaries)} component summaries")
    n_examples = [len(s.selected_examples) for s in component_summaries]
    print(
        f"num examples per component: mean {np.mean(n_examples):.2f} std {np.std(n_examples):.2f}"
    )

    report_dir = experiment_out_dir / "component_summary"
    print(f"writing report to {report_dir}")

    write_component_summaries_html(
        component_summaries,
        tokenizer,
        file_path=report_dir / "component_summaries.html",
        topk=topk_examples_to_render,
        context_tokens=n_context_tokens,
    )
