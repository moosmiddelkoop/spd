# %%
import html
from collections import defaultdict
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from time import time
from typing import cast

import numpy as np
import torch
import yaml
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast

from spd.configs import Config, LMTaskConfig
from spd.constants import CI_ALIVE_THRESHOLD
from spd.data import DatasetConfig, create_data_loader
from spd.models.component_model import ComponentModel
from spd.models.component_utils import calc_causal_importances
from spd.models.sigmoids import SigmoidTypes


@dataclass
class ComponentExample:
    tokens_S: torch.Tensor
    token_importances_S: torch.Tensor


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

        self.module_names = [m.replace("-", ".") for m in model.gates]

        self.Vs = {
            name.replace("-", "."): component.V for name, component in model.components.items()
        }

        self.gates = {name.replace("-", "."): gate for name, gate in model.gates.items()}

        self.sigmoid_type = sigmoid_type
        self.device = device

    def gather_component_summaries(
        self,
        # total_batches: int,
        budget: timedelta,
    ) -> dict[str, list[ComponentSummary]]:
        component_examples: dict[str, dict[int, list[ComponentExample]]] = defaultdict(
            lambda: defaultdict(list)
        )
        component_firing_counts: dict[str, dict[int, int]] = defaultdict(lambda: defaultdict(int))

        pbar = tqdm(unit="batches", desc="Gathering component summaries")
        start_time = time()
        tokens_seen = 0
        while time() - start_time < budget.total_seconds():
            tokens_BS = next(self.tokens).to(self.device)
            B, S = tokens_BS.shape
            _, preact_cache_BSD = self.model.forward_with_pre_forward_cache_hooks(
                tokens_BS, module_names=self.module_names
            )

            importance_BSM_by_module, _, _ = calc_causal_importances(
                pre_weight_acts=preact_cache_BSD,
                Vs=self.Vs,
                gates=self.gates,  # pyright: ignore[reportArgumentType]
                sigmoid_type=self.sigmoid_type,  # pyright: ignore[reportArgumentType]
                detach_inputs=False,
            )

            for module_name, imp_BSM in importance_BSM_by_module.items():
                mask = imp_BSM > CI_ALIVE_THRESHOLD  # shape (B, S, C)
                if not mask.any():
                    continue

                b_idx, s_idx, c_idx = torch.where(mask)

                comps_unique, comps_counts = torch.unique(c_idx, return_counts=True)
                for comp, cnt in zip(comps_unique.tolist(), comps_counts.tolist(), strict=True):
                    component_firing_counts[module_name][comp] += int(cnt)

                for b, s, c in zip(
                    b_idx.tolist(),
                    s_idx.tolist(),
                    c_idx.tolist(),
                    strict=True,
                ):
                    trail = 3
                    example = ComponentExample(
                        tokens_S=tokens_BS[b, : s + 1 + trail].detach().cpu(),
                        token_importances_S=imp_BSM[b, : s + 1 + trail, c].detach().cpu(),
                    )
                    component_examples[module_name][c].append(example)

            tokens_seen += B * S
            pbar.set_postfix(n_tokens=tokens_seen)
            pbar.update()

        summaries: dict[str, list[ComponentSummary]] = defaultdict(list)

        # sort for repeatability
        for module_name, components in component_examples.items():
            for comp_idx, examples in components.items():
                examples.sort(key=lambda x: x.token_importances_S[-1].item(), reverse=True)
                summaries[module_name].append(
                    ComponentSummary(
                        module_name=module_name,
                        component_idx=comp_idx,
                        selected_examples=examples,
                        density=component_firing_counts[module_name][comp_idx] / tokens_seen,
                    )
                )
            summaries[module_name].sort(key=lambda x: x.density, reverse=True)

        return summaries


def interpolate(bottom: float, top: float, x: float) -> float:
    """Interpolate between a and b using x, which is in [0, 1]."""
    return bottom + (top - bottom) * x


WHITE = (255, 255, 255)
LIGHT_GREEN = (160, 210, 160)


def _get_highlight_color(
    importance: float,
    color_upper: tuple[int, int, int] = LIGHT_GREEN,
    color_lower: tuple[int, int, int] = WHITE,
) -> str:
    """Get highlight color based on importance value."""
    importance_norm = min(max(importance, 0), 1)  # Clamp to [0, 1]
    r = int(interpolate(color_lower[0], color_upper[0], importance_norm))
    g = int(interpolate(color_lower[1], color_upper[1], importance_norm))
    b = int(interpolate(color_lower[2], color_upper[2], importance_norm))
    return f"rgba({r}, {g}, {b})"


def _apply_multi_token_highlights(
    token_ids: list[int],
    token_importances: list[float],
    tokenizer: PreTrainedTokenizerFast,
) -> str:
    assert len(token_ids) == len(token_importances), "length mismatch"
    token_strings = cast(list[str], tokenizer.convert_ids_to_tokens(token_ids))
    for i, tok in enumerate(token_strings):
        if tok.startswith("##"):
            token_strings[i] = tok[2:]
        else:
            token_strings[i] = " " + tok

    escaped_token_strings = [html.escape(ts, quote=False) for ts in token_strings]
    parts = []
    for imp, escaped_token_string in zip(token_importances, escaped_token_strings, strict=True):
        if imp > 0:
            bg_color = _get_highlight_color(imp)
            parts.append(
                f'<span style="background-color:{bg_color};" title="Importance: {imp:.3f}">{escaped_token_string}</span>'
            )
        else:
            parts.append(escaped_token_string)

    return "".join(parts)


def render_component_summary_html(
    summary: ComponentSummary,
    render_examples: Callable[[ComponentSummary], str],
) -> str:
    """Return a fully escaped HTML `<section>` for *one* `ComponentSummary`."""

    return (
        "<br/>"
        "<br/>"
        f"<section>\n"
        f"  <h2>Module: {html.escape(summary.module_name)} / "
        f"Component {summary.component_idx}</h2>\n"
        f"  <p>Density: {summary.density:.4%} &nbsp;|&nbsp; "
        f"Total examples: {len(summary.selected_examples)}</p>\n"
        f"{render_examples(summary)}"
    )


def format_component_summaries_html(
    summaries: list[ComponentSummary],
    render_examples: Callable[[ComponentSummary], str],
    page_title: str = "Component Example Visualisation",
) -> str:
    """Compile *all* summaries into a single styled HTML file and return the path."""

    sections = [render_component_summary_html(s, render_examples) for s in summaries[:100]]

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

    span { 
        white-space: pre-wrap;  /* keep highlighted token wrapping */
    }
    
    /* Custom instant tooltip */
    span[title] {
        position: relative;
    }
    
    span[title]:hover::after {
        content: attr(title);
        position: absolute;
        bottom: 100%;
        left: 50%;
        transform: translateX(-50%);
        background-color: #333;
        color: white;
        padding: 4px 8px;
        border-radius: 4px;
        font-size: 0.85em;
        white-space: nowrap;
        z-index: 1000;
        pointer-events: none;
        margin-bottom: 4px;
    }
    
    span[title]:hover::before {
        content: "";
        position: absolute;
        bottom: 100%;
        left: 50%;
        transform: translateX(-50%);
        border: 5px solid transparent;
        border-top-color: #333;
        z-index: 1000;
        pointer-events: none;
        margin-bottom: -1px;
    }
    """

    html_doc = (
        '<!DOCTYPE html>\n<html lang="en">\n<head>\n'
        f'  <meta charset="utf-8">\n  <title>{html.escape(page_title)}</title>\n'
        f"  <style>{css}</style>\n</head>\n<body>\n"
        f"  <h1>{html.escape(page_title)}</h1>\n  {''.join(sections)}\n</body>\n</html>\n"
    )

    return html_doc


def get_example_text(
    ex: ComponentExample,
    tokenizer: PreTrainedTokenizerFast,
    context_tokens: int = 20,
) -> str:
    ids_list = ex.tokens_S[-context_tokens:].tolist()
    importance_list = ex.token_importances_S[-context_tokens:].tolist()
    return _apply_multi_token_highlights(ids_list, importance_list, tokenizer)


def render_examples(
    summary: ComponentSummary,
    tokenizer: PreTrainedTokenizerFast,
    context_tokens: int = 20,
) -> str:
    N = 20
    top_examples = summary.selected_examples[:N]
    top_percentile_section = [
        get_example_text(ex, tokenizer, context_tokens) for ex in top_examples
    ]

    bottom_examples = summary.selected_examples[max(N, len(summary.selected_examples) - N) :]
    bottom_percentile_section = [
        get_example_text(ex, tokenizer, context_tokens) for ex in bottom_examples
    ]

    def row(ex: str) -> str:
        return f"<tr><td style='white-space:nowrap;'>{ex}</td></tr>"

    def section(title: str, examples: list[str]) -> str:
        return f"""
        <section>
            <p>{title}</p>
            <table>
                <thead>
                    <tr>
                        <th>Context</th>
                    </tr>
                </thead>
                <tbody>
                    {"\n".join(row(ex) for ex in examples)}
                </tbody>
            </table>
        </section>
        """

    top_section = section(f"Top {N} examples", top_percentile_section)
    bottom_section = section(
        f"Bottom {len(bottom_percentile_section)} (harvested) examples",
        bottom_percentile_section,
    )
    return top_section + bottom_section


def format_layer_component_summaries_html(
    component_summaries: list[ComponentSummary],
    tokenizer: PreTrainedTokenizerFast,
    context_tokens: int = 20,
) -> str:
    return format_component_summaries_html(
        component_summaries,
        lambda summary: render_examples(summary, tokenizer, context_tokens),
    )


def collect_importance_histograms(
    model: ComponentModel,
    data_loader: Iterator[dict[str, torch.Tensor]],
    budget: timedelta,
    device: torch.device,
    sigmoid_type: SigmoidTypes,
) -> dict[str, dict[int, list[float]]]:
    """Collect importance values for each component to create histograms."""
    module_names = list(model.gates.keys())
    module_names = [m.replace("-", ".") for m in module_names]

    Vs = {name.replace("-", "."): component.V for name, component in model.components.items()}
    gates = {name.replace("-", "."): gate for name, gate in model.gates.items()}

    # Store importance values for each module and component
    importance_values: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))

    pbar = tqdm(unit="batches", desc="Collecting importance values for histograms")
    start_time = time()
    tokens_seen = 0

    while time() - start_time < budget.total_seconds():
        try:
            tokens_BS = next(data_loader)["input_ids"].to(device)
        except StopIteration:
            break

        B, S = tokens_BS.shape

        # Get activations
        _, preact_cache_BSD = model.forward_with_pre_forward_cache_hooks(
            tokens_BS, module_names=module_names
        )

        # Calculate importance values
        importance_BSM_by_module, _, _ = calc_causal_importances(
            pre_weight_acts=preact_cache_BSD,
            Vs=Vs,
            gates=gates,
            sigmoid_type=sigmoid_type,
            detach_inputs=False,
        )

        # Collect all importance values
        for module_name, imp_BSM in importance_BSM_by_module.items():
            # imp_BSM shape: (B, S, C)
            for c in range(imp_BSM.shape[2]):
                # Get all importance values for this component
                comp_values = imp_BSM[:, :, c].flatten().detach().cpu().tolist()
                # Filter out very small values to reduce memory
                comp_values = [v for v in comp_values if v > 1e-6]
                importance_values[module_name][c].extend(comp_values)

        tokens_seen += B * S
        pbar.set_postfix(n_tokens=tokens_seen)
        pbar.update()

    pbar.close()
    return importance_values


def generate_histogram_plots(
    importance_values: dict[str, dict[int, list[float]]],
    output_dir: Path,
) -> None:
    """Generate histogram plots for each component using matplotlib."""
    import matplotlib.pyplot as plt

    # Create one figure per module
    for module_name, components in importance_values.items():
        if not components:
            continue

        # Count non-empty components
        non_empty_components = [(idx, vals) for idx, vals in sorted(components.items()) if vals]
        non_empty_components = non_empty_components[:40]
        if not non_empty_components:
            continue

        n_components = len(non_empty_components)
        n_cols = min(5, n_components)
        n_rows = (n_components + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
        axes = [axes] if n_components == 1 else axes.flatten()

        for i, (comp_idx, values) in tqdm(enumerate(non_empty_components), desc="Plotting histograms"):
            ax = axes[i]
            values_array = np.array(values)

            # Plot histogram
            ax.hist(
                values_array,
                bins=100,
                alpha=0.7,
                color="steelblue",
                edgecolor="black",
                linewidth=0.5,
            )
            # ax.set_yscale("log")
            ax.set_xlabel("Importance Value")
            ax.set_ylabel("Count")
            ax.set_xlim(0, 1)
            ax.set_title(
                f"Component {comp_idx}\nn={len(values):,}, mean={np.mean(values_array):.2e}"
            )
            ax.grid(True, alpha=0.3)

        # Hide unused subplots
        for i in range(len(non_empty_components), len(axes)):
            axes[i].set_visible(False)

        fig.suptitle(f"{module_name} - Importance Value Histograms", fontsize=16)
        fig.tight_layout()

        # Save figure
        output_path = output_dir / f"histograms_{module_name.replace('.', '_')}.png"
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved histogram plot to {output_path}")


def main_histograms(
    experiment_out_dir: Path,
    checkpoint_name: str,
    budget: timedelta,
):
    """Generate histograms of component importance values."""
    last_checkpoint_path = experiment_out_dir / checkpoint_name
    config_path = experiment_out_dir / "final_config.yaml"

    with open(config_path) as f:
        cfg_dict = yaml.safe_load(f)
        if cfg_dict["pnorm"] == "anneal-1-2":
            cfg_dict["pnorm"] = "anneal-2-1"
        cfg = Config(**cfg_dict)

    print("Creating model for histogram generation")
    llm = AutoModelForCausalLM.from_pretrained(
        cfg.pretrained_model_name_hf, torch_dtype=torch.bfloat16
    )
    model = ComponentModel(
        llm,
        cfg.target_module_patterns,
        cfg.C,
        cfg.gate_type,
        cfg.n_ci_mlp_neurons,
        cfg.gate_init_central,
        cfg.pretrained_model_output_attr,
        dtype=torch.bfloat16,
    )
    # model.sigmoid_type = cfg.sigmoid_type
    state_dict = torch.load(last_checkpoint_path, map_location="cuda")
    model.load_state_dict(state_dict)
    model.eval()
    model.to(DEVICE)

    print("Getting dataloader for histogram data")
    tc = cfg.task_config
    assert isinstance(tc, LMTaskConfig)
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

    print("Collecting importance values")
    importance_values = collect_importance_histograms(
        model=model,
        data_loader=iter(train_loader),
        budget=budget,
        device=DEVICE,
        sigmoid_type=cfg.sigmoid_type,
    )

    # Generate histogram plots
    report_dir = Path("reports") / experiment_out_dir.name
    report_dir.mkdir(parents=True, exist_ok=True)

    generate_histogram_plots(importance_values, report_dir)


if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available")

DEVICE = torch.device("cuda")


def main(
    experiment_out_dir: Path,
    checkpoint_name: str,
    budget: timedelta,
):
    last_checkpoint_path = experiment_out_dir / checkpoint_name
    config_path = experiment_out_dir / "final_config.yaml"

    with open(config_path) as f:
        cfg_dict = yaml.safe_load(f)
        if cfg_dict["pnorm"] == "anneal-1-2":
            cfg_dict["pnorm"] = "anneal-2-1"
        cfg = Config(**cfg_dict)

    print("creating model")
    llm = AutoModelForCausalLM.from_pretrained(
        cfg.pretrained_model_name_hf, torch_dtype=torch.bfloat16
    )
    model = ComponentModel(
        llm,
        cfg.target_module_patterns,
        cfg.C,
        cfg.gate_type,
        cfg.n_ci_mlp_neurons,
        cfg.gate_init_central,
        cfg.pretrained_model_output_attr,
        dtype=torch.bfloat16,
    )
    state_dict = torch.load(last_checkpoint_path, map_location="cuda")
    model.load_state_dict(state_dict)
    model.eval()
    model.to(DEVICE)

    print("getting tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(cfg.pretrained_model_name_hf)

    print("getting dataloader")
    tc = cfg.task_config
    assert isinstance(tc, LMTaskConfig)
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

    component_examiner = ComponentExaminer(
        model=model,
        tokens=(x["input_ids"] for x in train_loader),
        sigmoid_type=cfg.sigmoid_type,
        device=DEVICE,
    )

    component_summaries_by_module = component_examiner.gather_component_summaries(budget)

    n_examples = [
        len(s.selected_examples)
        for _, component_summaries in component_summaries_by_module.items()
        for s in component_summaries
    ]
    print(
        f"num examples per component: mean {np.mean(n_examples):.2f} std {np.std(n_examples):.2f}"
    )

    # %%
    report_dir = Path("reports") / experiment_out_dir.name
    report_dir.mkdir(parents=True, exist_ok=True)

    context_tokens = 20

    # %%

    for component_name, component_summaries in component_summaries_by_module.items():
        doc = format_layer_component_summaries_html(component_summaries, tokenizer, context_tokens)
        report_path = report_dir / f"{component_name}.html"
        report_path.write_text(doc, encoding="utf-8")
        print(f"wrote report to {report_path}")


# experiment_out_dir = Path(
#     "/mnt/polished-lake/home/oli/spd-gf/spd/experiments/lm/out/"
#     "20250629_194019_662_lm_nmasks1_stochrecon1.37e-01_stochreconlayer1.00e-01_"
#     "p1.50e+00_impmin2.49e-04_C4096_sd0_lr6.00e-05_bs16__"
#     "pretrainedSimpleStories_SimpleStories-1.25M_seq512"
# )
# checkpoint_name = "model_95000.pth"

# experiment_out_dir = Path(
#     "/mnt/polished-lake/home/oli/spd-gf/spd/experiments/lm/out/20250630_110913_309_distinctive-sweep-5"
# )
# checkpoint_name = "model_95000.pth"


experiments = [
    # (
    #     "/mnt/polished-lake/home/oli/spd-gf/spd/experiments/lm/out/20250701_022419_181_absurd-sweep-3",
    #     "model_25000.pth",
    # ),
    (
        "/mnt/polished-lake/home/oli/spd-gf/spd/experiments/lm/out/20250701_230936_457_vital-sweep-25",
        "model_20000.pth",
    )
]


for experiment_out_dir, model_cp in experiments:
    # main(
    #     budget=timedelta(minutes=1),
    #     experiment_out_dir=Path(experiment_out_dir),
    #     checkpoint_name=model_cp,
    #     importance_threshold=0.5,
    # )

    # Generate histograms
    main_histograms(
        experiment_out_dir=Path(experiment_out_dir),
        checkpoint_name=model_cp,
        budget=timedelta(minutes=3),
    )
