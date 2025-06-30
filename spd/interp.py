# %%
import html
from collections import defaultdict
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import yaml
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.tokenization_utils import PreTrainedTokenizer

from spd.configs import Config, LMTaskConfig
from spd.data import DatasetConfig, create_data_loader
from spd.models.component_model import ComponentModel
from spd.models.component_utils import calc_causal_importances
from spd.models.sigmoids import SigmoidTypes


@dataclass
class ComponentExample:
    tokens_S: torch.Tensor
    last_tok_importance: float
    token_importances_S: torch.Tensor  # Importance values for all tokens


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
        print("self.module_names", self.module_names)

        self.Vs = {
            name.replace("-", "."): component.V for name, component in model.components.items()
        }
        print("self.Vs", list(self.Vs.keys()))

        self.gates = {name.replace("-", "."): gate for name, gate in self.model.gates.items()}
        print("self.gates", list(self.gates.keys()))

        self.sigmoid_type = sigmoid_type
        self.device = device

    def gather_component_summaries(
        self,
        total_batches: int,
        importance_threshold: float,
    ) -> list[ComponentSummary]:
        component_examples: dict[tuple[str, int], list[ComponentExample]] = defaultdict(list)
        component_firing_counts: dict[tuple[str, int], int] = defaultdict(int)

        tokens_seen = 0

        pbar = tqdm(range(total_batches), desc="Gathering component summaries")

        for _ in pbar:
            tokens_BS = next(self.tokens).to(self.device)
            B, S = tokens_BS.shape
            _, preact_cache_BSD = self.model.forward_with_pre_forward_cache_hooks(
                tokens_BS, module_names=self.module_names
            )

            importance_BSM_by_module, _ = calc_causal_importances(
                pre_weight_acts=preact_cache_BSD,
                Vs=self.Vs,
                gates=self.gates,  # pyright: ignore[reportArgumentType]
                sigmoid_type=self.sigmoid_type,  # pyright: ignore[reportArgumentType]
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
                for comp, cnt in zip(comps_unique.tolist(), comps_counts.tolist(), strict=True):
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

                    # Get importance values for all tokens in this sequence
                    token_importances = imp_BSM[b, : s + 1, c].detach().cpu()
                    
                    example = ComponentExample(
                        tokens_S=tokens_BS[b, : s + 1].detach().cpu(),
                        last_tok_importance=float(imp),
                        token_importances_S=token_importances,
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


# These will be dynamically generated based on importance
_HIGHLIGHT_OPEN_TEMPLATE = '<span style="color:#fff;background-color:{color};">'
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


def _apply_highlight(escaped_text: str, escaped_target: str, importance: float) -> str:
    """Wrap **the last occurrence** of *escaped_target* in the highlight span.

    Both parameters **MUST** already be HTML-escaped.
    """
    pos = escaped_text.rfind(escaped_target)
    if pos == -1:
        # Fallback – shouldn’t normally happen, but fail gracefully
        return escaped_text
    highlight_color = _get_highlight_color(importance)
    highlight_open = _HIGHLIGHT_OPEN_TEMPLATE.format(color=highlight_color)
    
    return (
        escaped_text[:pos]
        + highlight_open
        + escaped_target
        + _HIGHLIGHT_CLOSE
        + escaped_text[pos + len(escaped_target) :]
    )


def _get_highlight_color(importance: float) -> str:
    """Get highlight color based on importance value."""
    importance_norm = min(max(importance, 0), 1)  # Clamp to [0, 1]
    # Interpolate from light green (144, 238, 144) to dark green (31, 122, 31)
    r = int(144 - (144 - 31) * importance_norm)
    g = int(238 - (238 - 122) * importance_norm)
    b = int(144 - (144 - 31) * importance_norm)
    return f"rgb({r}, {g}, {b})"


def _apply_multi_token_highlights(
    token_ids: list[int],
    token_importances: list[float],
    tokenizer: PreTrainedTokenizer,
) -> str:
    """Apply highlights to all tokens based on their importance values."""
    highlighted_parts = []
    
    for token_id, importance in zip(token_ids, token_importances, strict=True):
        # Decode single token
        token_str = tokenizer.decode([token_id], skip_special_tokens=True)
        
        # Make control chars visible
        visible_token = _make_visible(token_str)
        
        # Escape for HTML
        escaped_token = html.escape(visible_token, quote=False)
        
        # Apply highlight if importance > 0
        if importance > 0:
            highlight_color = _get_highlight_color(importance)
            highlight_open = _HIGHLIGHT_OPEN_TEMPLATE.format(color=highlight_color)
            highlighted_token = highlight_open + escaped_token + _HIGHLIGHT_CLOSE
        else:
            highlighted_token = escaped_token
            
        highlighted_parts.append(highlighted_token)
    
    return ''.join(highlighted_parts)


def render_component_summary_html(
    summary: "ComponentSummary",
    tokenizer: PreTrainedTokenizer,
    example_picker: Callable[[ComponentSummary], list[ComponentExample]],
    context_tokens: int = 20,
) -> str:
    """Return a fully escaped HTML `<section>` for *one* `ComponentSummary`."""

    rows: list[str] = []

    examples = example_picker(summary)

    for ex in examples:
        ids_list = ex.tokens_S[-context_tokens:].tolist()
        importance_list = ex.token_importances_S[-context_tokens:].tolist()
        
        highlighted_html = _apply_multi_token_highlights(ids_list, importance_list, tokenizer)
        # importance_bg_color = _get_highlight_color(ex.last_tok_importance)
        # text_color = "#fff"  # White text on green background

        rows.append(
            f'<tr>'
            # f'<td style="background-color:{importance_bg_color};color:{text_color};'
            f'font-weight:bold;">{ex.last_tok_importance:.3f}</td>'
            f'<td style="white-space:nowrap;">{highlighted_html}</td>'
            f'</tr>'
        )

    return (
        f"<section>\n"
        f"  <h2>Module: {html.escape(summary.module_name)} / "
        f"Component {summary.component_idx}</h2>\n"
        f"  <p>Density: {summary.density:.4%} &nbsp;|&nbsp; "
        f"Total examples: {len(summary.selected_examples)}</p>\n"
        f"  <table>\n    <thead><tr>"
        # "<th>Importance</th>"
        "<th>Context</th></tr></thead>\n    "
        f"<tbody>\n      {'\n      '.join(rows)}\n    </tbody>\n  </table>\n"
        f"</section>\n"
    )


def write_component_summaries_html(
    summaries: Iterable[ComponentSummary],
    tokenizer: PreTrainedTokenizer,
    *,
    file_path: str | Path = "component_summaries.html",
    example_picker: Callable[[ComponentSummary], list[ComponentExample]],
    context_tokens: int = 20,
    page_title: str = "Component Example Visualisation",
) -> Path:
    """Compile *all* summaries into a single styled HTML file and return the path."""

    file_path = Path(file_path)

    sections = [
        render_component_summary_html(s, tokenizer, example_picker, context_tokens=context_tokens)
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


# def write_summary_for_run(
#     experiment_out_dir: Path,
#     checkpoint_name: str,
#     total_batches: int = 1_000,
#     importance_threshold: float = 0.9,
#     n_context_tokens: int = 20,
#     # example_picker_cfg: dict[str, Any] = {},
#     # topk_examples_to_render: int = 30,
# ):
# %%

experiment_out_dir = Path(
    "/mnt/polished-lake/home/oli/spd-gf/spd/experiments/lm/out/"
    "20250629_194019_662_lm_nmasks1_stochrecon1.37e-01_stochreconlayer1.00e-01_"
    "p1.50e+00_impmin2.49e-04_C4096_sd0_lr6.00e-05_bs16__"
    "pretrainedSimpleStories_SimpleStories-1.25M_seq512"
)
checkpoint_name = "model_95000.pth"
total_batches = 1_000
importance_threshold = 0.9
n_context_tokens = 20

if __name__ == "__main__":
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
        cfg.gate_type,
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

    # %%

    component_summaries = component_examiner.gather_component_summaries(
        total_batches=total_batches,
        importance_threshold=importance_threshold,
    )

    print(f"gathered {len(component_summaries)} component summaries")
    n_examples = [len(s.selected_examples) for s in component_summaries]
    print(
        f"num examples per component: mean {np.mean(n_examples):.2f} std {np.std(n_examples):.2f}"
    )

    # %%

    report_path = experiment_out_dir / "component_summary.html"
    print(f"writing report to {report_path}")

    def example_picker(summary: ComponentSummary) -> list[ComponentExample]:
        out = []
        out.extend(summary.selected_examples[:10])
        out.extend(summary.selected_examples[-10:])
        return out

    write_component_summaries_html(
        # run_name=cfg.wandb_run_name,
        component_summaries,
        tokenizer,
        file_path=report_path,
        example_picker=example_picker,
        context_tokens=n_context_tokens,
    )

    # %%


# if __name__ == "__main__":
#     out_dir = (
#         "/mnt/polished-lake/home/oli/spd-gf/spd/experiments/lm/out/"
#         "20250629_194019_662_lm_nmasks1_stochrecon1.37e-01_stochreconlayer1.00e-01_"
#         "p1.50e+00_impmin2.49e-04_C4096_sd0_lr6.00e-05_bs16__"
#         "pretrainedSimpleStories_SimpleStories-1.25M_seq512"
#     )
#     model_name = "model_95000.pth"
#     write_summary_for_run(
#         experiment_out_dir=Path(out_dir),
#         checkpoint_name=model_name,
#         total_batches=10_000,
#         importance_threshold=0.5,
#         n_context_tokens=20,
#     )
# # %%
