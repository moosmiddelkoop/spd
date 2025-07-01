# %%
from collections import defaultdict
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import yaml
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast

from spd.configs import Config, LMTaskConfig
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

                comps_unique, comps_counts = torch.unique(c_idx, return_counts=True)
                for comp, cnt in zip(comps_unique.tolist(), comps_counts.tolist(), strict=True):
                    component_firing_counts[(module_name, comp)] += int(cnt)

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
                    component_examples[(module_name, c)].append(example)

            tokens_seen += B * S
            pbar.set_postfix(n_components=len(component_examples))

        summaries: list[ComponentSummary] = []
        for (module_name, comp_idx), examples in component_examples.items():
            # Ensure deterministic ordering for repeatability
            examples.sort(key=lambda x: x.token_importances_S[-1].item(), reverse=True)
            summaries.append(
                ComponentSummary(
                    module_name=module_name,
                    component_idx=comp_idx,
                    selected_examples=examples,
                    density=component_firing_counts[(module_name, comp_idx)] / tokens_seen,
                )
            )

        summaries.sort(key=lambda x: x.density, reverse=True)

        return summaries


def interpolate(bottom: float, top: float, x: float) -> float:
    """Interpolate between a and b using x, which is in [0, 1]."""
    return bottom + (top - bottom) * x


WHITE = (255, 255, 255, 255)
DARK_GREEN = (31, 122, 31, 255)
LIGHT_GREEN = (144, 238, 144, 255)


def _get_highlight_color(
    importance: float,
    color_upper: tuple[int, int, int, int] = DARK_GREEN,
    color_lower: tuple[int, int, int, int] = LIGHT_GREEN,
) -> str:
    """Get highlight color based on importance value."""
    importance_norm = min(max(importance, 0), 1)  # Clamp to [0, 1]
    r = int(interpolate(color_lower[0], color_upper[0], importance_norm))
    g = int(interpolate(color_lower[1], color_upper[1], importance_norm))
    b = int(interpolate(color_lower[2], color_upper[2], importance_norm))
    a = int(interpolate(color_lower[3], color_upper[3], importance_norm))
    return f"rgba({r}, {g}, {b}, {a})"


def _apply_multi_token_highlights(
    token_ids: list[int],
    token_importances: list[float],
    tokenizer: PreTrainedTokenizerFast,
) -> str:
    assert len(token_ids) == len(token_importances), "length mismatch"
    token_strings = tokenizer.convert_ids_to_tokens(token_ids)
    escaped_token_strings = [html.escape(ts, quote=False) for ts in token_strings]
    parts = []
    for imp, escaped_token_string in zip(token_importances, escaped_token_strings, strict=True):
        if imp > 0:
            color = _get_highlight_color(
                imp,
                color_lower=(DARK_GREEN[0], DARK_GREEN[1], DARK_GREEN[2], 0),
            )
            parts.append(
                f'<span style="color:#fff;background-color:{color};">{escaped_token_string}</span>'
            )
        else:
            parts.append(escaped_token_string)
    return " ".join(parts)


def render_component_summary_html(
    summary: ComponentSummary,
    # tokenizer: PreTrainedTokenizerFast,
    # example_picker:Callable[[ComponentSummary], list[ComponentExample]],
    render_examples: Callable[[ComponentSummary], str],
    # context_tokens: int = 20,
) -> str:
    """Return a fully escaped HTML `<section>` for *one* `ComponentSummary`."""

    return (
        f"<section>\n"
        f"  <h2>Module: {html.escape(summary.module_name)} / "
        f"Component {summary.component_idx}</h2>\n"
        f"  <p>Density: {summary.density:.4%} &nbsp;|&nbsp; "
        f"Total examples: {len(summary.selected_examples)}</p>\n"
        f"{render_examples(summary)}"
    )


def format_component_summaries_html(
    summaries: Iterable[ComponentSummary],
    render_examples: Callable[[ComponentSummary], str],
    page_title: str = "Component Example Visualisation",
) -> str:
    """Compile *all* summaries into a single styled HTML file and return the path."""

    sections = [render_component_summary_html(s, render_examples) for s in summaries]

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

    return html_doc


if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available")

DEVICE = torch.device("cuda")


# experiment_out_dir = Path(
#     "/mnt/polished-lake/home/oli/spd-gf/spd/experiments/lm/out/"
#     "20250629_194019_662_lm_nmasks1_stochrecon1.37e-01_stochreconlayer1.00e-01_"
#     "p1.50e+00_impmin2.49e-04_C4096_sd0_lr6.00e-05_bs16__"
#     "pretrainedSimpleStories_SimpleStories-1.25M_seq512"
# )
# checkpoint_name = "model_95000.pth"

experiment_out_dir = Path(
    "/mnt/polished-lake/home/oli/spd-gf/spd/experiments/lm/out/20250630_110913_309_distinctive-sweep-5"
)
checkpoint_name = "model_95000.pth"


# def main(
# experiment_out_dir: Path,
# checkpoint_name: str,
if __name__ == "__main__":
    total_batches: int = 2
    importance_threshold: float = 0.9
    n_context_tokens: int = 30
    # ):
    last_checkpoint_path = experiment_out_dir / checkpoint_name
    config_path = experiment_out_dir / "final_config.yaml"

    # %%

    with open(config_path) as f:
        cfg_dict = yaml.safe_load(f)

        if "gate_type" not in cfg_dict:
            raise ValueError("gate_type not in config")
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

    # %%

    print("gathering component summaries")
    component_examiner = ComponentExaminer(
        model=model,
        tokens=(x["input_ids"] for x in train_loader),
        sigmoid_type=cfg.sigmoid_type,
        device=DEVICE,
    )

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
    report_dir = Path("reports") / experiment_out_dir.name
    report_dir.mkdir(parents=True, exist_ok=True)

    report_path = report_dir / checkpoint_name.replace(".pth", ".html")
    print(f"writing report to {report_path}")

    context_tokens = 20

    # %%

    import matplotlib.pyplot as plt

    plt.hist([cs.density for cs in component_summaries], bins=100)
    plt.show()
    # %%

    def get_example_text(ex: ComponentExample) -> str:
        ids_list = ex.tokens_S[-context_tokens:].tolist()
        importance_list = ex.token_importances_S[-context_tokens:].tolist()
        return _apply_multi_token_highlights(ids_list, importance_list, tokenizer)

    def render_examples(summary: ComponentSummary) -> str:
        top_examples = summary.selected_examples[:20]
        top_percentile_section = [get_example_text(ex) for ex in top_examples]

        bottom_examples = summary.selected_examples[max(20, len(summary.selected_examples) - 20) :]
        bottom_percentile_section = [get_example_text(ex) for ex in bottom_examples]

        def row(ex: str) -> str:
            return f"<tr><td style='white-space:nowrap;'>{ex}</td></tr>"

        def section(title: str, examples: list[str]) -> str:
            return f"""
            <section>
                <h3>{title}</h3>
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

        return section("Top 20% of examples", top_percentile_section) + section(
            "Bottom 20% of examples", bottom_percentile_section
        )

    doc = format_component_summaries_html(component_summaries, render_examples)

    report_path.write_text(doc, encoding="utf-8")

    print(f"wrote report to {report_path}")

    # %%
    from IPython.display import HTML

    HTML(report_path.read_text(encoding="utf-8"))
    # %%


experiments = [
    "/mnt/polished-lake/home/oli/spd-gf/spd/experiments/lm/out/20250630_110913_309_distinctive-sweep-5",
    "/mnt/polished-lake/home/oli/spd-gf/spd/experiments/lm/out/20250630_050123_661_usual-sweep-12",
]

model_cp = "model_95000.pth"

for experiment_out_dir in experiments:
    main(
        total_batches=1,
        experiment_out_dir=Path(experiment_out_dir),
        checkpoint_name=model_cp,
    )
