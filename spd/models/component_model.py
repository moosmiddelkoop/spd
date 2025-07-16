import fnmatch
from contextlib import contextmanager
from functools import partial
from pathlib import Path
from typing import Any, override

import torch
import wandb
import yaml
from jaxtyping import Float, Int
from torch import Tensor, nn
from torch.utils.hooks import RemovableHandle
from wandb.apis.public import Run

from spd.configs import Config
from spd.models.components import (
    Components,
    ComponentsOrModule,
    EmbeddingComponents,
    GateMLPs,
    GateType,
    LinearComponents,
    VectorGateMLPs,
)
from spd.models.sigmoids import SIGMOID_TYPES, SigmoidTypes
from spd.spd_types import WANDB_PATH_PREFIX, ModelPath
from spd.utils.general_utils import load_pretrained
from spd.utils.wandb_utils import (
    download_wandb_file,
    fetch_latest_wandb_checkpoint,
    fetch_wandb_run_dir,
)


class ComponentModel(nn.Module):
    """Wrapper around an arbitrary model for running SPD.

    The underlying *base model* can be any subclass of `nn.Module` (e.g.
    `LlamaForCausalLM`, `AutoModelForCausalLM`) as long as its sub-module names
    match the patterns you pass in `target_module_patterns`.
    """

    def __init__(
        self,
        target_model: nn.Module,
        target_module_patterns: list[str],
        C: int,
        gate_type: GateType,
        gate_hidden_dims: list[int],
        pretrained_model_output_attr: str | None,
    ):
        super().__init__()
        self.target_model = target_model
        self.C = C
        self.pretrained_model_output_attr = pretrained_model_output_attr

        # where these did refer to the actual linear / embedding modules, they now refer to the
        # ComponentsOrModule objects. This still works for hooks
        self.target_module_paths = self._get_target_module_paths(
            target_model, target_module_patterns
        )

        components_or_modules = self.create_components_or_modules(
            target_model, self.target_module_paths, C
        )

        # just keep components_or_modules as a plain dict.
        # state_dict will pick it up via the target_model
        self.components_or_modules: dict[str, ComponentsOrModule] = components_or_modules

        self.gates = self.make_gates(gate_type, C, gate_hidden_dims, components_or_modules)
        self._gates = nn.ModuleDict({k.replace(".", "-"): v for k, v in self.gates.items()})

    @property
    def components(self) -> dict[str, Components]:
        return {name: cm.components for name, cm in self.components_or_modules.items()}

    def _get_target_module_paths(
        self, model: nn.Module, target_module_patterns: list[str]
    ) -> list[str]:
        matched_patterns: set[str] = set()
        for name, _ in model.named_modules():
            for pattern in target_module_patterns:
                if fnmatch.fnmatch(name, pattern):
                    print(f"Matched {name} to {pattern}")
                    matched_patterns.add(pattern)
        unmatched_patterns = set(target_module_patterns) - matched_patterns
        if unmatched_patterns:
            raise ValueError(
                f"The following patterns in target_module_patterns did not match any modules: "
                f"{sorted(unmatched_patterns)}"
            )

        return list(matched_patterns)

    @staticmethod
    def create_components_or_modules(
        target_model: nn.Module,
        target_module_paths: list[str],
        C: int,
    ) -> dict[str, ComponentsOrModule]:
        """Create target components for the model."""
        components_or_modules: dict[str, ComponentsOrModule] = {}

        for module_path in target_module_paths:
            module = target_model.get_submodule(module_path)

            if isinstance(module, nn.Linear):
                d_out, d_in = module.weight.shape
                component = LinearComponents(C=C, d_in=d_in, d_out=d_out, bias=module.bias)
                component.init_from_target_weight(module.weight)
            elif isinstance(module, nn.Embedding):
                component = EmbeddingComponents(
                    C=C,
                    vocab_size=module.num_embeddings,
                    embedding_dim=module.embedding_dim,
                )
                # NOTE(oli): Ensure that we're doing the right thing wrt how the old code does .T
                component.init_from_target_weight(module.weight)
            else:
                raise ValueError(
                    f"Module '{module_path}' matched pattern is not nn.Linear or "
                    f"nn.Embedding. Found type: {type(module)}"
                )

            replacement = ComponentsOrModule(original=module, components=component)

            target_model.set_submodule(module_path, replacement)

            components_or_modules[module_path] = replacement

        return components_or_modules

    @staticmethod
    def make_gates(
        gate_type: GateType,
        C: int,
        gate_hidden_dims: list[int],
        components_or_modules: dict[str, ComponentsOrModule],
    ) -> dict[str, nn.Module]:
        gates: dict[str, nn.Module] = {}
        for module_path, component in components_or_modules.items():
            # get input dim in case we're creating a vector gate
            if isinstance(component.original, nn.Linear):
                input_dim = component.original.weight.shape[1]
            elif isinstance(component.original, nn.Embedding):  # pyright: ignore[reportUnnecessaryIsInstance]
                input_dim = component.original.num_embeddings
            else:
                raise ValueError(f"Unknown component type: {type(component)}")

            if gate_type == "mlp":
                gate = GateMLPs(C=C, hidden_dims=gate_hidden_dims)
            else:
                gate = VectorGateMLPs(C=C, input_dim=input_dim, hidden_dims=gate_hidden_dims)

            gates[module_path] = gate

        return gates

    @override
    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Regular forward pass of the (target) model.

        If `model_output_attr` is set, return the attribute of the model's output.
        """
        raw_out = self.target_model(*args, **kwargs)
        if self.pretrained_model_output_attr is None:
            out = raw_out
        else:
            out = getattr(raw_out, self.pretrained_model_output_attr)
        return out

    @contextmanager
    def _replaced_modules(self, masks: dict[str, Tensor]):
        """Context manager for temporarily replacing modules with components.

        Args:
            masks: Optional dictionary mapping component names to masks
        """
        for module_name, component in self.components_or_modules.items():
            assert component.forward_mode is None, (
                f"Component must be in pristine state, but forward_mode is {component.forward_mode}"
            )
            assert component.mask is None, (
                "Component must be in pristine state, but mask is not None"
            )

            if module_name in masks:
                component.forward_mode = "components"
                component.mask = masks[module_name]
            else:
                component.forward_mode = "original"
                component.mask = None
        try:
            yield
        finally:
            for component in self.components_or_modules.values():
                component.forward_mode = None
                component.mask = None

    def forward_with_components(
        self,
        *args: Any,
        masks: dict[str, Float[Tensor, "... C"]],
        **kwargs: Any,
    ) -> Any:
        """Forward pass with temporary component replacements.

        Args:
            masks: Optional dictionary mapping component names to masks
        """
        with self._replaced_modules(masks):
            return self(*args, **kwargs)

    def forward_with_pre_forward_cache_hooks(
        self, *args: Any, module_names: list[str], **kwargs: Any
    ) -> tuple[Any, dict[str, Tensor]]:
        """Forward pass with caching at the input to the modules given by `module_names`.

        Args:
            module_names: List of module names to cache the inputs to.

        Returns:
            Tuple of (model output, cache dictionary)
        """
        cache = {}
        handles: list[RemovableHandle] = []

        def cache_hook(_: nn.Module, input: tuple[Tensor, ...], param_name: str) -> None:
            cache[param_name] = input[0]

        # Register hooks
        for module_name in module_names:
            module = self.target_model.get_submodule(module_name)
            assert module is not None, f"Module {module_name} not found"
            handles.append(
                module.register_forward_pre_hook(partial(cache_hook, param_name=module_name))
            )

        for module in self.components_or_modules.values():
            module.forward_mode = "original"

        try:
            out = self(*args, **kwargs)
            return out, cache
        finally:
            for handle in handles:
                handle.remove()

            for module in self.components_or_modules.values():
                module.forward_mode = None

    @staticmethod
    def _download_wandb_files(wandb_project_run_id: str) -> tuple[Path, Path]:
        """Download the relevant files from a wandb run.

        Returns:
            Tuple of (model_path, config_path)
        """
        api = wandb.Api()
        run: Run = api.run(wandb_project_run_id)

        checkpoint = fetch_latest_wandb_checkpoint(run, prefix="model")

        run_dir = fetch_wandb_run_dir(run.id)

        final_config_path = download_wandb_file(run, run_dir, "final_config.yaml")
        checkpoint_path = download_wandb_file(run, run_dir, checkpoint.name)

        return checkpoint_path, final_config_path

    @classmethod
    def from_pretrained(cls, path: ModelPath) -> tuple["ComponentModel", Config, Path]:
        """Load a trained ComponentModel checkpoint along with its original config.

        The method supports two storage schemes:
        1.  A direct local path to the checkpoint file (plus `final_config.yaml` in
            the same directory).
        2.  A WandB reference of the form ``wandb:<entity>/<project>/runs/<run_id>``.
        """

        if isinstance(path, str) and path.startswith(WANDB_PATH_PREFIX):
            wandb_path = path.removeprefix(WANDB_PATH_PREFIX)
            api = wandb.Api()
            run: Run = api.run(wandb_path)
            model_path, config_path = cls._download_wandb_files(wandb_path)
            out_dir = fetch_wandb_run_dir(run.id)
        else:
            model_path = Path(path)
            config_path = Path(path).parent / "final_config.yaml"
            out_dir = Path(path).parent

        model_weights = torch.load(model_path, map_location="cpu", weights_only=True)
        with open(config_path) as f:
            config = Config(**yaml.safe_load(f))

        assert config.pretrained_model_class is not None

        base_model_raw = load_pretrained(
            path_to_class=config.pretrained_model_class,
            model_path=config.pretrained_model_path,
            model_name_hf=config.pretrained_model_name_hf,
        )
        target_model = base_model_raw[0] if isinstance(base_model_raw, tuple) else base_model_raw

        comp_model = ComponentModel(
            target_model=target_model,
            target_module_patterns=config.target_module_patterns,
            C=config.C,
            gate_hidden_dims=config.gate_hidden_dims,
            gate_type=config.gate_type,
            pretrained_model_output_attr=config.pretrained_model_output_attr,
        )
        comp_model.load_state_dict(model_weights)
        return comp_model, config, out_dir

    def calc_causal_importances(
        self,
        pre_weight_acts: dict[str, Float[Tensor, "... d_in"] | Int[Tensor, "... pos"]],
        detach_inputs: bool = False,
        sigmoid_type: SigmoidTypes = "leaky_hard",
    ) -> tuple[dict[str, Float[Tensor, "... C"]], dict[str, Float[Tensor, "... C"]]]:
        """Calculate causal importances.

        Args:
            pre_weight_acts: The activations before each layer in the target model.
            detach_inputs: Whether to detach the inputs to the gates.
            sigmoid_type: Type of sigmoid to use.

        Returns:
            Tuple of (causal_importances, causal_importances_upper_leaky) dictionaries for each layer.
        """
        causal_importances = {}
        causal_importances_upper_leaky = {}

        for param_name in pre_weight_acts:
            acts = pre_weight_acts[param_name]
            gates = self.gates[param_name]

            if isinstance(gates, GateMLPs):
                # need to get the inner activation for GateMLP
                gate_input = self.components[param_name].get_inner_acts(acts)
            elif isinstance(gates, VectorGateMLPs):
                gate_input = acts
            else:
                raise ValueError(f"Unknown gate type: {type(gates)}")

            if detach_inputs:
                gate_input = gate_input.detach()

            gate_output = gates(gate_input)

            if sigmoid_type == "leaky_hard":
                causal_importances[param_name] = SIGMOID_TYPES["lower_leaky_hard"](gate_output)
                causal_importances_upper_leaky[param_name] = SIGMOID_TYPES["upper_leaky_hard"](
                    gate_output
                )
            else:
                # For other sigmoid types, use the same function for both
                sigmoid_fn = SIGMOID_TYPES[sigmoid_type]
                causal_importances[param_name] = sigmoid_fn(gate_output)
                # Use absolute value to ensure upper_leaky values are non-negative for importance minimality loss
                causal_importances_upper_leaky[param_name] = sigmoid_fn(gate_output).abs()

        return causal_importances, causal_importances_upper_leaky
