import fnmatch
from contextlib import contextmanager
from functools import partial
from pathlib import Path
from typing import Any, override

import einops
import torch
import wandb
import yaml
from jaxtyping import Float
from torch import Tensor, nn
from torch.utils.hooks import RemovableHandle
from wandb.apis.public import Run

from spd.configs import Config
from spd.models.components import (
    EmbeddingComponent,
    GateMLP,
    GateType,
    LinearComponent,
    ReplacedComponent,
    VectorGateMLP,
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
        base_model: nn.Module,
        target_module_patterns: list[str],
        C: int,
        gate_type: GateType,
        gate_hidden_dims: list[int],
        pretrained_model_output_attr: str | None,
    ):
        super().__init__()
        self.model = base_model
        self.C = C
        self.pretrained_model_output_attr = pretrained_model_output_attr

        replaced_components = self.create_replaced_components(base_model, target_module_patterns, C)
        self.replaced_components = replaced_components
        self._replaced_components = nn.ModuleDict(
            {k.replace(".", "-"): v for k, v in replaced_components.items()}
        )

        gates = self.make_gates(replaced_components, C, gate_type, gate_hidden_dims)
        self.gates = gates
        self._gates = nn.ModuleDict({k.replace(".", "-"): v for k, v in gates.items()})

    @staticmethod
    def make_gates(
        replaced_components: dict[str, ReplacedComponent],
        C: int,
        gate_type: GateType,
        gate_hidden_dims: list[int],
    ) -> dict[str, nn.Module]:
        gates = {}
        for component_name, component in replaced_components.items():
            if gate_type == "mlp":
                gates[component_name] = GateMLP(C=C, hidden_dims=gate_hidden_dims)
            else:
                input_dim = (
                    component.original.weight.shape[1]
                    if isinstance(component.original, nn.Linear)
                    else component.original.num_embeddings
                )
                gates[component_name] = VectorGateMLP(
                    C=C,
                    input_dim=input_dim,
                    hidden_dims=gate_hidden_dims,
                )
        return gates

    @staticmethod
    def create_replaced_components(
        model: nn.Module, target_module_patterns: list[str], C: int
    ) -> dict[str, ReplacedComponent]:
        """Create target components for the model."""
        components: dict[str, ReplacedComponent] = {}
        matched_patterns: set[str] = set()

        for name, module in model.named_modules():
            for pattern in target_module_patterns:
                if fnmatch.fnmatch(name, pattern):
                    matched_patterns.add(pattern)
                    if isinstance(module, nn.Linear):
                        d_out, d_in = module.weight.shape
                        component = LinearComponent(d_in=d_in, d_out=d_out, C=C, bias=module.bias)
                    elif isinstance(module, nn.Embedding):
                        component = EmbeddingComponent(
                            vocab_size=module.num_embeddings,
                            embedding_dim=module.embedding_dim,
                            C=C,
                        )
                    else:
                        raise ValueError(
                            f"Module '{name}' matched pattern '{pattern}' but is not nn.Linear or "
                            f"nn.Embedding. Found type: {type(module)}"
                        )
                    replaced_component = ReplacedComponent(original=module, replacement=component)

                    # Maybe a `.get_replaced` method
                    components[name] = replaced_component

        unmatched_patterns = set(target_module_patterns) - matched_patterns
        if unmatched_patterns:
            raise ValueError(
                f"The following patterns in target_module_patterns did not match any modules: "
                f"{sorted(unmatched_patterns)}"
            )

        if not components:
            raise ValueError(
                f"No modules found matching target_module_patterns: {target_module_patterns}"
            )

        return components

    @override
    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Regular forward pass of the (target) model.

        If `model_output_attr` is set, return the attribute of the model's output.
        """
        raw_out = self.model(*args, **kwargs)
        if self.pretrained_model_output_attr is None:
            out = raw_out
        else:
            out = getattr(raw_out, self.pretrained_model_output_attr)
        return out

    @contextmanager
    def _replaced_modules(self, masks_BxC: dict[str, Tensor]):
        """Context manager for temporarily replacing modules with components.

        Args:
            masks_BxC: Optional dictionary mapping component names to masks
        """
        for module_name, component in self.replaced_components.items():
            if module_name in masks_BxC:
                component.forward_mode = "replacement"
                component.mask_BxC = masks_BxC[module_name]
            else:
                component.forward_mode = "original"
                component.mask_BxC = None
        try:
            yield
        finally:
            for component in self.replaced_components.values():
                component.forward_mode = None
                component.mask_BxC = None

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
            module = self.model.get_submodule(module_name)
            assert module is not None, f"Module {module_name} not found"
            handles.append(
                module.register_forward_pre_hook(partial(cache_hook, param_name=module_name))
            )

        try:
            out = self(*args, **kwargs)
            return out, cache
        finally:
            for handle in handles:
                handle.remove()

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
        base_model = base_model_raw[0] if isinstance(base_model_raw, tuple) else base_model_raw

        comp_model = ComponentModel(
            base_model=base_model,
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
        pre_weight_acts: dict[str, Tensor],
        detach_inputs: bool = False,
        sigmoid_type: SigmoidTypes = "leaky_hard",
    ) -> tuple[dict[str, Float[Tensor, "... C"]], dict[str, Float[Tensor, "... C"]]]:
        """Calculate component activations and causal importances in one pass to save memory.

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
            acts_BxD = pre_weight_acts[param_name]
            gate = self.gates[param_name]
            V = self.replaced_components[param_name].replacement.V

            if isinstance(gate, GateMLP):
                # need to get the inner activation for GateMLP
                if not acts_BxD.dtype.is_floating_point:
                    # Embedding layer
                    inner_acts_BxC = V[acts_BxD]
                else:
                    # Linear layer
                    inner_acts_BxC = einops.einsum(acts_BxD, V, "... d_in, d_in C -> ... C")
                gate_input = inner_acts_BxC
            else:
                gate_input = acts_BxD

            if detach_inputs:
                gate_input = gate_input.detach()

            gate_output = gate(gate_input)

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
