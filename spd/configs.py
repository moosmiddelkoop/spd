"""Config classes of various types"""

from typing import Any, ClassVar, Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    NonNegativeFloat,
    NonNegativeInt,
    PositiveFloat,
    PositiveInt,
    model_validator,
)

from spd.log import logger
from spd.models.components import GateType
from spd.spd_types import ModelPath, Probability


class TMSTaskConfig(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", frozen=True)
    task_name: Literal["tms"] = Field(
        default="tms",
        description="Task identifier for TMS",
    )
    feature_probability: Probability = Field(
        ...,
        description="Probability that a given feature is active in generated data",
    )
    data_generation_type: Literal["exactly_one_active", "at_least_zero_active"] = Field(
        default="at_least_zero_active",
        description="Strategy for generating synthetic data for TMS training",
    )


class ResidualMLPTaskConfig(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", frozen=True)
    task_name: Literal["residual_mlp"] = Field(
        default="residual_mlp",
        description="Identifier for the residual-MLP decomposition task",
    )
    feature_probability: Probability = Field(
        ...,
        description="Probability that a given feature is active in generated data",
    )
    data_generation_type: Literal[
        "exactly_one_active", "exactly_two_active", "at_least_zero_active"
    ] = Field(
        default="at_least_zero_active",
        description="Strategy for generating synthetic data for residual-MLP training",
    )


class LMTaskConfig(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", frozen=True)
    task_name: Literal["lm"] = Field(
        default="lm",
        description="Identifier for the language-model decomposition task",
    )
    max_seq_len: PositiveInt = Field(
        default=512,
        description="Maximum sequence length to truncate or pad inputs to",
    )
    buffer_size: PositiveInt = Field(
        default=1000,
        description="Buffered sample count for streaming dataset shuffling",
    )
    dataset_name: str = Field(
        default="lennart-finke/SimpleStories",
        description="HuggingFace dataset identifier to use for the LM task",
    )
    column_name: str = Field(
        default="story",
        description="Dataset column that contains the text to train on",
    )
    train_data_split: str = Field(
        default="train",
        description="Name of the dataset split used for training",
    )
    eval_data_split: str = Field(
        default="test",
        description="Name of the dataset split used for evaluation",
    )


class MemorizationTaskConfig(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", frozen=True)
    task_name: Literal["memorization"] = Field(
        default="memorization",
        description="Identifier for the memorization decomposition task",
    )


class Config(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", frozen=True)
    # --- WandB
    wandb_project: str | None = Field(
        default=None,
        description="Weights & Biases project name (set to None to disable WandB logging)",
    )
    wandb_run_name: str | None = Field(
        default=None,
        description="Explicit name for the WandB run (None generates an automatic name)",
    )
    wandb_run_name_prefix: str = Field(
        default="",
        description="Prefix prepended to an auto-generated WandB run name",
    )

    # --- General ---
    seed: int = Field(default=0, description="Random seed for reproducibility")
    C: PositiveInt = Field(
        ...,
        description="The number of subcomponents per layer",
    )
    n_mask_samples: PositiveInt = Field(
        ...,
        description="Number of stochastic masks to sample when using stochastic recon losses",
    )
    gate_type: GateType = Field(
        default="vector_mlp",
        description="Type of gate used to calculate the causal importance.",
    )
    gate_hidden_dims: list[NonNegativeInt] = Field(
        default=[8],
        description="Hidden dimensions for the gate used to calculate the causal importance",
    )
    sigmoid_type: Literal[
        "normal", "hard", "leaky_hard", "upper_leaky_hard", "swish_hard", "scaled"
    ] = Field(
        default="leaky_hard",
        description="Type of sigmoid to use for causal importance calculation",
    )
    target_module_patterns: list[str] = Field(
        ...,
        description="List of fnmatch-style patterns that select modules to decompose",
    )

    # --- Loss Coefficients
    faithfulness_coeff: NonNegativeFloat | None = Field(
        default=1.0,
        description="Coefficient for matching parameters between components and target weights",
    )
    recon_coeff: NonNegativeFloat | None = Field(
        default=None,
        description="Coefficient for recon loss with a causal importance mask",
    )
    stochastic_recon_coeff: NonNegativeFloat | None = Field(
        default=None,
        description="Coefficient for recon loss with stochastically sampled masks",
    )
    recon_layerwise_coeff: NonNegativeFloat | None = Field(
        default=None,
        description="Coefficient for per-layer recon loss with a causal importance mask",
    )
    stochastic_recon_layerwise_coeff: NonNegativeFloat | None = Field(
        default=None,
        description="Coefficient for per-layer recon loss with stochastically sampled masks",
    )
    task_recon_coeff: NonNegativeFloat = Field(
        default=0.0,
        description="Coefficient for task reconstruction loss against ground truth targets",
    )
    task_recon_layerwise_coeff: NonNegativeFloat = Field(
        default=0.0,
        description="Coefficient for per-layer task reconstruction loss against ground truth targets",
    )
    task_stochastic_recon_coeff: NonNegativeFloat = Field(
        default=0.0,
        description="Coefficient for stochastic task reconstruction loss against ground truth targets",
    )
    task_stochastic_recon_layerwise_coeff: NonNegativeFloat = Field(
        default=0.0,
        description="Coefficient for per-layer stochastic task reconstruction loss against ground truth targets",
    )
    importance_minimality_coeff: NonNegativeFloat = Field(
        ...,
        description="Coefficient for importance minimality loss",
    )
    schatten_coeff: NonNegativeFloat | None = Field(
        default=None,
        description="Coefficient for Schatten-norm regularisation (LM only)",
    )
    out_recon_coeff: NonNegativeFloat | None = Field(
        default=None,
        description="Coefficient for output recon loss",
    )
    out_task_recon_coeff: NonNegativeFloat = Field(
        default=0.0,
        description="Coefficient for task recon loss when all masks are set to 1 against ground truth targets",
    )
    embedding_recon_coeff: float | None = Field(
        default=None,
        description="Coefficient for additional embedding recon loss (LM only)",
    )
    is_embed_unembed_recon: bool = Field(
        default=False,
        description="If True, apply embedding recon jointly to embed & unembed matrices",
    )
    pnorm: PositiveFloat = Field(
        ...,
        description="The p-value used for the importance minimality loss",
    )
    p_anneal_start_frac: Probability = Field(
        default=1.0,
        description="Fraction of training after which to start annealing p (1.0 = no annealing)",
    )
    p_anneal_final_p: PositiveFloat | None = Field(
        default=None,
        description="Final p value to anneal to (None = no annealing)",
    )
    output_loss_type: Literal["mse", "kl"] = Field(
        ...,
        description="Metric used to measure recon error between model outputs and targets",
    )

    # --- Training ---
    lr: PositiveFloat = Field(..., description="Learning rate for optimiser")
    steps: PositiveInt = Field(..., description="Total number of optimisation steps")
    batch_size: PositiveInt = Field(..., description="Mini-batch size used for optimisation")
    lr_schedule: Literal["linear", "constant", "cosine", "exponential"] = Field(
        default="constant",
        description="Type of learning-rate schedule to apply",
    )
    lr_exponential_halflife: PositiveFloat | None = Field(
        default=None,
        description="Half-life parameter when using an exponential LR schedule",
    )
    lr_warmup_pct: Probability = Field(
        default=0.0,
        description="Fraction of total steps to linearly warm up the learning rate",
    )
    n_eval_steps: PositiveInt = Field(
        ...,
        description="Frequency (in optimisation steps) at which to run evaluation",
    )

    # --- Logging & Saving ---
    image_freq: PositiveInt | None = Field(
        default=None,
        description="Interval (in steps) at which to log diagnostic images to WandB",
    )
    image_on_first_step: bool = Field(
        default=True,
        description="Whether to log images at optimisation step 0",
    )
    print_freq: PositiveInt = Field(
        ...,
        description="Interval (in steps) at which to print training metrics to stdout",
    )
    save_freq: PositiveInt | None = Field(
        default=None,
        description="Interval (in steps) at which to save model checkpoints (None disables saving "
        "until the end of training).",
    )
    log_ce_losses: bool = Field(
        default=False,
        description="If True, additionally track cross-entropy losses during training",
    )
    metrics_fns: list[str] = Field(
        default=[],
        description="List of local names of functions to use for computing metrics. These functions must be defined in the `spd.metrics_and_figs` module.",
    )
    figures_fns: list[str] = Field(
        default=[],
        description="List of local names of functions to use for creating figures. These functions must be defined in the `spd.metrics_and_figs` module.",
    )

    # --- Pretrained model info ---
    pretrained_model_class: str = Field(
        ...,
        description="Fully-qualified class name of the pretrained model to load. Can be defined "
        "locally or an in external package (e.g. 'transformers.LlamaForCausalLM' or "
        "'spd.experiments.resid_mlp.models.ResidualMLP').",
    )
    pretrained_model_path: ModelPath | None = Field(
        default=None,
        description="Model identifier. Local path or wandb reference "
        "(e.g. 'wandb:spd/runs/otxwx80v' or 'mnt/my_model/checkpoint.pth')",
    )
    pretrained_model_name_hf: str | None = Field(
        default=None,
        description="hf model identifier. E.g. 'SimpleStories/SimpleStories-1.25M'",
    )
    pretrained_model_output_attr: str | None = Field(
        default=None,
        description="Name of the attribute on the forward output that contains logits or activations",
    )
    tokenizer_name: str | None = Field(
        default=None,
        description="Name or path of the tokenizer to use when loading an LM",
    )

    # --- Task Specific ---
    task_config: TMSTaskConfig | ResidualMLPTaskConfig | LMTaskConfig | MemorizationTaskConfig = (
        Field(
            ...,
            discriminator="task_name",
            description="Nested task-specific configuration selected by the `task_name` discriminator",
        )
    )

    DEPRECATED_CONFIG_KEYS: ClassVar[list[str]] = []
    RENAMED_CONFIG_KEYS: ClassVar[dict[str, str]] = {}

    @model_validator(mode="before")
    def handle_deprecated_config_keys(cls, config_dict: dict[str, Any]) -> dict[str, Any]:
        """Remove deprecated config keys and change names of any keys that have been renamed."""
        for key in list(config_dict.keys()):
            val = config_dict[key]
            if key in cls.DEPRECATED_CONFIG_KEYS:
                logger.warning(f"{key} is deprecated, but has value: {val}. Removing from config.")
                del config_dict[key]
            elif key in cls.RENAMED_CONFIG_KEYS:
                logger.info(f"Renaming {key} to {cls.RENAMED_CONFIG_KEYS[key]}")
                config_dict[cls.RENAMED_CONFIG_KEYS[key]] = val
                del config_dict[key]
        return config_dict

    @model_validator(mode="after")
    def validate_model(self) -> Self:
        # If any of the coeffs are 0, raise a warning
        msg = "is 0, you may wish to instead set it to null to avoid calculating the loss"
        if self.recon_coeff == 0:
            logger.warning(f"recon_coeff {msg}")
        if self.importance_minimality_coeff == 0:
            logger.warning(f"importance_minimality_coeff {msg}")
        if self.faithfulness_coeff == 0:
            logger.warning(f"faithfulness_coeff {msg}")

        # Check that lr_exponential_halflife is not None if lr_schedule is "exponential"
        if self.lr_schedule == "exponential":
            assert self.lr_exponential_halflife is not None, (
                "lr_exponential_halflife must be set if lr_schedule is exponential"
            )

        return self
