"""Run spd on a TMS model.

Note that the first instance index is fixed to the identity matrix. This is done so we can compare
the losses of the "correct" solution during training.
"""

import json
from pathlib import Path
from typing import Any

import wandb

from spd.configs import Config, TMSTaskConfig
from spd.experiments.tms.models import TMSModel, TMSModelConfig
from spd.log import logger
from spd.run_spd import get_common_run_name_suffix, optimize
from spd.utils.data_utils import DatasetGeneratedDataLoader, SparseFeatureDataset
from spd.utils.general_utils import get_device, load_config, set_seed
from spd.utils.run_utils import get_output_dir, save_file
from spd.utils.wandb_utils import init_wandb


def get_run_name(config: Config, tms_model_config: TMSModelConfig) -> str:
    """Generate a run name based on the config."""
    if config.wandb_run_name:
        run_suffix = config.wandb_run_name
    else:
        run_suffix = get_common_run_name_suffix(config)
        run_suffix += f"ft{tms_model_config.n_features}_"
        run_suffix += f"hid{tms_model_config.n_hidden}"
        run_suffix += f"hid-layers{tms_model_config.n_hidden_layers}"
    exp_name = f"tms{tms_model_config.n_features}-{tms_model_config.n_hidden}"
    # TODO: Consolidate tms experiment names (previously we used tms_5-2-id)
    if tms_model_config.n_hidden_layers > 0:
        exp_name += f"-{tms_model_config.n_hidden_layers}"
    exp_name += "_"
    return config.wandb_run_name_prefix + exp_name + run_suffix


def save_target_model_info(
    save_to_wandb: bool,
    out_dir: Path,
    tms_model: TMSModel,
    tms_model_train_config_dict: dict[str, Any],
) -> None:
    save_file(tms_model.state_dict(), out_dir / "tms.pth")
    save_file(tms_model_train_config_dict, out_dir / "tms_train_config.yaml")

    if save_to_wandb:
        wandb.save(str(out_dir / "tms.pth"), base_path=out_dir, policy="now")
        wandb.save(str(out_dir / "tms_train_config.yaml"), base_path=out_dir, policy="now")


def main(
    config_path_or_obj: Path | str | Config,
    evals_id: str | None = None,
    sweep_id: str | None = None,
    sweep_params_json: str | None = None,
) -> None:
    sweep_params = (
        None if sweep_params_json is None else json.loads(sweep_params_json.removeprefix("json:"))
    )

    device = get_device()
    logger.info(f"Using device: {device}")

    config = load_config(config_path_or_obj, config_model=Config)

    if config.wandb_project:
        tags = ["tms"]
        if evals_id:
            tags.append(evals_id)
        if sweep_id:
            tags.append(sweep_id)
        config = init_wandb(config, config.wandb_project, tags=tags)

    # Get output directory (automatically uses wandb run ID if available)
    out_dir = get_output_dir()

    task_config = config.task_config
    assert isinstance(task_config, TMSTaskConfig)

    set_seed(config.seed)
    logger.info(config)

    assert config.pretrained_model_path, "pretrained_model_path must be set"
    target_model, target_model_train_config_dict = TMSModel.from_pretrained(
        config.pretrained_model_path,
    )
    target_model = target_model.to(device)
    target_model.eval()

    run_name = get_run_name(config=config, tms_model_config=target_model.config)
    if config.wandb_project:
        assert wandb.run, "wandb.run must be initialized before training"
        wandb.run.name = run_name

    save_file(config.model_dump(mode="json"), out_dir / "final_config.yaml")
    if sweep_params:
        save_file(sweep_params, out_dir / "sweep_params.yaml")
    if config.wandb_project:
        wandb.save(str(out_dir / "final_config.yaml"), base_path=out_dir, policy="now")
        if sweep_params:
            wandb.save(str(out_dir / "sweep_params.yaml"), base_path=out_dir, policy="now")

    save_target_model_info(
        save_to_wandb=config.wandb_project is not None,
        out_dir=out_dir,
        tms_model=target_model,
        tms_model_train_config_dict=target_model_train_config_dict,
    )

    synced_inputs = target_model_train_config_dict.get("synced_inputs", None)
    dataset = SparseFeatureDataset(
        n_features=target_model.config.n_features,
        feature_probability=task_config.feature_probability,
        device=device,
        data_generation_type=task_config.data_generation_type,
        value_range=(0.0, 1.0),
        synced_inputs=synced_inputs,
    )
    train_loader = DatasetGeneratedDataLoader(dataset, batch_size=config.batch_size, shuffle=False)
    eval_loader = DatasetGeneratedDataLoader(dataset, batch_size=config.batch_size, shuffle=False)

    tied_weights = None
    if target_model.config.tied_weights:
        tied_weights = [("linear1", "linear2")]

    optimize(
        target_model=target_model,
        config=config,
        device=device,
        train_loader=train_loader,
        eval_loader=eval_loader,
        n_eval_steps=config.n_eval_steps,
        out_dir=out_dir,
        tied_weights=tied_weights,
    )

    if config.wandb_project:
        wandb.finish()


if __name__ == "__main__":
    main(config_path_or_obj="spd/experiments/tms/tms_5-2_config.yaml")
    # fire.Fire(main)
