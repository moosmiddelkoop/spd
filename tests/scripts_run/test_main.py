"""Tests for the main() function and end-to-end execution workflows in spd/scripts/run.py.

This file focuses on testing the high-level integration behavior of the main() function,
including local execution, SLURM submission, and end-to-end parameter sweep workflows.

Lower-level functions like generate_grid_combinations and load_sweep_params are tested
in test_grid_search.py and test_run_sweep_params.py respectively.
"""

# pyright: reportUnknownParameterType=false, reportMissingParameterType=false, reportUnusedParameter=false

import json
from unittest.mock import Mock, patch

import pytest

from spd.scripts.run import (
    generate_commands,
    generate_run_id,
    main,
    resolve_sweep_params_path,
)


def get_valid_tms_config():
    """Get a valid TMS experiment config."""
    return {
        "wandb_project": "test",
        "C": 10,
        "n_mask_samples": 100,
        "target_module_patterns": ["layer1"],
        "importance_minimality_coeff": 0.1,
        "faithfulness_coeff": 1.0,
        "stochastic_recon_coeff": 1.0,
        "stochastic_recon_layerwise_coeff": 1.0,
        "pnorm": 2.0,
        "output_loss_type": "mse",
        "lr": 0.001,
        "steps": 1000,
        "batch_size": 32,
        "n_eval_steps": 100,
        "print_freq": 50,
        "pretrained_model_class": "spd.experiments.tms.models.TMSModel",
        "pretrained_model_path": "test_model.pth",
        "task_config": {
            "task_name": "tms",
            "feature_probability": 0.05,
            "data_generation_type": "at_least_zero_active",
        },
    }


def get_valid_resid_mlp_config():
    """Get a valid ResidMLP experiment config."""
    config = get_valid_tms_config()
    config["pretrained_model_class"] = "spd.experiments.resid_mlp.models.ResidualMLP"
    config["task_config"] = {
        "task_name": "residual_mlp",
        "feature_probability": 0.1,
        "data_generation_type": "exactly_two_active",
    }
    return config


class TestCommandGeneration:
    """Test command generation and utility functions.

    Tests run_id generation, path resolution, and the integration of
    generate_commands() with actual config loading and JSON serialization.
    Grid combination logic is tested separately in test_grid_search.py.
    """

    def test_generate_run_id_format(self):
        """Test that generated run IDs have the expected format."""
        run_id = generate_run_id()

        # Should start with 'run_' and have timestamp format
        assert run_id.startswith("run_")
        assert len(run_id) == 19  # run_YYYYMMDD_HHMMSS

        # Should be parseable as a timestamp format
        timestamp_part = run_id[4:]  # Remove 'run_' prefix
        assert len(timestamp_part) == 15  # YYYYMMDD_HHMMSS
        assert "_" in timestamp_part

    def test_resolve_sweep_params_path_simple_name(self):
        """Test resolving sweep params path with simple filename."""
        path = resolve_sweep_params_path("sweep_params.yaml")
        assert path.name == "sweep_params.yaml"
        assert "spd/scripts" in str(path)

    def test_resolve_sweep_params_path_with_directory(self):
        """Test resolving sweep params path with directory."""
        path = resolve_sweep_params_path("custom/my_sweep.yaml")
        assert path.name == "my_sweep.yaml"
        assert "custom" in str(path)

    @patch("spd.scripts.run.load_config")
    def test_generate_commands_without_sweep(self, mock_load_config):
        """Test generate_commands function without sweep parameters."""
        from spd.configs import Config

        # Mock config loading with proper Config structure
        mock_config = Mock(spec=Config)
        config_dict = get_valid_tms_config()
        mock_config.model_dump.return_value = config_dict
        mock_load_config.return_value = mock_config

        commands = generate_commands(
            experiments_list=["tms_5-2-id"],
            run_id="test_run_123",
            sweep_params_file=None,
            project="test-project",
        )

        assert len(commands) == 1
        command = commands[0]

        # Verify command structure
        assert "python" in command
        assert "tms_decomposition.py" in command
        assert "--sweep_id test_run_123" in command
        assert "--evals_id tms_5-2-id" in command
        assert "json:" in command

        # Extract and verify the JSON config
        json_start = command.find("json:") + 5
        json_end = command.find("' --sweep_id")
        config_json = json.loads(command[json_start:json_end])

        assert config_json["wandb_project"] == "test-project"  # Should be overridden
        assert config_json["C"] == 10
        assert config_json["task_config"]["task_name"] == "tms"

    @patch("spd.scripts.run.load_config")
    @patch("spd.scripts.run.load_sweep_params")
    def test_generate_commands_with_sweep(self, mock_load_sweep_params, mock_load_config):
        """Test generate_commands function with sweep parameters."""
        from spd.configs import Config

        # Mock config loading
        mock_config = Mock(spec=Config)
        config_dict = get_valid_tms_config()
        mock_config.model_dump.return_value = config_dict
        mock_load_config.return_value = mock_config

        # Mock sweep parameters
        mock_load_sweep_params.return_value = {"C": {"values": [5, 10]}}

        commands = generate_commands(
            experiments_list=["tms_5-2-id"],
            run_id="test_run_123",
            sweep_params_file="sweep_params.yaml",
            project="test-project",
        )

        assert len(commands) == 2  # Two C values

        for i, command in enumerate(commands):
            assert "python" in command
            assert "tms_decomposition.py" in command
            assert "--sweep_id test_run_123" in command
            assert "--evals_id tms_5-2-id" in command
            assert "--sweep_params_json" in command

            # Extract and verify the config JSON
            json_start = command.find("json:") + 5
            json_end = command.find("' --sweep_id")
            config_json = json.loads(command[json_start:json_end])

            # Check that C was overridden
            expected_c = [5, 10][i]
            assert config_json["C"] == expected_c


class TestLocalExecution:
    """Test local execution flow."""

    @patch("spd.scripts.run.create_wandb_report")
    @patch("spd.scripts.run.create_workspace_view")
    @patch("spd.scripts.run.ensure_project_exists")
    @patch("spd.scripts.run.logger")
    @patch("spd.scripts.run.subprocess.run")
    def test_local_run_single_experiment(
        self, mock_subprocess, mock_logger, mock_ensure_project, mock_workspace, mock_report
    ):
        """Test running a single experiment locally."""
        # Setup mocks
        mock_subprocess.return_value = Mock(returncode=0)
        mock_workspace.return_value = "https://wandb.ai/test/workspace/tms_5-2-id"

        # Call main with tms_5-2-id experiment in local mode
        main(experiments="tms_5-2-id", local=True)

        # Verify subprocess was called
        assert mock_subprocess.call_count == 1

        # Verify the command structure
        args = mock_subprocess.call_args[0][0]
        assert args[0] == "python"
        assert "tms_decomposition.py" in args[1]

        # Verify logger calls
        mock_logger.info.assert_any_call("Experiments: tms_5-2-id")
        mock_logger.section.assert_any_call("LOCAL EXECUTION: Running 1 tasks")

        # Verify workspace was created
        assert mock_workspace.call_count == 1
        assert mock_workspace.call_args[0][1] == "tms_5-2-id"

        # Verify no report for single experiment
        assert mock_report.call_count == 0

    @patch("spd.scripts.run.create_wandb_report")
    @patch("spd.scripts.run.create_workspace_view")
    @patch("spd.scripts.run.ensure_project_exists")
    @patch("spd.scripts.run.logger")
    @patch("spd.scripts.run.subprocess.run")
    def test_local_run_multiple_experiments(
        self, mock_subprocess, mock_logger, mock_ensure_project, mock_workspace, mock_report
    ):
        """Test running multiple experiments locally."""
        # Setup mocks
        mock_subprocess.return_value = Mock(returncode=0)
        mock_workspace.side_effect = (
            lambda run_id, exp, proj: f"https://wandb.ai/{proj}/{exp}/workspace/{run_id}"
        )
        mock_report.return_value = "https://wandb.ai/test/report"

        main(experiments="tms_5-2,resid_mlp1", local=True, create_report=True)

        # Should run 2 commands
        assert mock_subprocess.call_count == 2

        # Verify workspace views created for both
        assert mock_workspace.call_count == 2
        workspace_calls = mock_workspace.call_args_list
        experiments_created = [call[0][1] for call in workspace_calls]
        assert set(experiments_created) == {"tms_5-2", "resid_mlp1"}

        # Verify report created for multiple experiments
        assert mock_report.call_count == 1

    @patch("spd.scripts.run.create_wandb_report")
    @patch("spd.scripts.run.create_workspace_view")
    @patch("spd.scripts.run.ensure_project_exists")
    @patch("spd.scripts.run.subprocess.run")
    def test_local_run_with_failed_command(
        self, mock_subprocess, mock_ensure_project, mock_workspace, mock_report
    ):
        """Test handling of failed subprocess execution."""
        # Make subprocess fail
        mock_subprocess.return_value = Mock(returncode=1)

        # Should not raise exception
        main(experiments="tms_5-2-id", local=True)

        # Verify subprocess was still called
        assert mock_subprocess.call_count == 1

    @patch("spd.scripts.run.create_wandb_report")
    @patch("spd.scripts.run.create_workspace_view")
    @patch("spd.scripts.run.ensure_project_exists")
    @patch("spd.scripts.run.subprocess.run")
    def test_local_run_no_git_snapshot(
        self, mock_subprocess, mock_ensure_project, mock_workspace, mock_report
    ):
        """Test that git snapshot is not created in local mode."""
        # Setup mocks
        mock_subprocess.return_value = Mock(returncode=0)

        # This should raise an exception if create_git_snapshot is called
        with patch("spd.scripts.run.create_git_snapshot") as mock_git:
            mock_git.side_effect = Exception(
                "create_git_snapshot should not be called in local mode"
            )

            # Should complete without calling git snapshot
            main(experiments="tms_5-2-id", local=True)

            # Verify git snapshot was not called
            mock_git.assert_not_called()


class TestSLURMSubmission:
    """Test SLURM submission flow."""

    @patch("spd.scripts.run.submit_slurm_array")
    @patch("spd.scripts.run.create_slurm_array_script")
    @patch("spd.scripts.run.create_git_snapshot")
    @patch("spd.scripts.run.create_wandb_report")
    @patch("spd.scripts.run.create_workspace_view")
    @patch("spd.scripts.run.ensure_project_exists")
    def test_slurm_submission(
        self,
        mock_ensure_project,
        mock_workspace,
        mock_report,
        mock_git,
        mock_create_script,
        mock_submit,
    ):
        """Test basic SLURM submission."""
        # Setup mocks
        mock_git.return_value = "snapshot-20231215-120000"
        mock_submit.return_value = "12345"

        main(experiments="tms_5-2-id", local=False)

        # Verify git snapshot was created
        assert mock_git.call_count == 1

        # Verify SLURM script creation
        assert mock_create_script.call_count == 1
        script_args = mock_create_script.call_args[1]
        assert script_args["job_name"] == "spd"
        assert len(script_args["commands"]) == 1
        assert script_args["cpu"] is False
        assert script_args["snapshot_branch"] == "snapshot-20231215-120000"

        # Verify SLURM submission
        assert mock_submit.call_count == 1

    @patch("spd.scripts.run.submit_slurm_array")
    @patch("spd.scripts.run.create_slurm_array_script")
    @patch("spd.scripts.run.create_git_snapshot")
    @patch("spd.scripts.run.create_wandb_report")
    @patch("spd.scripts.run.create_workspace_view")
    @patch("spd.scripts.run.ensure_project_exists")
    def test_slurm_submission_with_n_agents(
        self,
        mock_ensure_project,
        mock_workspace,
        mock_report,
        mock_git,
        mock_create_script,
        mock_submit,
    ):
        """Test SLURM submission with n_agents specified."""
        # Setup mocks
        mock_git.return_value = "snapshot-20231215-120000"
        mock_submit.return_value = "12345"

        main(experiments="tms_5-2-id", local=False, n_agents=4)

        # Verify n_agents passed to SLURM script
        script_args = mock_create_script.call_args[1]
        assert script_args["max_concurrent_tasks"] == 4

    @patch("spd.scripts.run.submit_slurm_array")
    @patch("spd.scripts.run.create_slurm_array_script")
    @patch("spd.scripts.run.create_git_snapshot")
    @patch("spd.scripts.run.create_wandb_report")
    @patch("spd.scripts.run.create_workspace_view")
    @patch("spd.scripts.run.ensure_project_exists")
    @patch("spd.scripts.run.load_sweep_params")
    def test_sweep_requires_n_agents_for_slurm(
        self,
        mock_load_sweep_params,
        mock_ensure_project,
        mock_workspace,
        mock_report,
        mock_git,
        mock_create_script,
        mock_submit,
    ):
        """Test that sweeps require n_agents when not running locally."""
        mock_load_sweep_params.return_value = {"C": {"values": [5, 10]}}

        with pytest.raises(AssertionError, match="n_agents must be provided"):
            main(experiments="tms_5-2-id", sweep=True, local=False)

    @patch("spd.scripts.run.submit_slurm_array")
    @patch("spd.scripts.run.create_slurm_array_script")
    @patch("spd.scripts.run.create_git_snapshot")
    @patch("spd.scripts.run.create_wandb_report")
    @patch("spd.scripts.run.create_workspace_view")
    @patch("spd.scripts.run.ensure_project_exists")
    def test_job_suffix_and_cpu_flags(
        self,
        mock_ensure_project,
        mock_workspace,
        mock_report,
        mock_git,
        mock_create_script,
        mock_submit,
    ):
        """Test job_suffix and cpu flag parameters."""
        # Setup mocks
        mock_git.return_value = "snapshot-20231215-120000"
        mock_submit.return_value = "12345"

        main(experiments="tms_5-2-id", local=False, job_suffix="test-suffix", cpu=True)

        # Verify job name includes suffix
        script_args = mock_create_script.call_args[1]
        assert script_args["job_name"] == "spd-test-suffix"
        assert script_args["cpu"] is True


class TestParameterSweeps:
    """Test parameter sweep functionality."""

    @patch("spd.scripts.run.create_wandb_report")
    @patch("spd.scripts.run.create_workspace_view")
    @patch("spd.scripts.run.ensure_project_exists")
    @patch("spd.scripts.run.subprocess.run")
    @patch("spd.scripts.run.load_sweep_params")
    def test_local_sweep_with_default_file(
        self,
        mock_load_sweep_params,
        mock_subprocess,
        mock_ensure_project,
        mock_workspace,
        mock_report,
    ):
        """Test running a sweep with default sweep_params.yaml file."""
        # Setup mocks
        mock_load_sweep_params.return_value = {"C": {"values": [5, 10, 15]}}
        mock_subprocess.return_value = Mock(returncode=0)

        main(experiments="tms_5-2-id", sweep=True, local=True)

        # Should run 3 commands (one for each C value)
        assert mock_subprocess.call_count == 3

        # Verify load_sweep_params was called correctly
        mock_load_sweep_params.assert_called_once()
        args = mock_load_sweep_params.call_args[0]
        assert args[0] == "tms_5-2-id"
        assert "sweep_params.yaml" in str(args[1])

    @patch("spd.scripts.run.create_wandb_report")
    @patch("spd.scripts.run.create_workspace_view")
    @patch("spd.scripts.run.ensure_project_exists")
    @patch("spd.scripts.run.subprocess.run")
    @patch("spd.scripts.run.load_sweep_params")
    def test_local_sweep_with_custom_file(
        self,
        mock_load_sweep_params,
        mock_subprocess,
        mock_ensure_project,
        mock_workspace,
        mock_report,
    ):
        """Test running a sweep with custom sweep parameters file."""
        # Setup mocks
        mock_load_sweep_params.return_value = {"seed": {"values": [0, 1]}}
        mock_subprocess.return_value = Mock(returncode=0)

        main(experiments="tms_5-2-id", sweep="custom_sweep.yaml", local=True)

        # Verify custom file path was used
        args = mock_load_sweep_params.call_args[0]
        assert "custom_sweep.yaml" in str(args[1])

    @patch("spd.scripts.run.create_wandb_report")
    @patch("spd.scripts.run.create_workspace_view")
    @patch("spd.scripts.run.ensure_project_exists")
    @patch("spd.scripts.run.subprocess.run")
    @patch("spd.scripts.run.load_sweep_params")
    def test_sweep_allows_local_without_n_agents(
        self,
        mock_load_sweep_params,
        mock_subprocess,
        mock_ensure_project,
        mock_workspace,
        mock_report,
    ):
        """Test that sweeps work locally without specifying n_agents."""
        # Setup mocks
        mock_load_sweep_params.return_value = {"C": {"values": [5, 10]}}
        mock_subprocess.return_value = Mock(returncode=0)

        # Should not raise an error
        main(experiments="tms_5-2-id", sweep=True, local=True)

        assert mock_subprocess.call_count == 2


class TestLoggerOutput:
    """Test logger calls and messages."""

    @patch("spd.scripts.run.create_wandb_report")
    @patch("spd.scripts.run.create_workspace_view")
    @patch("spd.scripts.run.ensure_project_exists")
    @patch("spd.scripts.run.subprocess.run")
    @patch("spd.scripts.run.generate_run_id")
    @patch("spd.scripts.run.logger")
    def test_logger_output_for_local_run(
        self,
        mock_logger,
        mock_generate_run_id,
        mock_subprocess,
        mock_ensure_project,
        mock_workspace,
        mock_report,
    ):
        """Test all logger calls during local execution."""
        # Setup mocks
        mock_generate_run_id.return_value = "run_20240115_143022"
        mock_subprocess.return_value = Mock(returncode=0)
        mock_workspace.side_effect = (
            lambda run_id, exp, proj: f"https://wandb.ai/{proj}/{exp}/workspace/{run_id}"
        )
        mock_report.return_value = "https://wandb.ai/test/report"

        main(experiments="tms_5-2,resid_mlp1", local=True)

        # Verify initial info logs
        mock_logger.info.assert_any_call("Run ID: run_20240115_143022")
        mock_logger.info.assert_any_call("Experiments: tms_5-2, resid_mlp1")

        # Verify section logs
        mock_logger.section.assert_any_call("Creating workspace views...")
        mock_logger.section.assert_any_call("LOCAL EXECUTION: Running 2 tasks")

        # Verify execution logs (should have progress for each command)
        section_calls = [call[0][0] for call in mock_logger.section.call_args_list]
        execution_calls = [c for c in section_calls if "Executing:" in c]
        assert len(execution_calls) == 2

        # Verify completion log
        mock_logger.section.assert_any_call("LOCAL EXECUTION COMPLETE")

        # Verify workspace URLs logged
        values_calls = mock_logger.values.call_args_list
        workspace_call = next(c for c in values_calls if "workspace urls" in str(c))
        workspace_data = workspace_call[1]["data"]
        assert "tms_5-2" in workspace_data
        assert "resid_mlp1" in workspace_data

    @patch("spd.scripts.run.create_wandb_report")
    @patch("spd.scripts.run.create_workspace_view")
    @patch("spd.scripts.run.ensure_project_exists")
    @patch("spd.scripts.run.subprocess.run")
    @patch("spd.scripts.run.logger")
    def test_logger_warning_on_failure(
        self, mock_logger, mock_subprocess, mock_ensure_project, mock_workspace, mock_report
    ):
        """Test logger warning when command fails."""
        # Make subprocess fail
        mock_subprocess.return_value = Mock(returncode=1)

        main(experiments="tms_5-2-id", local=True)

        # Verify warning was logged
        warning_calls = mock_logger.warning.call_args_list
        assert len(warning_calls) == 1
        assert "exit code 1" in warning_calls[0][0][0]


class TestIntegration:
    """End-to-end tests with minimal mocking."""

    def test_invalid_experiment_name(self):
        """Test that invalid experiment names raise an error."""
        with pytest.raises(ValueError, match="Invalid experiments.*nonexistent_experiment"):
            main(experiments="nonexistent_experiment", local=True)

    def test_multiple_invalid_experiment_names(self):
        """Test that multiple invalid experiment names raise an error."""
        with pytest.raises(ValueError, match="Invalid experiments.*fake1.*fake2"):
            main(experiments="fake1,fake2", local=True)

    @patch("spd.scripts.run.create_wandb_report")
    @patch("spd.scripts.run.create_workspace_view")
    @patch("spd.scripts.run.ensure_project_exists")
    @patch("spd.scripts.run.subprocess.run")
    def test_run_all_experiments(
        self, mock_subprocess, mock_ensure_project, mock_workspace, mock_report
    ):
        """Test running all experiments when experiments=None."""
        # Setup mocks
        mock_subprocess.return_value = Mock(returncode=0)

        # Should run all experiments in registry
        main(experiments=None, local=True)

        # We have 7 active experiments in registry
        assert mock_subprocess.call_count == 7

    @patch("spd.scripts.run.create_wandb_report")
    @patch("spd.scripts.run.create_workspace_view")
    @patch("spd.scripts.run.ensure_project_exists")
    @patch("spd.scripts.run.subprocess.run")
    def test_mixed_experiment_types(
        self, mock_subprocess, mock_ensure_project, mock_workspace, mock_report
    ):
        """Test running mixed experiment types (TMS + ResidMLP + LM)."""
        # Setup mocks
        mock_subprocess.return_value = Mock(returncode=0)

        main(experiments="tms_5-2,resid_mlp1,resid_mlp2", local=True)

        # Should run 3 different experiment types
        assert mock_subprocess.call_count == 3

        # Verify different decomposition scripts were called
        commands = [call[0][0] for call in mock_subprocess.call_args_list]
        script_names = [cmd[1] for cmd in commands]  # cmd[1] is the script path

        assert any("tms_decomposition.py" in script for script in script_names)
        assert any("resid_mlp_decomposition.py" in script for script in script_names)

    @patch("spd.scripts.run.create_wandb_report")
    @patch("spd.scripts.run.create_workspace_view")
    @patch("spd.scripts.run.ensure_project_exists")
    @patch("spd.scripts.run.subprocess.run")
    def test_no_report_for_single_experiment(
        self, mock_subprocess, mock_ensure_project, mock_workspace, mock_report
    ):
        """Test that no report is created for single experiment even with create_report=True."""
        # Setup mocks
        mock_subprocess.return_value = Mock(returncode=0)

        main(experiments="tms_5-2-id", local=True, create_report=True)

        # Verify no report created
        assert mock_report.call_count == 0

    @patch("spd.scripts.run.create_wandb_report")
    @patch("spd.scripts.run.create_workspace_view")
    @patch("spd.scripts.run.ensure_project_exists")
    @patch("spd.scripts.run.subprocess.run")
    def test_create_report_false(
        self, mock_subprocess, mock_ensure_project, mock_workspace, mock_report
    ):
        """Test that no report is created when create_report=False."""
        # Setup mocks
        mock_subprocess.return_value = Mock(returncode=0)

        main(experiments="tms_5-2,resid_mlp1", local=True, create_report=False)

        # Verify no report created
        assert mock_report.call_count == 0

    @patch("spd.scripts.run.create_wandb_report")
    @patch("spd.scripts.run.create_workspace_view")
    @patch("spd.scripts.run.ensure_project_exists")
    @patch("spd.scripts.run.subprocess.run")
    @patch("spd.scripts.run.logger")
    def test_log_format_parameter(
        self, mock_logger, mock_subprocess, mock_ensure_project, mock_workspace, mock_report
    ):
        """Test log_format parameter is passed to logger."""
        # Setup mocks
        mock_subprocess.return_value = Mock(returncode=0)

        main(experiments="tms_5-2-id", local=True, log_format="terse")

        # Verify logger format was set
        mock_logger.set_format.assert_called_once_with("console", "terse")


class TestErrorHandling:
    """Test error propagation and edge cases."""

    @patch("spd.scripts.run.load_config")
    def test_config_validation_error_propagation(self, mock_load_config):
        """Test that config validation errors in main() are properly propagated."""
        from pydantic import ValidationError

        # Make load_config raise a ValidationError
        mock_load_config.side_effect = ValidationError.from_exception_data(
            "Config",
            [{"type": "missing", "loc": ("lr",)}],  # pyright: ignore[reportArgumentType]
        )

        with pytest.raises(ValidationError):
            main(experiments="tms_5-2-id", local=True)

    @patch("spd.scripts.run.load_config")
    def test_missing_config_file_error(self, mock_load_config):
        """Test handling of missing config files."""
        # Make load_config raise FileNotFoundError
        mock_load_config.side_effect = FileNotFoundError("Config file not found")

        with pytest.raises(FileNotFoundError, match="Config file not found"):
            main(experiments="tms_5-2-id", local=True)

    @patch("spd.scripts.run.submit_slurm_array")
    @patch("spd.scripts.run.create_slurm_array_script")
    @patch("spd.scripts.run.create_git_snapshot")
    @patch("spd.scripts.run.create_wandb_report")
    @patch("spd.scripts.run.create_workspace_view")
    @patch("spd.scripts.run.ensure_project_exists")
    def test_slurm_submission_failure(
        self,
        mock_ensure_project,
        mock_workspace,
        mock_report,
        mock_git,
        mock_create_script,
        mock_submit,
    ):
        """Test handling of SLURM submission failures."""
        # Setup mocks
        mock_git.return_value = "snapshot-20231215-120000"

        # Make SLURM submission fail
        mock_submit.side_effect = RuntimeError("SLURM submission failed")

        with pytest.raises(RuntimeError, match="SLURM submission failed"):
            main(experiments="tms_5-2-id", local=False)

    @patch("spd.scripts.run.create_git_snapshot")
    @patch("spd.scripts.run.create_wandb_report")
    @patch("spd.scripts.run.create_workspace_view")
    @patch("spd.scripts.run.ensure_project_exists")
    def test_git_snapshot_failure(self, mock_ensure_project, mock_workspace, mock_report, mock_git):
        """Test handling of git snapshot creation failures."""
        # Make git snapshot creation fail
        mock_git.side_effect = RuntimeError("Git snapshot failed")

        with pytest.raises(RuntimeError, match="Git snapshot failed"):
            main(experiments="tms_5-2-id", local=False)

    @patch("spd.scripts.run.load_sweep_params")
    def test_sweep_params_loading_error(self, mock_load_sweep_params):
        """Test handling of sweep parameters loading errors."""
        # Make sweep params loading fail
        mock_load_sweep_params.side_effect = FileNotFoundError("Sweep params file not found")

        with pytest.raises(FileNotFoundError, match="Sweep params file not found"):
            main(experiments="tms_5-2-id", sweep=True, local=True)

    @patch("spd.scripts.run.load_config")
    def test_invalid_task_config_discriminator(self, mock_load_config):
        """Test handling of invalid task_config discriminator values."""
        from pydantic import ValidationError

        # Create a simple ValidationError for testing error propagation
        try:
            from spd.configs import Config

            Config(task_config={"task_name": "invalid_task"})  # pyright: ignore[reportCallIssue]
        except ValidationError as e:
            mock_load_config.side_effect = e

        with pytest.raises(ValidationError):
            main(experiments="tms_5-2-id", local=True)

    @patch("spd.scripts.run.create_wandb_report")
    @patch("spd.scripts.run.create_workspace_view")
    @patch("spd.scripts.run.ensure_project_exists")
    @patch("spd.scripts.run.subprocess.run")
    @patch("spd.scripts.run.load_sweep_params")
    def test_large_parameter_grid_performance(
        self,
        mock_load_sweep_params,
        mock_subprocess,
        mock_ensure_project,
        mock_workspace,
        mock_report,
    ):
        """Test handling of very large parameter grids."""
        # Setup large parameter grid (2^6 = 64 combinations using valid Config fields)
        large_params = {
            "lr": {"values": [0.001, 0.01]},
            "batch_size": {"values": [16, 32]},
            "steps": {"values": [100, 200]},
            "seed": {"values": [0, 1]},
            "C": {"values": [5, 10]},
            "task_config": {"feature_probability": {"values": [0.05, 0.1]}},
        }
        mock_load_sweep_params.return_value = large_params
        mock_subprocess.return_value = Mock(returncode=0)

        # Should handle large grid without issues
        main(experiments="tms_5-2-id", sweep=True, local=True)

        # Verify all 64 combinations were generated
        assert mock_subprocess.call_count == 64
