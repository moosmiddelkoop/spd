"""Tests for the main function in spd/scripts/run.py."""

import json
from unittest.mock import Mock, patch

import pytest

from spd.scripts.run import (
    generate_commands,
    generate_grid_combinations,
    generate_run_id,
    main,
    resolve_sweep_params_path,
)


@patch("spd.scripts.run.subprocess.run")
@patch("spd.scripts.run.create_workspace_view")
@patch("spd.scripts.run.ensure_project_exists")
@patch("spd.scripts.run.generate_run_id")
@patch("spd.scripts.run.logger")
def test_local_run_single_experiment(
    mock_logger,
    mock_generate_run_id,
    mock_ensure_project,
    mock_create_workspace,
    mock_subprocess_run,
):
    """Test running a single experiment locally with subprocess mocking."""
    # Setup mocks
    mock_generate_run_id.return_value = "run_20231215_120000"
    mock_create_workspace.return_value = "https://wandb.ai/test/workspace"
    mock_subprocess_run.return_value = Mock(returncode=0)
    
    # Call main with tms_5-2-id experiment in local mode
    main(experiments="tms_5-2-id", local=True)
    
    # Verify subprocess.run was called once
    assert mock_subprocess_run.call_count == 1
    
    # Get the arguments passed to subprocess.run
    call_args = mock_subprocess_run.call_args[0][0]
    
    # Expected command structure based on the code
    # The command should be:
    # python <decomp_script> '<config_json>' --sweep_id <run_id> --evals_id <experiment>
    assert len(call_args) >= 6
    assert call_args[0] == "python"
    
    # Check decomp script path
    assert call_args[1].endswith("spd/experiments/tms/tms_decomposition.py")
    
    # Check config JSON
    config_json_str = call_args[2]
    assert config_json_str.startswith("json:")
    
    # Parse the config JSON
    config_dict = json.loads(config_json_str[5:])  # Remove 'json:' prefix
    assert config_dict["wandb_project"] == "spd"
    
    # Check sweep_id and evals_id flags
    assert "--sweep_id" in call_args
    assert "--evals_id" in call_args
    
    sweep_id_index = call_args.index("--sweep_id")
    evals_id_index = call_args.index("--evals_id")
    
    assert call_args[sweep_id_index + 1] == "run_20231215_120000"
    assert call_args[evals_id_index + 1] == "tms_5-2-id"
    
    # Verify other functions were called
    mock_ensure_project.assert_called_once_with("spd")
    mock_create_workspace.assert_called_once_with("run_20231215_120000", "tms_5-2-id", "spd")

@patch("spd.scripts.run.subprocess.run")
@patch("spd.scripts.run.create_workspace_view")
@patch("spd.scripts.run.ensure_project_exists")
@patch("spd.scripts.run.generate_run_id")
@patch("spd.scripts.run.logger")
def test_local_run_multiple_experiments(
    mock_logger,
    mock_generate_run_id,
    mock_ensure_project,
    mock_create_workspace,
    mock_subprocess_run,
):
    """Test running multiple experiments locally."""
    # Setup mocks
    mock_generate_run_id.return_value = "run_20231215_120000"
    mock_create_workspace.return_value = "https://wandb.ai/test/workspace"
    mock_subprocess_run.return_value = Mock(returncode=0)
    
    # Call main with multiple experiments
    main(experiments="tms_5-2,resid_mlp1", local=True)
    
    # Verify subprocess.run was called twice (once for each experiment)
    assert mock_subprocess_run.call_count == 2
    
    # Check first experiment call (tms_5-2)
    first_call_args = mock_subprocess_run.call_args_list[0][0][0]
    assert first_call_args[1].endswith("spd/experiments/tms/tms_decomposition.py")
    evals_id_index = first_call_args.index("--evals_id")
    assert first_call_args[evals_id_index + 1] == "tms_5-2"
    
    # Check second experiment call (resid_mlp1)
    second_call_args = mock_subprocess_run.call_args_list[1][0][0]
    assert second_call_args[1].endswith("spd/experiments/resid_mlp/resid_mlp_decomposition.py")
    evals_id_index = second_call_args.index("--evals_id")
    assert second_call_args[evals_id_index + 1] == "resid_mlp1"

@patch("spd.scripts.run.subprocess.run")
@patch("spd.scripts.run.create_workspace_view")
@patch("spd.scripts.run.ensure_project_exists")
@patch("spd.scripts.run.generate_run_id")
@patch("spd.scripts.run.logger")
def test_local_run_with_custom_project(
    mock_logger,
    mock_generate_run_id,
    mock_ensure_project,
    mock_create_workspace,
    mock_subprocess_run,
):
    """Test running with a custom W&B project."""
    # Setup mocks
    mock_generate_run_id.return_value = "run_20231215_120000"
    mock_create_workspace.return_value = "https://wandb.ai/test/workspace"
    mock_subprocess_run.return_value = Mock(returncode=0)
    
    # Call main with custom project
    main(experiments="tms_5-2-id", local=True, project="my-custom-project")
    
    # Verify the custom project was used
    mock_ensure_project.assert_called_once_with("my-custom-project")
    mock_create_workspace.assert_called_once_with(
        "run_20231215_120000", "tms_5-2-id", "my-custom-project"
    )
    
    # Check that the config JSON contains the custom project
    call_args = mock_subprocess_run.call_args[0][0]
    config_json_str = call_args[2]
    config_dict = json.loads(config_json_str[5:])  # Remove 'json:' prefix
    assert config_dict["wandb_project"] == "my-custom-project"

@patch("spd.scripts.run.subprocess.run")
@patch("spd.scripts.run.create_workspace_view")
@patch("spd.scripts.run.ensure_project_exists")
@patch("spd.scripts.run.generate_run_id")
@patch("spd.scripts.run.logger")
def test_local_run_with_failed_command(
    mock_logger,
    mock_generate_run_id,
    mock_ensure_project,
    mock_create_workspace,
    mock_subprocess_run,
):
    """Test handling of failed subprocess execution."""
    # Setup mocks
    mock_generate_run_id.return_value = "run_20231215_120000"
    mock_create_workspace.return_value = "https://wandb.ai/test/workspace"
    
    # Simulate a failed command
    mock_subprocess_run.return_value = Mock(returncode=1)
    
    # Call main - it should not raise an exception
    main(experiments="tms_5-2-id", local=True)
    
    # Verify warning was logged about the failure
    mock_logger.warning.assert_called_once()
    warning_msg = mock_logger.warning.call_args[0][0]
    assert "failed with exit code 1" in warning_msg

def test_invalid_experiment_name():
    """Test that invalid experiment names raise an error."""
    with pytest.raises(ValueError, match="Invalid experiments.*nonexistent_experiment"):
        main(experiments="nonexistent_experiment", local=True)

@patch("spd.scripts.run.create_slurm_array_script")
@patch("spd.scripts.run.submit_slurm_array")
@patch("spd.scripts.run.create_git_snapshot")
@patch("spd.scripts.run.create_workspace_view")
@patch("spd.scripts.run.ensure_project_exists")
@patch("spd.scripts.run.generate_run_id")
@patch("spd.scripts.run.logger")
def test_slurm_submission(
    mock_logger,
    mock_generate_run_id,
    mock_ensure_project,
    mock_create_workspace,
    mock_create_git_snapshot,
    mock_submit_slurm,
    mock_create_slurm_script,
):
    """Test SLURM submission path (non-local execution)."""
    # Setup mocks
    mock_generate_run_id.return_value = "run_20231215_120000"
    mock_create_workspace.return_value = "https://wandb.ai/test/workspace"
    mock_create_git_snapshot.return_value = "snapshot-20231215-120000"
    mock_submit_slurm.return_value = "12345"
    
    # Call main without local flag (should submit to SLURM)
    main(experiments="tms_5-2-id", local=False)
    
    # Verify git snapshot was created
    mock_create_git_snapshot.assert_called_once_with(branch_name_prefix="run")
    
    # Verify SLURM script was created
    mock_create_slurm_script.assert_called_once()
    call_kwargs = mock_create_slurm_script.call_args[1]
    assert call_kwargs["job_name"] == "spd"
    assert call_kwargs["cpu"] is False
    assert call_kwargs["snapshot_branch"] == "snapshot-20231215-120000"
    assert len(call_kwargs["commands"]) == 1  # Single experiment
    
    # Verify SLURM job was submitted
    mock_submit_slurm.assert_called_once()

@patch("spd.scripts.run.subprocess.run")
@patch("spd.scripts.run.create_workspace_view")
@patch("spd.scripts.run.ensure_project_exists")
@patch("spd.scripts.run.generate_run_id")
@patch("spd.scripts.run.logger")
@patch("spd.scripts.run.create_wandb_report")
def test_report_creation_for_multiple_experiments(
    mock_create_report,
    mock_logger,
    mock_generate_run_id,
    mock_ensure_project,
    mock_create_workspace,
    mock_subprocess_run,
):
    """Test that a report is created when running multiple experiments."""
    # Setup mocks
    mock_generate_run_id.return_value = "run_20231215_120000"
    mock_create_workspace.return_value = "https://wandb.ai/test/workspace"
    mock_subprocess_run.return_value = Mock(returncode=0)
    mock_create_report.return_value = "https://wandb.ai/test/report"
    
    # Call main with multiple experiments
    main(experiments="tms_5-2,resid_mlp1", local=True, create_report=True)
    
    # Verify report was created
    mock_create_report.assert_called_once_with(
        "run_20231215_120000", ["tms_5-2", "resid_mlp1"], "spd"
    )

@patch("spd.scripts.run.subprocess.run")
@patch("spd.scripts.run.create_workspace_view")
@patch("spd.scripts.run.ensure_project_exists")
@patch("spd.scripts.run.generate_run_id")
@patch("spd.scripts.run.logger")
@patch("spd.scripts.run.create_wandb_report")
def test_no_report_for_single_experiment(
    mock_create_report,
    mock_logger,
    mock_generate_run_id,
    mock_ensure_project,
    mock_create_workspace,
    mock_subprocess_run,
):
    """Test that no report is created when running a single experiment."""
    # Setup mocks
    mock_generate_run_id.return_value = "run_20231215_120000"
    mock_create_workspace.return_value = "https://wandb.ai/test/workspace"
    mock_subprocess_run.return_value = Mock(returncode=0)
    
    # Call main with single experiment
    main(experiments="tms_5-2-id", local=True, create_report=True)
    
    # Verify report was NOT created
    mock_create_report.assert_not_called()


class TestParameterSweeps:
    """Test parameter sweep functionality."""

    @patch("spd.scripts.run.subprocess.run")
    @patch("spd.scripts.run.create_workspace_view")
    @patch("spd.scripts.run.ensure_project_exists")
    @patch("spd.scripts.run.generate_run_id")
    @patch("spd.scripts.run.logger")
    @patch("spd.scripts.run.load_sweep_params")
    def test_local_sweep_with_default_file(
        self,
        mock_load_sweep_params,
        mock_logger,
        mock_generate_run_id,
        mock_ensure_project,
        mock_create_workspace,
        mock_subprocess_run,
    ):
        """Test running a sweep with default sweep_params.yaml file."""
        # Setup mocks
        mock_generate_run_id.return_value = "run_20231215_120000"
        mock_create_workspace.return_value = "https://wandb.ai/test/workspace"
        mock_subprocess_run.return_value = Mock(returncode=0)
        
        # Mock sweep parameters
        mock_sweep_params = {
            "seed": {"values": [0, 1]},
            "learning_rate": {"values": [0.001, 0.01]}
        }
        mock_load_sweep_params.return_value = mock_sweep_params
        
        # Call main with sweep enabled
        main(experiments="tms_5-2-id", sweep=True, local=True)
        
        # Verify sweep params were loaded
        mock_load_sweep_params.assert_called_once()
        args = mock_load_sweep_params.call_args
        assert args[0][0] == "tms_5-2-id"  # experiment name
        assert args[0][1].name == "sweep_params.yaml"  # default file
        
        # Verify multiple commands were generated (2 seeds × 2 learning rates = 4 combinations)
        assert mock_subprocess_run.call_count == 4
        
        # Check that sweep_params_json is included in commands
        for call in mock_subprocess_run.call_args_list:
            call_args = call[0][0]
            assert "--sweep_params_json" in call_args
            
    @patch("spd.scripts.run.subprocess.run")
    @patch("spd.scripts.run.create_workspace_view")
    @patch("spd.scripts.run.ensure_project_exists")
    @patch("spd.scripts.run.generate_run_id")
    @patch("spd.scripts.run.logger")
    @patch("spd.scripts.run.load_sweep_params")
    def test_local_sweep_with_custom_file(
        self,
        mock_load_sweep_params,
        mock_logger,
        mock_generate_run_id,
        mock_ensure_project,
        mock_create_workspace,
        mock_subprocess_run,
    ):
        """Test running a sweep with custom sweep parameters file."""
        # Setup mocks
        mock_generate_run_id.return_value = "run_20231215_120000"
        mock_create_workspace.return_value = "https://wandb.ai/test/workspace"
        mock_subprocess_run.return_value = Mock(returncode=0)
        
        # Mock sweep parameters
        mock_sweep_params = {"seed": {"values": [100, 200]}}
        mock_load_sweep_params.return_value = mock_sweep_params
        
        # Call main with custom sweep file
        main(experiments="tms_5-2-id", sweep="custom_sweep.yaml", local=True)
        
        # Verify custom sweep file was used
        mock_load_sweep_params.assert_called_once()
        args = mock_load_sweep_params.call_args
        assert args[0][0] == "tms_5-2-id"
        assert args[0][1].name == "custom_sweep.yaml"
        
        # Verify correct number of commands (2 seed values)
        assert mock_subprocess_run.call_count == 2

    def test_sweep_requires_n_agents_for_slurm(self):
        """Test that sweeps require n_agents when not running locally."""
        with pytest.raises(AssertionError, match="n_agents must be provided"):
            main(experiments="tms_5-2-id", sweep=True, local=False)
            
    @patch("spd.scripts.run.subprocess.run")
    @patch("spd.scripts.run.create_workspace_view")
    @patch("spd.scripts.run.ensure_project_exists")
    @patch("spd.scripts.run.generate_run_id")
    @patch("spd.scripts.run.logger")
    @patch("spd.scripts.run.load_sweep_params")
    def test_sweep_allows_local_without_n_agents(
        self,
        mock_load_sweep_params,
        mock_logger,
        mock_generate_run_id,
        mock_ensure_project,
        mock_create_workspace,
        mock_subprocess_run,
    ):
        """Test that sweeps work locally without specifying n_agents."""
        # Setup mocks
        mock_generate_run_id.return_value = "run_20231215_120000"
        mock_create_workspace.return_value = "https://wandb.ai/test/workspace"
        mock_subprocess_run.return_value = Mock(returncode=0)
        mock_load_sweep_params.return_value = {"seed": {"values": [1]}}
        
        # Should not raise an error
        main(experiments="tms_5-2-id", sweep=True, local=True)
        
        # Verify it executed
        assert mock_subprocess_run.call_count == 1


class TestHelperFunctions:
    """Test individual helper functions."""
    
    def test_generate_grid_combinations_simple(self):
        """Test grid combination generation with simple parameters."""
        parameters = {
            "seed": {"values": [0, 1]},
            "learning_rate": {"values": [0.001, 0.01]}
        }
        
        combinations = generate_grid_combinations(parameters)
        
        assert len(combinations) == 4  # 2 × 2
        expected_combinations = [
            {"seed": 0, "learning_rate": 0.001},
            {"seed": 0, "learning_rate": 0.01},
            {"seed": 1, "learning_rate": 0.001},
            {"seed": 1, "learning_rate": 0.01}
        ]
        assert combinations == expected_combinations
        
    def test_generate_grid_combinations_nested(self):
        """Test grid combination generation with nested parameters."""
        parameters = {
            "loss": {
                "faithfulness_weight": {"values": [0.1, 0.5]}
            },
            "seed": {"values": [0]}
        }
        
        combinations = generate_grid_combinations(parameters)
        
        assert len(combinations) == 2  # 2 × 1
        expected_combinations = [
            {"loss.faithfulness_weight": 0.1, "seed": 0},
            {"loss.faithfulness_weight": 0.5, "seed": 0}
        ]
        assert combinations == expected_combinations
        
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
        
    def test_generate_run_id_format(self):
        """Test that generated run IDs have the expected format."""
        run_id = generate_run_id()
        
        # Should start with 'run_' and have timestamp format
        assert run_id.startswith("run_")
        assert len(run_id) == 19  # run_YYYYMMDD_HHMMSS
        
        # Should be parseable as a timestamp format
        timestamp_part = run_id[4:]  # Remove 'run_' prefix
        assert len(timestamp_part) == 15  # YYYYMMDD_HHMMSS
        
    @patch("spd.scripts.run.load_config")
    def test_generate_commands_without_sweep(self, mock_load_config):
        """Test generate_commands function without sweep parameters."""
        from spd.configs import Config
        
        # Mock config loading
        mock_config = Mock(spec=Config)
        mock_config.model_dump.return_value = {"wandb_project": "test"}
        mock_load_config.return_value = mock_config
        
        commands = generate_commands(
            experiments_list=["tms_5-2-id"],
            run_id="test_run_123",
            sweep_params_file=None,
            project="test-project"
        )
        
        assert len(commands) == 1
        command = commands[0]
        assert "python" in command
        assert "tms_decomposition.py" in command
        assert "--sweep_id test_run_123" in command
        assert "--evals_id tms_5-2-id" in command
        assert "json:" in command
        
    @patch("spd.scripts.run.load_config")
    @patch("spd.scripts.run.load_sweep_params")
    def test_generate_commands_with_sweep(self, mock_load_sweep_params, mock_load_config):
        """Test generate_commands function with sweep parameters."""
        from spd.configs import Config
        
        # Mock config loading
        mock_config = Mock(spec=Config)
        mock_config.model_dump.return_value = {"wandb_project": "test"}
        mock_load_config.return_value = mock_config
        
        # Mock sweep parameters
        mock_load_sweep_params.return_value = {"seed": {"values": [0, 1]}}
        
        commands = generate_commands(
            experiments_list=["tms_5-2-id"],
            run_id="test_run_123",
            sweep_params_file="sweep_params.yaml",
            project="test-project"
        )
        
        assert len(commands) == 2  # Two seed values
        for command in commands:
            assert "python" in command
            assert "tms_decomposition.py" in command
            assert "--sweep_id test_run_123" in command
            assert "--evals_id tms_5-2-id" in command
            assert "--sweep_params_json" in command


class TestEdgeCases:
    """Test edge cases and additional parameters."""
    
    @patch("spd.scripts.run.subprocess.run")
    @patch("spd.scripts.run.create_workspace_view")
    @patch("spd.scripts.run.ensure_project_exists")
    @patch("spd.scripts.run.generate_run_id")
    @patch("spd.scripts.run.logger")
    def test_run_all_experiments(
        self,
        mock_logger,
        mock_generate_run_id,
        mock_ensure_project,
        mock_create_workspace,
        mock_subprocess_run,
    ):
        """Test running all experiments when experiments=None."""
        # Setup mocks
        mock_generate_run_id.return_value = "run_20231215_120000"
        mock_create_workspace.return_value = "https://wandb.ai/test/workspace"
        mock_subprocess_run.return_value = Mock(returncode=0)
        
        # Call main without specifying experiments (should run all)
        main(experiments=None, local=True)
        
        # Should run all experiments in the registry (currently 6 active experiments)
        # tms_5-2, tms_5-2-id, tms_40-10, tms_40-10-id, resid_mlp1, resid_mlp2, resid_mlp3
        expected_count = 7  # Based on current registry
        assert mock_subprocess_run.call_count == expected_count
        
    def test_n_agents_validation_non_sweep(self):
        """Test that n_agents defaults correctly for non-sweep runs."""
        # Should not raise an error for non-sweep runs without n_agents
        try:
            main(experiments="tms_5-2-id", local=False, sweep=False)
        except Exception as e:
            # Should not be an n_agents validation error
            assert "n_agents must be provided" not in str(e)
            
    @patch("spd.scripts.run.create_slurm_array_script")
    @patch("spd.scripts.run.submit_slurm_array")
    @patch("spd.scripts.run.create_git_snapshot")
    @patch("spd.scripts.run.create_workspace_view")
    @patch("spd.scripts.run.ensure_project_exists")
    @patch("spd.scripts.run.generate_run_id")
    @patch("spd.scripts.run.logger")
    def test_job_suffix_and_cpu_flags(
        self,
        mock_logger,
        mock_generate_run_id,
        mock_ensure_project,
        mock_create_workspace,
        mock_create_git_snapshot,
        mock_submit_slurm,
        mock_create_slurm_script,
    ):
        """Test job_suffix and cpu flag parameters."""
        # Setup mocks
        mock_generate_run_id.return_value = "run_20231215_120000"
        mock_create_workspace.return_value = "https://wandb.ai/test/workspace"
        mock_create_git_snapshot.return_value = "snapshot-20231215-120000"
        mock_submit_slurm.return_value = "12345"
        
        # Call main with job_suffix and cpu flags
        main(
            experiments="tms_5-2-id",
            local=False,
            job_suffix="test-suffix",
            cpu=True
        )
        
        # Verify SLURM script was created with correct parameters
        mock_create_slurm_script.assert_called_once()
        call_kwargs = mock_create_slurm_script.call_args[1]
        assert call_kwargs["job_name"] == "spd-test-suffix"
        assert call_kwargs["cpu"] is True
        
    @patch("spd.scripts.run.subprocess.run")
    @patch("spd.scripts.run.create_workspace_view")
    @patch("spd.scripts.run.ensure_project_exists")
    @patch("spd.scripts.run.generate_run_id")
    @patch("spd.scripts.run.logger")
    def test_log_format_parameter(
        self,
        mock_logger,
        mock_generate_run_id,
        mock_ensure_project,
        mock_create_workspace,
        mock_subprocess_run,
    ):
        """Test log_format parameter is passed to logger."""
        # Setup mocks
        mock_generate_run_id.return_value = "run_20231215_120000"
        mock_create_workspace.return_value = "https://wandb.ai/test/workspace"
        mock_subprocess_run.return_value = Mock(returncode=0)
        
        # Call main with terse log format
        main(experiments="tms_5-2-id", local=True, log_format="terse")
        
        # Verify logger.set_format was called with terse format
        mock_logger.set_format.assert_called_with("console", "terse")
        
    @patch("spd.scripts.run.subprocess.run")
    @patch("spd.scripts.run.create_workspace_view")
    @patch("spd.scripts.run.ensure_project_exists")
    @patch("spd.scripts.run.generate_run_id")
    @patch("spd.scripts.run.logger")
    @patch("spd.scripts.run.create_wandb_report")
    def test_create_report_false(
        self,
        mock_create_report,
        mock_logger,
        mock_generate_run_id,
        mock_ensure_project,
        mock_create_workspace,
        mock_subprocess_run,
    ):
        """Test that no report is created when create_report=False."""
        # Setup mocks
        mock_generate_run_id.return_value = "run_20231215_120000"
        mock_create_workspace.return_value = "https://wandb.ai/test/workspace"
        mock_subprocess_run.return_value = Mock(returncode=0)
        
        # Call main with multiple experiments but create_report=False
        main(
            experiments="tms_5-2,resid_mlp1",
            local=True,
            create_report=False
        )
        
        # Verify report was NOT created even with multiple experiments
        mock_create_report.assert_not_called()