"""Allowlisted repository commands exposed through structured browser forms."""

from __future__ import annotations

from .models import CommandSpec, ParameterSpec

AGENTS = ("random", "heuristic", "pimc", "ismcts", "ppo")
REPORT_AGENTS = ("random", "heuristic", "pimc", "ismcts", "ppo", "alphazero")
CONFIGS = ("config_test.yaml", "config.yaml")


def command_catalog() -> tuple[CommandSpec, ...]:
    """Return the complete immutable command catalog for this repository."""
    return (
        CommandSpec(
            "play-game",
            "Play Regicide",
            "Play",
            "Open the existing Pygame client in its own window.",
            ("-m", "ui"),
            risk="desktop",
            source="ui/__main__.py",
            quick_action=True,
            tags=("game", "pygame"),
        ),
        CommandSpec(
            "simulate-game",
            "Simulate one recorded game",
            "Play",
            "Run one random solo game and save its replay and summary.",
            ("-m", "scripts.log_game"),
            source="scripts/log_game.py",
            quick_action=True,
            tags=("game", "replay", "smoke"),
        ),
        _benchmark_command(),
        _evaluation_command(),
        _alphazero_command(),
        _config_command(
            "ppo-training",
            "Train PPO",
            "Train",
            "Train MaskablePPO with the selected repository configuration.",
            "solvers.train_rl",
            "solvers/train_rl.py",
            risk="heavy",
        ),
        _config_command(
            "ppo-pipeline",
            "PPO experiment pipeline",
            "Train",
            "Run PPO training followed by the existing policy analysis pipeline.",
            "solvers.orchestrator",
            "solvers/orchestrator.py",
            risk="heavy",
        ),
        _config_command(
            "hyperparameter-tuning",
            "Tune PPO hyperparameters",
            "Train",
            "Launch the Optuna tuning workflow with a bounded config profile.",
            "solvers.tune",
            "solvers/tune.py",
            risk="heavy",
        ),
        _bc_dataset_command(),
        _bc_training_command(),
        _policy_analysis_command(),
        _experimental_report_command(),
        _resume_report_command(),
        _raw_comparison_command(),
        _report_regeneration_command(),
        CommandSpec(
            "aggregate-runs",
            "Aggregate recorded games",
            "Analyze",
            "Compute aggregate results for every run or one selected run.",
            ("-m", "scripts.analyze_runs", "--artifacts-dir", "artifacts"),
            parameters=(
                ParameterSpec(
                    "run_id",
                    "Run ID",
                    "--run-id",
                    kind="identifier",
                    help="Leave empty to aggregate the full catalog.",
                ),
            ),
            source="scripts/analyze_runs.py",
            quick_action=True,
            tags=("analysis", "games"),
        ),
        CommandSpec(
            "action-space-analysis",
            "Inspect action space",
            "Analyze",
            "Persist the current global action-space dimensions and categories.",
            ("-m", "scripts.action_space_analyzer"),
            source="scripts/action_space_analyzer.py",
            tags=("diagnostic", "actions"),
        ),
        CommandSpec(
            "generate-docs",
            "Generate API documentation",
            "Quality",
            "Rebuild docs/api with pdoc while preserving curated documents.",
            ("-m", "scripts.generate_docs"),
            risk="maintenance",
            confirmation="This replaces the generated docs/api directory.",
            source="scripts/generate_docs.py",
            tags=("docs", "pdoc"),
        ),
        _test_command(
            "test-suite",
            "Run full test suite",
            (),
            "Run every repository test with compact output.",
            quick=True,
        ),
        _test_command(
            "test-game",
            "Test rules and agents",
            (
                "tests/test_game_rules.py",
                "tests/test_solo_rules.py",
                "tests/test_action_handler.py",
                "tests/test_env.py",
                "tests/test_alphazero.py",
            ),
            "Run the game, action-space, environment, and AlphaZero tests.",
        ),
        _test_command(
            "test-logger",
            "Test logger and storage",
            (
                "tests/test_ml_logger.py",
                "tests/test_ml_logger_runtime.py",
                "tests/test_ml_logger_dashboard.py",
                "tests/test_logger_config.py",
                "tests/test_logging_policy.py",
                "tests/test_terminal_logging.py",
            ),
            "Run the logging, catalog, dashboard, and runtime tests.",
        ),
        CommandSpec(
            "migrate-artifacts",
            "Migrate legacy artifacts",
            "Maintenance",
            "Move supported legacy output folders into the canonical artifact tree.",
            (
                "-m",
                "scripts.migrate_artifacts",
                "--workspace",
                ".",
                "--artifacts-dir",
                "artifacts",
            ),
            risk="maintenance",
            confirmation=(
                "This moves legacy output paths. Existing files are preserved, "
                "but their locations can change."
            ),
            source="scripts/migrate_artifacts.py",
            tags=("migration", "artifacts"),
        ),
    )


def command_map() -> dict[str, CommandSpec]:
    """Return command specifications keyed by their stable identifier."""
    return {command.command_id: command for command in command_catalog()}


def _benchmark_command() -> CommandSpec:
    """Build the benchmark command with safe workload bounds."""
    return CommandSpec(
        "benchmark",
        "Run benchmark",
        "Evaluate",
        "Benchmark rules, environment, parallel simulation, or training throughput.",
        ("benchmark.py",),
        parameters=(
            ParameterSpec(
                "mode",
                "Mode",
                "--mode",
                kind="choice",
                default="normal",
                choices=("normal", "env", "parallel", "cpu", "gpu", "all"),
            ),
            ParameterSpec(
                "games",
                "Games",
                "--games",
                kind="integer",
                default=100,
                minimum=1,
                maximum=1_000_000,
            ),
            ParameterSpec(
                "steps",
                "Training steps",
                "--steps",
                kind="integer",
                default=10_000,
                minimum=1,
                maximum=100_000_000,
            ),
            ParameterSpec(
                "jobs",
                "Workers",
                "--jobs",
                kind="integer",
                minimum=1,
                maximum=64,
                help="Leave empty to use the benchmark default.",
            ),
        ),
        risk="heavy",
        confirmation="CPU, GPU, and all modes can consume substantial resources.",
        source="benchmark.py",
        quick_action=True,
        tags=("performance", "evaluation"),
    )


def _evaluation_command() -> CommandSpec:
    """Build the multi-agent evaluation command."""
    return CommandSpec(
        "agent-evaluation",
        "Evaluate an agent",
        "Evaluate",
        "Run repeated games for a selected agent and persist metrics and plots.",
        ("-m", "solvers.train"),
        parameters=(
            ParameterSpec(
                "agent",
                "Agent",
                "--agent",
                kind="choice",
                default="random",
                choices=AGENTS,
            ),
            ParameterSpec(
                "episodes",
                "Evaluation loops",
                "--episodes",
                kind="integer",
                default=1,
                minimum=1,
                maximum=10_000,
            ),
            ParameterSpec(
                "games_per_episode",
                "Games per loop",
                "--games-per-episode",
                kind="integer",
                default=100,
                minimum=1,
                maximum=1_000_000,
            ),
            ParameterSpec(
                "jobs",
                "Workers",
                "--jobs",
                kind="integer",
                default=1,
                minimum=1,
                maximum=64,
            ),
            ParameterSpec(
                "iterations",
                "ISMCTS iterations",
                "--iterations",
                kind="integer",
                default=1000,
                minimum=1,
                maximum=10_000_000,
            ),
            ParameterSpec(
                "exploration",
                "ISMCTS exploration",
                "--exploration",
                kind="number",
                default=10.0,
                minimum=0,
                maximum=1000,
            ),
        ),
        risk="heavy",
        confirmation="Search agents and large game counts can run for a long time.",
        source="solvers/train.py",
        quick_action=True,
        tags=("agents", "metrics", "evaluation"),
    )


def _alphazero_command() -> CommandSpec:
    """Build the AlphaZero training command with smoke-safe defaults."""
    return CommandSpec(
        "alphazero-training",
        "Train AlphaZero",
        "Train",
        "Run self-play, network training, evaluation, and checkpoint promotion.",
        ("-m", "solvers.train", "--agent", "alphazero"),
        parameters=(
            _config_parameter(),
            ParameterSpec(
                "az_iterations",
                "Training iterations",
                "--az-iterations",
                kind="integer",
                default=1,
                minimum=1,
                maximum=100_000,
            ),
            ParameterSpec(
                "games_per_episode",
                "Self-play games",
                "--games-per-episode",
                kind="integer",
                default=1,
                minimum=1,
                maximum=1_000_000,
            ),
            ParameterSpec(
                "sims",
                "MCTS simulations",
                "--sims",
                kind="integer",
                default=1,
                minimum=1,
                maximum=1_000_000,
            ),
            ParameterSpec(
                "eval_games",
                "Evaluation games",
                "--eval-games",
                kind="integer",
                default=1,
                minimum=1,
                maximum=1_000_000,
            ),
            ParameterSpec(
                "device",
                "Device",
                "--device",
                kind="choice",
                default="cpu",
                choices=("cpu", "cuda"),
            ),
            ParameterSpec(
                "resume",
                "Resume checkpoint",
                "--resume",
                kind="path",
                path_mode="repo_path",
                placeholder="artifacts/runs/.../checkpoints/checkpoint",
            ),
        ),
        risk="heavy",
        confirmation=(
            "AlphaZero can consume significant CPU/GPU, memory, and disk. "
            "Defaults are intentionally a smoke run."
        ),
        source="solvers/train.py",
        tags=("alphazero", "self-play", "gpu"),
    )


def _config_command(
    command_id: str,
    title: str,
    category: str,
    description: str,
    module: str,
    source: str,
    risk: str,
) -> CommandSpec:
    """Build a command whose only variable input is a config profile."""
    return CommandSpec(
        command_id,
        title,
        category,
        description,
        ("-m", module),
        parameters=(_config_parameter(),),
        risk=risk,
        confirmation="The full profile can run for a long time.",
        source=source,
        tags=("config", "training"),
    )


def _config_parameter() -> ParameterSpec:
    """Return the shared repository-config selector."""
    return ParameterSpec(
        "config",
        "Configuration",
        "--config",
        kind="choice",
        default="config_test.yaml",
        choices=CONFIGS,
        help="The smoke profile is selected by default.",
    )


def _bc_dataset_command() -> CommandSpec:
    """Build the behavioral-cloning dataset generator command."""
    return CommandSpec(
        "bc-dataset",
        "Generate BC dataset",
        "Train",
        "Create a behavioral-cloning dataset from heuristic play.",
        ("-m", "solvers.generate_bc_data"),
        parameters=(
            ParameterSpec(
                "games",
                "Games",
                "--games",
                kind="integer",
                default=50,
                minimum=1,
                maximum=10_000_000,
            ),
            ParameterSpec(
                "output",
                "Output dataset",
                "--out",
                kind="path",
                default="artifacts/datasets/bc_data_control_panel.npz",
                path_mode="artifact_output",
            ),
        ),
        risk="heavy",
        confirmation="Large game counts can create sizeable datasets.",
        source="solvers/generate_bc_data.py",
        tags=("dataset", "behavioral-cloning"),
    )


def _bc_training_command() -> CommandSpec:
    """Build the behavioral-cloning pre-training command."""
    return CommandSpec(
        "bc-training",
        "Train from BC dataset",
        "Train",
        "Pre-train a PPO policy from an existing NPZ dataset.",
        ("-m", "solvers.bc_train"),
        parameters=(
            ParameterSpec(
                "data",
                "Dataset",
                "--data",
                kind="path",
                required=True,
                path_mode="existing_file",
                placeholder="artifacts/datasets/example.npz",
            ),
            _config_parameter(),
            ParameterSpec(
                "epochs",
                "Epochs",
                "--epochs",
                kind="integer",
                default=1,
                minimum=1,
                maximum=100_000,
            ),
            ParameterSpec(
                "batch",
                "Batch size",
                "--batch",
                kind="integer",
                default=64,
                minimum=1,
                maximum=1_000_000,
            ),
        ),
        risk="heavy",
        confirmation="Training can consume substantial compute resources.",
        source="solvers/bc_train.py",
        tags=("behavioral-cloning", "training"),
    )


def _policy_analysis_command() -> CommandSpec:
    """Build the trained PPO policy analysis command."""
    return CommandSpec(
        "policy-analysis",
        "Analyze trained policy",
        "Analyze",
        "Probe a trained PPO model and generate the existing analysis dashboard.",
        ("-m", "solvers.analysis.run_analysis"),
        parameters=(
            ParameterSpec(
                "model",
                "Model ZIP",
                "--model",
                kind="path",
                required=True,
                path_mode="existing_file",
                placeholder="artifacts/promoted_models/ppo_regicide.zip",
            ),
            ParameterSpec(
                "games",
                "Evaluation games",
                "--games",
                kind="integer",
                default=10,
                minimum=1,
                maximum=1_000_000,
            ),
            ParameterSpec(
                "logdir",
                "TensorBoard log directory",
                "--logdir",
                kind="path",
                path_mode="existing_dir",
            ),
        ),
        risk="heavy",
        confirmation="Policy probing runs full games and can take time.",
        source="solvers/analysis/run_analysis.py",
        tags=("ppo", "analysis", "plots"),
    )


def _experimental_report_command() -> CommandSpec:
    """Build the complete reproducible comparison pipeline command."""
    return CommandSpec(
        "experimental-report",
        "Build comparison report",
        "Analyze",
        "Evaluate selected agents on shared seeds and generate all statistics and plots.",
        ("-m", "scripts.experimental_report.orchestrator"),
        parameters=_report_parameters(include_resume=False),
        risk="heavy",
        confirmation=(
            "Search agents are expensive. The default selection avoids missing "
            "learned-model files."
        ),
        source="scripts/experimental_report/orchestrator.py",
        quick_action=True,
        tags=("comparison", "statistics", "report"),
    )


def _raw_comparison_command() -> CommandSpec:
    """Build the comparison-only runner without analysis generation."""
    return CommandSpec(
        "raw-comparison",
        "Run raw agent comparison",
        "Analyze",
        "Produce the shared-seed games dataset without regenerating the report.",
        ("-m", "scripts.experimental_report.runner"),
        parameters=_report_parameters(include_resume=False),
        risk="heavy",
        confirmation="Search agents and large samples can run for a long time.",
        source="scripts/experimental_report/runner.py",
        tags=("comparison", "dataset"),
    )


def _resume_report_command() -> CommandSpec:
    """Build the checkpoint-safe experimental report resume command."""
    return CommandSpec(
        "resume-experimental-report",
        "Resume comparison report",
        "Analyze",
        "Continue an interrupted comparison without repeating completed agent/seed pairs.",
        ("-m", "scripts.experimental_report.orchestrator"),
        parameters=(
            ParameterSpec(
                "resume_run",
                "Run directory",
                "--resume-run",
                kind="path",
                required=True,
                path_mode="existing_dir",
                placeholder="artifacts/runs/YYYY-MM-DD/experimental-report-...",
            ),
            ParameterSpec(
                "jobs",
                "Workers",
                "--jobs",
                kind="integer",
                default=1,
                minimum=1,
                maximum=64,
            ),
        ),
        risk="heavy",
        confirmation="The selected run will be reopened and its analysis outputs updated.",
        source="scripts/experimental_report/orchestrator.py",
        tags=("resume", "comparison", "report"),
    )


def _report_parameters(include_resume: bool) -> tuple[ParameterSpec, ...]:
    """Return shared experimental-report fields."""
    parameters = [
        ParameterSpec(
            "config",
            "Configuration",
            "--config",
            kind="choice",
            default="config.yaml",
            choices=CONFIGS,
        ),
        ParameterSpec(
            "agents",
            "Agents",
            "--agents",
            kind="multi_choice",
            default=["random", "heuristic"],
            choices=REPORT_AGENTS,
        ),
        ParameterSpec(
            "games",
            "Games per agent",
            "--games",
            kind="integer",
            default=10,
            minimum=1,
            maximum=1_000_000,
        ),
        ParameterSpec(
            "base_seed",
            "Base seed",
            "--base-seed",
            kind="integer",
            default=42,
            minimum=0,
            maximum=2_147_483_647,
        ),
        ParameterSpec(
            "jobs",
            "Workers",
            "--jobs",
            kind="integer",
            default=1,
            minimum=1,
            maximum=64,
        ),
    ]
    if include_resume:
        parameters.append(
            ParameterSpec(
                "resume_run",
                "Resume run directory",
                "--resume-run",
                kind="path",
                path_mode="existing_dir",
                placeholder="artifacts/runs/YYYY-MM-DD/experimental-report-...",
                help="When set, clear agents, games, and base seed in the form.",
            )
        )
    return tuple(parameters)


def _report_regeneration_command() -> CommandSpec:
    """Build the report-only regeneration command."""
    return CommandSpec(
        "regenerate-report",
        "Regenerate experiment report",
        "Analyze",
        "Rebuild statistics, tables, plots, and Markdown from an existing run.",
        ("-m", "scripts.experimental_report.analysis"),
        parameters=(
            ParameterSpec(
                "run_dir",
                "Run directory",
                kind="path",
                required=True,
                positional=True,
                path_mode="existing_dir",
                placeholder="artifacts/runs/YYYY-MM-DD/experimental-report-...",
            ),
            ParameterSpec(
                "config",
                "Optional configuration",
                "--config",
                kind="choice",
                choices=CONFIGS,
            ),
        ),
        risk="maintenance",
        confirmation="Existing files in the run's analysis directory may be replaced.",
        source="scripts/experimental_report/analysis.py",
        tags=("report", "plots", "statistics"),
    )


def _test_command(
    command_id: str,
    title: str,
    targets: tuple[str, ...],
    description: str,
    quick: bool = False,
) -> CommandSpec:
    """Build one fixed-scope pytest command."""
    return CommandSpec(
        command_id,
        title,
        "Quality",
        description,
        ("-m", "pytest", "-q", *targets),
        risk="standard",
        source="tests/",
        creates_run=False,
        quick_action=quick,
        tags=("tests", "quality"),
    )
