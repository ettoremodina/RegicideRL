---
name: ml-project-initializer
description: Use this skill when initializing or structuring a new Machine Learning or Reinforcement Learning project. It enforces the user's preferred directory structure, logging setup, configuration management, and autotraining orchestration.
---

# ML Project Initializer Guidelines

When starting a new Machine Learning or Reinforcement Learning project for this user, you MUST adhere to the following architectural patterns and preferences. The goal is to avoid hard-coded parameters, scattered logs, and manual pipelines.

## 1. Directory Organization
Always establish the following folder structure:
- `models/`: Where all final `.zip` or `.pt` model artifacts go.
- `runs/` or `experiments/`: Where training logs (like TensorBoard events) and experiment artifacts go.
- `solvers/` (or `src/`): Where all source code lives.
  - `analysis/`: Scripts specifically for parsing logs, plotting metrics, and generating reports.
  - `agents/` or `models/`: Network architectures and policy wrappers.

## 2. Configuration Management
**Never hardcode hyperparameters.**
- Create a `config.yaml` in the project root.
- The configuration file must dictate:
  - Model Hyperparameters (Learning rate, batch size, epochs, etc.)
  - Environment Settings (Rewards, penalties, observation shapes)
  - Training directvies (Total timesteps, eval frequencies, seed)
- Ensure all main scripts (training, evaluation, tuning) read from this centralized config file. Use `pyyaml` (or `omegaconf`).

## 3. Custom Logging and Callbacks
Default ML frameworks often omit domain-specific metrics.
- Always implement custom logging callbacks (e.g., `stable_baselines3.common.callbacks.BaseCallback` or custom PyTorch hooks).
- Log rich, domain-specific metrics to TensorBoard (e.g., win rates, action usage, invalid actions, scenario occurrences).
- The goal is to make TensorBoard the absolute source of truth during training.

## 4. The Autotraining Pipeline (Orchestrator)
The user requires an automated pipeline for rapid iteration. Implement an `orchestrator.py` that:
1. Creates a timestamped experiment folder (e.g., `experiments/trial_YYYYMMDD_HHMMSS/`).
2. Backs up a copy of the current `config.yaml` to that folder for reproducibility.
3. Spawns the training script (`train.py`) programmatically.
4. Spawns the analysis/evaluation script (`evaluate.py`) programmatically.
5. Saves all outputs (plots, reports, logs) to the timestamped folder.

## 5. Hyperparameter Optimization
Implement a `tune.py` script using `optuna`.
- Define a search space for the hyperparameters.
- Run multiple trials automatically.
- Evaluate the trials using the custom metrics and objective function.
- Save the best configuration found to a new YAML file.

## 6. Plotters and Visualizations
Always include visualization scripts.
- Use `matplotlib` and `seaborn`.
- Generate dashboards (e.g., `comprehensive_dashboard.png`) that summarize the training curve, evaluation metrics, action distributions, and model entropy in one image.

By following this skill, you ensure the user has a professional, scalable, and fully automated ML development environment from day one.
