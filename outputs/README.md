# Regicide Training Outputs

This directory contains organized outputs from Regicide training experiments.

## Structure

- **models/**: Global model storage (not experiment-specific)
- **plots/**: Global plot storage (not experiment-specific)  
- **checkpoints/**: Global checkpoint storage
- **logs/**: Global log storage
- **[experiment_name]/**: Individual experiment folders containing:
  - `models/`: Final trained models for this experiment
  - `plots/`: Training progress plots and visualizations
  - `checkpoints/`: Periodic model checkpoints during training
  - `logs/`: Training logs and debugging info
  - `experiment_summary.json`: Summary of experiment configuration and results

## Experiments

Each training run creates a new experiment folder with a timestamp.
You can also specify custom experiment names when creating a trainer.

## Usage

```python
from config import PathManager

# Create new experiment
path_manager = PathManager("my_experiment")

# Or use default timestamped name
path_manager = PathManager()

# Get organized paths
model_path = path_manager.get_model_path("my_model.pth")
plot_path = path_manager.get_plot_path("training_plot.png")
```
