"""Checkpoint path helpers shared by training and deployment."""

from pathlib import Path


def checkpoint_file(path: str | Path) -> Path:
    """Return a checkpoint path with exactly one ``.pt`` suffix."""
    checkpoint_path = Path(path)
    if checkpoint_path.suffix == ".pt":
        return checkpoint_path
    return Path(f"{checkpoint_path}.pt")
