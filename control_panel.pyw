"""Double-click launcher that silently selects the repository virtual environment."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def main() -> None:
    """Start the control panel with the repository's pythonw executable."""
    repository_root = Path(__file__).resolve().parent
    os.chdir(repository_root)
    virtual_python = repository_root / "venv" / "Scripts" / "pythonw.exe"
    current = Path(sys.executable).resolve()
    if virtual_python.exists() and current != virtual_python.resolve():
        subprocess.Popen(
            [str(virtual_python), "-m", "control_panel"],
            cwd=repository_root,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        return
    from control_panel.app import run_control_panel

    run_control_panel(repository_root)


if __name__ == "__main__":
    main()
