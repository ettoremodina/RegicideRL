"""Build the complete pdoc API reference without touching curated docs."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable

from ml_logger import get_logger, run_scope

logger = get_logger(__name__)

DOCUMENTED_MODULES = (
    "game",
    # ``agents`` exposes classes lazily through ``__getattr__``. Listing the
    # implementation modules explicitly lets pdoc resolve every class without
    # forcing optional ML dependencies to load when the package is imported.
    "agents.alphazero_agent",
    "agents.base_agent",
    "agents.determinize",
    "agents.heuristic_agent",
    "agents.ismcts_agent",
    "agents.pimc_agent",
    "agents.ppo_agent",
    "agents.random_agent",
    "solvers",
    "ml_logger",
    "integrations",
    "scripts",
    "ui",
    "benchmark",
)
DEFAULT_OUTPUT_DIRECTORY = Path("docs") / "api"


class DocumentationBuildWarning(RuntimeError):
    """Signal that pdoc rendered pages but emitted diagnostic warnings."""


def build_documentation(
    output_directory: str | Path = DEFAULT_OUTPUT_DIRECTORY,
    modules: Iterable[str] = DOCUMENTED_MODULES,
) -> Path:
    """Generate API pages for the selected modules in an isolated directory.

    Existing generated pages are removed first so renamed modules cannot leave
    stale HTML behind. Only the selected output directory is replaced; curated
    Markdown files elsewhere under ``docs`` are not modified.

    Args:
        output_directory: Directory that will contain the generated HTML.
        modules: Importable modules or packages passed to pdoc.

    Returns:
        Resolved output directory containing the generated reference.

    Raises:
        DocumentationBuildWarning: If pdoc emits any diagnostic warning.
        subprocess.CalledProcessError: If pdoc cannot import or render a module.
    """
    output_path = Path(output_directory)
    if output_path.exists():
        shutil.rmtree(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-m",
        "pdoc",
        *tuple(modules),
        "-o",
        str(output_path),
        "--docformat",
        "google",
    ]
    completed = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
    )
    if completed.stdout.strip():
        logger.info("pdoc: %s", completed.stdout.strip())
    if completed.stderr.strip():
        raise DocumentationBuildWarning(completed.stderr.strip())
    return output_path.resolve()


def main() -> None:
    """Generate the project API reference and record the build as a run."""
    with run_scope("documentation") as context:
        logger.info("Generating documentation with pdoc")
        output_directory = build_documentation()
        result = {
            "status": "completed",
            "output_dir": str(output_directory),
            "modules": list(DOCUMENTED_MODULES),
        }
        context.save_result("documentation.json", result)
        context.log_summary(result)
        logger.info("Documentation generated in %s", output_directory)


if __name__ == "__main__":
    main()
