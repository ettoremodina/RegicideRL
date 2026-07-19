"""Move legacy generated directories under the canonical artifacts root."""

import argparse
import shutil
from datetime import datetime, timezone
from pathlib import Path

from ml_logger import get_logger, start_run

logger = get_logger(__name__)
LEGACY_DIRECTORIES = (
    "runs",
    "logs",
    "models",
    "outputs",
    "experiments",
    "archive",
)
LEGACY_FILES = ("ui_game_log.txt",)


def migrate_legacy_artifacts(workspace=".", artifacts_dir="artifacts"):
    """Move known legacy outputs without overwriting or deleting data."""
    workspace_path = Path(workspace).resolve()
    artifacts_path = (workspace_path / artifacts_dir).resolve()
    if artifacts_path == workspace_path or workspace_path not in artifacts_path.parents:
        raise ValueError("Artifacts directory must be inside the workspace")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    destination = artifacts_path / "legacy" / timestamp
    moved = _promote_models(workspace_path, artifacts_path)
    for relative_path in (*LEGACY_DIRECTORIES, *LEGACY_FILES):
        source = (workspace_path / relative_path).resolve()
        if not source.exists() or source == artifacts_path:
            continue
        target = destination / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source), str(target))
        moved.append({"source": str(source), "target": str(target)})
        logger.info("Moved %s to %s", source, target)
    return moved


def _promote_models(workspace_path, artifacts_path):
    """Move reusable legacy model archives into ``promoted_models``."""
    models_dir = workspace_path / "models"
    if not models_dir.exists():
        return []
    promoted = []
    destination = artifacts_path / "promoted_models"
    destination.mkdir(parents=True, exist_ok=True)
    for model_path in models_dir.glob("*.zip"):
        target = destination / model_path.name
        if target.exists():
            logger.warning("Promoted model already exists, preserving %s", model_path)
            continue
        shutil.move(str(model_path), str(target))
        promoted.append({"source": str(model_path), "target": str(target)})
        logger.info("Promoted reusable model %s to %s", model_path, target)
    try:
        models_dir.rmdir()
    except OSError:
        pass
    return promoted


def main():
    """Migrate legacy artifact directories using command-line paths."""
    parser = argparse.ArgumentParser(description="Migrate legacy Regicide artifacts")
    parser.add_argument("--workspace", default=".")
    parser.add_argument("--artifacts-dir", default="artifacts")
    args = parser.parse_args()
    context = start_run(
        "artifact-migration",
        config=vars(args),
        root_dir=Path(args.workspace) / args.artifacts_dir,
    )
    try:
        moved = migrate_legacy_artifacts(args.workspace, args.artifacts_dir)
        report = context.save_result("migration.json", {"moved": moved})
        context.complete({"moved": len(moved), "report": str(report)})
    except Exception as error:
        context.fail(error)
        logger.exception("Artifact migration failed")
        raise


if __name__ == "__main__":
    main()
