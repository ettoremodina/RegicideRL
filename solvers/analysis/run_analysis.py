"""End-to-end policy analysis saved under a canonical run."""

import argparse
from pathlib import Path

from ml_logger import GameRecorder, RunContext, get_logger, start_run
from solvers.analysis.plotter import plot_dashboard
from solvers.analysis.probe import probe_policy
from solvers.analysis.reporter import generate_reports
from solvers.analysis.tb_extractor import extract_tb_logs, find_latest_run

logger = get_logger(__name__)


def run_analysis_pipeline(
    model_path,
    num_games=50,
    logdir=None,
    out_dir=None,
    run_context: RunContext | None = None,
):
    """Probe a model and persist reports, plots, and recorded games."""
    owns_context = run_context is None
    context = run_context or start_run(
        "policy-analysis",
        config={"model_path": model_path, "num_games": num_games, "logdir": logdir},
    )
    output_path = Path(out_dir) if out_dir else context.run_dir / "analysis"
    output_path.mkdir(parents=True, exist_ok=True)
    logger.info("Analysis output directory: %s", output_path)
    tensorboard_dir = logdir if logdir else find_latest_run()
    tensorboard_data = extract_tb_logs(tensorboard_dir)
    recorder = GameRecorder(context)
    probe_results = probe_policy(model_path, num_games, recorder=recorder)
    if not probe_results:
        if owns_context:
            context.fail("probing_error")
        logger.error("Analysis failed due to probing error")
        return False
    summary = generate_reports(probe_results, output_path)
    plot_dashboard(probe_results, tensorboard_data, output_path)
    context.save_result("policy_summary.json", summary)
    if owns_context:
        context.complete({"model_path": model_path})
    logger.info("Analysis complete; results saved to %s", output_path)
    return True


def main():
    parser = argparse.ArgumentParser(description="Analyze a trained Regicide policy")
    parser.add_argument("--model", required=True, help="Path to the trained .zip model")
    parser.add_argument("--games", type=int, default=50)
    parser.add_argument("--logdir")
    args = parser.parse_args()
    run_analysis_pipeline(args.model, args.games, args.logdir)


if __name__ == "__main__":
    main()
