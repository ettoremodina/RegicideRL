import os
import argparse
from datetime import datetime
from solvers.analysis.tb_extractor import find_latest_run, extract_tb_logs
from solvers.analysis.probe import probe_policy
from solvers.analysis.reporter import generate_reports
from solvers.analysis.plotter import plot_dashboard

def run_analysis_pipeline(model_path, num_games=50, logdir=None, out_dir=None):
    if out_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = f"policy_analysis_{timestamp}"
        
    os.makedirs(out_dir, exist_ok=True)
    print(f"Created analysis directory: {out_dir}")
    
    # Extract TensorBoard Logs
    logdir_to_use = logdir if logdir else find_latest_run()
    tb_data = extract_tb_logs(logdir_to_use)
    
    # Probe the Policy
    probe_results = probe_policy(model_path, num_games=num_games)
    
    if probe_results:
        # Generate Reports
        generate_reports(probe_results, out_dir)
        # Generate Plots
        plot_dashboard(probe_results, tb_data, out_dir)
        print(f"\n✅ Analysis complete! Check the '{out_dir}' folder for results.")
        return True
    else:
        print("\n❌ Analysis failed due to probing error.")
        return False

def main():
    parser = argparse.ArgumentParser(description="Analyze a trained Regicide RL policy.")
    parser.add_argument("--model", type=str, default="models/ppo_regicide_v1.zip", help="Path to the trained .zip model")
    parser.add_argument("--games", type=int, default=50, help="Number of evaluation games to play")
    parser.add_argument("--logdir", type=str, default=None, help="Specific TensorBoard log directory to extract (defaults to latest in runs/rl_logs)")
    args = parser.parse_args()
    
    run_analysis_pipeline(args.model, args.games, args.logdir)

if __name__ == "__main__":
    main()
