import os
import argparse
from datetime import datetime
from solvers.analysis.tb_extractor import find_latest_run, extract_tb_logs
from solvers.analysis.probe import probe_policy
from solvers.analysis.reporter import generate_reports
from solvers.analysis.plotter import plot_dashboard

def main():
    parser = argparse.ArgumentParser(description="Analyze a trained Regicide RL policy.")
    parser.add_argument("--model", type=str, default="models/ppo_regicide_v1.zip", help="Path to the trained .zip model")
    parser.add_argument("--games", type=int, default=50, help="Number of evaluation games to play")
    parser.add_argument("--logdir", type=str, default=None, help="Specific TensorBoard log directory to extract (defaults to latest in runs/rl_logs)")
    args = parser.parse_args()
    
    # 1. Create timestamped output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = f"policy_analysis_{timestamp}"
    os.makedirs(out_dir, exist_ok=True)
    print(f"Created analysis directory: {out_dir}")
    
    # 2. Extract TensorBoard Logs
    logdir = args.logdir if args.logdir else find_latest_run()
    tb_data = extract_tb_logs(logdir)
    
    # 3. Probe the Policy
    probe_results = probe_policy(args.model, num_games=args.games)
    
    if probe_results:
        # 4. Generate Reports
        generate_reports(probe_results, out_dir)
        
        # 5. Generate Plots
        plot_dashboard(probe_results, tb_data, out_dir)
        
        print(f"\n✅ Analysis complete! Check the '{out_dir}' folder for results.")
    else:
        print("\n❌ Analysis failed due to probing error.")

if __name__ == "__main__":
    main()
