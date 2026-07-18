import os
import json
from pathlib import Path

def analyze_runs(logs_dir="logs"):
    runs_file = Path(logs_dir) / "game_runs.jsonl"
    if not runs_file.exists():
        print(f"No runs file found at {runs_file}")
        return

    runs = []
    with open(runs_file, 'r', encoding='utf-8') as f:
        for line in f:
            runs.append(json.loads(line))

    total_runs = len(runs)
    if total_runs == 0:
        print("No runs to analyze.")
        return

    victories = sum(r.get('victory', 0) for r in runs)
    win_rate = (victories / total_runs) * 100

    print("=== Regicide Experimental Analysis ===")
    print(f"Total Games Played: {total_runs}")
    print(f"Total Victories:    {victories}")
    print(f"Win Rate:           {win_rate:.2f}%")
    
    if any('bosses_defeated' in r for r in runs):
        avg_bosses = sum(r.get('bosses_defeated', 0) for r in runs) / total_runs
        print(f"Avg Bosses Defeated: {avg_bosses:.2f}")

    if any('turns' in r for r in runs):
        avg_turns = sum(r.get('turns', 0) for r in runs) / total_runs
        print(f"Avg Turns per Game: {avg_turns:.2f}")

if __name__ == "__main__":
    analyze_runs()
