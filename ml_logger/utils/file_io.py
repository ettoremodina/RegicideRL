import json
import time
from pathlib import Path
from rich.text import Text

class FileWriter:
    def __init__(self, config: dict):
        self.enabled = config.get("enabled", True)
        self.save_dir = Path(config.get("save_dir", "./logs"))
        self.log_filename = config.get("log_filename", "run.log")
        self.metrics_filename = config.get("metrics_filename", "metrics.jsonl")
        self.telemetry_filename = config.get("telemetry_filename", "telemetry.jsonl")
        self.info_filename = config.get("info_filename", "run_info.json")
        self.game_runs_filename = config.get("game_runs_filename", "game_runs.jsonl")
        
        if self.enabled:
            # Create the save directory if it doesn't exist
            self.save_dir.mkdir(parents=True, exist_ok=True)
            self.log_path = self.save_dir / self.log_filename
            self.metrics_path = self.save_dir / self.metrics_filename
            self.telemetry_path = self.save_dir / self.telemetry_filename
            self.info_path = self.save_dir / self.info_filename
            self.game_runs_path = self.save_dir / self.game_runs_filename
            
            # Write a startup header to the log
            self._write_to_log("\n" + "="*50)
            self._write_to_log(f"--- New Run Started at {time.strftime('%Y-%m-%d %H:%M:%S')} ---")

    def _write_to_log(self, text: str):
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(text + "\n")

    def log_message(self, text_obj: Text):
        """Extracts plain text from a Rich Text object and saves it."""
        if self.enabled:
            self._write_to_log(text_obj.plain)

    def log_metric(self, category: str, name: str, value: any):
        """Saves a metric update to the JSONL file."""
        if self.enabled:
            entry = {
                "timestamp": time.time(),
                "category": category,
                "metric": name,
                "value": value
            }
            with open(self.metrics_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")

    def log_telemetry(self, cpu: float, ram: float, gpus: list):
        """Saves system hardware stats to the telemetry JSONL file."""
        if self.enabled:
            entry = {
                "timestamp": time.time(),
                "cpu_percent": cpu,
                "ram_percent": ram,
                "gpus": gpus
            }
            with open(self.telemetry_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")

    def log_metadata(self, info_dict: dict):
        """Saves static run information to a JSON file."""
        if self.enabled:
            # If the file already exists, we load it, update it, and save it back
            existing = {}
            if self.info_path.exists():
                try:
                    with open(self.info_path, "r", encoding="utf-8") as f:
                        existing = json.load(f)
                except json.JSONDecodeError:
                    pass
            
            existing.update(info_dict)
            
            with open(self.info_path, "w", encoding="utf-8") as f:
                json.dump(existing, f, indent=4)

    def log_game_run(self, run_data: dict):
        """Saves a complete game run record to the game runs JSONL file."""
        if self.enabled:
            run_data["timestamp"] = time.time()
            with open(self.game_runs_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(run_data) + "\n")
