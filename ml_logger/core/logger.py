import logging
import time
from collections import deque
from threading import Lock
from typing import Dict, Any

from rich.live import Live
from rich.panel import Panel

from ..configs.config_loader import load_config
from ..utils.system import get_cpu_usage, get_ram_usage, get_gpu_usage
from ..utils.formatting import Highlighter
from ..utils.file_io import FileWriter
from ..utils.export import export_run_to_csv
from .handler import DashboardLogHandler
from .layout import (
    create_command_center_layout, 
    render_metrics_table, 
    render_hardware_table, 
    render_logs_panel
)
from .progress import ProgressManager

class DashboardLogger:
    def __init__(self, config_path: str = None):
        # Load unified configuration
        self.config = load_config(config_path)
        
        self.refresh_rate = self.config.get('settings', {}).get('refresh_rate', 4)
        max_log_lines = self.config.get('settings', {}).get('max_log_lines', 50)
        
        # Telemetry
        self.telemetry_interval = self.config.get('saving', {}).get('telemetry_interval_sec', 10)
        self._last_telemetry_time = 0
        
        # Internal state
        self.metrics: Dict[str, Dict[str, Any]] = {}
        self.log_queue = deque(maxlen=max_log_lines)
        self.lock = Lock()
        self._running = False
        self._live = None
        self._layout = create_command_center_layout()
        self._plugins = []
        
        # Utilities
        self.file_writer = FileWriter(self.config.get('saving', {}))
        self.highlighter = Highlighter(self.config.get('highlights', []))
        self.progress_manager = ProgressManager()
        
        # Setup Root Logger Interception
        self.root_logger = logging.getLogger()
        self.root_logger.setLevel(logging.INFO)
        
        formatter = logging.Formatter("[dim]%(asctime)s[/dim] [%(levelname)s] %(name)s: %(message)s", datefmt="%H:%M:%S")
        self.handler = DashboardLogHandler(self.log_queue, self.lock, self.highlighter, self.file_writer)
        self.handler.setFormatter(formatter)
        
        # Remove existing handlers to avoid duplicates
        for h in self.root_logger.handlers[:]:
            self.root_logger.removeHandler(h)
        
        self.root_logger.addHandler(self.handler)

    def add_plugin(self, plugin):
        """Register a new plugin/hook."""
        self._plugins.append(plugin)

    # --- Progress API ---
    def start_progress(self, total_steps: int, description: str = "Training"):
        self.progress_manager.start(total_steps, description)
        self._layout["footer"].visible = True

    def step_progress(self, advance: int = 1, description: str = None):
        self.progress_manager.step(advance, description)

    # --- Core Logging API ---
    def log(self, level, msg, *args, **kwargs):
        self.root_logger.log(level, msg, *args, **kwargs)
        
    def info(self, msg, *args, **kwargs):
        self.root_logger.info(msg, *args, **kwargs)
        
    def warning(self, msg, *args, **kwargs):
        self.root_logger.warning(msg, *args, **kwargs)
        
    def error(self, msg, *args, **kwargs):
        self.root_logger.error(msg, *args, **kwargs)

    def update_metrics(self, category: str, name: str, value: Any):
        """Updates the metric table, saves the metric to disk, and broadcasts to plugins."""
        with self.lock:
            if category not in self.metrics:
                self.metrics[category] = {}
            self.metrics[category][name] = value
        
        # Save locally
        self.file_writer.log_metric(category, name, value)
        
        # Broadcast to plugins
        for plugin in self._plugins:
            plugin.on_metric_update(category, name, value)

    def log_metadata(self, info_dict: dict):
        """Saves static information (model architecture, configs) to JSON."""
        self.file_writer.log_metadata(info_dict)

    def log_game_run(self, run_data: dict):
        """Saves a complete game run record (e.g. states, actions, outcome)."""
        self.file_writer.log_game_run(run_data)
        
    def export_to_csv(self):
        """Exports all saved JSONL metrics and telemetry to CSV files."""
        save_dir = self.config.get('saving', {}).get('save_dir', './logs')
        export_run_to_csv(save_dir)

    # --- Layout Rendering ---
    def _update_and_get_layout(self):
        """Callback for rich.live to constantly fetch the new layout."""
        # 1. Update Logs
        with self.lock:
            queue_snapshot = list(self.log_queue)
            metrics_snapshot = {k: v.copy() for k, v in self.metrics.items()}
            
        self._layout["main_split"]["logs"].update(render_logs_panel(queue_snapshot))
        
        # 2. Update Metrics
        self._layout["main_split"]["sidebar"]["metrics"].update(
            Panel(render_metrics_table(metrics_snapshot), title="Metrics", border_style="magenta")
        )
        
        # 3. Update Hardware
        cpu = get_cpu_usage()
        ram = get_ram_usage()
        gpus = get_gpu_usage()
        
        self._layout["main_split"]["sidebar"]["hardware"].update(
            Panel(render_hardware_table(cpu, ram, gpus), title="System", border_style="cyan")
        )
        
        # 4. Save Telemetry periodically
        current_time = time.time()
        if current_time - self._last_telemetry_time >= self.telemetry_interval:
            self.file_writer.log_telemetry(cpu, ram, gpus)
            self._last_telemetry_time = current_time
        
        # 5. Update Progress Bar
        if self.progress_manager.is_active:
            self._layout["footer"].update(self.progress_manager.render())
        
        return self._layout

    # --- Lifecycle ---
    def start(self):
        self._running = True
        self._last_telemetry_time = time.time() # Reset telemetry timer
        
        for plugin in self._plugins:
            plugin.on_startup(self.config)
            
        self._live = Live(get_renderable=self._update_and_get_layout, refresh_per_second=self.refresh_rate, screen=True)
        self._live.start()

    def stop(self):
        self._running = False
        if self._live:
            self._live.stop()
            
        for plugin in self._plugins:
            plugin.on_shutdown()
