from rich.progress import Progress, TextColumn, BarColumn, TimeElapsedColumn, TimeRemainingColumn
from rich.panel import Panel

class ProgressManager:
    """Manages the rich progress bar for the dashboard."""
    def __init__(self):
        self.progress = Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            "[progress.percentage]{task.percentage:>3.0f}%",
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            expand=True
        )
        self.task_id = None
        self.is_active = False

    def start(self, total_steps: int, description: str = "Training"):
        self.task_id = self.progress.add_task(f"[magenta]{description}", total=total_steps)
        self.is_active = True

    def step(self, advance: int = 1, description: str = None):
        if self.task_id is not None:
            kwargs = {"advance": advance}
            if description:
                kwargs["description"] = f"[magenta]{description}"
            self.progress.update(self.task_id, **kwargs)

    def stop(self):
        self.is_active = False
        
    def render(self) -> Panel:
        return Panel(self.progress, title="Progress", border_style="blue")
