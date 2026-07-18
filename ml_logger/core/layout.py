from rich.layout import Layout
from rich.panel import Panel
from rich.table import Table
from rich import box
from rich.text import Text

def create_command_center_layout() -> Layout:
    """Creates the Layout 2: Command Center architecture."""
    layout = Layout()
    layout.split_column(
        Layout(name="main_split"),
        Layout(name="footer", size=3, visible=False)
    )
    layout["main_split"].split_row(
        Layout(name="logs", ratio=2),
        Layout(name="sidebar", ratio=1)
    )
    layout["main_split"]["sidebar"].split_column(
        Layout(name="metrics", ratio=3),
        Layout(name="hardware", ratio=2)
    )
    return layout

def render_metrics_table(metrics: dict) -> Table:
    """Renders the metrics dictionary into a rich Table."""
    table = Table(box=box.ROUNDED, expand=True)
    table.add_column("Category", style="cyan")
    table.add_column("Metric", style="magenta")
    table.add_column("Value", justify="right")
    
    for category, mets in metrics.items():
        for i, (name, val) in enumerate(mets.items()):
            cat_str = category if i == 0 else ""
            
            # Format value if it's a float
            if isinstance(val, float):
                formatted = f"{val:.4f}"
            else:
                formatted = str(val)
                
            table.add_row(cat_str, name, formatted)
            
    return table

def render_hardware_table(cpu: float, ram: float, gpus: list) -> Table:
    """Renders system metrics into a rich Table."""
    table = Table(box=box.SIMPLE, expand=True, show_header=False)
    table.add_column("Component", style="cyan")
    table.add_column("Usage", justify="right", style="green")
    
    table.add_row("CPU", f"{cpu}%")
    table.add_row("RAM", f"{ram}%")
    for i, gpu_stat in enumerate(gpus):
        table.add_row(f"GPU {i}", gpu_stat)
        
    return table

def render_logs_panel(log_queue: list) -> Panel:
    """Renders the queued log strings into a Panel."""
    content = Text("\n").join(log_queue)
    return Panel(content, title="Log Stream", border_style="white")
