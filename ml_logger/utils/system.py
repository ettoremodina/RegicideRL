import psutil
import subprocess
from typing import List

def get_cpu_usage() -> float:
    return psutil.cpu_percent()

def get_ram_usage() -> float:
    return psutil.virtual_memory().percent

def get_gpu_usage() -> List[str]:
    """Returns a list of strings representing GPU utilization."""
    try:
        # Requires nvidia-smi to be in PATH
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True
        )
        gpus = []
        for line in result.stdout.strip().split("\n"):
            if line:
                util, mem_used, mem_total = line.split(", ")
                gpus.append(f"{util}% ({mem_used}MB/{mem_total}MB)")
        return gpus
    except Exception:
        return ["N/A (No GPU found)"]
