# metrics/dashboard.py
import psutil
import time
from rich.console import Console
from rich.table import Table
from rich.live import Live
from rich.panel import Panel
from rich.columns import Columns
from rich import box
from typing import List
from workers.gpu_worker import GPUWorker
from metrics.collector import MetricsCollector

console = Console()


def get_gpu_stats() -> str:
    """Try to get GPU stats via nvidia-smi."""
    try:
        import subprocess
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=2
        )
        if result.returncode == 0:
            parts = result.stdout.strip().split(", ")
            util, mem_used, mem_total, temp = parts
            return f"GPU: {util}%  VRAM: {mem_used}MB/{mem_total}MB  Temp: {temp}°C"
    except Exception:
        pass
    return "GPU: N/A (nvidia-smi not available)"


def build_table(workers: List[GPUWorker], collector: MetricsCollector,
                strategy: str) -> Table:
    table = Table(box=box.ROUNDED, title=f"[bold cyan]DISTRIBUTED LLM SYSTEM[/bold cyan]",
                  title_justify="center")

    table.add_column("Worker", style="bold")
    table.add_column("Status")
    table.add_column("Active Conn", justify="center")
    table.add_column("Processed", justify="center")
    table.add_column("Avg Latency", justify="center")
    table.add_column("Failed", justify="center")

    for w in workers:
        status = "[green]🟢 alive[/green]" if w.status.is_alive else "[red]🔴 dead[/red]"
        table.add_row(
            f"Worker-{w.id}",
            status,
            str(w.status.active_connections),
            str(w.status.total_processed),
            f"{w.status.avg_latency:.2f}s",
            str(w.status.total_failed),
        )

    return table


def print_summary(workers: List[GPUWorker], collector: MetricsCollector, strategy: str):
    """Print final summary after load test completes."""
    summary = collector.get_summary()

    console.print("\n")
    console.rule("[bold cyan]FINAL RESULTS[/bold cyan]")

    table = Table(box=box.SIMPLE_HEAVY)
    table.add_column("Metric", style="bold cyan")
    table.add_column("Value", style="bold white")

    table.add_row("Strategy", strategy)
    table.add_row("Total Requests", str(summary["total"]))
    table.add_row("Completed", f"[green]{summary['completed']}[/green]")
    table.add_row("Failed", f"[red]{summary['failed']}[/red]")
    table.add_row("Avg Latency", f"{summary['avg_latency']}s")
    table.add_row("P95 Latency", f"{summary['p95_latency']}s")
    table.add_row("Throughput", f"{summary['throughput']} req/s")
    table.add_row("Dead Workers", str(summary["dead_workers"]))

    console.print(table)

    console.print("\n[bold]Per-Worker Breakdown:[/bold]")
    for worker_id, count in summary["worker_counts"].items():
        bar = "█" * min(count, 50)
        console.print(f"  Worker-{worker_id}: {bar} ({count})")

    console.print(f"\n[dim]{get_gpu_stats()}[/dim]")
    cpu = psutil.cpu_percent()
    ram = psutil.virtual_memory()
    console.print(f"[dim]CPU: {cpu}%  RAM: {ram.used // (1024**3)}GB / "
                  f"{ram.total // (1024**3)}GB[/dim]")
