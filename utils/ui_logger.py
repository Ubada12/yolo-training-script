"""
Centralized rich logger for training pipeline.

- Colored, timestamped logs
- Severity levels
- Section headers
- Checklist helpers
"""

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from datetime import datetime

console = Console()


def ts() -> str:
    return datetime.now().strftime("%H:%M:%S")


def banner(title: str, subtitle: str = ""):
    console.print(
        Panel.fit(
            f"[bold cyan]{title}[/bold cyan]\n[dim]{subtitle}[/dim]",
            border_style="cyan",
        )
    )


def section(title: str):
    console.print(f"\n[bold blue]▶ {title}[/bold blue]")


def info(msg: str):
    console.print(f"[dim]{ts()}[/dim] [bold cyan][INFO][/bold cyan] {msg}")


def success(msg: str):
    console.print(f"[dim]{ts()}[/dim] [bold green][OK][/bold green] {msg}")


def warn(msg: str):
    console.print(f"[dim]{ts()}[/dim] [bold yellow][WARN][/bold yellow] {msg}")


def error(msg: str):
    console.print(f"[dim]{ts()}[/dim] [bold red][ERROR][/bold red] {msg}")


def checklist(title: str, items: list[tuple[str, bool, str]]):
    """
    items: (label, passed, details)
    """
    table = Table(title=title, show_header=False, box=None)
    table.add_column("Status", width=3)
    table.add_column("Check")
    table.add_column("Details")

    for label, passed, details in items:
        icon = "[green]✔[/green]" if passed else "[red]✖[/red]"
        table.add_row(icon, label, details)

    console.print(table)
