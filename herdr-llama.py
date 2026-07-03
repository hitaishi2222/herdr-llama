#!/usr/bin/env python3
"""herdr-llama — Typer client for the daemon.

Runs the daemon if needed, reads its state, presents interactive choices,
sends commands, and exits. No CLI subcommands — just run it.
"""

import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

from rich.console import Console
from rich.prompt import Confirm, Prompt

PLUGIN_ID = "herdr.llama-server"
STATE_DIR = Path.home() / ".config" / "herdr" / "plugins" / "state" / PLUGIN_ID
SOCKET_PATH = STATE_DIR / "daemon.sock"
DAEMON_SCRIPT = Path(__file__).parent / "herdr-llama-daemon.py"

console = Console()


def _is_daemon_running() -> bool:
    try:
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.settimeout(2.0)
        s.connect(str(SOCKET_PATH))
        s.close()
        return True
    except (socket.error, FileNotFoundError):
        return False


def _ensure_daemon_running() -> bool:
    """Ensure the daemon is running, starting it if necessary."""
    if _is_daemon_running():
        return True
    # Try to start daemon
    if _start_daemon():
        return _wait_for_daemon()
    return False


def _send_command(command: str) -> dict:
    # Ensure daemon is running before sending command
    if not _ensure_daemon_running():
        return {"error": "Daemon is not running and could not be started"}
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.settimeout(10.0)
    try:
        s.connect(str(SOCKET_PATH))
        s.sendall(command.encode("utf-8") + b"\n")
        data = b""
        while True:
            chunk = s.recv(4096)
            if not chunk:
                break
            data += chunk
            if b"\n" in data:
                break
        s.close()
        return json.loads(data.decode("utf-8").strip())
    except Exception as e:
        return {"error": str(e)}
    finally:
        try:
            s.close()
        except Exception:
            pass


def _start_daemon() -> bool:
    """Start the daemon in the background."""
    console.print("[blue]Starting daemon...[/blue]")
    result = subprocess.run(
        [sys.executable, str(DAEMON_SCRIPT), "start"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        err = result.stderr.strip() or "unknown error"
        console.print(f"[red]Failed to start daemon: {err}[/red]")
        return False

    # Wait for socket
    for _ in range(20):
        if SOCKET_PATH.exists():
            return True
        import time
        time.sleep(0.5)

    console.print("[red]Daemon started but socket not ready[/red]")
    return False


def _wait_for_daemon() -> bool:
    """Wait for daemon to be ready."""
    for _ in range(20):
        if _is_daemon_running():
            return True
        import time
        time.sleep(0.5)
    return False


def _get_status() -> dict | None:
    """Get daemon status."""
    result = _send_command("status")
    if result.get("error"):
        return None
    return result


def _load_model(model_id: str) -> bool:
    result = _send_command(f"load {model_id}")
    return result.get("ok", False)


def _unload_model() -> bool:
    result = _send_command("unload")
    return result.get("ok", False)


def _stop_server() -> bool:
    result = _send_command("stop")
    return result.get("ok", False)


def _ask_select(message: str, choices: list[str], default: str | None = None) -> str | None:
    """Ask user to select from a list. Falls back to default in non-interactive mode."""
    if sys.stdin.isatty():
        import questionary
        return questionary.select(message, choices=choices, default=default).ask()
    try:
        return Prompt.ask(message, choices=choices, default=default)
    except EOFError:
        return default


def _ask_confirm(message: str, default: bool = False) -> bool:
    """Ask user a yes/no question."""
    if sys.stdin.isatty():
        import questionary
        return questionary.confirm(message, default=default).ask()
    try:
        return Confirm.ask(message, default=default)
    except EOFError:
        return default


def _get_model_list() -> list[dict]:
    """Get model list from config preset."""
    import configparser
    config_path = (
        Path.home() / ".config" / "herdr" / "plugins" / "config"
        / PLUGIN_ID / "herdr-llama.ini"
    )
    if not config_path.exists():
        console.print("[red]Config not found.[/red]")
        return []

    parser = configparser.ConfigParser()
    parser.read(str(config_path))
    models_preset = parser.get("server", "models-preset", fallback=None)
    if not models_preset:
        console.print("[red]No models-preset in config.[/red]")
        return []

    preset_path = Path(models_preset)
    if not preset_path.exists():
        console.print(f"[red]Preset not found: {preset_path}[/red]")
        return []

    # Reuse LlamaClient.get_model_list logic
    parser2 = configparser.ConfigParser()
    parser2.read(str(preset_path))
    models = []
    if parser2.has_section("models"):
        seen = set()
        for key in parser2.options("models"):
            if "." in key or key in seen:
                continue
            seen.add(key)
            name = parser2.get("models", key).strip()
            models.append({"id": key, "name": name})
    else:
        for section in parser2.sections():
            if section == "DEFAULT":
                continue
            name = parser2.get(section, "name", fallback=section)
            models.append({"id": section, "name": name})
    return models


def _display_status(status: dict) -> None:
    """Display current daemon status."""
    state = status.get("state", "unknown")
    model = status.get("model") or "none"
    tps = status.get("tokens_per_sec")
    ctx = status.get("context_usage")
    slots = status.get("slots_active", 0)
    error = status.get("error")

    console.print(f"State: [cyan]{state}[/cyan]")
    console.print(f"Model: [cyan]{model}[/cyan]")
    if tps is not None:
        console.print(f"Tokens/sec: [cyan]{tps:.1f}[/cyan]")
    if ctx is not None:
        console.print(f"Context: [cyan]{ctx * 100:.0f}%[/cyan]")
    console.print(f"Active slots: [cyan]{slots}[/cyan]")
    if error:
        console.print(f"Error: [red]{error}[/red]")


def _run_dashboard() -> None:
    """Main interactive dashboard loop."""
    # Ensure daemon is running
    if not _is_daemon_running():
        if not _start_daemon():
            return
        if not _wait_for_daemon():
            console.print("[red]Daemon not ready.[/red]")
            return

    # Get initial status
    status = _get_status()
    if not status:
        console.print("[red]Cannot connect to daemon.[/red]")
        return

    state = status.get("state")

    # Case 1: No server
    if state == "no_server":
        console.print("\n[blue]llama-server is not running.[/blue]")
        choice = _ask_select(
            "Start server?",
            choices=["Yes", "No"],
            default="Yes",
        )
        if choice != "Yes":
            return

        # Start server
        result = _send_command("start")
        if not result.get("ok"):
            console.print(f"[red]Failed to start: {result.get('error')}[/red]")
            return

        status = _get_status()
        if not status:
            console.print("[red]Cannot connect to daemon.[/red]")
            return
        state = status.get("state")

    # Case 2: Server running, no model
    elif state in ("server_running", "server_starting"):
        models = _get_model_list()
        if not models:
            console.print("[red]No models configured.[/red]")
            return

        choices = [m["name"] for m in models]
        selected = _ask_select(
            "Select a model",
            choices=choices,
        )
        if not selected:
            return

        # Find model id
        model_id = next(m["id"] for m in models if m["name"] == selected)
        console.print(f"\n[blue]Loading {selected}...[/blue]")
        if _load_model(model_id):
            console.print(f"[green]Loading initiated. You'll see a notification when ready.[/green]")
        else:
            console.print(f"[red]Failed to load {selected}[/red]")
        return

    # Case 3: Model loaded
    elif state == "model_loaded":
        model = status.get("model", "unknown")
        tps = status.get("tokens_per_sec")
        ctx = status.get("context_usage")

        console.print(f"\n[green]Model loaded: {model}[/green]")
        if tps is not None:
            console.print(f"Tokens/sec: [cyan]{tps:.1f}[/cyan]")
        if ctx is not None:
            console.print(f"Context: [cyan]{ctx * 100:.0f}%[/cyan]")

        choice = _ask_select(
            "What would you like to do?",
            choices=["Unload model", "Stop server", "Quit"],
            default="Unload model",
        )

        if choice == "Unload model":
            if _unload_model():
                console.print("[green]Model unloaded.[/green]")
            else:
                console.print("[red]Failed to unload.[/red]")
        elif choice == "Stop server":
            confirm = _ask_select(
                "Stop server (close pane)?",
                choices=["Yes", "No"],
                default="No",
            )
            if confirm == "Yes":
                if _stop_server():
                    console.print("[green]Server stopped.[/green]")
                else:
                    console.print("[red]Failed to stop.[/red]")
        # Quit does nothing — just exit
        return

    # Case 4: Model loading
    elif state == "model_loading":
        console.print("\n[blue]Model is loading...[/blue]")
        console.print("[yellow]Waiting for completion...[/yellow]")
        for _ in range(60):
            time.sleep(2)
            status = _get_status()
            if not status:
                break
            new_state = status.get("state")
            if new_state == "model_loaded":
                console.print(f"[green]Model loaded: {status.get('model')}[/green]")
                return
            if new_state == "server_running":
                console.print("[red]Model load failed.[/red]")
                return
        console.print("[red]Timed out waiting for model load.[/red]")
        return

    # Unknown state
    else:
        console.print(f"[yellow]Unknown state: {state}[/yellow]")
        _display_status(status)


def main() -> None:
    try:
        _run_dashboard()
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted.[/yellow]")
        sys.exit(0)
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)


if __name__ == "__main__":
    main()
