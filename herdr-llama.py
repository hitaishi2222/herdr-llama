"""herdr-llama — single decision point for all llama-server interactions.

Handles:
  - Daemon lifecycle (start/stop)
  - Server lifecycle (start/stop via herdr CLI)
  - Model loading/unloading (via daemon socket or direct HTTP)
  - Tab/pane management (via herdr CLI)
  - Interactive dashboard

Daemon (herdr-llama-daemon.py) handles only:
  - Polling llama-server HTTP API
  - Calling report-agent / report-metadata
  - Exposing state via Unix socket
"""

import configparser
import json
import logging
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

from InquirerPy import inquirer

logger = logging.getLogger("herdr-llama")

PLUGIN_ID = "herdr-llama"
CONFIG_FILENAME = "herdr-llama.ini"
STATE_DIR = Path.home() / ".config" / "herdr" / "plugins" / "state" / PLUGIN_ID
SOCKET_PATH = STATE_DIR / "daemon.sock"
DAEMON_SCRIPT = Path(__file__).parent / "herdr-llama-daemon.py"
CONFIG_DIR = Path.home() / ".config" / "herdr" / "plugins" / "config" / PLUGIN_ID
CONFIG_PATH = CONFIG_DIR / CONFIG_FILENAME


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class ConfigError(Exception):
    pass


def _config_path() -> Path:
    return CONFIG_DIR / CONFIG_FILENAME


def load_config() -> dict:
    path = _config_path()
    if not path.exists():
        raise ConfigError(f"Config file not found: {path}")
    parser = configparser.ConfigParser()
    parser.read(str(path))
    if not parser.has_section("server"):
        raise ConfigError("Missing [server] section")
    if not parser.has_option("server", "llama-server-path"):
        raise ConfigError("Missing llama-server-path in [server]")
    if not parser.has_option("server", "port"):
        raise ConfigError("Missing port in [server]")
    return {
        "llama_server_path": parser.get("server", "llama-server-path"),
        "models_preset": parser.get("server", "models-preset", fallback=None),
        "port": int(parser.get("server", "port")),
        "extra_args": parser.get("server", "extra-args", fallback=None),
        "default_model": parser.get("server", "default-model", fallback=None),
        "open_method": parser.get("server", "open-method", fallback="tab"),
    }


# ---------------------------------------------------------------------------
# Herdr CLI wrapper — all herdr commands
# ---------------------------------------------------------------------------


class Herdr:
    """Wraps the herdr CLI."""

    def __init__(self):
        self.herdr_bin = os.environ.get("HERDR_BIN_PATH", "herdr")

    def _herdr(self, *args: str, timeout: int = 30) -> subprocess.CompletedProcess:
        return subprocess.run(
            [self.herdr_bin, *args],
            capture_output=True,
            text=True,
            timeout=timeout,
        )

    def _json(self, result: subprocess.CompletedProcess) -> dict | None:
        try:
            data = json.loads(result.stdout)
            return data.get("result", {})
        except (json.JSONDecodeError, KeyError):
            return None

    def refresh(self) -> dict[str, dict]:
        """Fetch workspace/tab/pane tree."""
        tree: dict[str, dict] = {}
        ws_result = self._herdr("workspace", "list")
        if ws_result.returncode != 0:
            return tree

        ws_data = self._json(ws_result)
        if not ws_data:
            return tree

        for ws in ws_data.get("workspaces", []):
            ws_id = ws.get("workspace_id")
            if not ws_id:
                continue
            tree[ws_id] = {"tabs": []}

            panes_result = self._herdr("pane", "list", "--workspace", ws_id)
            if panes_result.returncode != 0:
                continue
            panes_data = self._json(panes_result)
            if not panes_data:
                continue

            tab_panes: dict[str, list] = {}
            for pane in panes_data.get("panes", []):
                tab_id = pane.get("tab_id")
                if not tab_id:
                    continue
                tab_panes.setdefault(tab_id, []).append(pane)

            for tab_id, panes in tab_panes.items():
                tree[ws_id]["tabs"].append({"tab_id": tab_id, "panes": panes})

        return tree

    def find_llama_tab(self, config_port: int) -> str | None:
        """Find tab running llama-server on config port, claim it, rename to 'llama-server'."""
        tree = self.refresh()
        for ws_id, ws in tree.items():
            for tab in ws.get("tabs", []):
                tab_id = tab.get("tab_id")
                if not tab_id:
                    continue
                for pane in tab.get("panes", []):
                    pane_id = pane.get("pane_id")
                    if not pane_id:
                        continue
                    proc_result = self._herdr("pane", "process-info", "--pane", pane_id)
                    if proc_result.returncode != 0:
                        continue
                    proc_data = self._json(proc_result)
                    if not proc_data:
                        continue
                    for proc in proc_data.get("process_info", {}).get(
                        "foreground_processes", []
                    ):
                        if proc.get("name") == "llama-server":
                            args = proc.get("argv", [])
                            port = self._extract_port(args)
                            if port == config_port:
                                self._herdr("tab", "rename", tab_id, "llama-server")
                                return tab_id
        return None

    @staticmethod
    def _extract_port(args: list[str]) -> int:
        for i, arg in enumerate(args):
            if arg == "--port" and i + 1 < len(args):
                try:
                    return int(args[i + 1])
                except ValueError:
                    continue
        for arg in args:
            if arg.startswith("--port="):
                try:
                    return int(arg.split("=")[1])
                except (ValueError, IndexError):
                    continue
        return 0

    def create_workspace(self, label: str = "") -> tuple[str | None, str | None, str | None]:
        """Create a workspace, return (ws_id, tab_id, pane_id) from the JSON output."""
        args = ["workspace", "create"]
        if label:
            args.extend(["--label", label])
        result = self._herdr(*args)
        if result.returncode != 0:
            return None, None, None
        data = self._json(result)
        if not data:
            return None, None, None
        ws = data.get("workspace", {})
        rp = data.get("root_pane", {})
        return (
            ws.get("workspace_id"),
            ws.get("active_tab_id"),
            rp.get("pane_id"),
        )

    def create_tab(self, ws_id: str, label: str = "") -> str | None:
        args = ["tab", "create", "--workspace", ws_id]
        if label:
            args.extend(["--label", label])
        result = self._herdr(*args)
        if result.returncode != 0:
            return None
        data = self._json(result)
        return data.get("tab", {}).get("tab_id") if data else None

    def close_tab(self, tab_id: str) -> bool:
        result = self._herdr("tab", "close", tab_id)
        return result.returncode == 0

    def find_pane(self, tab_id: str) -> dict | None:
        tree = self.refresh()
        for ws_tabs in tree.values():
            for tab in ws_tabs.get("tabs", []):
                if tab.get("tab_id") == tab_id:
                    panes = tab.get("panes", [])
                    if panes:
                        return panes[0]
        return None

    def run_in_pane(self, pane_id: str, cmd: str) -> bool:
        result = self._herdr("pane", "run", pane_id, cmd)
        return result.returncode == 0

    def close_pane(self, pane_id: str) -> bool:
        result = self._herdr("pane", "close", pane_id)
        return result.returncode == 0

    def focus_tab(self, tab_id: str) -> bool:
        result = self._herdr("tab", "focus", tab_id)
        return result.returncode == 0

    def focused_workspace_id(self) -> str | None:
        ws_result = self._herdr("workspace", "list")
        if ws_result.returncode != 0:
            return None
        data = self._json(ws_result)
        if not data:
            return None
        for ws in data.get("workspaces", []):
            if ws.get("focused"):
                return ws.get("workspace_id")
        return None

    def read_pane_source(self, pane_id: str, source: str = "recent-unwrapped", lines: int = 5) -> str:
        """Read recent pane output. Uses Popen streaming to avoid capturing
        the full pane buffer into memory."""
        cmd = [
            self.herdr_bin,
            "pane",
            "read",
            pane_id,
            "--source",
            source,
            "--lines",
            str(lines),
        ]
        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                bufsize=1,
            )
            last_lines: list[str] = []
            for line in proc.stdout:  # type: ignore[union-attr]
                last_lines.append(line.rstrip("\n"))
                if len(last_lines) > lines:
                    last_lines.pop(0)
            trailing = proc.stdout.read() if proc.stdout else ""
            if trailing:
                last_lines.append(trailing.rstrip("\n"))
                if len(last_lines) > lines:
                    last_lines.pop(0)
            proc.wait(timeout=10)
            return "\n".join(last_lines).strip()
        except (subprocess.TimeoutExpired, OSError):
            return ""

# ---------------------------------------------------------------------------
# Server Manager
# ---------------------------------------------------------------------------

HEALTH_CHECK_TIMEOUT = 5
HEALTH_CHECK_INTERVAL = 1.0


class ServerManager:
    """Starts/stops llama-server via herdr CLI."""

    def __init__(self, config: dict, herdr: Herdr):
        self.config = config
        self.herdr = herdr
        self.tab_id: str | None = None
        self.pane_id: str | None = None

    def start(self) -> bool:
        server_path = self.config["llama_server_path"]
        if not os.path.isfile(server_path):
            print(f"[red]llama-server not found: {server_path}[/red]")
            return False
        if not os.access(server_path, os.X_OK):
            print(f"[red]llama-server not executable: {server_path}[/red]")
            return False

        if self.config["open_method"] == "workspace":
            self.tab_id = self._start_in_workspace()
        else:
            self.tab_id = self._start_in_focused_tab()

        if not self.tab_id:
            return False

        if not self._health_check():
            print(f"[red]Health check failed on port {self.config['port']}[/red]")
            if self.tab_id:
                self.herdr.close_tab(self.tab_id)
            return False

        return True

    def stop(self) -> bool:
        self._kill_llama_server()
        if self.pane_id:
            self.herdr.close_pane(self.pane_id)
        if self.tab_id:
            self.herdr.close_tab(self.tab_id)
        self.tab_id = None
        self.pane_id = None
        return True

    def _kill_llama_server(self) -> None:
        if self.pane_id:
            proc_result = self.herdr._herdr(
                "pane", "process-info", "--pane", self.pane_id
            )
            if proc_result.returncode == 0:
                proc_data = self.herdr._json(proc_result)
                if proc_data:
                    pids = [
                        p.get("pid")
                        for p in proc_data.get("process_info", {}).get(
                            "foreground_processes", []
                        )
                        if p.get("pid")
                    ]
                    if self._kill_by_pids(pids):
                        return

        try:
            result = subprocess.run(
                ["pgrep", "-f", f"llama-server.*--port\\s+{self.config['port']}"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0 and result.stdout.strip():
                pids = [int(p) for p in result.stdout.strip().split()]
                self._kill_by_pids(pids)
        except Exception:
            pass

    @staticmethod
    def _kill_by_pids(pids: list[int]) -> bool:
        import signal as sig

        killed = False
        for pid in pids:
            if not pid:
                continue
            try:
                os.kill(pid, 0)
            except OSError:
                continue
            try:
                os.kill(pid, sig.SIGTERM)
            except ProcessLookupError:
                continue
            except PermissionError:
                logger.warning("Permission denied: cannot kill process %d", pid)
                continue
            killed = True
            for _ in range(10):
                try:
                    os.kill(pid, 0)
                    time.sleep(0.5)
                except OSError:
                    break
            else:
                try:
                    os.kill(pid, sig.SIGKILL)
                except ProcessLookupError:
                    pass
        return killed

    def _start_in_focused_tab(self) -> str | None:
        ws_id = self.herdr.focused_workspace_id()
        if not ws_id:
            print("[red]No focused workspace[/red]")
            return None
        tab_id = self.herdr.create_tab(ws_id, "llama-server")
        if not tab_id:
            return None
        pane = self.herdr.find_pane(tab_id)
        if not pane:
            return None
        self.pane_id = pane.get("pane_id")
        cmd = self._build_command()
        if not self.herdr.run_in_pane(self.pane_id, " ".join(cmd)):
            print("[red]Failed to run llama-server in pane[/red]")
            return None
        return tab_id

    def _start_in_workspace(self) -> str | None:
        ws_id, tab_id, pane_id = self.herdr.create_workspace("llama-server")
        if not ws_id or not tab_id or not pane_id:
            return None
        self.pane_id = pane_id
        self.tab_id = tab_id
        cmd = self._build_command()
        result = self.herdr._herdr("pane", "run", pane_id, " ".join(cmd))
        if result.returncode != 0:
            print("[red]Failed to run llama-server in pane[/red]")
            return None
        return tab_id

    def _build_command(self) -> list[str]:
        cmd = [self.config["llama_server_path"]]
        if self.config["models_preset"]:
            cmd.extend(["--models-preset", self.config["models_preset"]])
        cmd.extend(["--port", str(self.config["port"])])
        if self.config["extra_args"]:
            cmd.extend(self.config["extra_args"].split())
        if "--metrics" not in cmd:
            cmd.append("--metrics")
        return cmd

    def _health_check(self) -> bool:
        import httpx

        url = f"http://127.0.0.1:{self.config['port']}/health"
        deadline = time.monotonic() + HEALTH_CHECK_TIMEOUT
        attempts = 0
        while time.monotonic() < deadline:
            try:
                resp = httpx.get(url, timeout=2.0)
                if resp.status_code == 200:
                    return True
            except httpx.ConnectError:
                attempts += 1
                if attempts % 5 == 0:
                    print(f"Waiting for server... (attempt {attempts})")
            except httpx.TimeoutException:
                attempts += 1
            time.sleep(HEALTH_CHECK_INTERVAL)
        return False


# ---------------------------------------------------------------------------
# Llama HTTP Client (same as daemon's, for direct model load/unload)
# ---------------------------------------------------------------------------


class LlamaClient:
    def __init__(self, port: int = 8080):
        self.port = port
        self._client = __import__("httpx").Client(timeout=10.0)

    def close(self):
        self._client.close()

    def is_running(self) -> bool:
        try:
            resp = self._client.get(f"http://127.0.0.1:{self.port}/health")
            return resp.status_code == 200
        except Exception:
            return False

    def is_model_loaded(self) -> bool:
        try:
            resp = self._client.get(f"http://127.0.0.1:{self.port}/models")
            resp.raise_for_status()
        except Exception:
            return False
        for model in resp.json().get("data", []):
            status = model.get("status", {})
            if isinstance(status, dict) and status.get("value") == "loaded":
                return True
        return False

    def get_model_info(self) -> dict | None:
        try:
            resp = self._client.get(f"http://127.0.0.1:{self.port}/models")
            resp.raise_for_status()
        except Exception:
            return None
        data = resp.json()
        models = data.get("data", [])
        for model in models:
            status = model.get("status", {})
            if isinstance(status, dict) and status.get("value") == "loaded":
                return model
        return None

    def load_model(self, model_id: str) -> bool:
        try:
            resp = self._client.post(
                f"http://127.0.0.1:{self.port}/models/load",
                json={"model": model_id},
            )
            resp.raise_for_status()
            return True
        except Exception as e:
            print(f"[red]Failed to load {model_id}: {e}[/red]")
            return False

    def unload_model(self, model_id: str) -> bool:
        try:
            resp = self._client.post(
                f"http://127.0.0.1:{self.port}/models/unload",
                json={"model": model_id},
            )
            resp.raise_for_status()
            return True
        except Exception as e:
            print(f"[red]Failed to unload {model_id}: {e}[/red]")
            return False

    @staticmethod
    def get_model_list(preset_path: str | Path) -> list[dict]:
        path = Path(preset_path)
        if not path.exists():
            return []
        parser = configparser.ConfigParser()
        parser.read(str(path))
        models = []
        if parser.has_section("models"):
            seen = set()
            for key in parser.options("models"):
                if "." in key or key in seen:
                    continue
                seen.add(key)
                name = parser.get("models", key).strip()
                ctx_raw = parser.get("models", f"{key}.context_size", fallback=None)
                url = parser.get("models", f"{key}.url", fallback=None)
                ctx = None
                if ctx_raw:
                    try:
                        ctx = int(ctx_raw)
                    except ValueError:
                        pass
                models.append(
                    {"id": key, "name": name, "context_size": ctx, "url": url}
                )
        else:
            for section in parser.sections():
                if section == "DEFAULT":
                    continue
                name = parser.get(section, "name", fallback=section)
                models.append({"id": section, "name": name})
        return models


# ---------------------------------------------------------------------------
# Process checking — verify llama-server is actually alive
# ---------------------------------------------------------------------------


def _find_llama_server_process(config_port: int) -> dict | None:
    """Check system processes for a running llama-server on the config port.

    Returns dict with {"pid": ..., "tab_id": ..., "pane_id": ...} if found,
    or None if no matching process exists.
    """
    try:
        result = subprocess.run(
            ["pgrep", "-af", f"llama-server.*--port\\s+{config_port}"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode != 0 or not result.stdout.strip():
            return None

        # Found at least one process. Now find which herdr pane owns it.
        herdr = Herdr()
        tree = herdr.refresh()
        for ws_id, ws in tree.items():
            for tab in ws.get("tabs", []):
                tab_id = tab.get("tab_id")
                if not tab_id:
                    continue
                for pane in tab.get("panes", []):
                    pane_id = pane.get("pane_id")
                    if not pane_id:
                        continue
                    proc_result = herdr._herdr(
                        "pane", "process-info", "--pane", pane_id
                    )
                    if proc_result.returncode != 0:
                        continue
                    proc_data = herdr._json(proc_result)
                    if not proc_data:
                        continue
                    for proc in proc_data.get("process_info", {}).get(
                        "foreground_processes", []
                    ):
                        if proc.get("name") == "llama-server":
                            args = proc.get("argv", [])
                            port = Herdr._extract_port(args)
                            if port == config_port:
                                # Also grab the PID from pgrep output
                                pid = None
                                for line in result.stdout.strip().splitlines():
                                    parts = line.split(None, 1)
                                    if parts:
                                        try:
                                            pid = int(parts[0])
                                        except ValueError:
                                            pass
                                        break
                                return {
                                    "pid": pid,
                                    "tab_id": tab_id,
                                    "pane_id": pane_id,
                                }
        return None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Daemon management
# ---------------------------------------------------------------------------


def _is_daemon_running() -> bool:
    try:
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.settimeout(2.0)
        s.connect(str(SOCKET_PATH))
        s.close()
        return True
    except (socket.error, FileNotFoundError):
        return False


def _close_daemon_gracefully() -> bool:
    """Stop daemon via socket, confirm process is gone."""
    try:
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.settimeout(3.0)
        s.connect(str(SOCKET_PATH))
        s.sendall(b"stop\n")
        s.recv(4096)
        s.close()
    except (socket.error, FileNotFoundError, OSError):
        pass

    # Wait for daemon process to exit (not just socket file)
    for _ in range(12):  # 6 seconds total
        if not _is_daemon_running():
            return True
        time.sleep(0.5)

    # Fallback: check if process is actually gone via pgrep
    result = subprocess.run(
        ["pgrep", "-f", "herdr-llama-daemon.py"],
        capture_output=True,
        text=True,
    )
    return result.returncode != 0


def _send_command(command: str) -> dict:
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
        return json.loads(data.decode("utf-8").strip())
    except Exception as e:
        return {"error": str(e)}
    finally:
        try:
            s.close()
        except Exception:
            pass


def _start_daemon() -> bool:
    result = subprocess.run(
        [sys.executable, str(DAEMON_SCRIPT), "start"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        timeout=5,
    )
    if result.returncode != 0:
        _notify("Daemon start failed")
        return False

    for _ in range(6):
        if _is_daemon_running():
            return True
        time.sleep(0.5)
    _notify("Daemon socket not ready")
    return False


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------


def _ask_select(
    message: str, choices: list[str], default: str | None = None
) -> str | None:
    return inquirer.select(message, choices=choices, default=default).execute()


def _notify(body: str) -> None:
    subprocess.run(
        ["herdr", "notification", "show", "herdr-llama", "--body", body],
        capture_output=True,
    )


def _get_model_list(config: dict) -> list[dict]:
    models_preset = config.get("models_preset")
    if not models_preset:
        _notify("No models-preset in config")
        return []
    preset_path = Path(models_preset)
    if not preset_path.exists():
        _notify(f"Preset not found: {preset_path}")
        return []
    client = LlamaClient(port=config["port"])
    models = client.get_model_list(preset_path)
    client.close()
    return models


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------


def _get_system_state(config_port: int) -> tuple[bool, bool]:
    """Check if daemon and llama-server are running.

    Returns (daemon_running, server_running).
    """
    daemon_running = _is_daemon_running()
    server_running = _find_llama_server_process(config_port) is not None
    return daemon_running, server_running


def _claim_existing_server(config: dict, herdr: Herdr) -> tuple[str | None, str | None]:
    """Find and claim an existing llama-server tab/pane.

    Returns (tab_id, pane_id) or (None, None).
    """
    tab_id = herdr.find_llama_tab(config["port"])
    if not tab_id:
        return None, None
    pane = herdr.find_pane(tab_id)
    if not pane:
        return tab_id, None
    pane_id = pane.get("pane_id")
    return tab_id, pane_id


def _tell_daemon_pane(
    daemon_running: bool, tab_id: str | None, pane_id: str | None
) -> None:
    """Tell daemon which pane to report to."""
    if daemon_running and tab_id and pane_id:
        _send_command(f"set-pane {tab_id} {pane_id}")


def _run_dashboard() -> None:
    config = load_config()
    herdr = Herdr()
    config_port = config["port"]

    # Determine which case we're in
    daemon_running, server_running = _get_system_state(config_port)
    # -----------------------------------------------------------------------
    # Case 1: Daemon running, llama-server NOT running
    # -----------------------------------------------------------------------
    if daemon_running and not server_running:
        _notify("Daemon running but server not found. Restarting...")
        _close_daemon_gracefully()

        server_mgr = ServerManager(config, herdr)
        if not server_mgr.start():
            _notify("Failed to start server")
            return

        tab_id = server_mgr.tab_id
        pane_id = server_mgr.pane_id

        if not _start_daemon():
            _notify("Failed to start daemon")
            return

        _tell_daemon_pane(True, tab_id, pane_id)
        print("(--metrics auto-enabled for TPS stats)")
        _notify("Server and daemon running")
        return

    # -----------------------------------------------------------------------
    # Case 2: llama-server running, daemon NOT running
    # -----------------------------------------------------------------------
    elif not daemon_running and server_running:
        _notify("Server running but daemon not found. Starting daemon...")

        if not _start_daemon():
            _notify("Failed to start daemon")
            return

        tab_id, pane_id = _claim_existing_server(config, herdr)
        if tab_id and pane_id:
            herdr._herdr("tab", "rename", tab_id, "llama-server")
            _tell_daemon_pane(True, tab_id, pane_id)
        _notify("Daemon started")
        return

    # -----------------------------------------------------------------------
    # Case 3: Both daemon and server running
    # -----------------------------------------------------------------------
    elif daemon_running and server_running:
        tab_id, pane_id = _claim_existing_server(config, herdr)
        status = _send_command("status")
        if status.get("error"):
            _notify("Daemon not responding")
            return

        daemon_tab_id = status.get("tab_id")
        daemon_pane_id = status.get("pane_id")

        if not (tab_id == daemon_tab_id and pane_id == daemon_pane_id):
            _tell_daemon_pane(True, tab_id, pane_id)
            status = _send_command("status")
            if status.get("error"):
                _notify("Daemon not responding after updating pane")
                return

        state = status.get("state")

    # -----------------------------------------------------------------------
    # Case 4: Neither daemon nor server running
    # -----------------------------------------------------------------------
    else:
        _notify("Starting server and daemon...")

        server_mgr = ServerManager(config, herdr)
        if not server_mgr.start():
            _notify("Failed to start server")
            return

        tab_id = server_mgr.tab_id
        pane_id = server_mgr.pane_id

        if not _start_daemon():
            _notify("Failed to start daemon")
            return

        _tell_daemon_pane(True, tab_id, pane_id)
        print("(--metrics auto-enabled for TPS stats)")
        _notify("Server and daemon started")
        return

    # ---------------------------------------------------------------
    # Act on state (only Case 3 reaches here)
    # ---------------------------------------------------------------

    if state == "no_server":
        _notify("Server not detected by daemon")
        return

    # Server running, no model
    if state in ("server_running", "server_starting"):
        models = _get_model_list(config)
        if not models:
            return

        choices = [m["name"] for m in models]
        selected = _ask_select("Select a model", choices=choices)
        if not selected:
            return

        model_id = next(m["id"] for m in models if m["name"] == selected)
        _notify(f"Loading {selected}...")
        result = _send_command(f"load {model_id}")
        if result.get("ok"):
            _notify("Loading initiated. You'll see a notification when ready.")
        else:
            _notify(f"Failed to load {selected}")
        return

    # Model loaded
    if state == "model_loaded":
        model = status.get("model", "unknown")
        tps = status.get("tokens_per_sec")
        ctx = status.get("context_usage")

        _notify(f"Model loaded: {model}")
        if tps is not None:
            _notify(f"Tokens/sec: {tps:.1f}")
        if ctx is not None:
            _notify(f"Context: {ctx * 100:.0f}%")

        choice = _ask_select(
            "What would you like to do?",
            choices=["Unload model", "Stop server", "Quit"],
            default="Unload model",
        )

        if choice == "Unload model":
            result = _send_command("unload")
            if result.get("ok"):
                _notify("Model unloaded")
            else:
                _notify("Failed to unload")
        elif choice == "Stop server":
            confirm = _ask_select(
                "Stop server (close pane)?", choices=["Yes", "No"], default="Yes"
            )
            if confirm == "Yes":
                # Kill the llama-server process and close the pane/tab
                server_mgr = ServerManager(config, herdr)
                server_mgr.tab_id = tab_id
                server_mgr.pane_id = pane_id
                if server_mgr.stop():
                    _close_daemon_gracefully()
                    _notify("Server stopped")
                else:
                    _notify("Failed to stop")
        # Quit
        return

    # Model loading
    if state == "model_loading":
        _notify("Model is loading...")
        return

    # Unknown state
    _notify(f"Unknown state: {state}")


def main() -> None:
    try:
        _run_dashboard()
    except KeyboardInterrupt:
        _notify("Interrupted")
        sys.exit(0)
    except Exception as e:
        _notify(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
