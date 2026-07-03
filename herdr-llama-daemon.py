#!/usr/bin/env python3
"""herdr-llama daemon — manages llama-server lifecycle, watcher, and socket server.

Usage:
    python herdr-llama-daemon.py start   # Start daemon (background)
    python herdr-llama-daemon.py stop    # Stop daemon
    python herdr-llama-daemon.py status  # Check daemon status
    python herdr-llama-daemon.py run     # Run daemon in foreground (for debugging)
"""

import configparser
import json
import logging
import os
import re
import signal
import socket
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import httpx
import typer
from rich.console import Console

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

REQUIRED_SECTIONS = ["server"]
REQUIRED_KEYS = ["llama-server-path", "port"]
DEFAULT_PORT = 8080

PLUGIN_ID = "herdr.llama-server"
CONFIG_FILENAME = "herdr-llama.ini"
STATE_DIR = Path.home() / ".config" / "herdr" / "plugins" / "state" / PLUGIN_ID
SOCKET_PATH = STATE_DIR / "daemon.sock"
LOG_FILE = STATE_DIR / "daemon.log"


class ConfigError(Exception):
    """Configuration error."""


@dataclass
class Config:
    llama_server_path: str
    models_preset: str | None = None
    port: int = DEFAULT_PORT
    extra_args: str | None = None
    default_model: str | None = None
    open_method: str = "tab"

    def get_extra_args_list(self) -> list[str]:
        if not self.extra_args:
            return []
        return self.extra_args.split()


def _config_path() -> Path:
    return (
        Path.home()
        / ".config"
        / "herdr"
        / "plugins"
        / "config"
        / PLUGIN_ID
        / CONFIG_FILENAME
    )


def load_config() -> Config:
    path = _config_path()
    if not path.exists():
        raise ConfigError(
            f"Config file not found: {path}\nCreate it per README instructions."
        )
    parser = configparser.ConfigParser()
    parser.read(str(path))

    for section in REQUIRED_SECTIONS:
        if not parser.has_section(section):
            raise ConfigError(f"Missing required section: [{section}]")

    for key in REQUIRED_KEYS:
        if not parser.has_option("server", key):
            raise ConfigError(f"Missing required key '{key}' in [server] section")

    extra_args = parser.get("server", "extra-args", fallback=None)
    default_model = parser.get("server", "default-model", fallback=None)
    models_preset = parser.get("server", "models-preset", fallback=None)
    open_method = parser.get("server", "open-method", fallback="tab")

    port_str = parser.get("server", "port")
    try:
        port = int(port_str)
    except ValueError:
        raise ConfigError(f"Invalid port value: {port_str!r}")

    return Config(
        llama_server_path=parser.get("server", "llama-server-path"),
        models_preset=models_preset,
        port=port,
        extra_args=extra_args,
        default_model=default_model,
        open_method=open_method,
    )


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


def setup_logging() -> logging.Logger:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("herdr-llama-daemon")
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    # File handler
    fh = logging.FileHandler(str(LOG_FILE))
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    # Stderr handler (for debugging)
    sh = logging.StreamHandler(sys.stderr)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    return logger


logger = setup_logging()


# ---------------------------------------------------------------------------
# Herdr CLI wrapper
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
            logger.warning("Failed to list workspaces: %s", ws_result.stderr)
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
                # Check if any pane in this tab runs llama-server on config port
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
                            # Check if process is running on config port
                            args = proc.get("argv", [])
                            port = self._extract_port(args)
                            if port == config_port:
                                # Claim this tab — rename to 'llama-server'
                                self._herdr("tab", "rename", tab_id, "llama-server")
                                logger.info("Claimed tab %s (port %d)", tab_id, port)
                                return tab_id
        return None

    @staticmethod
    def _extract_port(args: list[str]) -> int:
        """Extract port from llama-server process args."""
        for i, arg in enumerate(args):
            if arg == "--port" and i + 1 < len(args):
                try:
                    return int(args[i + 1])
                except ValueError:
                    continue
        # Fallback: look for --port=<value> format
        for arg in args:
            if arg.startswith("--port="):
                try:
                    return int(arg.split("=")[1])
                except (ValueError, IndexError):
                    continue
        return 0  # No port found

    def create_workspace(self, label: str = "") -> str | None:
        args = ["workspace", "create", "--focus"]
        if label:
            args.extend(["--label", label])
        result = self._herdr(*args)
        if result.returncode != 0:
            logger.error("Failed to create workspace: %s", result.stderr)
            return None
        data = self._json(result)
        return data.get("workspace", {}).get("id") if data else None

    def create_tab(self, ws_id: str, label: str = "") -> str | None:
        args = ["tab", "create", "--workspace", ws_id]
        if label:
            args.extend(["--label", label])
        result = self._herdr(*args)
        if result.returncode != 0:
            logger.error("Failed to create tab: %s", result.stderr)
            return None
        data = self._json(result)
        return data.get("tab", {}).get("tab_id") if data else None

    def close_tab(self, tab_id: str) -> bool:
        result = self._herdr("tab", "close", tab_id)
        if result.returncode != 0:
            logger.error("Failed to close tab: %s", result.stderr)
            return False
        return True

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
        if result.returncode != 0:
            logger.error("Failed to run in pane: %s", result.stderr)
            return False
        return True

    def close_pane(self, pane_id: str) -> bool:
        result = self._herdr("pane", "close", pane_id)
        if result.returncode != 0:
            logger.error("Failed to close pane: %s", result.stderr)
            return False
        return True

    def focus_tab(self, tab_id: str) -> bool:
        result = self._herdr("tab", "focus", tab_id)
        if result.returncode != 0:
            logger.warning("Failed to focus tab %s: %s", tab_id, result.stderr)
            return False
        return True

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

    def read_pane_source(self, pane_id: str, source: str, lines: int = 2) -> str:
        result = self._herdr(
            "pane", "read", pane_id, "--source", source, "--lines", str(lines)
        )
        if result.returncode != 0:
            return ""
        return result.stdout.strip()

    def report_agent(
        self,
        pane_id: str,
        agent: str,
        state: str,
        message: str = "",
        state_level: str = "on",
    ) -> bool:
        cmd = [
            self.herdr_bin,
            "pane",
            "report-agent",
            pane_id,
            "--source",
            "herdr:llama-server",
            "--agent",
            agent,
            "--state",
            state,
            "--state-level",
            state_level,
        ]
        if message:
            cmd.extend(["--message", message])
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        if result.returncode != 0:
            logger.warning("report-agent failed: %s", result.stderr.strip())
            return False
        return True

    def report_metadata(
        self,
        pane_id: str,
        title: str | None = None,
        custom_status: str | None = None,
        state_label: str | None = None,
    ) -> bool:
        cmd = [
            self.herdr_bin,
            "pane",
            "report-metadata",
            pane_id,
            "--source",
            "herdr:llama-server",
        ]
        if title is not None:
            cmd.extend(["--title", title])
        if custom_status is not None:
            cmd.extend(["--custom-status", custom_status])
        if state_label is not None:
            cmd.extend(["--state-label", state_label])
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        if result.returncode != 0:
            logger.debug("report-metadata failed: %s", result.stderr.strip())
            return False
        return True

    def notification_show(self, message: str) -> None:
        cmd = [self.herdr_bin, "notification", "show", message]
        try:
            subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        except Exception:
            logger.debug("Failed to send notification")


# ---------------------------------------------------------------------------
# Llama Client
# ---------------------------------------------------------------------------


class LlamaClient:
    def __init__(self, base_url: str = "http://127.0.0.1", port: int = 8080):
        self.base_url = base_url.rstrip("/")
        self.port = port
        self._client = httpx.Client(timeout=10.0)

    def close(self) -> None:
        self._client.close()

    def is_running(self) -> bool:
        try:
            resp = self._client.get(f"{self.base_url}:{self.port}/health")
            return resp.status_code == 200
        except httpx.ConnectError:
            return False

    def get_model_info(self) -> dict | None:
        try:
            resp = self._client.get(f"{self.base_url}:{self.port}/models")
            resp.raise_for_status()
        except httpx.HTTPError as e:
            logger.debug("Failed to get /models: %s", e)
            return None
        data = resp.json()
        models = data.get("data", [])
        for model in models:
            status = model.get("status", {})
            if isinstance(status, dict) and status.get("value") == "loaded":
                return model
        for model in models:
            status = model.get("status", {})
            if isinstance(status, dict) and status.get("value") not in (
                "unloaded",
                "error",
                None,
            ):
                return model
        return None

    def is_model_loaded(self) -> bool:
        info = self.get_model_info()
        if not info:
            return False
        status = info.get("status", {})
        if isinstance(status, dict):
            return status.get("value") == "loaded"
        return False

    def wait_for_model_load(self, model_id: str, timeout: float = 120.0) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            info = self.get_model_info()
            if info and info.get("id") == model_id and self.is_model_loaded():
                return True
            time.sleep(1.0)
        return False

    def load_model(self, model_id: str) -> bool:
        try:
            resp = self._client.post(
                f"{self.base_url}:{self.port}/models/load",
                json={"model": model_id},
            )
            resp.raise_for_status()
            return True
        except httpx.HTTPError as e:
            logger.error("Failed to load model %s: %s", model_id, e)
            return False

    def unload_model(self, model_id: str) -> bool:
        try:
            resp = self._client.post(
                f"{self.base_url}:{self.port}/models/unload",
                json={"model": model_id},
            )
            resp.raise_for_status()
            return True
        except httpx.HTTPError as e:
            logger.error("Failed to unload model %s: %s", model_id, e)
            return False

    def get_completions(self) -> dict | None:
        try:
            resp = self._client.get(f"{self.base_url}:{self.port}/completion")
            resp.raise_for_status()
        except httpx.HTTPError as e:
            logger.debug("Failed to get /completion: %s", e)
            return None
        return resp.json()

    def get_slots(self) -> list[dict]:
        try:
            resp = self._client.get(f"{self.base_url}:{self.port}/slots")
            resp.raise_for_status()
        except httpx.HTTPError as e:
            logger.debug("Failed to get /slots: %s", e)
            return []
        return resp.json()

    @staticmethod
    def get_model_list(preset_path: str | Path) -> list[dict]:
        path = Path(preset_path)
        if not path.exists():
            logger.warning("Preset file not found: %s", path)
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
                ctx_raw = parser.get(section, "context_size", fallback=None)
                url = parser.get(section, "url", fallback=None)
                ctx = None
                if ctx_raw:
                    try:
                        ctx = int(ctx_raw)
                    except ValueError:
                        pass
                models.append(
                    {"id": section, "name": name, "context_size": ctx, "url": url}
                )
        return models


# ---------------------------------------------------------------------------
# Server Manager
# ---------------------------------------------------------------------------

HEALTH_CHECK_TIMEOUT = 30
HEALTH_CHECK_INTERVAL = 1.0


class ServerManager:
    """Manages llama-server lifecycle."""

    def __init__(self, config: Config, herdr: Herdr):
        self.config = config
        self.herdr = herdr
        self.tab_id: str | None = None
        self.pane_id: str | None = None

    def start(self) -> bool:
        """Start llama-server. Returns tab_id or None."""
        server_path = self.config.llama_server_path
        if not os.path.isfile(server_path):
            logger.error("llama-server binary not found: %s", server_path)
            return False
        if not os.access(server_path, os.X_OK):
            logger.error("llama-server not executable: %s", server_path)
            return False

        if self.config.open_method == "workspace":
            self.tab_id = self._start_in_workspace()
        else:
            self.tab_id = self._start_in_focused_tab()

        if not self.tab_id:
            return False

        # Wait for health check
        if not self._health_check():
            logger.error("Health check failed on port %d", self.config.port)
            if self.tab_id:
                self.herdr.close_tab(self.tab_id)
            return False

        logger.info(
            "llama-server started in tab %s, pane %s", self.tab_id, self.pane_id
        )
        return True

    def stop(self) -> bool:
        """Stop llama-server by closing the tab."""
        if self.tab_id:
            self.herdr.close_tab(self.tab_id)
            logger.info("Tab closed")
        self.tab_id = None
        self.pane_id = None
        return True

    def _start_in_focused_tab(self) -> str | None:
        ws_id = self.herdr.focused_workspace_id()
        if not ws_id:
            logger.error("No focused workspace")
            return None
        tab_id = self.herdr.create_tab(ws_id, "llama-server")
        if not tab_id:
            return None
        pane = self.herdr.find_pane(tab_id)
        if not pane:
            return None
        self.pane_id = pane.get("pane_id")
        if not self.pane_id:
            return None
        # Actually run llama-server in the pane
        cmd = self._build_command()
        if not self.herdr.run_in_pane(self.pane_id, " ".join(cmd)):
            logger.error("Failed to run llama-server in pane")
            return None
        return tab_id

    def _start_in_workspace(self) -> str | None:
        ws_id = self.herdr.create_workspace("llama-server")
        if not ws_id:
            return None
        # Get workspace info to find the default tab
        result = self.herdr._herdr("workspace", "get", ws_id)
        if result.returncode != 0:
            return None
        data = self.herdr._json(result)
        if not data:
            return None
        tabs = data.get("workspace", {}).get("tabs", [])
        if not tabs:
            return None
        self.tab_id = tabs[0].get("tab_id")
        if not self.tab_id:
            return None
        pane = self.herdr.find_pane(self.tab_id)
        if not pane:
            return None
        self.pane_id = pane.get("pane_id")
        if not self.pane_id:
            return None
        # Actually run llama-server in the pane
        cmd = self._build_command()
        if not self.herdr.run_in_pane(self.pane_id, " ".join(cmd)):
            logger.error("Failed to run llama-server in pane")
            return None
        return self.tab_id

    def _build_command(self) -> list[str]:
        cmd = [self.config.llama_server_path]
        if self.config.models_preset:
            cmd.extend(["--models-preset", self.config.models_preset])
        cmd.extend(["--port", str(self.config.port)])
        cmd.extend(self.config.get_extra_args_list())
        return cmd

    def _health_check(self) -> bool:
        url = f"http://127.0.0.1:{self.config.port}/health"
        deadline = time.monotonic() + HEALTH_CHECK_TIMEOUT
        attempts = 0
        while time.monotonic() < deadline:
            try:
                resp = httpx.get(url, timeout=2.0)
                if resp.status_code == 200:
                    return True
            except httpx.ConnectError:
                attempts += 1
                if attempts % 5 == 0:  # Log every 5 attempts
                    logger.info("Waiting for server to be ready (attempt %d)...", attempts)
            except httpx.TimeoutException:
                attempts += 1
            time.sleep(HEALTH_CHECK_INTERVAL)
        logger.error("Health check failed after %d attempts on port %d", attempts, self.config.port)
        return False


# ---------------------------------------------------------------------------
# Watcher
# ---------------------------------------------------------------------------

POLL_INTERVAL = 2.0
CONNECTION_RETRY_INTERVAL = 5.0
CONNECTION_RETRY_MAX = 10


@dataclass
class WatcherStats:
    tokens_per_sec: float | None = None
    context_usage: float | None = None
    model_name: str | None = None
    model_id: str | None = None
    slots_active: int = 0
    server_running: bool = False
    error: str | None = None
    state: str = "idle"  # "idle" | "working" | "blocked"


class Watcher:
    """Polls llama-server for stats and state changes."""

    def __init__(self, client: LlamaClient, herdr: Herdr, tab_id: str, pane_id: str, daemon: Optional["Daemon"] = None):
        self.client = client
        self.herdr = herdr
        self.tab_id = tab_id
        self.pane_id = pane_id
        self.daemon = daemon
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._stats = WatcherStats()
        self._last_state = "idle"
        self._last_model: str | None = None
        self._consecutive_errors = 0

    @property
    def stats(self) -> WatcherStats:
        return self._stats

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        logger.info("Watcher started")

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)
        self._thread = None
        logger.info("Watcher stopped")

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                self._poll_once()
                self._consecutive_errors = 0
                time.sleep(POLL_INTERVAL)
            except Exception as e:
                self._consecutive_errors += 1
                self._stats.error = str(e)
                logger.warning("Watcher poll error: %s", e)
                if self._consecutive_errors >= CONNECTION_RETRY_MAX:
                    self.herdr.notification_show(
                        "llama-server connection lost. Check if the server is running."
                    )
                    logger.error("Max connection retries reached. Stopping watcher.")
                    break
                time.sleep(CONNECTION_RETRY_INTERVAL)

    def _poll_once(self) -> None:
        server_was_online = self._stats.server_running
        if not self.client.is_running():
            self._stats.server_running = False
            self._stats.error = "server not responding"
            self._report_state("blocked", "server offline")
            # ponytail: transition daemon state on server crash
            if self.daemon and server_was_online:
                logger.info("Watcher detected server offline, transitioning to NO_SERVER")
                self.daemon.state = DaemonState.NO_SERVER
            return

        self._stats.server_running = True
        self._stats.error = None

        # Get model info
        model_info = self.client.get_model_info()
        current_model = model_info.get("id") if model_info else None
        if current_model:
            self._stats.model_name = current_model
            self._stats.model_id = current_model
        else:
            self._stats.model_name = None
            self._stats.model_id = None

        # Detect model load/unload transitions
        if self.daemon:
            if current_model and self._last_model != current_model:
                logger.info("Watcher detected model loaded: %s", current_model)
                self.daemon.current_model = current_model
                self.daemon.state = DaemonState.MODEL_LOADED
            elif not current_model and self._last_model:
                logger.info("Watcher detected model unloaded")
                self.daemon.current_model = None
                self.daemon.state = DaemonState.SERVER_RUNNING
            self._last_model = current_model

        # Get completions
        completion = self.client.get_completions()
        if completion:
            timings = completion.get("timings", {})
            self._stats.tokens_per_sec = timings.get("predict_per_token_ms")
            ctx_size = completion.get("context_size")
            ctx_used = completion.get("n_past")
            if ctx_size and ctx_used:
                self._stats.context_usage = ctx_used / ctx_size

        # Get slots
        slots = self.client.get_slots()
        self._stats.slots_active = sum(1 for s in slots if s.get("running"))

        # Determine state from pane output
        state = self._determine_state_from_output()
        self._stats.state = state

        # Report metadata
        self._report_metadata(state)

    def _determine_state_from_output(self) -> str:
        """Determine state by parsing llama-server pane output."""
        output = self.herdr.read_pane_source(self.pane_id, "recent-unwrapped", lines=2)
        if not output:
            return self._last_state

        lines = [l.strip() for l in output.splitlines() if l.strip()]
        if not lines:
            return self._last_state

        last_line = lines[-1]

        # Check for generation progress
        if re.search(r"tg\s*=\s*[\d.]+\s*t/s", last_line, re.IGNORECASE):
            return "working"
        if re.search(r"progress\s*=\s*[\d.]+", last_line, re.IGNORECASE):
            return "working"

        # Check for errors
        output_lower = output.lower()
        if "error" in output_lower or "fail" in output_lower:
            return "blocked"

        return self._last_state

    def _determine_state_level(self, state: str) -> str:
        """Parse state-level from pane output."""
        output = self.herdr.read_pane_source(self.pane_id, "recent-unwrapped", lines=2)
        if not output:
            return "on"

        lines = [l.strip() for l in output.splitlines() if l.strip()]
        if not lines:
            return "on"
        last_line = lines[-1]

        tg_match = re.search(r"tg\s*=\s*([\d.]+)\s*t/s", last_line, re.IGNORECASE)
        if tg_match:
            return f"{tg_match.group(1)} tps"

        progress_match = re.search(r"progress\s*=\s*([\d.]+)", last_line, re.IGNORECASE)
        if progress_match:
            pct = float(progress_match.group(1)) * 100
            return f"{pct:.0f}% (progress)"

        output_lower = output.lower()
        if "error" in output_lower or "fail" in output_lower:
            return "error"
        if "warn" in output_lower:
            return "warning"
        return "on"

    def _report_state(self, state: str, message: str) -> None:
        """Report agent state via report-agent."""
        model_name = self._stats.model_name or "unknown"
        state_level = self._determine_state_level(state)
        self.herdr.report_agent(self.pane_id, model_name, state, message, state_level)
        self._last_state = state

    def _report_metadata(self, state: str) -> None:
        """Report metadata via report-metadata."""
        model_name = self._stats.model_name or "unknown"
        state_level = self._determine_state_level(state)

        if state == "working":
            title = f"{model_name} — working"
            custom_status = (
                f"{self._stats.tokens_per_sec or 0:.1f} tok/s"
                if self._stats.tokens_per_sec
                else state_level
            )
        elif state == "idle":
            title = f"{model_name} — idle"
            custom_status = (
                "ready"
                if not self._stats.context_usage
                else f"{self._stats.context_usage * 100:.0f}% context"
            )
        else:
            title = f"{model_name} — {state}"
            custom_status = self._stats.error or state_level

        self.herdr.report_metadata(
            self.pane_id,
            title=title,
            custom_status=custom_status,
            state_label=state_level,
        )


# ---------------------------------------------------------------------------
# Daemon State Machine
# ---------------------------------------------------------------------------


class DaemonState:
    """Tracks daemon state."""

    NO_SERVER = "no_server"
    SERVER_STARTING = "server_starting"
    SERVER_RUNNING = "server_running"
    MODEL_LOADING = "model_loading"
    MODEL_LOADED = "model_loaded"


class Daemon:
    """Main daemon process."""

    def __init__(self):
        self.config = load_config()
        self.herdr = Herdr()
        self.client = LlamaClient(port=self.config.port)
        self.server_manager = ServerManager(self.config, self.herdr)
        self.watcher: Watcher | None = None
        self.state = DaemonState.NO_SERVER
        self.current_model: str | None = None
        self._running = False

    def start(self) -> bool:
        """Start the daemon."""
        if self._running:
            logger.warning("Daemon already running")
            return False

        logger.info("Starting daemon")
        self._running = True

        # Check if llama-server is already running on config port
        tab_id = self.herdr.find_llama_tab(self.config.port)
        if tab_id:
            logger.info("llama-server already running in tab %s", tab_id)
            self.server_manager.tab_id = tab_id
            pane = self.herdr.find_pane(tab_id)
            if pane:
                self.server_manager.pane_id = pane.get("pane_id")
            self.state = DaemonState.SERVER_RUNNING
            # Check if model is loaded
            if self.client.is_model_loaded():
                model_info = self.client.get_model_info()
                self.current_model = model_info.get("id") if model_info else None
                self.state = DaemonState.MODEL_LOADED
            self._start_watcher()
            return True

        # Direct health check — server might be running outside herdr's tab tree
        if self.client.is_running():
            logger.info("llama-server already responding on port %d", self.config.port)
            # Try to find and claim the tab
            tab_id = self.herdr.find_llama_tab(self.config.port)
            if tab_id:
                self.server_manager.tab_id = tab_id
                pane = self.herdr.find_pane(tab_id)
                if pane:
                    self.server_manager.pane_id = pane.get("pane_id")
            self.state = DaemonState.SERVER_RUNNING
            # Check if model is loaded
            if self.client.is_model_loaded():
                model_info = self.client.get_model_info()
                self.current_model = model_info.get("id") if model_info else None
                self.state = DaemonState.MODEL_LOADED
            self._start_watcher()
            return True

        # Start server
        logger.info("Starting llama-server")
        self.state = DaemonState.SERVER_STARTING
        if not self.server_manager.start():
            logger.error("Failed to start server")
            self.state = DaemonState.NO_SERVER
            return False

        self.state = DaemonState.SERVER_RUNNING
        logger.info("Server started, state: %s", self.state)
        # Check if a model is already loaded before starting watcher
        if self.client.is_model_loaded():
            model_info = self.client.get_model_info()
            self.current_model = model_info.get("id") if model_info else None
            self.state = DaemonState.MODEL_LOADED
            logger.info("Existing model detected: %s", self.current_model)
        self._start_watcher()
        return True

    def stop(self) -> bool:
        """Stop the daemon."""
        logger.info("Stopping daemon")
        self._running = False

        if self.watcher:
            self.watcher.stop()
            self.watcher = None

        if self.server_manager.tab_id:
            self.server_manager.stop()

        self.client.close()
        self.state = DaemonState.NO_SERVER
        logger.info("Daemon stopped")
        return True

    def load_model(self, model_id: str) -> bool:
        """Load a model (fire-and-forget). Returns immediately."""
        if self.state not in (DaemonState.SERVER_RUNNING, DaemonState.MODEL_LOADED):
            logger.error("Cannot load model in state %s", self.state)
            return False

        # ponytail: skip load if model already loaded — server returns 400
        if self.current_model == model_id and self.state == DaemonState.MODEL_LOADED:
            logger.info("Model %s already loaded", model_id)
            self.herdr.notification_show(f"Loading {model_id}…")
            self._start_watcher()
            return True

        logger.info("Loading model: %s", model_id)
        self.state = DaemonState.MODEL_LOADING
        self.current_model = model_id

        if self.client.load_model(model_id):
            # ponytail: fire-and-forget load — watcher polls /models and
            # detects the loaded transition, no blocking wait here
            self.herdr.notification_show(f"Loading {model_id}…")
            self._start_watcher()
            return True
        else:
            logger.error("Failed to load model")
            self.state = DaemonState.SERVER_RUNNING
            return False

    def unload_model(self) -> bool:
        """Unload current model."""
        if not self.current_model:
            logger.warning("No model loaded")
            return False

        logger.info("Unloading model: %s", self.current_model)
        if self.client.unload_model(self.current_model):
            self.current_model = None
            self.state = DaemonState.SERVER_RUNNING
            if self.watcher:
                self.watcher.stop()
            self.herdr.report_metadata(
                self.server_manager.pane_id,
                title="No model",
                custom_status="idle",
                state_label="on",
            )
            return True
        return False

    def get_status(self) -> dict:
        """Get current daemon status, synced with actual server state."""
        # Sync state with actual server before returning
        if self.server_manager.tab_id:
            if self.client.is_model_loaded():
                model_info = self.client.get_model_info()
                self.current_model = model_info.get("id") if model_info else None
                self.state = DaemonState.MODEL_LOADED
            elif self.state == DaemonState.MODEL_LOADED:
                # Model was unloaded externally
                self.current_model = None
                self.state = DaemonState.SERVER_RUNNING

        stats = self.watcher.stats if self.watcher else WatcherStats()
        # No model to report when server is down
        model = self.current_model if self.state != DaemonState.NO_SERVER else None
        return {
            "state": self.state,
            "model": model,
            "tokens_per_sec": stats.tokens_per_sec,
            "context_usage": stats.context_usage,
            "slots_active": stats.slots_active,
            "server_running": stats.server_running,
            "error": stats.error,
        }

    def _start_watcher(self) -> None:
        """Start the watcher if not already running."""
        if self.watcher and self.watcher._thread and self.watcher._thread.is_alive():
            return
        if not self.server_manager.pane_id:
            return
        self.watcher = Watcher(
            client=self.client,
            herdr=self.herdr,
            tab_id=self.server_manager.tab_id,
            pane_id=self.server_manager.pane_id,
            daemon=self,
        )
        self.watcher.start()


# ---------------------------------------------------------------------------
# Socket Server
# ---------------------------------------------------------------------------


def handle_client(conn: socket.socket, daemon: Daemon) -> None:
    """Handle a client connection."""
    try:
        data = conn.recv(4096)
        if not data:
            return

        command = data.decode("utf-8").strip()
        logger.debug("Received command: %s", command)

        parts = command.split(maxsplit=1)
        cmd = parts[0]
        arg = parts[1] if len(parts) > 1 else None

        response = _handle_command(cmd, arg, daemon)
        conn.sendall(json.dumps(response).encode("utf-8") + b"\n")
    except Exception as e:
        logger.error("Error handling client: %s", e)
        try:
            conn.sendall(json.dumps({"error": str(e)}).encode("utf-8") + b"\n")
        except Exception:
            pass
    finally:
        conn.close()


def _handle_command(cmd: str, arg: str | None, daemon: Daemon) -> dict:
    """Handle a command from the Typer app."""
    if cmd == "status":
        return daemon.get_status()

    elif cmd == "start":
        if daemon.start():
            return {"ok": True, "state": daemon.state}
        return {"error": "Failed to start daemon"}

    elif cmd == "load":
        if not arg:
            return {"error": "Model ID required"}
        if daemon.load_model(arg):
            return {"ok": True, "state": daemon.state, "model": arg}
        return {"error": "Failed to load model"}

    elif cmd == "unload":
        if daemon.unload_model():
            return {"ok": True, "state": daemon.state}
        return {"error": "Failed to unload model"}

    elif cmd == "stop":
        if daemon.stop():
            return {"ok": True}
        return {"error": "Failed to stop daemon"}

    else:
        return {"error": f"Unknown command: {cmd}"}


def start_socket_server(daemon: Daemon) -> socket.socket:
    """Start the Unix socket server."""
    SOCKET_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Remove old socket if exists
    if SOCKET_PATH.exists():
        SOCKET_PATH.unlink()

    server_socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server_socket.bind(str(SOCKET_PATH))
    server_socket.listen(5)
    server_socket.settimeout(1.0)  # Allow periodic checks

    logger.info("Socket server listening on %s", SOCKET_PATH)
    return server_socket


def socket_loop(server_socket: socket.socket, daemon: Daemon) -> None:
    """Main socket accept loop."""
    while daemon._running:
        try:
            conn, _ = server_socket.accept()
            threading.Thread(
                target=handle_client, args=(conn, daemon), daemon=True
            ).start()
        except socket.timeout:
            continue
        except OSError:
            break


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

app = typer.Typer(name="herdr-llama-daemon", help="Llama Server Daemon")
console = Console()


def _is_daemon_running() -> bool:
    """Check if daemon is already running."""
    try:
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.settimeout(2.0)
        s.connect(str(SOCKET_PATH))
        s.close()
        return True
    except (socket.error, FileNotFoundError):
        return False


def _check_config() -> Config | None:
    """Validate config and return it, or None with a print error."""
    try:
        return load_config()
    except ConfigError as e:
        print(f"[red]Config error: {e}[/red]", file=sys.stderr)
        return None


def _send_command(command: str) -> dict:
    """Send a command to the daemon."""
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


@app.command()
def start():
    """Start daemon in background."""
    if _is_daemon_running():
        console.print("[yellow]Daemon already running[/yellow]")
        sys.exit(1)

    STATE_DIR.mkdir(parents=True, exist_ok=True)

    # Fork to background
    try:
        pid = os.fork()
        if pid > 0:
            # Parent
            console.print("[green]Daemon started[/green]")
            sys.exit(0)
    except OSError as e:
        console.print(f"[red]Fork failed: {e}[/red]")
        sys.exit(1)

    # Child
    os.setsid()

    # Redirect stdin/stdout/stderr
    sys.stdin = open(os.devnull, "r")
    sys.stdout = open(LOG_FILE, "a")
    sys.stderr = open(LOG_FILE, "a")

    # Validate config before forking
    if _check_config() is None:
        sys.exit(1)

    # Run daemon
    try:
        daemon = Daemon()
        if not daemon.start():
            logger.error("Failed to start daemon")
            sys.exit(1)

        server_socket = start_socket_server(daemon)

        # Handle signals
        def signal_handler(signum, frame):
            logger.info("Signal %d received", signum)
            daemon.stop()
            server_socket.close()
            if SOCKET_PATH.exists():
                SOCKET_PATH.unlink()
            sys.exit(0)

        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)

        logger.info("Daemon running (PID: %d)", os.getpid())
        socket_loop(server_socket, daemon)

    except Exception as e:
        logger.error("Daemon error: %s", e, exc_info=True)
        sys.exit(1)


@app.command()
def stop():
    """Stop daemon."""
    if not _is_daemon_running():
        console.print("[yellow]Daemon not running[/yellow]")
        sys.exit(0)

    result = _send_command("stop")
    if result.get("ok"):
        console.print("[green]Daemon stopped[/green]")
    else:
        console.print(f"[red]Failed to stop: {result.get('error')}[/red]")
        sys.exit(1)


@app.command()
def status():
    """Check daemon status."""
    if not _is_daemon_running():
        console.print("[yellow]Daemon not running[/yellow]")
        sys.exit(1)

    result = _send_command("status")
    state = result.get("state", "unknown")
    model = result.get("model", "none")
    tps = result.get("tokens_per_sec")

    console.print(f"State: [cyan]{state}[/cyan]")
    console.print(f"Model: [cyan]{model}[/cyan]")
    if tps is not None:
        console.print(f"Tokens/sec: [cyan]{tps:.1f}[/cyan]")
    if result.get("error"):
        console.print(f"Error: [red]{result['error']}[/red]")


@app.command()
def run():
    """Run daemon in foreground (for debugging)."""
    try:
        daemon = Daemon()
        if not daemon.start():
            console.print("[red]Failed to start daemon[/red]")
            sys.exit(1)

        server_socket = start_socket_server(daemon)

        def signal_handler(signum, frame):
            logger.info("Signal %d received", signum)
            daemon.stop()
            server_socket.close()
            if SOCKET_PATH.exists():
                SOCKET_PATH.unlink()
            sys.exit(0)

        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)

        logger.info("Daemon running in foreground (PID: %d)", os.getpid())
        socket_loop(server_socket, daemon)

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        logger.error("Daemon error", exc_info=True)
        sys.exit(1)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
