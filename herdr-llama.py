#!/usr/bin/env python3
"""herdr-llama — single-file plugin app.

Usage:
    python herdr-llama.py dashboard    # Open the dashboard overlay
"""

import configparser
import json
import logging
import os
import re
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import httpx
import questionary
import typer
from rich.console import Console

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("herdr-llama")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

REQUIRED_SECTIONS = ["server"]
REQUIRED_KEYS = ["llama-server-path", "port"]
DEFAULT_PORT = 8080

PLUGIN_ID = "herdr.llama-server"
CONFIG_FILENAME = "herdr-llama.ini"


class ConfigError(Exception):
    """Configuration error."""


@dataclass
class Config:
    llama_server_path: str
    models_preset: str | None = None
    port: int = DEFAULT_PORT
    extra_args: str | None = None
    default_model: str | None = None
    open_method: str = "tab"  # "tab" or "workspace"

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
            f"Config file not found: {path}\n" "Create it per README instructions."
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
# Herdr State Manager
# ---------------------------------------------------------------------------


@dataclass
class Herdr:
    """Wraps the herdr CLI — keeps a snapshot of workspaces/tabs/panes and
    running processes for easy interaction."""

    tree: dict[str, dict] = field(default_factory=dict)
    process: list[tuple[str, str]] = field(default_factory=list)

    def __post_init__(self):
        self.tree = {}
        self.process = []
        self.refresh()

    # -- CLI helpers -------------------------------------------------------

    def _herdr(self, *args: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            ["herdr", *args],
            capture_output=True,
            text=True,
            timeout=30,
        )

    def _json(self, result: subprocess.CompletedProcess) -> dict | None:
        try:
            data = json.loads(result.stdout)
            return data.get("result", {})
        except (json.JSONDecodeError, KeyError):
            logger.debug("Invalid JSON from herdr")
            return None

    # -- Tree refresh ------------------------------------------------------

    def refresh(self) -> None:
        """Fetch the full workspace/tab/pane tree and process list."""
        self.tree = {}
        self.process = []

        ws_result = self._herdr("workspace", "list")
        if ws_result.returncode != 0:
            logger.warning("Failed to list workspaces: %s", ws_result.stderr)
            return

        ws_data = self._json(ws_result)
        if not ws_data:
            return

        for ws in ws_data.get("workspaces", []):
            ws_id = ws.get("workspace_id")
            if not ws_id:
                continue
            self.tree[ws_id] = {"tabs": []}

            # List panes for this workspace
            panes_result = self._herdr("pane", "list", "--workspace", ws_id)
            if panes_result.returncode != 0:
                continue
            panes_data = self._json(panes_result)
            if not panes_data:
                continue

            # Group panes by tab
            tab_panes: dict[str, list] = {}
            for pane in panes_data.get("panes", []):
                tab_id = pane.get("tab_id")
                if not tab_id:
                    continue
                tab_panes.setdefault(tab_id, []).append(pane)

            for tab_id, panes in tab_panes.items():
                self.tree[ws_id]["tabs"].append({"tab_id": tab_id, "panes": panes})
                for pane in panes:
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
                        name = proc.get("name", "")
                        if name:
                            self.process.append((name, tab_id))

    # -- Focus / navigation ------------------------------------------------

    def focus_tab(self, tab_id: str) -> bool:
        result = self._herdr("tab", "focus", tab_id)
        if result.returncode != 0:
            logger.warning("Failed to focus tab %s: %s", tab_id, result.stderr)
            return False
        self.refresh()
        logger.info("Focused tab %s", tab_id)
        return True

    # -- Workspace ---------------------------------------------------------

    def create_workspace(self, label: str = "") -> str | None:
        args = ["workspace", "create", "--focus"]
        if label:
            args.extend(["--label", label])
        result = self._herdr(*args)
        if result.returncode != 0:
            logger.error("Failed to create workspace: %s", result.stderr)
            return None
        self.refresh()
        data = self._json(result)
        if not data:
            return None
        return data.get("workspace", {}).get("id")

    def get_workspace(self, ws_id: str) -> dict | None:
        result = self._herdr("workspace", "get", ws_id)
        if result.returncode != 0:
            logger.warning("Failed to get workspace %s: %s", ws_id, result.stderr)
            return None
        data = self._json(result)
        if not data:
            return None
        return data.get("workspace", {})

    # -- Tab ---------------------------------------------------------------

    def create_tab(self, ws_id: str, label: str = "") -> str | None:
        args = ["tab", "create", "--workspace", ws_id]
        if label:
            args.extend(["--label", label])
        result = self._herdr(*args)
        if result.returncode != 0:
            logger.error("Failed to create tab: %s", result.stderr)
            return None
        self.refresh()
        data = self._json(result)
        if not data:
            return None
        return data.get("tab", {}).get("tab_id")

    def close_tab(self, tab_id: str) -> bool:
        logger.info("Closing tab: %s", tab_id)
        result = self._herdr("tab", "close", tab_id)
        if result.returncode != 0:
            logger.error("Failed to close tab: %s", result.stderr)
            return False
        self.refresh()
        logger.info("Tab %s closed", tab_id)
        return True

    # -- Pane --------------------------------------------------------------

    def find_pane(self, tab_id: str) -> dict | None:
        """Return the first pane in a tab."""
        for ws_tabs in self.tree.values():
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
        self.refresh()
        return True

    def close_pane(self, pane_id: str) -> bool:
        result = self._herdr("pane", "close", pane_id)
        if result.returncode != 0:
            logger.error("Failed to close pane: %s", result.stderr)
            return False
        self.refresh()
        return True

    # -- Process -----------------------------------------------------------

    def find_process(self, name: str) -> str | None:
        """Return the first tab_id running a process with the given name."""
        for proc_name, tab_id in self.process:
            if proc_name == name:
                return tab_id
        return None

    # -- Agent reporting ---------------------------------------------------

    def read_pane_source(self, pane_id: str, source: str, lines: int = 2) -> str:
        """Read recent output from a pane source."""
        result = self._herdr("pane", "read", pane_id, "--source", source, "--lines", str(lines))
        if result.returncode != 0:
            logger.debug("Failed to read pane %s source %s: %s", pane_id, source, result.stderr.strip())
            return ""
        return result.stdout.strip()

    def report_agent(
        self,
        pane_id: str,
        agent: str,
        state: str,
        message: str = "",
        state_level: str = "info",
    ) -> bool:
        cmd = [
            "pane",
            "report-agent",
            pane_id,
            "--source",
            AGENT_SOURCE,
            "--agent",
            agent,
            "--state",
            state,
            "--state-level",
            state_level,
        ]
        if message:
            cmd.extend(["--message", message])
        result = self._herdr(*cmd)
        if result.returncode != 0:
            logger.warning(
                "herdr report-agent failed (rc=%d): %s",
                result.returncode,
                result.stderr.strip(),
            )
            return False
        return True

    # -- Convenience queries -----------------------------------------------

    def find_llama_tab(self, model_names: list[str] | None = None) -> str | None:
        """Return tab_id running llama-server. Optionally filter by model agent."""
        return self.find_process("llama-server")

    @property
    def focused_workspace_id(self) -> str | None:
        """Find the currently focused workspace."""
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


# ---------------------------------------------------------------------------
# Agent Reporter
# ---------------------------------------------------------------------------

AGENT_SOURCE = "herdr:llama-server"


class ReporterError(Exception):
    """Agent reporter error."""


@dataclass
class AgentState:
    agent: str
    state: str  # "idle" | "working" | "blocked"
    message: str = ""
    state_level: str = "on"  # "on" | "warning" | "error" | <tps> | <percent> (progress)


class AgentReporter:
    def __init__(self, herdr: Herdr | None = None, herdr_bin_path: str | None = None, tab_id: str | None = None):
        self.herdr = herdr
        self.herdr_bin = herdr_bin_path or os.environ.get("HERDR_BIN_PATH", "herdr")
        self.tab_id = tab_id or os.environ.get("HERDR_PANE_ID", "")

    def report(self, state: AgentState) -> bool:
        """Report semantic agent state (idle/working/blocked)."""
        if not self.tab_id:
            logger.warning("No tab_id set, skipping report-agent")
            return False
        cmd = [
            self.herdr_bin,
            "pane",
            "report-agent",
            self.tab_id,
            "--source",
            AGENT_SOURCE,
            "--agent",
            state.agent,
            "--state",
            state.state,
            "--state-level",
            state.state_level,
        ]
        if state.message:
            cmd.extend(["--message", state.message])

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            if result.returncode != 0:
                logger.warning(
                    "herdr report-agent failed (rc=%d): %s",
                    result.returncode,
                    result.stderr.strip(),
                )
                return False
            return True
        except FileNotFoundError:
            logger.error("herdr binary not found: %s", self.herdr_bin)
            return False
        except subprocess.TimeoutExpired:
            logger.error("herdr report-agent timed out")
            return False
        except Exception as e:
            logger.error("Failed to report agent state: %s", e)
            return False



    def notify(self, message: str) -> None:
        cmd = [self.herdr_bin, "notification", "show", message]
        try:
            subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        except Exception:
            logger.debug("Failed to send notification")

    def report_metadata(
        self,
        title: str | None = None,
        custom_status: str | None = None,
        state_label: str | None = None,
    ) -> bool:
        """Report display metadata (title, custom-status, state-labels)."""
        if not self.tab_id:
            return False
        cmd = [
            self.herdr_bin,
            "pane",
            "report-metadata",
            self.tab_id,
            "--source",
            AGENT_SOURCE,
        ]
        if title is not None:
            cmd.extend(["--title", title])
        if custom_status is not None:
            cmd.extend(["--custom-status", custom_status])
        if state_label is not None:
            cmd.extend(["--state-label", state_label])

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            if result.returncode != 0:
                logger.debug(
                    "herdr report-metadata failed (rc=%d): %s",
                    result.returncode,
                    result.stderr.strip(),
                )
                return False
            return True
        except FileNotFoundError:
            logger.error("herdr binary not found: %s", self.herdr_bin)
            return False
        except subprocess.TimeoutExpired:
            logger.error("herdr report-metadata timed out")
            return False
        except Exception as e:
            logger.error("Failed to report metadata: %s", e)
            return False


# ---------------------------------------------------------------------------
# Llama Client
# ---------------------------------------------------------------------------


class ClientError(Exception):
    """API client error."""


class ServerNotRunningError(ClientError):
    """Server is not responding."""


@dataclass
class CompletionStats:
    tokens_per_sec: float | None = None
    context_size: int | None = None
    context_used: int | None = None


@dataclass
class SlotInfo:
    id: int
    running: bool
    prompt_tokens: int = 0
    predicted_tokens: int = 0


@dataclass
class ModelInfo:
    name: str | None = None
    model_id: str | None = None
    context_size: int | None = None
    architecture: str | None = None
    loaded: bool = False


@dataclass
class PresetModel:
    id: str
    name: str
    context_size: int | None = None
    url: str | None = None


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

    def get_completions(self) -> CompletionStats | None:
        try:
            resp = self._client.get(f"{self.base_url}:{self.port}/completion")
            resp.raise_for_status()
        except httpx.HTTPError as e:
            logger.debug("Failed to get completions: %s", e)
            return None
        data = resp.json()
        return CompletionStats(
            tokens_per_sec=data.get("timings", {}).get("predict_per_token_ms"),
            context_size=data.get("context_size"),
            context_used=data.get("n_past"),
        )

    def get_context_usage(self) -> float | None:
        stats = self.get_completions()
        if not stats or not stats.context_size or not stats.context_used:
            return None
        return stats.context_used / stats.context_size

    def get_slots(self) -> list[SlotInfo]:
        try:
            resp = self._client.get(f"{self.base_url}:{self.port}/slots")
            resp.raise_for_status()
        except httpx.HTTPError as e:
            logger.debug("Failed to get slots: %s", e)
            return []
        slots = []
        for slot in resp.json():
            slots.append(
                SlotInfo(
                    id=slot.get("id", 0),
                    running=slot.get("running", False),
                    prompt_tokens=slot.get("tokens_prompt", 0),
                    predicted_tokens=slot.get("tokens_predicted", 0),
                )
            )
        return slots

    def get_model_info(self) -> ModelInfo | None:
        # /models returns {"data": [{"id": ..., "status": {"value": "loaded"}, ...}]}
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
                return ModelInfo(
                    name=model.get("id"),
                    model_id=model.get("id"),
                    loaded=True,
                )
        # ponytail: if no model is explicitly "loaded", check if any model
        # is in a non-error, non-unloaded state (e.g., "loading" → might be done).
        for model in models:
            status = model.get("status", {})
            if isinstance(status, dict) and status.get("value") not in (
                "unloaded",
                "error",
                None,
            ):
                return ModelInfo(
                    name=model.get("id"),
                    model_id=model.get("id"),
                    loaded=True,
                )
        return None

    def is_model_loaded(self) -> bool:
        info = self.get_model_info()
        return info is not None and info.loaded

    def wait_for_model_load(self, model_id: str, timeout: float = 120.0) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            info = self.get_model_info()
            if info and info.name == model_id and info.loaded:
                return True
            time.sleep(2.0)
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

    @staticmethod
    def get_model_list(preset_path: str | Path) -> list[PresetModel]:
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
                ctx: int | None = None
                if ctx_raw:
                    try:
                        ctx = int(ctx_raw)
                    except ValueError:
                        ctx = None
                models.append(PresetModel(id=key, name=name, context_size=ctx, url=url))
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
                        ctx = None
                models.append(
                    PresetModel(id=section, name=name, context_size=ctx, url=url)
                )
        return models


# ---------------------------------------------------------------------------
# Pane Management (llama-server runs in an external tab)
# ---------------------------------------------------------------------------

HEALTH_CHECK_TIMEOUT = 30
HEALTH_CHECK_INTERVAL = 1.0


def _start_llama_in_tab(config: Config, herdr: Herdr) -> str | None:
    """Create a tab for llama-server based on open_method config."""
    if config.open_method == "workspace":
        return _start_llama_in_workspace(config, herdr)
    else:
        return _start_llama_in_focused_tab(config, herdr)


def _start_llama_in_focused_tab(config: Config, herdr: Herdr) -> str | None:
    """Create a tab in the focused workspace and run llama-server."""
    ws_id = herdr.focused_workspace_id
    if not ws_id:
        logger.error("No focused workspace found")
        return None

    tab_id = herdr.create_tab(ws_id, "llama-server")
    if not tab_id:
        logger.error("Failed to create tab in workspace %s", ws_id)
        return None

    pane = herdr.find_pane(tab_id)
    if not pane:
        logger.error("No pane found in created tab %s", tab_id)
        return None

    pane_id = pane.get("pane_id")
    if not pane_id:
        logger.error("No pane_id in tab %s", tab_id)
        return None
    return _run_llama_in_tab(config, pane_id, herdr)


def _start_llama_in_workspace(config: Config, herdr: Herdr) -> str | None:
    """Create a new workspace and run llama-server in its default tab."""
    ws_id = herdr.create_workspace("llama-server")
    if not ws_id:
        logger.error("Failed to create workspace")
        return None

    ws_info = herdr.get_workspace(ws_id)
    if not ws_info:
        logger.error("Failed to get workspace %s", ws_id)
        return None
    tabs = ws_info.get("tabs", [])
    if not tabs:
        logger.error("No tabs in created workspace %s", ws_id)
        return None

    tab_id = tabs[0].get("tab_id")
    if not tab_id:
        logger.error("No tab_id in workspace %s", ws_id)
        return None

    pane = herdr.find_pane(tab_id)
    if not pane:
        logger.error("No pane found in workspace tab %s", tab_id)
        return None

    pane_id = pane.get("pane_id")
    if not pane_id:
        logger.error("No pane_id in workspace tab %s", tab_id)
        return None
    return _run_llama_in_tab(config, pane_id, herdr)


def _run_llama_in_tab(config: Config, pane_id: str, herdr: Herdr) -> str | None:
    """Run llama-server in the given pane and wait for health check."""
    cmd = [config.llama_server_path]
    if config.models_preset:
        cmd.extend(["--models-preset", config.models_preset])
    cmd.extend(["--port", str(config.port)])
    cmd.extend(config.get_extra_args_list())

    cmd_str = " ".join(cmd)
    if not herdr.run_in_pane(pane_id, cmd_str):
        logger.error("Failed to run in pane %s: command=%s", pane_id, cmd_str)
        return None

    logger.info("llama-server started in pane %s", pane_id)

    if not _health_check(config.port):
        logger.error(
            "llama-server health check failed on port %d (command: %s)",
            config.port,
            cmd_str,
        )
        herdr.close_pane(pane_id)
        return None

    return pane_id


def _health_check(port: int, timeout: float = HEALTH_CHECK_TIMEOUT) -> bool:
    url = f"http://127.0.0.1:{port}/health"
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            resp = httpx.get(url, timeout=2.0)
            if resp.status_code == 200:
                return True
        except httpx.ConnectError:
            pass
        time.sleep(HEALTH_CHECK_INTERVAL)
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
    slots_active: int = 0
    server_running: bool = False
    error: str | None = None


class Watcher:
    def __init__(
        self,
        client: LlamaClient,
        reporter: AgentReporter,
        model_name: str = "unknown",
        herdr: Herdr | None = None,
    ):
        self.client = client
        self.reporter = reporter
        self.model_name = model_name
        self.herdr = herdr
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._stats = WatcherStats()

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
        consecutive_errors = 0
        while not self._stop_event.is_set():
            try:
                self._poll_once()
                consecutive_errors = 0
                time.sleep(POLL_INTERVAL)
            except ServerNotRunningError:
                consecutive_errors += 1
                self._stats.error = "server not responding"
                # ponytail: use report-agent for state change to blocked
                self.reporter.report(
                    AgentState(
                        agent=self.model_name,
                        state="blocked",
                        message=f"server offline (retry {consecutive_errors})",
                        state_level="error",
                    )
                )
                if consecutive_errors >= CONNECTION_RETRY_MAX:
                    self.reporter.notify(
                        "llama-server connection lost. Check if the server is running."
                    )
                    logger.error("Max connection retries reached. Stopping watcher.")
                    break
                time.sleep(CONNECTION_RETRY_INTERVAL)
            except Exception as e:
                consecutive_errors += 1
                self._stats.error = str(e)
                logger.warning("Watcher poll error: %s", e)
                time.sleep(POLL_INTERVAL)

    def _poll_once(self) -> None:
        if not self.client.is_running():
            raise ServerNotRunningError("Server not responding")

        self._stats.server_running = True
        self._stats.error = None

        # ponytail: only check /models — /completion and /slots may not exist
        # on every llama-server build, and erroring every cycle is noise.
        model_info = self.client.get_model_info()
        if model_info and model_info.name:
            self._stats.model_name = model_info.name
            self._stats.tokens_per_sec = None
            self._stats.context_usage = None
            self._stats.slots_active = 0

        state = self._determine_state()
        message = self._format_message(state)
        state_level = self._determine_state_level()

        # ponytail: use report-metadata for polling updates — state stays
        # whatever report-agent last set (on/working/blocked)
        title = f"{self.model_name} — {state}"
        custom_status = message or state_level
        self.reporter.report_metadata(title=title, custom_status=custom_status, state_label=state_level)

    def _determine_state(self) -> str:
        # ponytail: use pane output to determine state — more reliable than
        # /completion endpoint which may not exist on all builds
        tab_id = self.reporter.tab_id
        if tab_id and self.herdr:
            pane = self.herdr.find_pane(tab_id)
            if pane:
                pane_id = pane.get("pane_id")
                if pane_id:
                    output = self.herdr.read_pane_source(pane_id, "recent-unwrapped", lines=2)
                    if output:
                        lines = [l.strip() for l in output.splitlines() if l.strip()]
                        if lines:
                            last_line = lines[-1]
                            if re.search(r"tg\s*=\s*[\d.]+\s*t/s", last_line, re.IGNORECASE):
                                return "working"
                            if re.search(r"progress\s*=\s*[\d.]+", last_line, re.IGNORECASE):
                                return "working"
                            output_lower = output.lower()
                            if "error" in output_lower or "fail" in output_lower:
                                return "blocked"

        if self._stats.tokens_per_sec and self._stats.tokens_per_sec > 0:
            return "working"
        return "idle"

    def _determine_state_level(self) -> str:
        """Determine state-level from recent pane output.

        Parses the last non-empty line for llama-server timing patterns:
        - `tg = X.XX t/s` → `{X.XX} tps`
        - `progress = X.XX` → `{X:.0f}% (progress)`
        Falls back to keyword matching for error/warning.
        """
        tab_id = self.reporter.tab_id
        if not tab_id or not self.herdr:
            return "on"
        pane = self.herdr.find_pane(tab_id)
        if not pane:
            return "on"
        pane_id = pane.get("pane_id")
        if not pane_id:
            return "on"
        output = self.herdr.read_pane_source(pane_id, "recent-unwrapped", lines=2)
        if not output:
            return "on"

        # Get last non-empty line
        lines = [l.strip() for l in output.splitlines() if l.strip()]
        if not lines:
            return "on"
        last_line = lines[-1]

        # ponytail: parse llama-server timing output for state-level
        # Match: tg = 18.75 t/s
        tg_match = re.search(r"tg\s*=\s*([\d.]+)\s*t/s", last_line, re.IGNORECASE)
        if tg_match:
            return f"{tg_match.group(1)} tps"

        # Match: progress = 0.93
        progress_match = re.search(r"progress\s*=\s*([\d.]+)", last_line, re.IGNORECASE)
        if progress_match:
            pct = float(progress_match.group(1)) * 100
            return f"{pct:.0f}% (progress)"

        # Fallback: keyword matching
        output_lower = output.lower()
        if "error" in output_lower or "fail" in output_lower:
            return "error"
        if "warn" in output_lower:
            return "warning"
        return "on"

    def _format_message(self, state: str) -> str:
        if state == "working":
            tps = self._stats.tokens_per_sec or 0
            return f"{tps:.1f} tok/s"
        elif state == "idle":
            return ""
        return ""


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------

app = typer.Typer(name="herdr-llama", help="Llama Server Agent plugin for Herdr")
console = Console()


def _print_error(message: str) -> None:
    console.print(f"[red]Error:[/red] {message}")


def _print_info(message: str) -> None:
    console.print(f"[blue]Info:[/blue] {message}")


def _load_models(config: Config) -> list[PresetModel]:
    if not config.models_preset:
        _print_error("No models-preset configured. Edit config to add one.")
        return []
    models = LlamaClient.get_model_list(config.models_preset)
    # Filter: only show models that exist in the preset (server may have
    # additional models loaded via API that aren't in the preset).
    return models


def _prompt_model_choice(models: list[PresetModel]) -> str | None:
    if not models:
        return None

    choices = [f"{m.name}" for m in models]
    selected = questionary.select(
        "Select a model",
        choices=choices,
    ).ask()
    if not selected:
        return None
    for m in models:
        if m.name == selected:
            return m.id
    return None


def _start_server(config: Config, herdr: Herdr) -> str | None:
    """Start llama-server in a new tab. Returns tab_id or None."""
    server_path = config.llama_server_path
    if not os.path.isfile(server_path):
        _print_error(f"llama-server binary not found: {server_path}")
        return None
    if not os.access(server_path, os.X_OK):
        _print_error(f"llama-server not executable: {server_path}")
        return None

    tab_id = _start_llama_in_tab(config, herdr)
    if tab_id:
        _print_info(f"Server started in tab {tab_id}")
    else:
        _print_error(
            f"Failed to start server in tab (check: binary exists, "
            f"port {config.port} available, herdr pane run works). "
            f"See logs for details."
        )
    return tab_id


def _load_model(client: LlamaClient, model_id: str) -> bool:
    _print_info(f"Loading model: {model_id}")
    if client.load_model(model_id):
        _print_info("Model load initiated. Waiting...")
        if client.wait_for_model_load(model_id):
            _print_info("Model loaded successfully")
            return True
        _print_error("Model load timed out")
        return False
    _print_error("Failed to load model")
    return False


def _unload_model(client: LlamaClient, model_id: str) -> bool:
    _print_info(f"Unloading model: {model_id}")
    if client.unload_model(model_id):
        _print_info("Model unloaded")
        return True
    _print_error("Failed to unload model")
    return False


def _report_agent(reporter: AgentReporter, model_id: str, pane_id: str, herdr: Herdr | None, state: str = "working") -> None:
    # ponytail: poll pane for recent output to determine state-level
    if not herdr:
        return
    output = herdr.read_pane_source(pane_id, "recent-unwrapped", lines=2)
    state_level = _parse_state_level(output)
    agent_state = AgentState(agent=model_id, state=state, state_level=state_level)
    reporter.report(agent_state)


def _parse_state_level(output: str) -> str:
    """Parse state-level from pane output using the same logic as Watcher."""
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


def _start_watcher(
    client: LlamaClient,
    reporter: AgentReporter,
    model_id: str,
    tab_id: str,
    original_tab_id: str | None,
    herdr: Herdr | None = None,
) -> Watcher:
    # ponytail: wait for model to be fully loaded before reporting
    if not client.wait_for_model_load(model_id, timeout=120.0):
        _print_error(f"Model {model_id} failed to load within 120s")
        raise RuntimeError(f"Model {model_id} failed to load")
    _print_info(f"Model {model_id} loaded, starting watcher")
    reporter.tab_id = tab_id
    # ponytail: agent already reported at server start — just update metadata
    reporter.report_metadata(title=model_id, custom_status="loaded")
    watcher = Watcher(client=client, reporter=reporter, model_name=model_id, herdr=herdr)
    watcher.start()
    _print_info(f"Dashboard closing. Watcher running for {model_id} in tab {tab_id}.")

    # Refocus the original tab so the overlay doesn't block the screen
    if original_tab_id:
        _focus_tab(original_tab_id)
    else:
        _print_info("No original tab_id to refocus — dashboard closed.")
    return watcher


def _get_current_tab_id() -> str | None:
    """Get the original tab_id set by the bin wrapper."""
    return os.environ.get("HERDR_TAB_ID") or None


def _focus_tab(tab_id: str) -> None:
    """Focus a tab with `herdr tab focus <tab_id>`."""
    herdr = Herdr()
    herdr.focus_tab(tab_id)


def _run_dashboard() -> None:
    try:
        config = load_config()
    except ConfigError as e:
        _print_error(str(e))
        return

    herdr = Herdr()

    client = LlamaClient(port=config.port)
    reporter = AgentReporter(herdr=herdr)
    models = _load_models(config)

    # Capture the original tab_id before we do anything
    original_tab_id = _get_current_tab_id()

    # Track watcher/tab for quit cleanup
    watcher: Watcher | None = None
    server_tab_id: str | None = None

    try:
        # Find existing llama-server tab
        tab_id = herdr.find_llama_tab()
        logger.info("find_llama_tab returned: %s", tab_id)

        if not tab_id:
            _print_info("llama-server is not running, starting...")
            new_tab_id = _start_server(config, herdr)
            if not new_tab_id:
                return

            tab_id = new_tab_id
            server_tab_id = tab_id
            reporter.tab_id = tab_id

            # ponytail: report initial state — agent is "Not-loaded", state is idle
            _report_agent(reporter, "Not-loaded", tab_id, herdr, state="idle")

            _print_info(f"Server started in tab {tab_id}")

            if not models:
                _print_error("No models available. Add a models-preset to config.")
                return

            model_id = _prompt_model_choice(models)
            if not model_id:
                return

            if _load_model(client, model_id):
                reporter.report_metadata(title=model_id, custom_status="loaded")
                watcher = _start_watcher(
                    client,
                    reporter,
                    model_id,
                    tab_id=tab_id,
                    original_tab_id=original_tab_id,
                )
                return
        else:
            _print_info(f"llama-server running in tab {tab_id}")
            server_tab_id = tab_id
            reporter.tab_id = tab_id

            model_info = client.get_model_info()
            if model_info and model_info.loaded:
                loaded_name = model_info.name or "unknown"
                _print_info(f"Model loaded: {loaded_name}")
                choice = questionary.select(
                    "What would you like to do? (Esc to exit)",
                    choices=["unload", "stop-server"],
                    default="unload",
                ).ask()

                if choice == "unload":
                    unload_id = model_info.model_id or loaded_name
                    if _unload_model(client, unload_id):
                        _print_info("Model unloaded. Dashboard closing.")
                        return
                elif choice == "stop-server":
                    confirm = questionary.select(
                        "Stop server (close pane)?",
                        choices=["Yes", "No"],
                        default="No",
                    ).ask()
                    if confirm == "Yes":
                        herdr.refresh()
                        pane = herdr.find_pane(tab_id)
                        if pane:
                            pane_id = pane.get("pane_id")
                            if watcher:
                                watcher.stop()
                            if pane_id and herdr.close_pane(pane_id):
                                _print_info("Server stopped (pane closed).")
                                return
                        else:
                            _print_error(f"No pane found in tab {tab_id}")
                            return
            else:
                if not models:
                    _print_error("No models available. Add a models-preset to config.")
                    return

                model_id = _prompt_model_choice(models)
                if model_id and _load_model(client, model_id):
                    reporter.report_metadata(title=model_id, custom_status="loaded")
                    _start_watcher(
                        client,
                        reporter,
                        model_id,
                        tab_id=tab_id,
                        original_tab_id=original_tab_id,
                    )
                    return
    finally:
        client.close()


@app.command()
def dashboard():
    """Open the llama-server dashboard."""
    _run_dashboard()


def main() -> None:
    app()


if __name__ == "__main__":
    main()
