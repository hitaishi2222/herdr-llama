#!/usr/bin/env python3
"""herdr-llama daemon — polls llama-server and reports state via report-agent/report-metadata.

The daemon does NOT manage server lifecycle or run herdr CLI commands.
All decisions (start/stop server, load/unload models, claim tabs) are made
by herdr-llama.py. The daemon only:
  1. Polls llama-server HTTP API for state
  2. Calls report-agent / report-metadata based on state
  3. Exposes current state via Unix socket for herdr-llama.py to read
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
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import httpx

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

REQUIRED_SECTIONS = ["server"]
REQUIRED_KEYS = ["llama-server-path", "port"]
DEFAULT_PORT = 8080

PLUGIN_ID = "herdr-llama"
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
    log_file_size: int = 5  # KB
    update_rate: float = 1.0  # seconds

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
    log_file_size_raw = parser.get("server", "log-file-size", fallback="5")
    update_rate_raw = parser.get("server", "update-rate", fallback="1")

    port_str = parser.get("server", "port")
    try:
        port = int(port_str)
    except ValueError:
        raise ConfigError(f"Invalid port value: {port_str!r}")

    try:
        log_file_size = int(log_file_size_raw)
    except ValueError:
        raise ConfigError(f"Invalid log-file-size value: {log_file_size_raw!r}")

    try:
        update_rate = float(update_rate_raw)
        if update_rate <= 0:
            raise ValueError("update-rate must be positive")
    except ValueError:
        raise ConfigError(f"Invalid update-rate value: {update_rate_raw!r}")

    return Config(
        llama_server_path=parser.get("server", "llama-server-path"),
        models_preset=models_preset,
        port=port,
        extra_args=extra_args,
        default_model=default_model,
        open_method=open_method,
        log_file_size=log_file_size,
        update_rate=update_rate,
    )


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


def setup_logging() -> logging.Logger:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("herdr-llama-daemon")
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    fh = logging.FileHandler(str(LOG_FILE))
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    sh = logging.StreamHandler(sys.stderr)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    return logger


logger = setup_logging()


# ---------------------------------------------------------------------------
# Minimal Herdr wrapper — only report-agent and report-metadata
# ---------------------------------------------------------------------------


class Herdr:
    """Minimal herdr CLI wrapper. Only report-agent and report-metadata."""

    def __init__(self):
        self.herdr_bin = os.environ.get("HERDR_BIN_PATH", "herdr")

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
        ]
        if message:
            cmd.extend(["--message", message])
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        if result.returncode != 0:
            return False
        return True

    def report_metadata(
        self,
        pane_id: str,
        title: str | None = None,
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
        if state_label is not None:
            cmd.extend(["--state-label", state_label])
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        if result.returncode != 0:
            return False
        return True

    def read_pane_source(
        self, pane_id: str, source: str = "visible", lines: int = 10
    ) -> str:
        """Read recent pane output to detect inference activity."""
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
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        return result.stdout.strip() if result.returncode == 0 else ""


# ---------------------------------------------------------------------------
# Llama Client (HTTP, not herdr)
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
# Watcher
# ---------------------------------------------------------------------------

POLL_INTERVAL = 1.0
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
    state: str = "idle"
    progress: float | None = None


class Watcher:
    """Polls llama-server for stats and state changes."""

    def __init__(
        self,
        client: LlamaClient,
        herdr: Herdr,
        pane_id: str,
        daemon: Optional["Daemon"] = None,
        update_rate: float = 1.0,
    ):
        self.client = client
        self.herdr = herdr
        self.pane_id = pane_id
        self.daemon = daemon
        self.update_rate = update_rate
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._stats = WatcherStats()
        self._last_state = "idle"
        self._last_model: str | None = None
        self._consecutive_errors = 0
        self._last_reported_agent_state: str | None = None
        self._last_reported_label: str | None = None

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
        start_time = time.monotonic()
        while not self._stop_event.is_set():
            try:
                self._poll_once()
                self._consecutive_errors = 0
                time.sleep(self.update_rate)
            except Exception as e:
                self._consecutive_errors += 1
                self._stats.error = str(e)
                logger.warning("Watcher poll error: %s", e)
                if self._consecutive_errors >= CONNECTION_RETRY_MAX:
                    elapsed = time.monotonic() - start_time
                    logger.error(
                        "Max connection retries reached (%d errors over %.0fs). Stopping watcher.",
                        self._consecutive_errors,
                        elapsed,
                    )
                    break
                time.sleep(CONNECTION_RETRY_INTERVAL)

    def _poll_once(self) -> None:
        server_was_online = self._stats.server_running
        if not self.client.is_running():
            self._stats.server_running = False
            self._stats.error = "server not responding"
            if self.pane_id:
                self._report_state("blocked", "server offline")
            if self.daemon and server_was_online:
                logger.info("Watcher detected server offline")
                self.daemon.state = DaemonState.NO_SERVER
            return

        self._stats.server_running = True
        self._stats.error = None

        # Server recovered from offline — re-read pane state and report
        # so the label updates from any stale "blocked" label.
        if self.pane_id and not server_was_online:
            state, label = self._read_pane_state()
            self._stats.state = state
            if state != self._last_reported_agent_state:
                self._report_state(state, "")
            if label != self._last_reported_label:
                self._report_metadata(state, label)

        model_info = self.client.get_model_info()
        current_model = model_info.get("id") if model_info else None
        if current_model:
            self._stats.model_name = current_model
            self._stats.model_id = current_model
        else:
            self._stats.model_name = None
            self._stats.model_id = None

        if self.daemon:
            if current_model and self._last_model != current_model:
                logger.info(
                    "[STATE-TRANSITION] model changed: %s -> %s, loaded=%s",
                    self._last_model,
                    current_model,
                    self.client.is_model_loaded(),
                )
                if self.client.is_model_loaded():
                    logger.info("Watcher detected model loaded: %s", current_model)
                    with self.daemon._lock:
                        self.daemon.current_model = current_model
                        self.daemon.state = DaemonState.MODEL_LOADED
                    # Register agent with model name and set state to idle
                    self._report_state("idle", "ready")
                else:
                    with self.daemon._lock:
                        self.daemon.state = DaemonState.MODEL_LOADING
                        self.daemon.current_model = current_model
            elif not current_model and self._last_model:
                logger.info("Watcher detected model unloaded")
                with self.daemon._lock:
                    self.daemon.current_model = None
                    self.daemon.state = DaemonState.SERVER_RUNNING
            elif current_model and self._last_model is None:
                with self.daemon._lock:
                    self.daemon.current_model = current_model
                    # Honor the state set by load_model() (MODEL_LOADING)
                    # or detect immediate load. Don't override to SERVER_RUNNING.
                    if self.daemon.state == DaemonState.MODEL_LOADING:
                        pass  # keep MODEL_LOADING, watcher will transition when loaded
                    elif self.client.is_model_loaded():
                        self.daemon.state = DaemonState.MODEL_LOADED
                    else:
                        self.daemon.state = DaemonState.MODEL_LOADING
            elif not current_model and self._last_model is None:
                with self.daemon._lock:
                    if self.daemon.state == DaemonState.NO_SERVER:
                        self.daemon.state = DaemonState.SERVER_RUNNING
            elif current_model and self._last_model == current_model:
                with self.daemon._lock:
                    if (
                        self.daemon.state == DaemonState.MODEL_LOADING
                        and self.client.is_model_loaded()
                    ):
                        logger.info(
                            "Watcher detected model finished loading: %s", current_model
                        )
                        self.daemon.state = DaemonState.MODEL_LOADED
                        # Confirm via /models — report loaded state
                        self._report_state("unknown", "")
                        self._report_metadata("unknown", "Loaded")
            elif self.daemon.state == DaemonState.MODEL_LOADING and not current_model:
                with self.daemon._lock:
                    logger.info("Watcher detected model load failed")
                    self.daemon.current_model = None
                    self.daemon.state = DaemonState.SERVER_RUNNING
            self._last_model = current_model

        completion = self.client.get_completions()
        if completion:
            timings = completion.get("timings", {})
            self._stats.tokens_per_sec = timings.get("predict_per_token_ms")
            ctx_size = completion.get("context_size")
            ctx_used = completion.get("n_past")
            if ctx_size and ctx_used:
                self._stats.context_usage = ctx_used / ctx_size

        slots = self.client.get_slots()
        self._stats.slots_active = sum(1 for s in slots if s.get("running"))

        if self.pane_id:
            state, label = self._read_pane_state()
            self._stats.state = state

            # Only report if state or label actually changed
            agent_state_changed = state != self._last_reported_agent_state
            label_changed = label != self._last_reported_label
            if agent_state_changed:
                self._report_state(state, "")
            if label_changed:
                self._report_metadata(state, label)

    def _read_pane_state(self) -> tuple[str, str]:
        """Find all signals in output, return the last (most recent) match."""
        if not self._stats.server_running or not self.pane_id:
            return "idle", "on"

        output = self.herdr.read_pane_source(self.pane_id, "visible", lines=5)
        lines = output.splitlines()
        if not lines:
            if self._last_reported_agent_state:
                return self._last_reported_agent_state, self._last_reported_label or "on"
            return "idle", "on"

        last_line = lines[-1].strip()

        # Parse the last line only.
        tg = re.search(r"tg\s*=\s*([\d.]+)\s*t/s", last_line)
        if tg:
            return "working", f"{float(tg.group(1)):.1f} tps"

        proc = re.search(r"process\s*=\s*([\d.]+)\s*t/s", last_line)
        if proc:
            return "working", f"{float(proc.group(1)):.1f} tps"

        progress = re.search(r"progress\s*=\s*([\d.]+)", last_line)
        if progress:
            return "working", f"Processing({int(float(progress.group(1)) * 100)}%)"

        m = re.search(r"\{.*\}", last_line)
        if m:
            try:
                data = json.loads(m.group())
                if data.get("state") == "loading":
                    val = data.get("payload", {}).get("value")
                    if isinstance(val, (int, float)):
                        return "working", f"Loading({int(val * 100)}%)"
            except (json.JSONDecodeError, ValueError):
                pass

        if "stop processing" in last_line:
            return "idle", "ready"

        # No signal — keep last known state.
        if self._last_reported_agent_state:
            return self._last_reported_agent_state, self._last_reported_label or "on"
        return "idle", "on"

    def _report_state(self, state: str, message: str) -> None:
        """Update agent state via report-agent (required before state-label)."""
        model_name = self._stats.model_name or "unknown"
        valid_state = (
            state if state in ("idle", "working", "blocked", "unknown") else "unknown"
        )

        self.herdr.report_agent(self.pane_id, model_name, valid_state, message)
        self._last_state = state
        self._last_reported_agent_state = valid_state

    def _report_metadata(self, state: str, label: str) -> None:
        """Update metadata with state-label (state must be set first via report-agent)."""
        model_name = self._stats.model_name or "unknown"
        title = f"{model_name} — {state}"
        label_value = f"{state}={label}"

        self.herdr.report_metadata(
            self.pane_id,
            title=title,
            state_label=label_value,
        )
        self._last_reported_label = label_value


# ---------------------------------------------------------------------------
# Daemon State Machine
# ---------------------------------------------------------------------------


class DaemonState:
    NO_SERVER = "no_server"
    SERVER_STARTING = "server_starting"
    SERVER_RUNNING = "server_running"
    MODEL_LOADING = "model_loading"
    MODEL_LOADED = "model_loaded"


class Daemon:
    """Daemon: polls llama-server and reports state. No server lifecycle."""

    def __init__(self):
        self.config = load_config()
        self.herdr = Herdr()
        self.client = LlamaClient(port=self.config.port)
        self.watcher: Watcher | None = None
        self._lock = threading.Lock()
        self.state = DaemonState.NO_SERVER
        self.current_model: str | None = None
        self._running = False
        self.pane_id: str | None = None
        self.tab_id: str | None = None
        self.server_socket: socket.socket | None = None

    def set_pane(self, tab_id: str, pane_id: str) -> None:
        """Tell daemon which pane to report to (set by herdr-llama.py after starting server)."""
        with self._lock:
            self.tab_id = tab_id
            self.pane_id = pane_id
        logger.info("Pane set: tab=%s pane=%s", tab_id, pane_id)

        # Register agent once with herdr (creates the glacial widget overview)
        self.herdr.report_agent(
            pane_id,
            agent="llama-server",
            state="idle",
            message="ready",
        )

        self._start_watcher()

        # If a model is already loaded, report metadata immediately so the
        # title shows up as soon as we know which pane to report to.
        if self.client.is_model_loaded():
            self._report_state("idle", "ready")
            self._report_metadata(self._stats.state or "idle")

    def start(self) -> bool:
        """Start the daemon polling loop."""
        if self._running:
            logger.warning("Daemon already running")
            return False

        logger.info("Starting daemon")
        with self._lock:
            self._running = True
            # Check if llama-server is already running on config port
            if self.client.is_running():
                self.state = DaemonState.SERVER_RUNNING
                if self.client.is_model_loaded():
                    model_info = self.client.get_model_info()
                    self.current_model = model_info.get("id") if model_info else None
                    self.state = DaemonState.MODEL_LOADED
                self._start_watcher()
                return True

            self.state = DaemonState.NO_SERVER
        return True

    def stop(self) -> bool:
        logger.info("Stopping daemon")
        with self._lock:
            self._running = False
            self.state = DaemonState.NO_SERVER
        if self.watcher:
            self.watcher.stop()
            self.watcher = None
        self.client.close()
        logger.info("Daemon stopped")
        return True

    def load_model(self, model_id: str) -> bool:
        with self._lock:
            if self.state not in (DaemonState.SERVER_RUNNING, DaemonState.MODEL_LOADED):
                logger.error("Cannot load model in state %s", self.state)
                return False

            if (
                self.current_model == model_id
                and self.state == DaemonState.MODEL_LOADED
            ):
                logger.info("Model %s already loaded", model_id)
                return True

            logger.info("Loading model: %s", model_id)
            self.state = DaemonState.MODEL_LOADING
            self.current_model = model_id

        if self.client.load_model(model_id):
            self._start_watcher()
            return True
        else:
            logger.error("Failed to load model")
            with self._lock:
                self.state = DaemonState.SERVER_RUNNING
            return False

    def unload_model(self) -> bool:
        with self._lock:
            if not self.current_model:
                logger.warning("No model loaded")
                return False

            logger.info("Unloading model: %s", self.current_model)
            model_id = self.current_model
            self.current_model = None
            self.state = DaemonState.SERVER_RUNNING

        if self.client.unload_model(model_id):
            if self.watcher:
                self.watcher.stop()
            return True
        # Restore state on failure
        with self._lock:
            self.current_model = model_id
            self.state = DaemonState.MODEL_LOADED
        return False

    def get_status(self) -> dict:
        """Get current daemon status."""
        with self._lock:
            # Sync with actual server state if we have a pane
            if self.pane_id:
                if self.client.is_model_loaded():
                    model_info = self.client.get_model_info()
                    self.current_model = model_info.get("id") if model_info else None
                    self.state = DaemonState.MODEL_LOADED
                elif self.state == DaemonState.MODEL_LOADED:
                    self.current_model = None
                    self.state = DaemonState.SERVER_RUNNING

            state = self.state
            model = self.current_model
            tab_id = self.tab_id
            pane_id = self.pane_id

        stats = self.watcher.stats if self.watcher else WatcherStats()
        error = stats.error if state != DaemonState.NO_SERVER else None
        return {
            "state": state,
            "model": model if state != DaemonState.NO_SERVER else None,
            "tokens_per_sec": stats.tokens_per_sec,
            "context_usage": stats.context_usage,
            "slots_active": stats.slots_active,
            "server_running": stats.server_running,
            "error": error,
            "tab_id": tab_id,
            "pane_id": pane_id,
        }

    def _start_watcher(self) -> None:
        if self.watcher and self.watcher._thread and self.watcher._thread.is_alive():
            return
        if not self.pane_id:
            logger.warning("No pane_id set — watcher cannot report state")
            return
        self.watcher = Watcher(
            client=self.client,
            herdr=self.herdr,
            pane_id=self.pane_id,
            daemon=self,
            update_rate=self.config.update_rate,
        )
        self.watcher.start()


# ---------------------------------------------------------------------------
# Socket Server
# ---------------------------------------------------------------------------


def handle_client(
    conn: socket.socket, daemon: Daemon, server_socket: socket.socket
) -> None:
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

        if cmd == "stop":
            daemon.stop()
            server_socket.close()
            if SOCKET_PATH.exists():
                SOCKET_PATH.unlink()
            logger.info("Daemon stopped via stop command")
    except Exception as e:
        logger.error("Error handling client: %s", e)
        try:
            conn.sendall(json.dumps({"error": str(e)}).encode("utf-8") + b"\n")
        except Exception:
            pass
    finally:
        conn.close()


def _handle_command(cmd: str, arg: str | None, daemon: Daemon) -> dict:
    if cmd == "status":
        return daemon.get_status()

    elif cmd == "set-pane":
        if not arg:
            return {"error": "Expected: set-pane <tab_id> <pane_id>"}
        parts = arg.split(maxsplit=1)
        if len(parts) < 2:
            return {"error": "Expected: set-pane <tab_id> <pane_id>"}
        tab_id, pane_id = parts[0], parts[1]
        daemon.set_pane(tab_id, pane_id)
        return {"ok": True}

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
        return {"ok": True}

    else:
        return {"error": f"Unknown command: {cmd}"}


def start_socket_server(daemon: Daemon) -> socket.socket:
    SOCKET_PATH.parent.mkdir(parents=True, exist_ok=True)
    if SOCKET_PATH.exists():
        SOCKET_PATH.unlink()

    server_socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server_socket.bind(str(SOCKET_PATH))
    server_socket.listen(5)
    server_socket.settimeout(1.0)

    logger.info("Socket server listening on %s", SOCKET_PATH)
    return server_socket


def socket_loop(server_socket: socket.socket, daemon: Daemon) -> None:
    client_threads: list[threading.Thread] = []
    while daemon._running:
        try:
            conn, _ = server_socket.accept()
            t = threading.Thread(
                target=handle_client, args=(conn, daemon, server_socket), daemon=False
            )
            t.start()
            client_threads.append(t)
        except socket.timeout:
            client_threads = [t for t in client_threads if t.is_alive()]
            continue
        except OSError:
            break
    for t in client_threads:
        t.join(timeout=5.0)


# ---------------------------------------------------------------------------
# Main
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


def _check_config() -> Config | None:
    try:
        return load_config()
    except ConfigError as e:
        print(f"Config error: {e}", file=sys.stderr)
        return None


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
        s.close()
        return json.loads(data.decode("utf-8").strip())
    except Exception as e:
        return {"error": str(e)}
    finally:
        try:
            s.close()
        except Exception:
            pass


def _check_log_size() -> None:
    """Check if log file exceeds size limit and truncate if necessary."""
    if not LOG_FILE.exists():
        return

    size_bytes = LOG_FILE.stat().st_size
    size_kb = size_bytes / 1024

    # Get config to check the limit
    try:
        config = load_config()
        limit_kb = config.log_file_size
    except ConfigError:
        limit_kb = 5  # Default if config not available

    if size_kb > limit_kb:
        logger.info(
            "Log file too large (%.1f KB > %d KB), truncating", size_kb, limit_kb
        )
        # Notify user
        try:
            subprocess.run(
                [
                    "herdr",
                    "notification",
                    "show",
                    "herdr-llama",
                    "--body",
                    f"Log file was {size_kb:.0f} KB (limit: {limit_kb} KB). Truncated for this session.",
                ],
                capture_output=True,
                timeout=10,
            )
        except Exception:
            pass
        # Truncate the log file
        LOG_FILE.write_text("")


def _start_daemon():
    """Start daemon in background."""
    if _is_daemon_running():
        print("Daemon already running")
        sys.exit(1)

    STATE_DIR.mkdir(parents=True, exist_ok=True)

    try:
        pid = os.fork()
        if pid > 0:
            print("Daemon started")
            sys.exit(0)
    except OSError as e:
        print(f"Fork failed: {e}")
        sys.exit(1)

    os.setsid()
    sys.stdin = open(os.devnull, "r")
    sys.stdout = open(LOG_FILE, "a")
    sys.stderr = open(LOG_FILE, "a")

    if _check_config() is None:
        sys.exit(1)

    # Check log file size and truncate if necessary
    _check_log_size()

    try:
        daemon = Daemon()
        if not daemon.start():
            logger.error("Failed to start daemon")
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

        logger.info("Daemon running (PID: %d)", os.getpid())
        socket_loop(server_socket, daemon)

    except Exception as e:
        logger.error("Daemon error: %s", e, exc_info=True)
        sys.exit(1)


def _stop_daemon():
    """Stop daemon."""
    if not _is_daemon_running():
        print("Daemon not running")
        sys.exit(0)

    result = _send_command("stop")
    if result.get("ok"):
        print("Daemon stopped")
    else:
        print(f"Failed to stop: {result.get('error')}")
        sys.exit(1)


def _show_status():
    """Check daemon status."""
    if not _is_daemon_running():
        print("Daemon not running")
        sys.exit(1)

    result = _send_command("status")
    state = result.get("state", "unknown")
    model = result.get("model", "none")
    tps = result.get("tokens_per_sec")
    tab_id = result.get("tab_id")
    pane_id = result.get("pane_id")

    if state == DaemonState.NO_SERVER:
        print("Daemon not running")
    else:
        print(f"State: {state}")
        print(f"Model: {model}")
        if tps is not None:
            print(f"Tokens/sec: {tps:.1f}")
        if tab_id:
            print(f"Tab: {tab_id}")
        if pane_id:
            print(f"Pane: {pane_id}")
        if not tab_id and not pane_id:
            print("No herdr tab/pane claimed for llama-server")
    if result.get("error"):
        print(f"Error: {result['error']}")


def _run_foreground():
    """Run daemon in foreground (for debugging)."""
    # Check log file size and truncate if necessary
    _check_log_size()

    try:
        daemon = Daemon()
        if not daemon.start():
            print("Failed to start daemon")
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
        print(f"Error: {e}")
        logger.error("Daemon error", exc_info=True)
        sys.exit(1)


def main() -> None:
    if len(sys.argv) > 1:
        cmd = sys.argv[1]
        if cmd == "start":
            _start_daemon()
        elif cmd == "stop":
            _stop_daemon()
        elif cmd == "status":
            _show_status()
        elif cmd == "run":
            _run_foreground()
        else:
            print(f"Unknown command: {cmd}")
            sys.exit(1)
    else:
        _start_daemon()


if __name__ == "__main__":
    main()
