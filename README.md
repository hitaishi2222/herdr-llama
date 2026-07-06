# herdr-llama

A [Herdr](https://github.com/nicehash/herdr) plugin that integrates [llama-server](https://github.com/ggml-org/llama.cpp/tree/master/examples/server) as an agent with a real-time stats dashboard.

**Trigger**: `alt+l` opens an overlay for model selection and server control. A background daemon continuously polls llama-server and reports tokens/sec, processing state, and slot status to Herdr.

## Architecture

Two processes work together:

- **Daemon** (`herdr-llama-daemon.py`): Long-lived background process that polls llama-server HTTP API (`/health` every 10s, `/models` every 1s) and reads the herdr pane output for TPS/processing state, calls `report-agent`/`report-metadata` based on state, exposes state via Unix socket. Does NOT manage server lifecycle.
- **InquirerPy app** (`herdr-llama.py`): Handles server lifecycle (start/stop via herdr CLI), daemon lifecycle, and the interactive dashboard. Starts the daemon if not running, sends commands via Unix socket.

```
alt+l pressed
    │
    ▼
herdr-llama.py (InquirerPy app)
    ├── Determines which of 4 cases applies (daemon/server running/not running)
    ├── Starts server + daemon if needed (via ServerManager)
    ├── Sends commands to daemon via Unix socket
    └── Presents interactive choices based on daemon state (InquirerPy)
          │
          ▼
herdr-llama-daemon.py (Daemon)
    ├── Polls /health every 10s, /models every 1s, reads pane output for TPS/processing (Watcher thread)
    ├── Calls report-agent / report-metadata based on state
    └── Exposes state via Unix socket for InquirerPy app
```

## Features

- **One-shot dashboard**: Start server, load/unload models, stop server — all from a single overlay
- **Real-time stats**: Background daemon reads the herdr pane output for tokens/sec and processing state, reports to Herdr agent system
- **State-aware**: Agent state maps to server/model state (idle/working/blocked)
- **Crash detection**: Daemon detects server crashes via health check, sends notification
- **`--metrics` auto-enable**: Plugin automatically adds `--metrics` flag when starting llama-server
- **Persistent daemon**: Daemon keeps running after dashboard closes, continues polling

## Installation

### Prerequisites

1. [llama-server](https://github.com/ggml-org/llama.cpp/releases) installed and accessible
2. A [model preset file](#model-preset) in `.ini` format
3. Herdr plugin system (v0.7.0+)

### Setup

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Create the config file:

```bash
cat > ~/.config/herdr/plugins/config/herdr-llama/herdr-llama.ini << 'EOF'
[server]
llama-server-path = /usr/local/bin/llama-server
models-preset = /path/to/your/models.ini
port = 8080
extra-args = --threads 4
default-model = llama-3-8b
open-method = tab
update-rate = 1
EOF
```

3. Register the plugin with Herdr (see Herdr docs for plugin registration).

### Keybind (Optional)

The plugin registers `alt+l` as the default keybind in `herdr-plugin.toml`. You can also configure it in `~/.config/herdr/config.toml`:

```toml
[keys]
"alt+l" = "herdr-llama.open-dashboard"
```

## Configuration

### Config File (`herdr-llama.ini`)

```ini
[server]
llama-server-path = /usr/local/bin/llama-server
models-preset = /path/to/models.ini
port = 8080
extra-args = --threads 4
default-model = llama-3-8b
open-method = tab
log-file-size = 5
update-rate = 1
```

| Key | Required | Default | Description |
|-----|----------|---------|-------------|
| `llama-server-path` | Yes | — | Path to the `llama-server` binary |
| `models-preset` | No | — | Path to a `.ini` model preset file |
| `port` | Yes | `8080` | Port for llama-server to listen on |
| `extra-args` | No | — | Additional CLI args (space-separated). `--metrics` is auto-added for TPS polling, no need to include it here. |
| `default-model` | No | — | Default model name for agent reporting |
| `open-method` | No | `tab` | Where to run server: `tab` (focused workspace) or `workspace` (new workspace) |
| `log-file-size` | No | `5` | Daemon log rotation threshold in KB |
| `update-rate` | No | `1` | Polling interval in seconds (accepts float, e.g. `0.5` for 500ms) |
| `logs` | No | `false` | Enable daemon log file. Set to `true` to enable file logging |

### Model Preset

llama-server uses `.ini` files to declare available models. See the [official preset docs](https://github.com/ggml-org/llama.cpp/blob/master/docs/preset.md) for the full format.

Two formats are supported:

**Format 1** — `[models]` section with dotted keys:

```ini
[models]
llama-3-8b = Llama-3-8B-Instruct
llama-3-8b.context_size = 8192
llama-3-8b.url = https://huggingface.co/meta-llama/Llama-3-8B/resolve/main/q4_k_m.gguf

llama-3-70b = Llama-3-70B-Instruct
llama-3-70b.context_size = 8192
llama-3-70b.url = https://huggingface.co/meta-llama/Llama-3-70B/resolve/main/q4_k_m.gguf
```

**Format 2** — One section per model:

```ini
[llama-3-8b]
name = Llama-3-8B-Instruct
context_size = 8192
url = https://huggingface.co/meta-llama/Llama-3-8B/resolve/main/q4_k_m.gguf
```

## Usage

Press `alt+l` to open the dashboard overlay:

1. **Server not running** → Daemon starts server, daemon starts, dashboard closes
2. **Server running, no model** → Select a model → model loads, dashboard closes
3. **Model loaded** → Choose: Unload / Stop server / Quit

After loading a model, the daemon's watcher thread polls every 1 second and reads the herdr pane output to report tokens/sec and processing state to Herdr via `report-agent`/`report-metadata`.

### Startup Cases

The InquirerPy app handles 4 cases automatically:

| Case | Daemon | Server | Action |
|------|--------|--------|--------|
| 1 | Running | Not running | Restart server + daemon |
| 2 | Not running | Running | Start daemon, claim tab |
| 3 | Running | Running | Sync pane IDs, act on state |
| 4 | Not running | Not running | Start server + daemon |

## Daemon Socket Protocol

The InquirerPy app communicates with the daemon via a Unix socket:

```
~/.config/herdr/plugins/state/herdr-llama/daemon.sock
```

### Commands

| Command | Args | Response |
|---------|------|----------|
| `status` | — | `{"state": "...", "model": "...", "tokens_per_sec": ..., "error": null, "tab_id": ..., "pane_id": ...}` |
| `set-pane` | `<tab_id> <pane_id>` | `{"ok": true}` |
| `start` | — | `{"ok": true, "state": "..."}` or `{"error": "..."}` |
| `load` | `<model_id>` | `{"ok": true, "state": "...", "model": "..."}` or `{"error": "..."}` |
| `unload` | — | `{"ok": true, "state": "..."}` or `{"error": "..."}` |
| `stop` | — | `{"ok": true}` (then daemon shuts down) |

**Response format**: JSON on a single line, newline-terminated.

**Important**: The `error` key is always present, even on success (value is `null`). Check `result.get("error")` instead of `"error" in result`.

### Debugging

Test the socket manually:

```bash
# Check if daemon is running
nc -U ~/.config/herdr/plugins/state/herdr-llama/daemon.sock

# Send status command
echo "status" | nc -U ~/.config/herdr/plugins/state/herdr-llama/daemon.sock

# Send load command
echo "load llama-3-8b" | nc -U ~/.config/herdr/plugins/state/herdr-llama/daemon.sock
```

## Daemon State Machine

```
[NO_SERVER] → [SERVER_RUNNING] → [MODEL_LOADING] → [MODEL_LOADED]
     ↑                                      │
     └────────── server crash ◄─────────────┘
```

- **NO_SERVER**: Daemon started, llama-server not detected
- **SERVER_RUNNING**: llama-server detected via HTTP health check, no model loaded
- **MODEL_LOADING**: `POST /models/load` issued, model status is "loading"
- **MODEL_LOADED**: Model status is "loaded", watcher active, reporting via `report-metadata`

State transitions are driven by the Watcher thread polling every 1s.

## Agent State Mapping

| llama-server State | Herdr Agent State | Custom Status |
|---|---|---|
| Server not running | `blocked` | "server offline" |
| Loading model | `working` | "Loading(XX%)" |
| Model loaded, idle | `idle` | "ready" |
| Processing request | `working` | "<tokens>/s tps" |
| Error | `blocked` | "server offline" |

## API Endpoints Used

| Endpoint | Method | Purpose |
|---|---|---|
| `/health` | GET | Server health check (every 10s) |
| `/models` | GET | Current model info + model list (every 1s, cached after confirmed load) |
| `/models/load` | POST | Load a model (`{"model": "<id>"}`) |
| `/models/unload` | POST | Unload a model (`{"model": "<id>"}`) |

TPS and processing state are read from the herdr pane output (not from `/metrics` or `/slots`).

## Directory Structure

```
herdr-llama/
├── herdr-plugin.toml          # Plugin manifest
├── requirements.txt           # Python dependencies
├── README.md                  # This file
├── herdr-llama.py             # InquirerPy app (server lifecycle + dashboard)
├── herdr-llama-daemon.py      # Daemon (polls + reports + socket server)
└── config/                    # User config directory
    └── herdr-llama.ini        # User-created config
```

## Daemon Logging

File logging is disabled by default. Set `logs = true` in config to enable logging to `~/.config/herdr/plugins/state/herdr-llama/daemon.log`.

Log rotation: the daemon periodically checks log size every ~60s. When it exceeds `log-file-size` KB (default 5KB), the daemon sends a herdr notification and truncates the log.

## Troubleshooting

### "llama-server binary not found"
Verify `llama-server-path` in config points to a valid, executable binary:
```bash
ls -la $(herdr config get llama-server-path)
```

### "Server failed to become healthy"
Check the log file at `~/.config/herdr/plugins/state/herdr-llama/daemon.log` for errors. Common causes:
- Port already in use (change `port` in config)
- Model failed to load (check model path/permissions)

### "No models available"
Set `models-preset` in config to a valid `.ini` file path.

### Server crashes
No auto-restart. Check `~/.config/herdr/plugins/state/herdr-llama/daemon.log` and restart via `alt+l`.

### Agent state shows "blocked"
The watcher can't reach the server. Verify the server is running:
```bash
curl http://127.0.0.1:8080/health
```

### Daemon not responding
Check if daemon is running:
```bash
echo "status" | nc -U ~/.config/herdr/plugins/state/herdr-llama/daemon.sock
```

If the socket doesn't exist, the daemon isn't running. Restart via `alt+l`.

### Pane ID mismatch
If the daemon reports the wrong pane, the InquirerPy app auto-syncs pane IDs on startup. If issues persist, stop and restart:
```bash
echo "stop" | nc -U ~/.config/herdr/plugins/state/herdr-llama/daemon.sock
# Then press alt+l to restart
```

## License

MIT
