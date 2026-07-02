# herdr-llama

A [Herdr](https://github.com/nicehash/herdr) plugin that integrates [llama-server](https://github.com/ggml-org/llama.cpp/tree/master/examples/server) as an agent with a real-time stats dashboard.

**Trigger**: `alt+l` opens an overlay for model selection and server control. A background watcher continuously reports tokens/sec, context usage, and slot status to Herdr.

## Features

- **One-shot dashboard**: Start server, load/unload models, stop server — all from a single overlay
- **Real-time stats**: Background watcher polls every 2s and reports to Herdr agent system
- **State-aware**: Agent state maps to server/model state (idle/working/blocked)
- **Crash detection**: Notifications on server crashes (no auto-restart)

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
cat > ~/.config/herdr/plugins/config/llama-server.ini << 'EOF'
[server]
llama-server-path = /usr/local/bin/llama-server
models-preset = /path/to/your/models.ini
port = 8080
extra-args = --threads 4
default-model = llama-3-8b
EOF
```

3. Register the plugin with Herdr (see Herdr docs for plugin registration).

## Configuration

### Config File (`llama-server.ini`)

```ini
[server]
llama-server-path = /usr/local/bin/llama-server
models-preset = /path/to/models.ini
port = 8080
extra-args = --threads 4
default-model = llama-3-8b
```

| Key | Required | Default | Description |
|-----|----------|---------|-------------|
| `llama-server-path` | Yes | — | Path to the `llama-server` binary |
| `models-preset` | No | — | Path to a `.ini` model preset file |
| `port` | Yes | `8080` | Port for llama-server to listen on |
| `extra-args` | No | — | Additional CLI args (space-separated) |
| `default-model` | No | — | Default model name for agent reporting |

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

1. **Server not running** → "Start server with model?" (Y/n)
   - **Y** → Select a model → server starts, model loads, dashboard closes
   - **n** → Server starts empty, dashboard closes
2. **Server running, no model** → Select a model → model loads, dashboard closes
3. **Model loaded** → Choose: Unload / Stop / Quit

After loading a model, a background watcher runs in a separate pane, polling every 2 seconds and reporting stats to Herdr.

## Agent State Mapping

| llama-server State | Herdr Agent State | Custom Status |
|---|---|---|
| Server not running | `blocked` | "server offline" |
| Loading model | `working` | "loading: <model>" |
| Model loaded, idle | `idle` | "ready: <model>" |
| Processing request | `working` | "generating: <tokens>/s" |
| Error | `blocked` | "error: <message>" |

## API Endpoints Used

| Endpoint | Method | Purpose |
|---|---|---|
| `/health` | GET | Server health check |
| `/completion` | GET | Last completion stats (tokens/sec, context) |
| `/slots` | GET | Slot status (active/inactive) |
| `/models` | GET | Current model info + model list |
| `/models/load` | POST | Load a model (`{"model": "<id>"}`) |
| `/models/unload` | POST | Unload a model (`{"model": "<id>"}`) |

## Directory Structure

```
herdr-llama/
├── herdr-plugin.toml      # Plugin manifest
├── requirements.txt       # Python dependencies
├── README.md              # This file
├── src/
│   ├── main.py            # Entry point (Typer CLI)
│   ├── dashboard.py       # Rich-based overlay UI
│   ├── llama_client.py    # llama-server API client (sync httpx)
│   ├── llama_runner.py    # Server process management (subprocess)
│   ├── agent_reporter.py  # Herdr agent state reporting
│   ├── config.py          # Configuration management
│   └── watcher.py         # Background stats watcher
└── tests/
    ├── test_config.py
    ├── test_llama_client.py
    ├── test_llama_runner.py
    ├── test_agent_reporter.py
    ├── test_watcher.py
    └── test_dashboard.py
```

## Troubleshooting

### "llama-server binary not found"
Verify `llama-server-path` in config points to a valid, executable binary:
```bash
ls -la $(herdr config get llama-server-path)
```

### "Server failed to become healthy"
Check the log file at `~/.local/state/herdr-llama/logs/server.log` for errors. Common causes:
- Port already in use (change `port` in config)
- Model failed to load (check model path/permissions)

### "No models available"
Set `models-preset` in config to a valid `.ini` file path.

### Server crashes
No auto-restart. Check `~/.local/state/herdr-llama/logs/server.log` and restart via `alt+l`.

### Agent state shows "blocked"
The watcher can't reach the server. Verify the server is running:
```bash
curl http://127.0.0.1:8080/health
```

## License

MIT
