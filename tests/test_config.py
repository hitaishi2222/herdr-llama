"""Tests for config loading."""

import sys
from pathlib import Path

import pytest

# Import from the single-file app (hyphenated name needs importlib).
sys.modules["herdr_llama"] = __import__("importlib").import_module("herdr-llama")
import herdr_llama  # noqa: E402
from herdr_llama import Config, ConfigError, load_config  # noqa: E402

SAMPLE_CONFIG = """\
[server]
llama-server-path = /usr/local/bin/llama-server
models-preset = /home/user/models.ini
port = 8080
extra-args = --threads 4
default-model = llama-3-8b
"""


def _write_config(tmpdir: Path, content: str = SAMPLE_CONFIG) -> Path:
    config_file = tmpdir / "herdr-llama.ini"
    config_file.write_text(content)
    return config_file


def test_load_config_success(tmp_path, monkeypatch):
    _write_config(tmp_path)
    monkeypatch.setattr("herdr_llama._config_path", lambda: tmp_path / "herdr-llama.ini")
    config = load_config()
    assert config.llama_server_path == "/usr/local/bin/llama-server"
    assert config.models_preset == "/home/user/models.ini"
    assert config.port == 8080
    assert config.extra_args == "--threads 4"
    assert config.default_model == "llama-3-8b"
    assert config.get_extra_args_list() == ["--threads", "4"]


def test_load_config_missing_file(tmp_path, monkeypatch):
    monkeypatch.setattr("herdr_llama._config_path", lambda: tmp_path / "nonexistent.ini")
    with pytest.raises(ConfigError, match="not found"):
        load_config()


def test_load_config_missing_section(tmp_path, monkeypatch):
    config_file = tmp_path / "herdr-llama.ini"
    config_file.write_text("[other]\nfoo = bar\n")
    monkeypatch.setattr("herdr_llama._config_path", lambda: tmp_path / "herdr-llama.ini")
    with pytest.raises(ConfigError, match="Missing required section"):
        load_config()


def test_load_config_missing_key(tmp_path, monkeypatch):
    config_file = tmp_path / "herdr-llama.ini"
    config_file.write_text("[server]\nllama-server-path = /bin/llama-server\n")
    monkeypatch.setattr("herdr_llama._config_path", lambda: tmp_path / "herdr-llama.ini")
    with pytest.raises(ConfigError, match="Missing required key 'port'"):
        load_config()


def test_load_config_invalid_port(tmp_path, monkeypatch):
    config_file = tmp_path / "herdr-llama.ini"
    config_file.write_text("[server]\nllama-server-path = /bin/llama-server\nport = abc\n")
    monkeypatch.setattr("herdr_llama._config_path", lambda: tmp_path / "herdr-llama.ini")
    with pytest.raises(ConfigError, match="Invalid port"):
        load_config()


def test_load_config_defaults(tmp_path, monkeypatch):
    config_file = tmp_path / "herdr-llama.ini"
    config_file.write_text(
        "[server]\nllama-server-path = /bin/llama-server\nport = 8080\n"
    )
    monkeypatch.setattr("herdr_llama._config_path", lambda: tmp_path / "herdr-llama.ini")
    config = load_config()
    assert config.models_preset is None
    assert config.extra_args is None
    assert config.default_model is None
    assert config.get_extra_args_list() == []


def test_config_extra_args_parsing():
    config = Config(
        llama_server_path="/bin/llama-server",
        extra_args="--threads 4 --ctx-size 4096",
    )
    assert config.get_extra_args_list() == ["--threads", "4", "--ctx-size", "4096"]
