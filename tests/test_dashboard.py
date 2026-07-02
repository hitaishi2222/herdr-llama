"""Tests for the dashboard UI helpers."""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.modules["herdr_llama"] = __import__("importlib").import_module("herdr-llama")
import herdr_llama  # noqa: E402
from herdr_llama import Config  # noqa: E402
from herdr_llama import PresetModel  # noqa: E402
from herdr_llama import _load_models, _print_error, _print_info  # noqa: E402


def test_load_models_no_preset(tmp_path):
    config = Config(llama_server_path=str(tmp_path / "llama-server"), port=8080)
    models = _load_models(config)
    assert models == []


def test_load_models_with_preset(tmp_path):
    preset = tmp_path / "models.ini"
    preset.write_text(
        "[models]\nllama-3-8b = Llama-3-8B\nllama-3-8b.context_size = 8192\n"
    )
    config = Config(
        llama_server_path=str(tmp_path / "llama-server"),
        port=8080,
        models_preset=str(preset),
    )
    models = _load_models(config)
    assert len(models) == 1
    assert models[0].id == "llama-3-8b"
    assert models[0].name == "Llama-3-8B"


@patch("herdr_llama.console")
def test_print_error(mock_console):
    _print_error("test error")
    mock_console.print.assert_called_once()
    call_text = mock_console.print.call_args[0][0]
    assert "test error" in call_text


@patch("herdr_llama.console")
def test_print_info(mock_console):
    _print_info("test info")
    mock_console.print.assert_called_once()
    call_text = mock_console.print.call_args[0][0]
    assert "test info" in call_text
