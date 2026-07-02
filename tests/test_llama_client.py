"""Tests for the llama-server API client and preset parser."""

import sys
from pathlib import Path

import pytest

sys.modules["herdr_llama"] = __import__("importlib").import_module("herdr-llama")
import herdr_llama  # noqa: E402
from herdr_llama import LlamaClient, PresetModel  # noqa: E402

SAMPLE_PRESET = """\
[models]
llama-3-8b = Llama-3-8B-Instruct
llama-3-8b.context_size = 8192
llama-3-8b.url = https://huggingface.co/meta-llama/Llama-3-8B/resolve/main/q4_k_m.gguf

llama-3-70b = Llama-3-70B-Instruct
llama-3-70b.context_size = 8192
llama-3-70b.url = https://huggingface.co/meta-llama/Llama-3-70B/resolve/main/q4_k_m.gguf
"""

SECTION_PRESET = """\
[llama-3-8b]
name = Llama-3-8B-Instruct
context_size = 8192
url = https://example.com/model.gguf

[llama-3-70b]
name = Llama-3-70B-Instruct
context_size = 8192
url = https://example.com/model-70b.gguf
"""


def test_get_model_list_models_section(tmp_path):
    preset = tmp_path / "models.ini"
    preset.write_text(SAMPLE_PRESET)
    models = LlamaClient.get_model_list(preset)
    assert len(models) == 2
    assert models[0].id == "llama-3-8b"
    assert models[0].name == "Llama-3-8B-Instruct"
    assert models[0].context_size == 8192
    assert models[1].id == "llama-3-70b"


def test_get_model_list_section_format(tmp_path):
    preset = tmp_path / "models.ini"
    preset.write_text(SECTION_PRESET)
    models = LlamaClient.get_model_list(preset)
    assert len(models) == 2
    assert models[0].id == "llama-3-8b"
    assert models[0].name == "Llama-3-8B-Instruct"
    assert models[1].id == "llama-3-70b"


def test_get_model_list_missing_file(tmp_path, caplog):
    models = LlamaClient.get_model_list(tmp_path / "nonexistent.ini")
    assert models == []


def test_get_model_list_empty_file(tmp_path):
    preset = tmp_path / "empty.ini"
    preset.write_text("")
    models = LlamaClient.get_model_list(preset)
    assert models == []


def test_client_init():
    client = LlamaClient(port=9090)
    assert client.base_url == "http://127.0.0.1"
    assert client.port == 9090
    client.close()


def test_client_base_url_trailing_slash():
    client = LlamaClient(base_url="http://127.0.0.1/")
    assert client.base_url == "http://127.0.0.1"
    client.close()
