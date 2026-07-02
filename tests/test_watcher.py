"""Tests for the background watcher."""

import sys
from unittest.mock import MagicMock, patch

import pytest

sys.modules["herdr_llama"] = __import__("importlib").import_module("herdr-llama")
import herdr_llama  # noqa: E402
from herdr_llama import (  # noqa: E402
    AgentReporter,
    CompletionStats,
    LlamaClient,
    ModelInfo,
    ServerNotRunningError,
    SlotInfo,
    Watcher,
)


def _make_watcher(mock_client=None, mock_reporter=None) -> Watcher:
    client = mock_client or MagicMock(spec=LlamaClient)
    reporter = mock_reporter or MagicMock(spec=AgentReporter)
    return Watcher(client=client, reporter=reporter, model_name="test-model")


def test_watcher_initial_state():
    watcher = _make_watcher()
    assert watcher.stats.server_running is False
    assert watcher.stats.model_name is None
    assert watcher.stats.error is None


@patch("herdr_llama.time")
def test_poll_once_idle(mock_time, tmp_path):
    mock_client = MagicMock(spec=LlamaClient)
    mock_client.is_running.return_value = True
    mock_client.get_model_info.return_value = ModelInfo(name="test-model", loaded=True)
    mock_client.get_completions.return_value = CompletionStats(
        tokens_per_sec=0, context_size=4096, context_used=100
    )
    mock_client.get_slots.return_value = [SlotInfo(id=0, running=False)]

    mock_reporter = MagicMock(spec=AgentReporter)
    watcher = _make_watcher(mock_client, mock_reporter)

    watcher._poll_once()

    assert watcher.stats.server_running is True
    assert watcher.stats.model_name == "test-model"
    assert watcher.stats.slots_active == 0
    mock_reporter.report.assert_called_once()
    state = mock_reporter.report.call_args[0][0]
    assert state.state == "idle"
    assert "ready: test-model" in state.custom_status


@patch("herdr_llama.time")
def test_poll_once_working(mock_time, tmp_path):
    mock_client = MagicMock(spec=LlamaClient)
    mock_client.is_running.return_value = True
    mock_client.get_model_info.return_value = ModelInfo(name="test-model", loaded=True)

    mock_reporter = MagicMock(spec=AgentReporter)
    watcher = _make_watcher(mock_client, mock_reporter)

    watcher._poll_once()

    assert watcher.stats.model_name == "test-model"
    assert watcher.stats.tokens_per_sec is None
    assert watcher.stats.slots_active == 0
    state = mock_reporter.report.call_args[0][0]
    assert state.state == "idle"


def test_start_stop_watcher():
    watcher = _make_watcher()
    watcher.start()
    assert watcher._thread is not None
    assert watcher._thread.is_alive()
    watcher.stop()
    assert watcher._thread is None


def test_start_idempotent():
    watcher = _make_watcher()
    watcher.start()
    thread1 = watcher._thread
    watcher.start()
    assert watcher._thread is thread1  # same thread, not restarted


def test_poll_once_server_not_running():
    mock_client = MagicMock(spec=LlamaClient)
    mock_client.is_running.return_value = False

    mock_reporter = MagicMock(spec=AgentReporter)
    watcher = _make_watcher(mock_client, mock_reporter)

    with pytest.raises(ServerNotRunningError):
        watcher._poll_once()
