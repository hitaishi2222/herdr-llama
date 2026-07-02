"""Tests for the agent state reporter."""

import sys
from unittest.mock import patch

import pytest

sys.modules["herdr_llama"] = __import__("importlib").import_module("herdr-llama")
import herdr_llama  # noqa: E402
from herdr_llama import AGENT_SOURCE, AgentReporter, AgentState  # noqa: E402


def test_agent_state_defaults():
    state = AgentState(agent="llama-3-8b", state="idle")
    assert state.agent == "llama-3-8b"
    assert state.state == "idle"
    assert state.message == ""
    assert state.custom_status == ""


def test_agent_state_with_details():
    state = AgentState(
        agent="llama-3-8b",
        state="working",
        message="45%",
        custom_status="62% context",
    )
    assert state.message == "45%"
    assert state.custom_status == "62% context"


@patch("subprocess.run")
def test_report_success(mock_run):
    mock_run.return_value.returncode = 0
    reporter = AgentReporter(herdr_bin_path="/usr/bin/herdr")
    state = AgentState(agent="llama-3-8b", state="idle")
    result = reporter.report(state)
    assert result is True
    mock_run.assert_called_once()
    cmd = mock_run.call_args[0][0]
    assert cmd[0] == "/usr/bin/herdr"
    assert "--agent" in cmd
    assert "llama-3-8b" in cmd
    assert "--state" in cmd
    assert "idle" in cmd
    assert "--source" in cmd
    assert AGENT_SOURCE in cmd


@patch("subprocess.run")
def test_report_with_message(mock_run):
    mock_run.return_value.returncode = 0
    reporter = AgentReporter(herdr_bin_path="/usr/bin/herdr")
    state = AgentState(
        agent="llama-3-8b",
        state="working",
        message="45%",
        custom_status="62% context",
    )
    reporter.report(state)
    cmd = mock_run.call_args[0][0]
    assert "--message" in cmd
    assert "45%" in cmd
    assert "--custom-status" in cmd
    assert "62% context" in cmd


@patch("subprocess.run")
def test_report_failure(mock_run, caplog):
    mock_run.return_value.returncode = 1
    mock_run.return_value.stderr = "error message"
    reporter = AgentReporter(herdr_bin_path="/usr/bin/herdr")
    state = AgentState(agent="llama-3-8b", state="blocked")
    result = reporter.report(state)
    assert result is False


@patch("subprocess.run", side_effect=FileNotFoundError)
def test_report_binary_not_found(mock_run, caplog):
    reporter = AgentReporter(herdr_bin_path="/nonexistent/herdr")
    state = AgentState(agent="llama-3-8b", state="idle")
    result = reporter.report(state)
    assert result is False


@patch("subprocess.run", side_effect=FileNotFoundError)
def test_notify(mock_run):
    reporter = AgentReporter(herdr_bin_path="/usr/bin/herdr")
    reporter.notify("Server crashed!")
    cmd = mock_run.call_args[0][0]
    assert cmd == ["/usr/bin/herdr", "notification", "show", "Server crashed!"]
