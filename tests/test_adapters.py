from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

import drummer.adapters as adapters_module
from drummer.adapters import (
    AdapterExecutionDisabled,
    ClaudeCLIAdapter,
    CodexCLIAdapter,
    LocalOpenAIAdapter,
)


class RecordingRunner:
    def __init__(self, stdout: str, *, returncode: int = 0, stderr: str = "") -> None:
        self.stdout = stdout
        self.returncode = returncode
        self.stderr = stderr
        self.calls: list[tuple[list[str], dict[str, object]]] = []

    def __call__(self, args: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        self.calls.append((args, kwargs))
        assert Path(str(kwargs["cwd"])).is_dir()
        return subprocess.CompletedProcess(args, self.returncode, self.stdout, self.stderr)


def ticking_clock() -> callable:
    values = iter((10.0, 10.25, 20.0, 20.5, 30.0, 30.75))
    return lambda: next(values)


def test_claude_cli_is_isolated_tool_free_and_reports_native_usage() -> None:
    runner = RecordingRunner(
        json.dumps(
            {
                "type": "result",
                "subtype": "success",
                "result": '{"status":"ok"}',
                "usage": {
                    "input_tokens": 31,
                    "output_tokens": 7,
                    "cache_read_input_tokens": 11,
                    "cache_creation_input_tokens": 3,
                },
                "modelUsage": {"claude-sonnet-4-6": {"inputTokens": 31}},
            }
        )
    )
    adapter = ClaudeCLIAdapter(
        executable="claude",
        model="sonnet",
        runner=runner,
        clock=ticking_clock(),
        allow_live=True,
    )

    result = adapter.generate("synthetic handoff", timeout_seconds=12)

    args, call = runner.calls[0]
    assert args[0] == "claude"
    for flag in (
        "--safe-mode",
        "--restricted",
        "--strict-mcp-config",
        "--no-session-persistence",
        "--disable-slash-commands",
        "--no-chrome",
    ):
        assert flag in args
    assert args[args.index("--tools") + 1] == ""
    assert args[args.index("--mcp-config") + 1] == '{"mcpServers":{}}'
    assert args[args.index("--output-format") + 1] == "json"
    assert args[-1] == "-p"
    assert call["input"] == "synthetic handoff"
    assert call["shell"] is False
    assert call["timeout"] == 12
    assert Path(str(call["cwd"])) != Path.cwd()
    env = call["env"]
    assert isinstance(env, dict)
    assert "ANTHROPIC_API_KEY" not in env
    assert "CLAUDE_CODE_USE_BEDROCK" not in env
    assert result.text == '{"status":"ok"}'
    assert result.usage.input_tokens == 31
    assert result.usage.output_tokens == 7
    assert result.usage.cached_input_tokens == 11
    assert result.usage.cache_creation_input_tokens == 3
    assert result.usage.total_tokens == 38
    assert result.elapsed_seconds == pytest.approx(0.25)
    assert result.retries == 0
    assert result.errors == ()
    assert result.setup["tools"] == "disabled"
    assert result.setup["customizations"] == "safe-mode"
    assert result.setup["provider_reported_models"] == ("claude-sonnet-4-6",)


def test_codex_cli_is_ephemeral_context_free_and_parses_jsonl_usage() -> None:
    stdout = "\n".join(
        (
            json.dumps({"type": "thread.started", "thread_id": "ignored"}),
            json.dumps(
                {
                    "type": "item.completed",
                    "item": {"type": "agent_message", "text": '{"status":"ok"}'},
                }
            ),
            json.dumps(
                {
                    "type": "turn.completed",
                    "usage": {
                        "input_tokens": 41,
                        "cached_input_tokens": 13,
                        "output_tokens": 9,
                    },
                }
            ),
        )
    )
    runner = RecordingRunner(stdout)
    adapter = CodexCLIAdapter(
        executable="codex",
        model="gpt-5.4-mini",
        runner=runner,
        clock=ticking_clock(),
        allow_live=True,
    )

    result = adapter.generate("synthetic handoff", timeout_seconds=17)

    args, call = runner.calls[0]
    assert args[:2] == ["codex", "exec"]
    for flag in (
        "--ignore-user-config",
        "--ignore-rules",
        "--ephemeral",
        "--skip-git-repo-check",
        "--strict-config",
        "--json",
    ):
        assert flag in args
    assert args[args.index("--sandbox") + 1] == "read-only"
    assert "mcp_servers={}" in args
    assert 'web_search="disabled"' in args
    disabled = {args[index + 1] for index, value in enumerate(args[:-1]) if value == "--disable"}
    assert {
        "shell_tool",
        "unified_exec",
        "hooks",
        "plugins",
        "apps",
        "browser_use",
        "computer_use",
        "multi_agent",
        "view_image",
    } <= disabled
    assert args[-1] == "-"
    assert call["input"] == "synthetic handoff"
    assert call["shell"] is False
    env = call["env"]
    assert isinstance(env, dict)
    assert "OPENAI_API_KEY" not in env
    assert result.text == '{"status":"ok"}'
    assert result.usage.input_tokens == 41
    assert result.usage.output_tokens == 9
    assert result.usage.cached_input_tokens == 13
    assert result.usage.total_tokens == 50
    assert result.setup["project_doc_max_bytes"] == 0
    assert result.setup["web_search"] == "disabled"
    assert result.setup["strict_config"] is True


def test_cli_execution_requires_an_explicit_live_gate() -> None:
    runner = RecordingRunner("{}")
    adapter = ClaudeCLIAdapter(runner=runner)

    with pytest.raises(AdapterExecutionDisabled):
        adapter.generate("must not run", timeout_seconds=1)

    assert runner.calls == []


def test_cli_failure_is_reported_without_provider_fallback() -> None:
    runner = RecordingRunner("", returncode=2, stderr="authentication unavailable")
    adapter = CodexCLIAdapter(runner=runner, allow_live=True)

    result = adapter.generate("synthetic handoff", timeout_seconds=2)

    assert result.text == ""
    assert result.retries == 0
    assert result.errors == ("process exited with status 2: authentication unavailable",)
    assert result.usage.total_tokens is None
    assert len(runner.calls) == 1


class FakeResponse:
    def __init__(self, payload: dict[str, object], *, status: int = 200) -> None:
        self.payload = payload
        self.status = status

    def __enter__(self) -> FakeResponse:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")


class RecordingURLopener:
    def __init__(self, responses: list[FakeResponse]) -> None:
        self.responses = iter(responses)
        self.calls: list[tuple[object, float]] = []

    def __call__(self, request: object, *, timeout: float) -> FakeResponse:
        self.calls.append((request, timeout))
        return next(self.responses)


def test_local_openai_checks_health_uses_timeouts_and_reports_native_usage() -> None:
    opener = RecordingURLopener(
        [
            FakeResponse({"data": [{"id": "qwen2.5-1.5b"}]}),
            FakeResponse(
                {
                    "model": "qwen2.5-1.5b",
                    "choices": [{"message": {"content": '{"status":"ok"}'}}],
                    "usage": {
                        "prompt_tokens": 23,
                        "completion_tokens": 4,
                        "total_tokens": 27,
                        "prompt_tokens_details": {"cached_tokens": 6},
                    },
                }
            ),
        ]
    )
    adapter = LocalOpenAIAdapter(
        base_url="http://127.0.0.1:1234/v1",
        model="qwen2.5-1.5b",
        urlopen=opener,
        clock=ticking_clock(),
        allow_live=True,
    )

    result = adapter.generate("synthetic handoff", timeout_seconds=5)

    assert len(opener.calls) == 2
    health_request, health_timeout = opener.calls[0]
    generation_request, generation_timeout = opener.calls[1]
    assert health_request.full_url == "http://127.0.0.1:1234/v1/models"
    assert generation_request.full_url == "http://127.0.0.1:1234/v1/chat/completions"
    assert health_timeout == 5
    assert generation_timeout == 5
    assert result.text == '{"status":"ok"}'
    assert result.usage.input_tokens == 23
    assert result.usage.output_tokens == 4
    assert result.usage.total_tokens == 27
    assert result.usage.cached_input_tokens == 6
    assert result.setup["health"] == "passed"
    assert result.setup["endpoint_scope"] == "loopback-only"
    assert result.setup["provider_reported_model"] == "qwen2.5-1.5b"


@pytest.mark.parametrize(
    "url",
    (
        "https://api.openai.com/v1",
        "http://example.com/v1",
        "http://192.168.1.20:1234/v1",
    ),
)
def test_local_openai_rejects_non_loopback_endpoints(url: str) -> None:
    with pytest.raises(ValueError, match="loopback"):
        LocalOpenAIAdapter(base_url=url, model="qwen")


def test_local_openai_accepts_only_an_explicitly_allowlisted_pi() -> None:
    adapter = LocalOpenAIAdapter(
        base_url="http://192.168.0.100:8080/v1",
        model="qwen2.5-1.5b",
        trusted_hosts=("192.168.0.100",),
    )

    assert adapter.endpoint_scope == "explicit-host:192.168.0.100"


def test_allowlist_does_not_turn_local_adapter_into_public_provider():
    with pytest.raises(ValueError, match="private"):
        LocalOpenAIAdapter(base_url="https://example.com/v1", model="qwen", trusted_hosts=["example.com"])
    with pytest.raises(ValueError, match="private"):
        LocalOpenAIAdapter(base_url="http://8.8.8.8/v1", model="qwen", trusted_hosts=["8.8.8.8"])


def test_client_environment_omits_unrelated_secrets(monkeypatch):
    monkeypatch.setenv("DRUMMER_UNRELATED_SECRET", "not-for-the-client")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "not-for-the-client")
    environment = adapters_module._isolated_environment(set())
    assert "DRUMMER_UNRELATED_SECRET" not in environment
    assert "ANTHROPIC_API_KEY" not in environment


def test_failed_cli_run_preserves_reported_partial_usage() -> None:
    runner = RecordingRunner(
        json.dumps(
            {
                "type": "result",
                "subtype": "error_during_execution",
                "is_error": True,
                "result": "failed",
                "usage": {"input_tokens": 19, "output_tokens": 2},
            }
        ),
        returncode=1,
        stderr="provider failed",
    )
    adapter = ClaudeCLIAdapter(runner=runner, allow_live=True)

    result = adapter.generate("synthetic", timeout_seconds=4)

    assert result.errors
    assert result.usage.input_tokens == 19
    assert result.usage.output_tokens == 2
    assert result.usage.total_tokens == 21


def test_adapter_environment_does_not_mutate_parent_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-only")
    runner = RecordingRunner(
        "\n".join(
            (
                json.dumps(
                    {
                        "type": "item.completed",
                        "item": {"type": "agent_message", "text": "ok"},
                    }
                ),
                json.dumps({"type": "turn.completed", "usage": {}}),
            )
        )
    )
    adapter = CodexCLIAdapter(runner=runner, allow_live=True)

    adapter.generate("synthetic", timeout_seconds=1)

    assert os.environ["OPENAI_API_KEY"] == "test-only"
    assert "OPENAI_API_KEY" not in runner.calls[0][1]["env"]


def test_default_process_runner_kills_the_whole_child_group_on_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    killed: list[tuple[int, int]] = []

    class FakeProcess:
        pid = 4321
        returncode = -9

        def __init__(self) -> None:
            self.communications = 0

        def communicate(
            self, input: str | None = None, timeout: float | None = None
        ) -> tuple[str, str]:
            self.communications += 1
            if self.communications == 1:
                raise subprocess.TimeoutExpired(["fake"], timeout)
            return "partial stdout", "partial stderr"

    def fake_popen(*args: object, **kwargs: object) -> FakeProcess:
        assert kwargs["start_new_session"] is True
        assert kwargs["shell"] is False
        assert kwargs["stdin"] is subprocess.PIPE
        return FakeProcess()

    monkeypatch.setattr(adapters_module.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(adapters_module.os, "killpg", lambda pid, sig: killed.append((pid, sig)))

    with pytest.raises(subprocess.TimeoutExpired):
        adapters_module.run_process_group(
            ["fake"],
            input="prompt",
            text=True,
            capture_output=True,
            timeout=2,
            check=False,
            shell=False,
        )

    assert killed == [(4321, adapters_module.signal.SIGKILL)]
