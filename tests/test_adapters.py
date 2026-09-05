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
    assert result.usage.input_tokens == 45
    assert result.usage.uncached_input_tokens == 31
    assert result.usage.output_tokens == 7
    assert result.usage.cached_input_tokens == 11
    assert result.usage.cache_creation_input_tokens == 3
    assert result.usage.total_tokens == 52
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
    assert 0 < generation_timeout <= 5
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


def test_local_schema_is_optional_copied_and_recorded():
    schema = {"type": "object", "properties": {"value": {"type": "string"}},
              "required": ["value"], "additionalProperties": False}
    opener = RecordingURLopener([
        FakeResponse({"data": [{"id": "qwen"}]}),
        FakeResponse({"choices": [{"message": {"content": '{"value":"x"}'}}]})])
    adapter = LocalOpenAIAdapter(model="qwen", allow_live=True, urlopen=opener,
                                 response_schema=schema, reasoning_effort="none")
    schema["properties"]["value"]["type"] = "integer"
    result = adapter.generate("synthetic", timeout_seconds=5)
    payload = json.loads(opener.calls[1][0].data)
    assert payload["response_format"]["json_schema"]["schema"]["properties"]["value"]["type"] == "string"
    assert payload["reasoning_effort"] == "none"
    assert result.setup["response_mode"] == "schema-guided"
    assert result.setup["reasoning_effort"] == "none"
    assert "enum" not in json.dumps(payload["response_format"])


def test_invalid_local_reasoning_effort():
    with pytest.raises(ValueError, match="reasoning"):
        LocalOpenAIAdapter(model="qwen", reasoning_effort="unbounded")


def test_health_and_generation_share_one_deadline():
    now = [0.0]
    timeouts = []

    def opener(request, *, timeout):
        timeouts.append(timeout)
        now[0] += timeout * 0.9
        if request.full_url.endswith("/models"):
            return FakeResponse({"data": [{"id": "qwen"}]})
        return FakeResponse({"choices": [{"message": {"content": "{}"}}]})

    adapter = LocalOpenAIAdapter(model="qwen", allow_live=True, urlopen=opener, clock=lambda: now[0])
    result = adapter.generate("synthetic", timeout_seconds=10)
    assert timeouts == [10, 1]
    assert result.elapsed_seconds == pytest.approx(9.9)


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
    assert result.usage.input_tokens is None
    assert result.usage.uncached_input_tokens is None
    assert result.usage.output_tokens is None
    assert result.usage.total_tokens is None
    assert result.setup["reported_usage_subtotal"]["uncached_input_tokens"] == 19
    assert result.setup["reported_usage_subtotal"]["output_tokens"] == 2
    assert result.setup["usage_coverage"] == "incomplete_or_unknown"


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

    with pytest.raises(subprocess.TimeoutExpired) as caught:
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
    assert caught.value.output == "partial stdout"
    assert caught.value.stderr == "partial stderr"


@pytest.mark.parametrize("failure", [KeyboardInterrupt("stop"), SystemExit(7),
                                    GeneratorExit("stop"), RuntimeError("read failed")])
def test_process_runner_cleans_up_on_baseexception_and_preserves_original(failure, monkeypatch):
    calls = []

    class FakeProcess:
        pid = 4321
        returncode = -9

        def communicate(self, **kwargs):
            calls.append(("communicate", kwargs))
            if len(calls) == 1:
                raise failure
            return "partial", ""

    monkeypatch.setattr(adapters_module.subprocess, "Popen", lambda *args, **kwargs: FakeProcess())
    monkeypatch.setattr(adapters_module.os, "killpg", lambda pid, sig: calls.append(("killpg", pid, sig)))
    with pytest.raises(type(failure)) as caught:
        adapters_module.run_process_group(["fake"], input="once", text=True, capture_output=True,
                                          timeout=1, shell=False)
    assert caught.value is failure
    assert calls[1] == ("killpg", 4321, adapters_module.signal.SIGKILL)
    assert len([call for call in calls if call[0] == "communicate"]) == 2
    assert calls[-1][1].get("input") is None  # Drain only; never resend the prompt.


def test_process_runner_missing_group_and_interrupted_drain_do_not_mask_original(monkeypatch):
    original = KeyboardInterrupt("first interrupt")
    calls = []

    class FakeProcess:
        pid = 4321
        returncode = -9

        def communicate(self, **kwargs):
            calls.append("communicate")
            if calls.count("communicate") == 1:
                raise original
            raise KeyboardInterrupt("second interrupt during drain")

        def kill(self):
            calls.append("kill")

        def wait(self, *, timeout):
            assert 0 < timeout <= 2
            calls.append("wait")
            return self.returncode

    def gone(pid, sig):
        calls.append("killpg")
        raise ProcessLookupError("group already ended")

    monkeypatch.setattr(adapters_module.subprocess, "Popen", lambda *args, **kwargs: FakeProcess())
    monkeypatch.setattr(adapters_module.os, "killpg", gone)
    with pytest.raises(KeyboardInterrupt) as caught:
        adapters_module.run_process_group(["fake"], capture_output=True, timeout=1, shell=False)
    assert caught.value is original
    assert "wait" in calls


def test_timeout_keeps_partial_output_if_cleanup_is_incomplete(monkeypatch):
    original = subprocess.TimeoutExpired(["fake"], 1, output=b"first", stderr=b"error")
    calls = []

    class FakeProcess:
        pid = 4321
        returncode = -9

        def communicate(self, **kwargs):
            calls.append("communicate")
            if len(calls) == 1:
                raise original
            raise subprocess.TimeoutExpired(["fake"], 1, output=b"first plus later", stderr=None)

        def kill(self):
            calls.append("kill")

        def wait(self, *, timeout):
            calls.append("wait")
            return self.returncode

    monkeypatch.setattr(adapters_module.subprocess, "Popen", lambda *args, **kwargs: FakeProcess())
    monkeypatch.setattr(adapters_module.os, "killpg", lambda *args: None)
    with pytest.raises(subprocess.TimeoutExpired) as caught:
        adapters_module.run_process_group(["fake"], capture_output=True, timeout=1, shell=False)
    assert caught.value is original
    assert caught.value.output == b"first plus later"
    assert caught.value.stderr == b"error"
    assert "wait" in calls


def test_cleanup_signal_wait_and_close_failures_never_replace_original_exception(monkeypatch):
    original = KeyboardInterrupt("original")

    class FailedPipe:
        def close(self):
            raise OSError("synthetic close failure")

    class FakeProcess:
        pid = 4321
        returncode = None
        stdout = FailedPipe()
        communications = 0

        def communicate(self, **kwargs):
            self.communications += 1
            if self.communications == 1:
                raise original
            assert kwargs["timeout"] == 1.0
            raise RuntimeError("synthetic drain failure")

        def kill(self):
            raise OSError("synthetic kill failure")

        def wait(self, *, timeout):
            assert timeout == 1.0
            raise KeyboardInterrupt("synthetic repeated interrupt while reaping")

    def failed_group(*args):
        raise PermissionError("synthetic group failure")

    monkeypatch.setattr(adapters_module.subprocess, "Popen", lambda *args, **kwargs: FakeProcess())
    monkeypatch.setattr(adapters_module.os, "killpg", failed_group)
    with pytest.raises(KeyboardInterrupt) as caught:
        adapters_module.run_process_group(["fake"], capture_output=True, timeout=1, shell=False)
    assert caught.value is original
    assert len(original.__notes__) == 5
    assert any("reap incomplete" in note for note in original.__notes__)


def test_interrupt_terminates_real_authored_child_group_and_reaps_leader(monkeypatch):
    import select
    import sys
    import time

    real_popen = subprocess.Popen
    launched = []
    descendant = []
    original = KeyboardInterrupt("synthetic interruption, not a model call")

    class InterruptingProcess:
        def __init__(self, *args, **kwargs):
            self.process = real_popen(*args, **kwargs)
            self.communications = 0
            launched.append(self.process)

        def __getattr__(self, name):
            return getattr(self.process, name)

        def communicate(self, **kwargs):
            self.communications += 1
            if self.communications == 1:
                assert select.select([self.process.stdout], [], [], 2)[0], "trusted child did not start"
                descendant.append(int(self.process.stdout.readline()))
                raise original
            return self.process.communicate(**kwargs)

    monkeypatch.setattr(adapters_module.subprocess, "Popen", InterruptingProcess)
    program = ("import subprocess, sys, time\n"
               "child = subprocess.Popen([sys.executable, '-I', '-S', '-B', '-c', 'import time; time.sleep(30)'])\n"
               "print(child.pid, flush=True)\n"
               "time.sleep(30)\n")
    try:
        with pytest.raises(KeyboardInterrupt) as caught:
            adapters_module.run_process_group([sys.executable, "-I", "-S", "-B", "-c", program],
                                              text=True, capture_output=True, timeout=3, shell=False,
                                              env={"LC_ALL": "C"})
        assert caught.value is original
        assert len(launched) == 1 and launched[0].returncode == -adapters_module.signal.SIGKILL
        assert descendant
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            try:
                os.kill(descendant[0], 0)
            except ProcessLookupError:
                break
            time.sleep(0.01)
        else:
            pytest.fail("trusted descendant remained after process-group interruption cleanup")
    finally:
        # Keep the regression itself leak-free even against the old broken code.
        for process in launched:
            if process.poll() is None:
                try:
                    os.killpg(process.pid, adapters_module.signal.SIGKILL)
                except ProcessLookupError:
                    pass
            process.wait(timeout=2)
            for stream in (process.stdin, process.stdout, process.stderr):
                if stream is not None:
                    stream.close()


def response_shape():
    return {"type": "object", "additionalProperties": False,
            "properties": {"value": {"type": "string"}}, "required": ["value"]}


def claude_structured_payload(**changes):
    return {"type": "result", "subtype": "success", "is_error": False,
            "result": "Unstructured commentary is not the structured answer.",
            "structured_output": {"value": "é e\u0301"}, "num_turns": 3,
            "stop_reason": "end_turn", "terminal_reason": "completed",
            "total_cost_usd": 0.012,
            "usage": {"input_tokens": 7, "output_tokens": 5,
                      "cache_read_input_tokens": 2, "cache_creation_input_tokens": 1},
            "modelUsage": {"main-model": {"inputTokens": 7, "outputTokens": 5},
                           "auxiliary-model": {"inputTokens": 2, "outputTokens": 1}}, **changes}


def test_claude_native_schema_selects_validated_structured_field_and_retains_native_metadata():
    schema = response_shape()
    payload = claude_structured_payload()
    runner = RecordingRunner(json.dumps(payload))
    adapter = ClaudeCLIAdapter(runner=runner, allow_live=True, response_schema=schema)
    schema["properties"]["value"]["type"] = "integer"
    adapter.response_schema["properties"]["value"]["type"] = "boolean"
    result = adapter.generate("synthetic", timeout_seconds=5)
    args = runner.calls[0][0]
    assert json.loads(args[args.index("--json-schema") + 1]) == response_shape()
    assert json.loads(result.text) == payload["structured_output"]
    assert not result.errors and result.usage.total_tokens == 15
    assert result.setup["native_turns"] == 3
    assert result.setup["native_repairs"] is None  # Turn count does not identify repairs.
    assert result.setup["provider_model_usage"] == payload["modelUsage"]
    assert result.setup["native_stop_reason"] == "end_turn"
    assert result.setup["native_result_text"] == payload["result"]
    assert result.setup["native_structured_output"] == payload["structured_output"]
    assert result.setup["response_schema_utf8_bytes"] > 0
    assert result.setup["response_schema_sha256"]
    assert result.setup["tools"] == "disabled"


@pytest.mark.parametrize("change", [{"structured_output": {"value": 3}},
                                    {"structured_output": {"value": "x", "extra": "bad"}},
                                    {"structured_output": None},
                                    {"subtype": "error_max_structured_output_retries", "is_error": True}])
def test_claude_schema_failure_preserves_usage_without_repair_or_prose_fallback(change):
    runner = RecordingRunner(json.dumps(claude_structured_payload(**change)))
    result = ClaudeCLIAdapter(runner=runner, allow_live=True, response_schema=response_shape()).generate(
        "synthetic", timeout_seconds=5)
    assert result.errors and result.text == "" and result.usage.total_tokens is None
    assert result.setup["reported_usage_subtotal"]["total_tokens"] == 15
    assert result.setup["native_turns"] == 3
    assert len(runner.calls) == 1 and result.retries == 0


def test_claude_schema_missing_structured_field_never_uses_even_valid_prose():
    payload = claude_structured_payload(result='{"value":"looks valid"}')
    del payload["structured_output"]
    result = ClaudeCLIAdapter(runner=RecordingRunner(json.dumps(payload)), allow_live=True,
                              response_schema=response_shape()).generate("synthetic", timeout_seconds=5)
    assert result.text == "" and "missing" in result.errors[0]
    assert result.usage.total_tokens is None
    assert result.setup["reported_usage_subtotal"]["total_tokens"] == 15


def test_codex_native_schema_file_exists_only_during_call_and_preserves_isolation():
    observed = []

    def runner(args, **kwargs):
        path = Path(args[args.index("--output-schema") + 1])
        assert path.parent == Path(kwargs["cwd"])
        assert json.loads(path.read_text()) == response_shape()
        observed.append(path)
        return subprocess.CompletedProcess(args, 0, '\n'.join([
            json.dumps({"type": "item.completed", "item": {"type": "agent_message", "text": '{"value":"ok"}'}}),
            json.dumps({"type": "turn.completed", "usage": {"input_tokens": 10, "output_tokens": 5}}),
        ]), "")

    schema = response_shape()
    adapter = CodexCLIAdapter(runner=runner, allow_live=True, response_schema=schema)
    schema["properties"]["value"]["type"] = "integer"
    result = adapter.generate("synthetic", timeout_seconds=5)
    assert result.text == '{"value":"ok"}' and not result.errors
    assert len(observed) == 1 and not observed[0].exists()
    assert result.setup["native_completed_turns"] == 1


@pytest.mark.parametrize("text", ['{"value":3}', '{"value":"a","value":"b"}',
                                  '{"value":"a","extra":"b"}', '```json\n{"value":"a"}\n```'])
def test_codex_native_schema_is_independently_validated_without_relaxation(text):
    stdout = '\n'.join([
        json.dumps({"type": "item.completed", "item": {"type": "agent_message", "text": text}}),
        json.dumps({"type": "turn.completed", "usage": {"input_tokens": 10, "output_tokens": 5}}),
    ])
    result = CodexCLIAdapter(runner=RecordingRunner(stdout), allow_live=True,
                             response_schema=response_shape()).generate("synthetic", timeout_seconds=5)
    assert result.errors and result.text == "" and result.usage.total_tokens is None
    assert result.setup["reported_usage_subtotal"]["total_tokens"] == 15
    assert result.setup["native_agent_messages"] == [text]


def test_codex_all_native_turn_usage_survives_later_failure_and_partial_timeout():
    stdout = '\n'.join([
        json.dumps({"type": "turn.completed", "usage": {"input_tokens": 10, "output_tokens": 5}}),
        json.dumps({"type": "turn.failed", "usage": {"input_tokens": 3, "output_tokens": 2}}),
    ])
    result = CodexCLIAdapter(runner=RecordingRunner(stdout), allow_live=True).generate("synthetic", timeout_seconds=5)
    assert result.errors and result.usage.total_tokens is None
    assert result.setup["reported_usage_subtotal"]["total_tokens"] == 20
    assert result.setup["native_terminal_events"] == ["turn.completed", "turn.failed"]

    def runner(*args, **kwargs):
        raise subprocess.TimeoutExpired(args[0], kwargs["timeout"], output=stdout.encode())

    partial = CodexCLIAdapter(runner=runner, allow_live=True).generate("synthetic", timeout_seconds=5)
    assert partial.errors and partial.usage.total_tokens is None
    assert partial.setup["reported_usage_subtotal"]["total_tokens"] == 20
    assert partial.setup["native_completed_turns"] == 1


@pytest.mark.parametrize("timeout", [True, False])
def test_completed_codex_turn_followed_by_unreported_turn_never_claims_complete_usage(timeout):
    stdout = '\n'.join([
        json.dumps({"type": "turn.started"}),
        json.dumps({"type": "item.completed", "item": {"type": "agent_message", "text": "earlier answer"}}),
        json.dumps({"type": "turn.completed", "usage": {"input_tokens": 10, "output_tokens": 5}}),
        json.dumps({"type": "turn.started"}),
        # The later turn has no terminal event and no reported token counts.
    ])

    def runner(args, **kwargs):
        if timeout:
            raise subprocess.TimeoutExpired(args, kwargs["timeout"], output=stdout)
        return subprocess.CompletedProcess(args, 0, stdout, "")

    result = CodexCLIAdapter(runner=runner, allow_live=True).generate("synthetic", timeout_seconds=5)
    assert result.errors and result.text == ""
    assert all(getattr(result.usage, key) is None for key in result.usage.__dataclass_fields__)
    assert result.setup["usage_coverage"] == "incomplete_or_unknown"
    assert result.setup["reported_usage_subtotal"]["total_tokens"] == 15
    assert result.setup["native_completed_turns"] == 1


@pytest.mark.parametrize("kind", [ClaudeCLIAdapter, CodexCLIAdapter])
def test_cli_invalid_schema_is_rejected_before_execution(kind):
    with pytest.raises(Exception, match="not valid"):
        kind(response_schema={"type": "invented"})
