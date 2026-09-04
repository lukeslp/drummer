"""Isolated, opt-in adapters for synthetic handoff experiments.

The adapters never select another provider when one fails.  Live execution is
disabled by default so importing the benchmark or constructing an adapter has
no external effect.
"""

from __future__ import annotations

import ipaddress
import hashlib
import json
import os
import signal
import shutil
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Mapping, Protocol, Sequence
from urllib.parse import urlparse


class AdapterExecutionDisabled(RuntimeError):
    """Raised when a caller has not explicitly enabled live execution."""


class ProcessRunner(Protocol):
    def __call__(self, args: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]: ...


def run_process_group(args: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
    """Run one CLI in a fresh process group and reap that group on timeout."""

    if kwargs.get("shell") is not False:
        raise ValueError("agent CLIs must run with shell=False")
    input_text = kwargs.pop("input", None)
    timeout = kwargs.pop("timeout", None)
    check = bool(kwargs.pop("check", False))
    capture_output = bool(kwargs.pop("capture_output", False))
    if input_text is not None:
        kwargs["stdin"] = subprocess.PIPE
    if capture_output:
        kwargs["stdout"] = subprocess.PIPE
        kwargs["stderr"] = subprocess.PIPE
    process = subprocess.Popen(args, start_new_session=True, **kwargs)
    try:
        stdout, stderr = process.communicate(input=input_text, timeout=timeout)
    except subprocess.TimeoutExpired as error:
        os.killpg(process.pid, signal.SIGKILL)
        stdout, stderr = process.communicate()
        error.output = stdout
        error.stderr = stderr
        raise
    completed = subprocess.CompletedProcess(args, process.returncode, stdout, stderr)
    if check:
        completed.check_returncode()
    return completed


@dataclass(frozen=True)
class TokenUsage:
    """Counts reported by the provider; ``None`` means it did not report one."""

    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    cached_input_tokens: int | None = None
    cache_creation_input_tokens: int | None = None
    uncached_input_tokens: int | None = None


@dataclass(frozen=True)
class AdapterResult:
    text: str
    usage: TokenUsage
    elapsed_seconds: float
    retries: int = 0
    errors: tuple[str, ...] = ()
    setup: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class AdapterHealth:
    available: bool
    elapsed_seconds: float
    models: tuple[str, ...] = ()
    error: str | None = None


_ANTHROPIC_BILLING_ENV = {
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_AUTH_TOKEN",
    "ANTHROPIC_BASE_URL",
    "CLAUDE_CODE_USE_BEDROCK",
    "CLAUDE_CODE_USE_VERTEX",
    "CLAUDE_CODE_USE_FOUNDRY",
}

_OPENAI_BILLING_ENV = {
    "OPENAI_API_KEY",
    "OPENAI_BASE_URL",
    "AZURE_OPENAI_API_KEY",
    "AZURE_OPENAI_ENDPOINT",
}


def _isolated_environment(removed: set[str]) -> dict[str, str]:
    # Preserve the installed clients' subscription authentication location, not
    # arbitrary task secrets or provider routing. Filesystem access remains
    # controlled by the clients' tool-free isolated invocation.
    allowed = {"PATH", "HOME", "USER", "LOGNAME", "TMPDIR", "LANG", "LC_ALL", "TZ",
               "SYSTEMROOT", "WINDIR"}
    environment = {key: value for key, value in os.environ.items()
                   if key in allowed and key not in removed}
    environment.update({"CI": "1", "NO_COLOR": "1", "TERM": "dumb"})
    return environment


def _integer(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int) and value >= 0:
        return value
    return None


def _usage(
    *,
    input_tokens: object = None,
    output_tokens: object = None,
    total_tokens: object = None,
    cached_input_tokens: object = None,
    cache_creation_input_tokens: object = None,
    uncached_input_tokens: object = None,
) -> TokenUsage:
    reported_input = _integer(input_tokens)
    reported_output = _integer(output_tokens)
    reported_total = _integer(total_tokens)
    if reported_total is None and reported_input is not None and reported_output is not None:
        reported_total = reported_input + reported_output
    return TokenUsage(
        input_tokens=reported_input,
        output_tokens=reported_output,
        total_tokens=reported_total,
        cached_input_tokens=_integer(cached_input_tokens),
        cache_creation_input_tokens=_integer(cache_creation_input_tokens),
        uncached_input_tokens=_integer(uncached_input_tokens),
    )


def _claude_usage(raw: Mapping[str, object]) -> TokenUsage:
    # Claude reports uncached, cache-read, and cache-write input separately.
    # Missing components are unknown, never assumed zero.
    parts = [_integer(raw.get(key)) for key in (
        "input_tokens", "cache_read_input_tokens", "cache_creation_input_tokens")]
    total_input = sum(parts) if all(value is not None for value in parts) else None
    return _usage(input_tokens=total_input, output_tokens=raw.get("output_tokens"),
                  uncached_input_tokens=raw.get("input_tokens"),
                  cached_input_tokens=raw.get("cache_read_input_tokens"),
                  cache_creation_input_tokens=raw.get("cache_creation_input_tokens"))


def _error_text(returncode: int, stderr: str) -> str:
    detail = " ".join(stderr.strip().split())
    prefix = f"process exited with status {returncode}"
    return f"{prefix}: {detail}" if detail else prefix


def _strict_json(text: str) -> object:
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result

    def invalid_constant(value):
        raise ValueError("non-finite JSON constant")

    return json.loads(text, object_pairs_hook=unique, parse_constant=invalid_constant)


class _CLIAdapter:
    adapter_name = "cli"
    billing_environment: set[str] = set()

    def __init__(
        self,
        *,
        executable: str,
        model: str | None,
        runner: ProcessRunner | None = None,
        clock: Callable[[], float] = time.perf_counter,
        allow_live: bool = False,
        response_schema: Mapping[str, object] | None = None,
    ) -> None:
        self.executable = executable
        self.model = model
        self._runner = runner or run_process_group
        self._clock = clock
        self.allow_live = allow_live
        self._schema_json: str | None = None
        if response_schema is not None:
            from jsonschema import Draft202012Validator

            if not isinstance(response_schema, Mapping):
                raise ValueError("native CLI schema must be a JSON object")
            serialized = json.dumps(response_schema, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":"), allow_nan=False)
            Draft202012Validator.check_schema(json.loads(serialized))
            self._schema_json = serialized

    @property
    def response_schema(self) -> dict[str, object] | None:
        # Returning a copy also protects against mutation through the adapter itself.
        return json.loads(self._schema_json) if self._schema_json is not None else None

    def build_command(self, *, schema_path: str | None = None) -> list[str]:
        raise NotImplementedError

    def _parse(self, stdout: str) -> tuple[str, TokenUsage]:
        raise NotImplementedError

    def _partial_usage(self, stdout: str) -> TokenUsage:
        try:
            return self._parse(stdout)[1]
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            return TokenUsage()

    def _reported_setup(self, stdout: str) -> dict[str, object]:
        return {}

    def _setup(self) -> dict[str, object]:
        return {
            "adapter": self.adapter_name,
            "model": self.model,
            "requested_model": self.model,
            "tools": "disabled",
            "mcp": "disabled",
            "hooks": "disabled",
            "project_context": "disabled",
            "cwd": "isolated-temporary-directory",
            "paid_api_environment": "removed",
            "response_mode": "native-schema-guided" if self._schema_json else "prompt-only",
            "response_schema_sha256": hashlib.sha256(self._schema_json.encode()).hexdigest()
            if self._schema_json is not None else None,
            "response_schema_utf8_bytes": len(self._schema_json.encode())
            if self._schema_json is not None else 0,
            "application_repairs": 0,
            "native_repairs": None,
            "usage_coverage": "not_established",
            "structured_return_mechanism": "client-managed; not an action-tool grant"
            if self._schema_json is not None else "not requested",
        }

    def _usage_setup(self, stdout: str, *, complete: bool) -> dict[str, object]:
        setup = self._setup()
        setup.update(self._reported_setup(stdout))
        setup.update(
            usage_coverage="complete_client_report" if complete else "incomplete_or_unknown",
            reported_usage_subtotal=asdict(self._partial_usage(stdout)),
            reported_usage_subtotal_interpretation=(
                "Successful terminal client report; individual unreported fields remain unknown."
                if complete else
                "Reported portions only, not the complete invocation; later or failed work may be unreported."
            ),
        )
        return setup

    def generate(self, prompt: str, *, timeout_seconds: float) -> AdapterResult:
        if not self.allow_live:
            raise AdapterExecutionDisabled(
                f"{self.adapter_name} execution is disabled; an integration gate must opt in"
            )
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")

        started = self._clock()
        try:
            with tempfile.TemporaryDirectory(prefix="drummer-handoff-") as temporary_cwd:
                schema_path = None
                if self._schema_json is not None:
                    schema_file = Path(temporary_cwd) / "response.schema.json"
                    schema_file.write_text(self._schema_json, encoding="utf-8")
                    schema_path = str(schema_file)
                command = self.build_command(schema_path=schema_path)
                completed = self._runner(
                    command,
                    input=prompt,
                    text=True,
                    capture_output=True,
                    cwd=temporary_cwd,
                    timeout=timeout_seconds,
                    check=False,
                    shell=False,
                    env=_isolated_environment(self.billing_environment),
                )
        except subprocess.TimeoutExpired as error:
            elapsed = self._clock() - started
            partial = error.stdout or ""
            if isinstance(partial, bytes):
                partial = partial.decode("utf-8", errors="replace")
            setup = self._usage_setup(partial, complete=False)
            return AdapterResult(
                text="",
                usage=TokenUsage(),
                elapsed_seconds=elapsed,
                errors=(f"process timed out after {timeout_seconds:g} seconds",),
                setup=setup,
            )
        except FileNotFoundError:
            elapsed = self._clock() - started
            return AdapterResult(
                text="",
                usage=TokenUsage(),
                elapsed_seconds=elapsed,
                errors=(f"executable not found: {self.executable}",),
                setup=self._setup(),
            )

        elapsed = self._clock() - started
        if completed.returncode != 0:
            setup = self._usage_setup(completed.stdout, complete=False)
            return AdapterResult(
                text="",
                usage=TokenUsage(),
                elapsed_seconds=elapsed,
                errors=(_error_text(completed.returncode, completed.stderr),),
                setup=setup,
            )
        try:
            text, usage = self._parse(completed.stdout)
            if self._schema_json is not None:
                from jsonschema import Draft202012Validator

                parsed = _strict_json(text)
                if not Draft202012Validator(self.response_schema).is_valid(parsed):
                    raise ValueError("native structured output failed independent schema validation")
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            setup = self._usage_setup(completed.stdout, complete=False)
            return AdapterResult(
                text="",
                usage=TokenUsage(),
                elapsed_seconds=elapsed,
                errors=(f"invalid {self.adapter_name} output: {error}",),
                setup=setup,
            )
        setup = self._usage_setup(completed.stdout, complete=True)
        return AdapterResult(
            text=text,
            usage=usage,
            elapsed_seconds=elapsed,
            setup=setup,
        )


class ClaudeCLIAdapter(_CLIAdapter):
    """Claude Code print-mode adapter with all customizations and tools disabled."""

    adapter_name = "claude-cli"
    billing_environment = _ANTHROPIC_BILLING_ENV

    def __init__(
        self,
        *,
        executable: str = "claude",
        model: str | None = None,
        runner: ProcessRunner | None = None,
        clock: Callable[[], float] = time.perf_counter,
        allow_live: bool = False,
        response_schema: Mapping[str, object] | None = None,
    ) -> None:
        super().__init__(
            executable=executable,
            model=model,
            runner=runner,
            clock=clock,
            allow_live=allow_live,
            response_schema=response_schema,
        )

    def build_command(self, *, schema_path: str | None = None) -> list[str]:
        command = [
            self.executable,
            "--safe-mode",
            "--restricted",
            "--tools",
            "",
            "--strict-mcp-config",
            "--mcp-config",
            '{"mcpServers":{}}',
            "--no-session-persistence",
            "--disable-slash-commands",
            "--no-chrome",
            "--permission-mode",
            "dontAsk",
            "--input-format",
            "text",
            "--output-format",
            "json",
        ]
        if self.model:
            command.extend(("--model", self.model))
        if self._schema_json is not None:
            command.extend(("--json-schema", self._schema_json))
        command.append("-p")
        return command

    def _setup(self) -> dict[str, object]:
        setup = super()._setup()
        setup.update(
            {
                "customizations": "safe-mode",
                "restricted_mode": True,
                "strict_mcp_config": True,
                "session_persistence": "disabled",
            }
        )
        return setup

    def _parse(self, stdout: str) -> tuple[str, TokenUsage]:
        payload = _strict_json(stdout)
        if not isinstance(payload, dict):
            raise TypeError("top-level result is not an object")
        if payload.get("is_error") is True or payload.get("subtype") not in (None, "success"):
            raise ValueError(str(payload.get("result") or "Claude reported an error"))
        if self._schema_json is not None:
            if "structured_output" not in payload:
                raise ValueError("Claude structured_output is missing; no prose fallback")
            text = json.dumps(payload["structured_output"], ensure_ascii=False,
                              separators=(",", ":"), allow_nan=False)
        else:
            text = payload.get("result")
            if not isinstance(text, str):
                raise TypeError("result text is missing")
        raw_usage = payload.get("usage", {})
        if not isinstance(raw_usage, dict):
            raw_usage = {}
        usage = _claude_usage(raw_usage)
        return text, usage

    def _partial_usage(self, stdout: str) -> TokenUsage:
        try:
            payload = json.loads(stdout)
        except (TypeError, ValueError, json.JSONDecodeError):
            return TokenUsage()
        if not isinstance(payload, dict) or not isinstance(payload.get("usage"), dict):
            return TokenUsage()
        raw_usage = payload["usage"]
        return _claude_usage(raw_usage)

    def _reported_setup(self, stdout: str) -> dict[str, object]:
        try:
            payload = json.loads(stdout)
        except (TypeError, ValueError, json.JSONDecodeError):
            return {}
        if not isinstance(payload, dict):
            return {}
        reported: list[str] = []
        if isinstance(payload.get("model"), str):
            reported.append(payload["model"])
        model_usage = payload.get("modelUsage")
        if isinstance(model_usage, dict):
            reported.extend(key for key in model_usage if isinstance(key, str))
        raw_usage = payload.get("usage", {})
        counts = {key: _integer(raw_usage.get(key)) for key in (
            "input_tokens", "output_tokens", "cache_read_input_tokens", "cache_creation_input_tokens"
        )} if isinstance(raw_usage, dict) else {}
        return {"provider_reported_models": tuple(dict.fromkeys(reported)),
                "provider_usage": counts, "input_accounting": "uncached+cache_read+cache_creation",
                "provider_model_usage": model_usage if isinstance(model_usage, dict) else None,
                "top_level_usage_auxiliary_coverage": "unverified; preserve modelUsage separately",
                "native_turns": _integer(payload.get("num_turns")),
                "native_subtype": payload.get("subtype"),
                "native_stop_reason": payload.get("stop_reason"),
                "native_terminal_reason": payload.get("terminal_reason"),
                "provider_total_cost_usd": payload.get("total_cost_usd"),
                "native_structured_output_present": "structured_output" in payload,
                "native_result_text": payload.get("result"),
                "native_structured_output": payload.get("structured_output")}


_CODEX_DISABLED_FEATURES: tuple[str, ...] = (
    "apps",
    "artifact",
    "browser_use",
    "browser_use_external",
    "browser_use_full_cdp_access",
    "code_mode",
    "code_mode_host",
    "computer_use",
    "deferred_executor",
    "goals",
    "hooks",
    "image_generation",
    "in_app_browser",
    "in_app_local_automation",
    "multi_agent",
    "multi_agent_v2",
    "network_proxy",
    "plugins",
    "shell_tool",
    "skill_mcp_dependency_install",
    "skill_search",
    "sleep_tool",
    "tool_suggest",
    "unified_exec",
    "view_image",
    "web_search_cached",
    "web_search_request",
    "workspace_dependencies",
)


class CodexCLIAdapter(_CLIAdapter):
    """Codex exec adapter isolated from tools, rules, MCP, and repository context."""

    adapter_name = "codex-cli"
    billing_environment = _OPENAI_BILLING_ENV

    def __init__(
        self,
        *,
        executable: str = "codex",
        model: str | None = None,
        runner: ProcessRunner | None = None,
        clock: Callable[[], float] = time.perf_counter,
        allow_live: bool = False,
        response_schema: Mapping[str, object] | None = None,
    ) -> None:
        super().__init__(
            executable=executable,
            model=model,
            runner=runner,
            clock=clock,
            allow_live=allow_live,
            response_schema=response_schema,
        )

    def build_command(self, *, schema_path: str | None = None) -> list[str]:
        command = [
            self.executable,
            "exec",
            "--ignore-user-config",
            "--ignore-rules",
            "--ephemeral",
            "--skip-git-repo-check",
            "--strict-config",
            "--sandbox",
            "read-only",
            "--config",
            "approval_policy=\"never\"",
            "--config",
            "mcp_servers={}",
            "--config",
            'web_search="disabled"',
            "--config",
            "project_doc_max_bytes=0",
        ]
        for feature in _CODEX_DISABLED_FEATURES:
            command.extend(("--disable", feature))
        if self.model:
            command.extend(("--model", self.model))
        if self._schema_json is not None:
            if schema_path is None:
                raise ValueError("Codex native schema requires a temporary schema path")
            command.extend(("--output-schema", schema_path))
        command.extend(("--json", "-"))
        return command

    def _setup(self) -> dict[str, object]:
        setup = super()._setup()
        setup.update(
            {
                "strict_config": True,
                "project_doc_max_bytes": 0,
                "web_search": "disabled",
                "sandbox": "read-only",
                "disabled_features": _CODEX_DISABLED_FEATURES,
            }
        )
        return setup

    def _parse(self, stdout: str) -> tuple[str, TokenUsage]:
        answer: str | None = None
        completed = False
        open_turn = False
        for line in stdout.splitlines():
            if not line.strip():
                continue
            event = json.loads(line)
            if not isinstance(event, dict):
                continue
            if event.get("type") == "item.completed":
                item = event.get("item")
                if isinstance(item, dict) and item.get("type") == "agent_message":
                    candidate = item.get("text")
                    if isinstance(candidate, str):
                        answer = candidate
            if event.get("type") in {"turn.failed", "error"}:
                raise ValueError("Codex reported a failed turn")
            if event.get("type") == "turn.started":
                open_turn = True
            if event.get("type") == "turn.completed":
                completed = True
                open_turn = False
        if answer is None:
            raise ValueError("final agent message is missing")
        if not completed:
            raise ValueError("Codex completed-turn event is missing")
        if open_turn:
            raise ValueError("Codex later turn has no terminal event")
        return answer, self._partial_usage(stdout)

    def _partial_usage(self, stdout: str) -> TokenUsage:
        # Native turn reports are per-turn. Keep all of them, including usage
        # preceding a terminal failure, instead of silently returning only the last.
        reports = []
        for line in stdout.splitlines():
            try:
                event = json.loads(line)
            except (ValueError, TypeError):
                continue
            if not isinstance(event, dict) or event.get("type") not in {"turn.completed", "turn.failed"}:
                continue
            raw = event.get("usage")
            if not isinstance(raw, dict):
                raw = {}
            reports.append(_usage(**{key: raw.get(key) for key in TokenUsage.__dataclass_fields__}))
        if not reports:
            return TokenUsage()
        totals = {}
        for key in TokenUsage.__dataclass_fields__:
            values = [getattr(item, key) for item in reports]
            totals[key] = sum(values) if all(value is not None for value in values) else None
        return TokenUsage(**totals)

    def _reported_setup(self, stdout: str) -> dict[str, object]:
        reported: list[str] = []
        turn_usages: list[dict[str, object]] = []
        terminal_events: list[str] = []
        messages: list[str] = []
        for line in stdout.splitlines():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(event, dict) and isinstance(event.get("model"), str):
                reported.append(event["model"])
            if isinstance(event, dict) and event.get("type") == "item.completed":
                item = event.get("item")
                if isinstance(item, dict) and item.get("type") == "agent_message" and isinstance(item.get("text"), str):
                    messages.append(item["text"])
            if isinstance(event, dict) and event.get("type") in {"turn.completed", "turn.failed", "error"}:
                terminal_events.append(event["type"])
                if isinstance(event.get("usage"), dict):
                    turn_usages.append(event["usage"])
        return {"provider_reported_models": tuple(dict.fromkeys(reported)),
                "native_turn_usage_records": turn_usages,
                "native_terminal_events": terminal_events,
                "native_completed_turns": terminal_events.count("turn.completed"),
                "native_agent_messages": messages,
                "native_repairs": None}


def _local_base_url(value: str, trusted_hosts: Sequence[str]) -> tuple[str, str]:
    parsed = urlparse(value)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username:
        raise ValueError("local endpoint must be an HTTP(S) loopback or explicitly trusted URL")
    hostname = parsed.hostname
    is_loopback = hostname.lower() == "localhost"
    if not is_loopback:
        try:
            is_loopback = ipaddress.ip_address(hostname).is_loopback
        except ValueError:
            is_loopback = False
    trusted = {candidate.lower().strip("[]") for candidate in trusted_hosts}
    if not is_loopback and hostname.lower() not in trusted:
        raise ValueError("local endpoint must use a loopback or explicitly trusted host")
    if not is_loopback:
        try:
            address = ipaddress.ip_address(hostname)
        except ValueError as exc:
            raise ValueError("trusted local endpoints must use a literal private IP, not public DNS") from exc
        if not (address.is_private or address.is_link_local) or address.is_unspecified or address.is_multicast:
            raise ValueError("trusted local endpoints must use a private or link-local address")
    scope = "loopback-only" if is_loopback else f"explicit-host:{hostname.lower()}"
    return value.rstrip("/"), scope


class _NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(
        self,
        request: urllib.request.Request,
        file_pointer: object,
        code: int,
        message: str,
        headers: object,
        new_url: str,
    ) -> None:
        raise urllib.error.HTTPError(
            request.full_url,
            code,
            f"redirect refused (destination was {urlparse(new_url).hostname or 'unknown'})",
            headers,  # type: ignore[arg-type]
            file_pointer,  # type: ignore[arg-type]
        )


class LocalOpenAIAdapter:
    """OpenAI-compatible loopback adapter with an explicit health preflight."""

    adapter_name = "local-openai-compatible"

    def __init__(
        self,
        *,
        base_url: str = "http://127.0.0.1:1234/v1",
        model: str,
        trusted_hosts: Sequence[str] = (),
        urlopen: Callable[..., object] | None = None,
        clock: Callable[[], float] = time.perf_counter,
        allow_live: bool = False,
        max_retries: int = 0,
        max_tokens: int = 2048,
        temperature: float = 0.0,
        response_schema: Mapping[str, object] | None = None,
        reasoning_effort: str | None = None,
    ) -> None:
        if not model:
            raise ValueError("model must be explicit")
        if max_retries < 0:
            raise ValueError("max_retries cannot be negative")
        self.base_url, self.endpoint_scope = _local_base_url(base_url, trusted_hosts)
        self.model = model
        self._urlopen = urlopen or urllib.request.build_opener(_NoRedirectHandler()).open
        self._clock = clock
        self.allow_live = allow_live
        self.max_retries = max_retries
        if not 1 <= max_tokens <= 8192 or not 0 <= temperature <= 2:
            raise ValueError("generation settings exceed the bounded local experiment range")
        self.max_tokens = max_tokens
        self.temperature = temperature
        if reasoning_effort not in {None, "none", "low", "medium", "high", "max"}:
            raise ValueError("unsupported local reasoning effort")
        self.reasoning_effort = reasoning_effort
        # Copy the schema so later caller mutations cannot change this run.
        self.response_schema = None
        if response_schema is not None:
            from jsonschema import Draft202012Validator

            self.response_schema = json.loads(json.dumps(response_schema, allow_nan=False))
            Draft202012Validator.check_schema(self.response_schema)

    def _require_live(self) -> None:
        if not self.allow_live:
            raise AdapterExecutionDisabled(
                "local endpoint execution is disabled; an integration gate must opt in"
            )

    def _request_json(
        self,
        path: str,
        *,
        timeout_seconds: float,
        payload: Mapping[str, object] | None = None,
    ) -> dict[str, object]:
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}{path}",
            data=data,
            headers={"Content-Type": "application/json"} if data is not None else {},
            method="POST" if data is not None else "GET",
        )
        with self._urlopen(request, timeout=timeout_seconds) as response:
            final_url = getattr(response, "geturl", lambda: request.full_url)()
            if urlparse(final_url).hostname != urlparse(request.full_url).hostname:
                raise RuntimeError("cross-host redirect refused")
            status = getattr(response, "status", 200)
            if not 200 <= status < 300:
                raise RuntimeError(f"local endpoint returned HTTP {status}")
            decoded = json.loads(response.read())
        if not isinstance(decoded, dict):
            raise TypeError("local endpoint returned a non-object JSON value")
        return decoded

    def _models(self, *, timeout_seconds: float) -> tuple[str, ...]:
        payload = self._request_json("/models", timeout_seconds=timeout_seconds)
        data = payload.get("data", [])
        if not isinstance(data, list):
            raise TypeError("model catalog data is not a list")
        models: list[str] = []
        for item in data:
            if isinstance(item, dict) and isinstance(item.get("id"), str):
                models.append(item["id"])
        return tuple(models)

    def health(self, *, timeout_seconds: float) -> AdapterHealth:
        self._require_live()
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        started = self._clock()
        try:
            models = self._models(timeout_seconds=timeout_seconds)
        except (OSError, RuntimeError, TypeError, ValueError, json.JSONDecodeError) as error:
            return AdapterHealth(
                available=False,
                elapsed_seconds=self._clock() - started,
                error=f"{type(error).__name__}: {error}",
            )
        return AdapterHealth(
            available=self.model in models,
            elapsed_seconds=self._clock() - started,
            models=models,
            error=None if self.model in models else f"model is not loaded: {self.model}",
        )

    def generate(self, prompt: str, *, timeout_seconds: float) -> AdapterResult:
        self._require_live()
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        started = self._clock()
        errors: list[str] = []
        try:
            models = self._models(timeout_seconds=timeout_seconds)
        except (OSError, RuntimeError, TypeError, ValueError, json.JSONDecodeError) as error:
            return AdapterResult(
                text="",
                usage=TokenUsage(),
                elapsed_seconds=self._clock() - started,
                errors=(f"health check failed: {type(error).__name__}: {error}",),
                setup=self._setup("failed"),
            )
        if self.model not in models:
            return AdapterResult(
                text="",
                usage=TokenUsage(),
                elapsed_seconds=self._clock() - started,
                errors=(f"health check failed: model is not loaded: {self.model}",),
                setup=self._setup("failed"),
            )

        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }
        if self.response_schema is not None:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "drummer_response", "strict": True,
                                "schema": self.response_schema},
            }
        if self.reasoning_effort is not None:
            payload["reasoning_effort"] = self.reasoning_effort
        response: dict[str, object] | None = None
        retries = 0
        for attempt in range(self.max_retries + 1):
            remaining = timeout_seconds - (self._clock() - started)
            if remaining <= 0:
                errors.append("local generation deadline exhausted during health check or retries")
                break
            try:
                response = self._request_json(
                    "/chat/completions", timeout_seconds=remaining, payload=payload
                )
                break
            except (OSError, RuntimeError, TypeError, ValueError, json.JSONDecodeError) as error:
                errors.append(f"{type(error).__name__}: {error}")
                if attempt < self.max_retries:
                    retries += 1
        elapsed = self._clock() - started
        if response is None:
            return AdapterResult(
                text="",
                usage=TokenUsage(),
                elapsed_seconds=elapsed,
                retries=retries,
                errors=tuple(errors),
                setup=self._setup("passed"),
            )
        try:
            choices = response["choices"]
            if not isinstance(choices, list) or not choices:
                raise TypeError("choices is empty")
            choice = choices[0]
            if not isinstance(choice, dict) or not isinstance(choice.get("message"), dict):
                raise TypeError("message is missing")
            content = choice["message"].get("content")
            if not isinstance(content, str):
                raise TypeError("message content is missing")
            raw_usage = response.get("usage", {})
            if not isinstance(raw_usage, dict):
                raw_usage = {}
            details = raw_usage.get("prompt_tokens_details", {})
            if not isinstance(details, dict):
                details = {}
            usage = _usage(
                input_tokens=raw_usage.get("prompt_tokens"),
                output_tokens=raw_usage.get("completion_tokens"),
                total_tokens=raw_usage.get("total_tokens"),
                cached_input_tokens=details.get("cached_tokens"),
            )
        except (KeyError, TypeError) as error:
            return AdapterResult(
                text="",
                usage=TokenUsage(),
                elapsed_seconds=elapsed,
                retries=retries,
                errors=tuple((*errors, f"invalid completion response: {error}")),
                setup=self._setup("passed"),
            )
        setup = self._setup("passed")
        if isinstance(response.get("model"), str):
            setup["provider_reported_model"] = response["model"]
        return AdapterResult(
            text=content,
            usage=usage,
            elapsed_seconds=elapsed,
            retries=retries,
            errors=tuple(errors),
            setup=setup,
        )

    def _setup(self, health: str) -> dict[str, object]:
        return {
            "adapter": self.adapter_name,
            "model": self.model,
            "health": health,
            "endpoint_scope": self.endpoint_scope,
            "tools": "not-applicable",
            "paid_api_fallback": "disabled",
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "response_mode": "schema-guided" if self.response_schema is not None else "prompt-only",
            "reasoning_effort": self.reasoning_effort,
        }


def installed_cli_status(executables: Sequence[str] = ("codex", "claude")) -> dict[str, bool]:
    """Return executable presence without invoking a model or reading credentials."""

    return {name: shutil.which(name) is not None for name in executables}
