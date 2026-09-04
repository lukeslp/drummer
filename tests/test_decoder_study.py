from dataclasses import asdict
import json

import pytest

from drummer.adapters import AdapterExecutionDisabled, AdapterResult, TokenUsage
from drummer.decoder_study import DecoderStudyConfig, run_decoder_study, sum_usage
from drummer.functional_handoffs import build_functional_prompt, functional_handoff_cases


class FixtureAdapter:
    model = "qwen2.5:1.5b"
    max_retries = 0

    def __init__(self, fail_first=False, transport_error=False):
        self.calls = []
        self.fail_first = fail_first
        self.transport_error = transport_error
        case = functional_handoff_cases()[0]
        self.prompt = build_functional_prompt(case, representation="full-english", condition="packet-context")

    def generate(self, text, *, timeout_seconds):
        self.calls.append(text)
        assert timeout_seconds > 0
        if self.transport_error:
            return AdapterResult("", TokenUsage(), 1, errors=("timeout",))
        if self.fail_first and len(self.calls) == 1:
            return AdapterResult("not json", TokenUsage(2, 1, 3, 0), 0.1)
        return AdapterResult(json.dumps(self.prompt.delivered_expected), TokenUsage(7, 3, 10, 2), 0.2)


def config(**overrides):
    values = dict(model="qwen2.5:1.5b", endpoint="http://192.168.0.100:11434/v1",
                  representations=("full-english",), conditions=("packet-context",), case_limit=1)
    return DecoderStudyConfig(**(values | overrides))


def execute(path, adapter, **overrides):
    return run_decoder_study(path, config(**overrides), allow_live=True, adapter=adapter,
                             snapshot={"model": adapter.model, "artifact_digest": "a" * 64}, require_clean=False)


def test_live_optin_before_any_output_or_network(tmp_path):
    with pytest.raises(AdapterExecutionDisabled):
        run_decoder_study(tmp_path / "run", config())
    assert not (tmp_path / "run").exists()


def test_exact_scoring_and_total_repair_accounting(tmp_path):
    adapter = FixtureAdapter(fail_first=True)
    report = execute(tmp_path / "run", adapter, repair_limit=1)
    assert report["status"] == "complete" and report["calls"] == 2
    record = report["records"][0]
    assert record["first_pass"]["schema_valid"] is False
    assert record["final_pass"]["delivered_fidelity_exact"] is True
    assert record["usage_all_attempts"]["total_tokens"] == 13
    assert report["usage_all_attempts"]["total_tokens"] == 13
    assert json.loads((tmp_path / "run" / "study.json").read_text())["status"] == "complete"
    # No label feedback is added to the repair request.
    assert "Previous response" in adapter.calls[1]
    assert json.dumps(adapter.prompt.delivered_expected) not in adapter.calls[1]
    with pytest.raises(ValueError, match="exists"):
        execute(tmp_path / "run", FixtureAdapter())


def test_transport_failure_is_retained_and_never_retried(tmp_path):
    adapter = FixtureAdapter(transport_error=True)
    report = execute(tmp_path / "run", adapter, repair_limit=1)
    assert report["status"] == "transport_stopped"
    assert report["calls"] == 1 and len(adapter.calls) == 1
    assert report["usage_all_attempts"]["total_tokens"] is None


def test_call_budget_counts_repair(tmp_path):
    report = execute(tmp_path / "run", FixtureAdapter(fail_first=True), repair_limit=1, max_calls=1)
    assert report["calls"] == 1
    assert report["records"][0]["final_pass"]["schema_valid"] is False
    assert report["status"] == "budget_exhausted"


def test_unknown_usage_is_not_free():
    assert sum_usage([{"usage": asdict(TokenUsage(2, 1, 3))},
                      {"usage": asdict(TokenUsage())}])["total_tokens"] is None


@pytest.mark.parametrize("kwargs", [dict(max_calls=0), dict(timeout_seconds=121),
                                   dict(repair_limit=2), dict(endpoint="https://public.example/v1"),
                                   dict(model="download-me"), dict(max_seconds=float("nan"))])
def test_invalid_bounds(kwargs):
    with pytest.raises(ValueError):
        config(**kwargs)
