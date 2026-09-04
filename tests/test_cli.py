import pytest

from drummer.cli import main, parser


def test_default_handoff_does_not_call_models():
    args = parser().parse_args(["handoff"])
    assert args.live is False
    assert args.limit == 1


def test_mixed_model_pair_is_available_without_live_default():
    args = parser().parse_args(["pair", "--sender", "codex", "--receiver", "qwen-1.5b"])
    assert args.live is False
    assert args.receiver == "qwen-1.5b"


def test_cloud_requires_verification_and_ledger():
    with pytest.raises(SystemExit):
        parser().parse_args(["cloud-smoke"])


def test_budget_cli(tmp_path, capsys):
    with pytest.raises(SystemExit) as result:
        main(["budget", "--ledger", str(tmp_path / "b.sqlite")])
    assert result.value.code == 0
    assert '"committed_micro_usd": 0' in capsys.readouterr().out


def test_local_bench_is_offline_and_diagnostics_cannot_select_test():
    args = parser().parse_args(["compression-bench"])
    assert args.live is False and args.limit == 24
    with pytest.raises(SystemExit):
        parser().parse_args(["channel-diagnostics", "--checkpoint", "x", "--corpus", "x",
                            "--output", "x", "--split", "test"])


def test_autopsy_cannot_overwrite_input(tmp_path):
    path = tmp_path / "evidence.json"
    path.write_text("{}")
    with pytest.raises(SystemExit) as result:
        main(["autopsy", "--report", str(path), "--output", str(path)])
    assert result.value.code != 0
    assert path.read_text() == "{}"
