import pytest

from drummer.cli import main, parser


def test_default_handoff_does_not_call_models():
    args = parser().parse_args(["handoff"])
    assert args.live is False
    assert args.limit == 1


def test_cloud_requires_verification_and_ledger():
    with pytest.raises(SystemExit):
        parser().parse_args(["cloud-smoke"])


def test_budget_cli(tmp_path, capsys):
    with pytest.raises(SystemExit) as result:
        main(["budget", "--ledger", str(tmp_path / "b.sqlite")])
    assert result.value.code == 0
    assert '"committed_micro_usd": 0' in capsys.readouterr().out
