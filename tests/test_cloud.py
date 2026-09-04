from types import SimpleNamespace

import pytest

from drummer.budget import BudgetError, BudgetLedger
from drummer.cloud import IMAGE, reconcile, worker_command


def test_pinned_image_and_no_command_injection():
    assert "@sha256:" in IMAGE
    command = worker_command("a" * 40)
    assert command[:2] == ["python", "-c"]
    assert "--frozen" in command[2]
    with pytest.raises(BudgetError):
        worker_command("main;echo nope")


def test_reconcile_never_releases_unknown(tmp_path):
    ledger = BudgetLedger(tmp_path / "b.sqlite")
    item = ledger.reserve("smoke", 1)
    ledger.uncertain(item, "timeout")
    api = SimpleNamespace(list_jobs=lambda **kw: [])
    assert reconcile(ledger, api=api)["entries"][0]["status"] == "uncertain"


def test_reconcile_charges_failed_job_maximum(tmp_path):
    ledger = BudgetLedger(tmp_path / "b.sqlite")
    item = ledger.reserve("smoke", 1)
    ledger.submitted(item, "job")
    api = SimpleNamespace(inspect_job=lambda **kw: SimpleNamespace(status=SimpleNamespace(stage="ERROR")))
    result = reconcile(ledger, api=api)
    assert result["committed_micro_usd"] == 1000000
    assert result["entries"][0]["status"] == "settled"
