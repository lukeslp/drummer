from concurrent.futures import ThreadPoolExecutor

import pytest

from drummer.budget import BudgetError, BudgetLedger, micro_usd


def test_reserve_settle_and_failed_runs_count(tmp_path):
    ledger = BudgetLedger(tmp_path / "budget.sqlite")
    item = ledger.reserve("smoke", "0.40002")
    ledger.submitted(item, "job")
    with pytest.raises(BudgetError):
        ledger.reserve("smoke", 1)
    ledger.settle(item, evidence="Job failed; conservative maximum")
    assert ledger.snapshot()["committed_micro_usd"] == 400020
    assert ledger.snapshot()["entries"][0]["metadata"]["cost_basis"] == "reserved_upper_bound"


def test_ceiling_and_tranche(tmp_path):
    ledger = BudgetLedger(tmp_path / "budget.sqlite")
    with pytest.raises(BudgetError):
        ledger.reserve("smoke", "20.000001")
    for tranche, value in [("smoke", 20), ("training", 100), ("evaluation", 40), ("interop", 25), ("reserve", 65)]:
        ledger.reserve(tranche, value, kind="expense")
    with pytest.raises(BudgetError):
        ledger.reserve("interop", "0.01", kind="expense")
    assert ledger.snapshot()["remaining_micro_usd"] == 0


def test_uncertain_submission_retains_slot_and_funds(tmp_path):
    ledger = BudgetLedger(tmp_path / "budget.sqlite")
    item = ledger.reserve("smoke", "0.4")
    ledger.uncertain(item, "timeout")
    with pytest.raises(BudgetError):
        ledger.reserve("smoke", "0.4")
    assert ledger.snapshot()["committed_micro_usd"] == 400000


def test_two_process_like_connections_cannot_double_submit(tmp_path):
    path = tmp_path / "budget.sqlite"
    BudgetLedger(path)
    def attempt(_):
        try:
            return BudgetLedger(path).reserve("smoke", 1)
        except BudgetError:
            return None
    with ThreadPoolExecutor(4) as pool:
        assert sum(x is not None for x in pool.map(attempt, range(4))) == 1


@pytest.mark.parametrize("value", [-1, "NaN", "Infinity"])
def test_invalid_costs(value):
    with pytest.raises(BudgetError):
        micro_usd(value)


def test_round_up():
    assert micro_usd("0.0000001") == 1
