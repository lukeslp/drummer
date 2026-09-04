import json
from types import SimpleNamespace

from drummer.pilot import run_pilot, select_pressure


def test_selection_preserves_quality_then_minimizes_bits():
    records = [
        {"pressure": .01, "success": .99, "probe_bits": 5},
        {"pressure": .03, "success": .97, "probe_bits": 4},
        {"pressure": .1, "success": .90, "probe_bits": 1},
    ]
    assert select_pressure(records, .99)["pressure"] == .03


def test_failed_quality_does_not_auto_scale():
    assert select_pressure([{"pressure": .01, "success": .65, "probe_bits": 1}], .99) is None


def test_deadline_prevents_first_training_call(tmp_path, monkeypatch):
    monkeypatch.setattr("drummer.training.train", lambda config: (_ for _ in ()).throw(AssertionError("must not train")))
    result = run_pilot({}, data_root=tmp_path / "data", output_root=tmp_path / "runs", deadline_unix=0)
    assert result["status"] == "stopped_deadline"
    assert result["test_unsealed"] is False


def test_failed_calibration_keeps_test_sealed(tmp_path, monkeypatch):
    monkeypatch.setattr("drummer.world.generate_corpus", lambda *args: {})
    monkeypatch.setattr("drummer.training.train", lambda config: SimpleNamespace(
        best_checkpoint="unused", to_dict=lambda: {"seed": config["seed"]}))
    monkeypatch.setattr("drummer.evaluation.evaluate", lambda *args: {"success": .70})
    monkeypatch.setattr("drummer.world.unseal_test", lambda *args: (_ for _ in ()).throw(AssertionError("sealed")))
    result = run_pilot({}, data_root=tmp_path / "data", output_root=tmp_path / "runs")
    assert result["status"] == "stopped_quality_gate"
    assert len(result["runs"]) == 2
    assert json.loads((tmp_path / "runs/pilot_report.json").read_text())["test_unsealed"] is False
