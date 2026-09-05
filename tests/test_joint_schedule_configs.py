"""The prospective schedule comparison varies one declared config field."""

import json
import math
from pathlib import Path

import pytest

from drummer.joint_study import StudyConfig


def test_prospective_schedules_are_valid_and_otherwise_identical():
    root = Path(__file__).resolve().parents[1] / "configs"
    control = json.loads((root / "joint-schedule-control-v2.json").read_text())
    slow = json.loads((root / "joint-schedule-slow-v2.json").read_text())
    assert {key for key in control if control[key] != slow[key]} == {"anneal_steps"}
    assert set(control) == set(slow)
    configs = [StudyConfig(**values) for values in (control, slow)]
    for config, anneal in zip(configs, (1500, 4500), strict=True):
        assert config.arms == ["entropy_annealed"]
        assert config.seeds == [101]
        assert config.steps == 6000 and config.anneal_steps == anneal
        assert config.batch_size == 128 and config.evaluate_every == 250
        assert config.coefficient == 0.1 and config.threads == 2
        assert config.max_seconds_per_arm == 1500
        coefficient_mass = sum(
            config.coefficient * max(1 - step / anneal, 0) for step in range(config.steps)
        )
        assert coefficient_mass == pytest.approx({1500: 75.05, 4500: 225.05}[anneal])
    size = 100_000
    batches = math.ceil(size / configs[0].batch_size)
    whole, tail = divmod(configs[0].steps, batches)
    assert whole * size + tail * configs[0].batch_size == 767_328
    assert configs[0].steps <= 10 * batches
    assert configs[0].steps // configs[0].evaluate_every + 1 == 25
