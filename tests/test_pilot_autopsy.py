import ast
import hashlib
import json
import math
from pathlib import Path

import pytest
from markdown_it import MarkdownIt

from drummer.pilot_autopsy import build_autopsy, render_autopsy_markdown, symbol_information


def matrix():
    return [[0] * 64 for _ in range(64)]


def artifact(tmp_path, value, name="pilot_report.json"):
    path = tmp_path / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")
    return path


def pilot():
    counts = matrix()
    counts[0][0] = 1
    counts[1][1] = 1
    return {
        "status": "stopped_quality_gate", "stage": "calibration_compulsory",
        "test_unsealed": False, "selected_pressure": None, "runs": [],
        "compulsory_validation": {
            "split": "validation", "split_logical_sha256": "a" * 64,
            "success": 0.5, "examples": 4,
            "by_condition": {
                "valid_repeat": {"count": 2, "success": 1.0},
                "dropped_grounding": {"count": 1, "success": 0.0},
                "new_reference": {"count": 1, "success": 0.0},
            },
            "counterfactual": {"sender_entropy": 4e-7},
            "packet_content": {
                "nonrepeat_examples": 2, "nonrepeat_sent": 2,
                "alignment": {"source_split": "validation", "counts": counts,
                              "aligned_examples": 2, "active_symbols": 2,
                              "identities_with_evidence": 2},
            },
        },
    }


def test_information_distinguishes_independence_code_and_constant():
    independent = matrix()
    for s in (0, 1):
        for identity in (0, 1):
            independent[s][identity] = 1
    stats = symbol_information(independent)
    assert stats["symbol_entropy_bits"] == 1
    assert stats["identity_entropy_bits"] == 1
    assert stats["mutual_information_bits"] == 0
    perfect = matrix()
    for i in range(64):
        perfect[i][i] = 3
    assert symbol_information(perfect)["mutual_information_bits"] == 6
    constant = matrix()
    constant[9] = [1] * 64
    stats = symbol_information(constant)
    assert stats["active_symbols"] == 1
    assert stats["symbol_entropy_bits"] == 0
    assert stats["identity_entropy_bits"] == 6
    assert stats["mutual_information_bits"] == 0


def test_missing_and_empty_counts_do_not_invent_entropy():
    for counts in (None, []):
        assert all(value is None for value in symbol_information(counts).values())
    empty = symbol_information(matrix())
    assert empty["sample_count"] == empty["active_symbols"] == 0
    assert empty["symbol_entropy_bits"] is None
    assert empty["identity_entropy_bits"] is None
    assert empty["mutual_information_bits"] is None


@pytest.mark.parametrize("bad", [-1, 1.2, True, None, "2", 2**53])
def test_invalid_count_cell_rejected(bad):
    counts = matrix()
    counts[0][0] = bad
    with pytest.raises(ValueError):
        symbol_information(counts)


@pytest.mark.parametrize("counts", [[[1]], [[0] * 64] * 63, [[0] * 63] * 64, "bad"])
def test_invalid_matrix_shape_rejected(counts):
    with pytest.raises(ValueError, match="64x64"):
        symbol_information(counts)


def test_read_only_reproducible_autopsy_and_markdown(tmp_path):
    source = artifact(tmp_path, pilot())
    before = source.read_bytes()
    first = build_autopsy(source)
    assert first == build_autopsy(source)
    assert source.read_bytes() == before
    assert list(tmp_path.iterdir()) == [source]
    assert first["artifacts"][0]["sha256"] == hashlib.sha256(before).hexdigest()
    assert first["gate"]["passed"] is False
    assert first["promotion"] == "not_eligible"
    assert first["symbol_information"]["symbol_entropy_bits"] == 1.0
    assert first["symbol_information"]["mutual_information_bits"] == 1.0
    assert first["sender_policy"]["conditional_entropy_nats"] == 4e-7
    markdown = render_autopsy_markdown(first)
    assert "Dropped grounding repeats" in markdown
    assert "bias it upward" in markdown
    assert "near-zero conditional entropy alone" in markdown
    rendered = MarkdownIt("commonmark", {"html": False}).enable("table").render(markdown)
    assert "<table>" in rendered
    assert "Luke Steuber" in rendered
    json.dumps(first, allow_nan=False)


def test_missing_optional_metrics_are_unavailable(tmp_path):
    source = artifact(tmp_path, {"status": "stopped_deadline"})
    report = build_autopsy(source)
    assert report["gate"]["observed"] is report["gate"]["passed"] is None
    assert report["sender_policy"]["conditional_entropy_nats"] is None
    assert report["symbol_information"]["mutual_information_bits"] is None
    assert report["validation"]["by_condition"]["valid_repeat"]["count"] is None
    assert "Unavailable" in render_autopsy_markdown(report)


@pytest.mark.parametrize("change", [
    lambda p: p.update(status="passed"),
    lambda p: p.update(test_unsealed=1),
    lambda p: p["compulsory_validation"].update(split="test"),
    lambda p: p["compulsory_validation"].update(success=1.1),
    lambda p: p["compulsory_validation"].update(examples=-1),
    lambda p: p["compulsory_validation"]["counterfactual"].update(sender_entropy=math.inf),
    lambda p: p["compulsory_validation"]["counterfactual"].update(sender_entropy=-.1),
    lambda p: p["compulsory_validation"]["packet_content"].update(nonrepeat_sent=3),
    lambda p: p["compulsory_validation"]["packet_content"]["alignment"].update(active_symbols=3),
    lambda p: p["compulsory_validation"]["packet_content"]["alignment"].update(source_split="test"),
    lambda p: p["compulsory_validation"]["packet_content"]["alignment"].update(
        source_logical_sha256="b" * 64),
    lambda p: p["compulsory_validation"]["by_condition"]["new_reference"].update(count=2),
    lambda p: p["compulsory_validation"]["by_condition"]["new_reference"].update(success=.5),
    lambda p: p.update(corpus={"num_identities": 32}),
    lambda p: p.update(corpus={"attribute_cardinalities": [2, 2, 2, 2, 8]}),
])
def test_malformed_and_inconsistent_evidence_rejected(tmp_path, change):
    report = pilot()
    change(report)
    with pytest.raises(ValueError):
        build_autopsy(artifact(tmp_path, report))


def test_matrix_alignment_and_correct_count_are_checked(tmp_path):
    report = pilot()
    alignment = report["compulsory_validation"]["packet_content"]["alignment"]
    alignment.update(symbol_to_identity=list(range(64)), aligned_correct=2, aligned_accuracy=1)
    build_autopsy(artifact(tmp_path, report))
    alignment["aligned_correct"] = 1
    with pytest.raises(ValueError, match="aligned_correct"):
        build_autopsy(artifact(tmp_path, report))


@pytest.mark.parametrize("change", [
    lambda p: p["compulsory_validation"].update(success=10**400),
    lambda p: p["compulsory_validation"]["packet_content"].update(causal_swap_examples=3),
    lambda p: p["compulsory_validation"]["packet_content"]["alignment"].update(
        aligned_correct=1, aligned_accuracy=1),
    lambda p: p["compulsory_validation"]["packet_content"]["alignment"].update(
        symbol_to_identity=list(range(64)), identity_to_symbol=[0] * 64),
    lambda p: p["compulsory_validation"]["packet_content"].update(
        aligned_exact_match=.5, alignment={"aligned_accuracy": 1}),
    lambda p: p.update(corpus={"splits": {"validation": {"size": 4,
        "condition_counts": {"valid_repeat": 2, "dropped_grounding": 2, "new_reference": 2}}}}),
])
def test_additional_inconsistent_counts_rejected(tmp_path, change):
    report = pilot()
    change(report)
    with pytest.raises(ValueError):
        build_autopsy(artifact(tmp_path, report))


def test_no_metric_average_when_no_examples(tmp_path):
    report = pilot()
    report["compulsory_validation"]["by_condition"]["new_reference"]["count"] = 0
    with pytest.raises(ValueError, match="zero-example"):
        build_autopsy(artifact(tmp_path, report))


def test_reject_duplicate_json_keys(tmp_path):
    path = tmp_path / "pilot_report.json"
    path.write_text('{"status":"failed","status":"running"}')
    with pytest.raises(ValueError, match="duplicate JSON key"):
        build_autopsy(path)


def test_discovery_curves_source_bindings_and_corpus_consistency(tmp_path):
    source = artifact(tmp_path, pilot())
    curves = [{"epoch": 1, "global_step": 4, "validation": {
        "success": .5, "objective": .9, "entropy": .01}}]
    training = {
        "status": "complete", "training": {"seed": 101, "mode": "compulsory"},
        "best_checkpoint": "/unavailable/step.safetensors", "learning_curves": curves,
        "runtime": {"source": {"revision": "c" * 40}, "uv_lock_sha256": "d" * 64},
        "corpus_logical_sha256": {"validation": "a" * 64},
    }
    run = artifact(tmp_path, training, "training/calibration/training_report.json")
    artifact(run.parent, curves, "learning_curves.json")
    artifact(run.parent, {"status": "complete", "corpus_logical_sha256": {
        "validation": "a" * 64}}, "run_manifest.json")
    report = build_autopsy(source, run_root=tmp_path, training_reports=[run])
    assert len(report["training_runs"]) == 1
    assert len(report["artifacts"]) == 4
    assert report["training_runs"][0]["source"]["revision"] == "c" * 40
    assert report["training_runs"][0]["curves"][0]["validation"]["success"] == .5
    assert report["training_runs"][0]["best_checkpoint"] == "/unavailable/step.safetensors"
    artifact(run.parent, {"status": "complete", "corpus_logical_sha256": {
        "validation": "b" * 64}}, "run_manifest.json")
    with pytest.raises(ValueError, match="corpus"):
        build_autopsy(source, run_root=tmp_path)


def test_conflicting_adjacent_curves_rejected(tmp_path):
    source = artifact(tmp_path, {"status": "failed"})
    run = artifact(tmp_path, {"status": "failed", "learning_curves": []},
                   "training/x/training_report.json")
    artifact(run.parent, [{"epoch": 1}], "learning_curves.json")
    with pytest.raises(ValueError, match="learning curves"):
        build_autopsy(source, training_reports=[run])


def test_discovery_does_not_follow_symlink_outside_root(tmp_path):
    root = tmp_path / "root"
    source = artifact(root, {"status": "failed"})
    outside = artifact(tmp_path, {"status": "complete"}, "outside/training_report.json")
    (root / "training").mkdir()
    (root / "training/escape").symlink_to(outside.parent, target_is_directory=True)
    with pytest.raises(ValueError, match="escapes"):
        build_autopsy(source, run_root=root)


def test_adjacent_artifacts_cannot_escape_training_directory(tmp_path):
    source = artifact(tmp_path, {"status": "failed"})
    run = artifact(tmp_path, {"status": "complete"}, "training/run/training_report.json")
    unrelated = artifact(tmp_path, [], "unrelated.json")
    (run.parent / "learning_curves.json").symlink_to(unrelated)
    with pytest.raises(ValueError, match="escape"):
        build_autopsy(source, training_reports=[run])


def test_markdown_escapes_untrusted_artifact_text(tmp_path):
    report = pilot()
    report["reason"] = '<script>alert(1)</script> [link](https://example.com) | x'
    summary = build_autopsy(artifact(tmp_path, report))
    rendered = MarkdownIt("commonmark", {"html": False}).enable("table").render(
        render_autopsy_markdown(summary))
    assert "<script>" not in rendered
    assert '<a href="https://example.com"' not in rendered


def test_module_has_only_standard_library_imports():
    import drummer.pilot_autopsy as module

    tree = ast.parse(Path(module.__file__).read_text())
    roots = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            roots.add(node.module.split(".")[0])
    assert roots <= {"__future__", "hashlib", "html", "json", "math", "collections", "pathlib", "typing"}
