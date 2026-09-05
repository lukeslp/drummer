import ast
from dataclasses import asdict, fields, replace
import json

import pytest

from drummer.workflow_fixtures import (
    AcknowledgedReference, CommunicationTraceBoundary, FixtureFile, VisibleEvidence, build_observation,
    canonical_json, expected_results, get_fixture, public_task, score_results,
    trusted_verifier, workflow_fixtures,
)


def test_two_frozen_independent_fixtures_have_public_contracts_and_valid_source_syntax():
    fixtures = workflow_fixtures()
    assert fixtures == workflow_fixtures()
    assert {fixture.task_id for fixture in fixtures} == {"expiry-boundary", "refresh-integrity"}
    assert len({fixture.definition_sha256 for fixture in fixtures}) == 2
    for fixture in fixtures:
        assert len(fixture.editable_paths) == 1
        assert fixture.requirements and fixture.visible_examples
        assert {file.path for file in fixture.files} == {"README.md", fixture.editable_paths[0]}
        for file in fixture.files:
            if file.path.endswith(".py"):
                parsed = ast.parse(file.text)  # Do not import or execute candidate/fixture source.
                assert not any(isinstance(node, (ast.Import, ast.ImportFrom)) for node in ast.walk(parsed))
        assert fixture.definition_sha256 == get_fixture(fixture.task_id).definition_sha256
    assert "TTLCache" not in get_fixture("refresh-integrity").files[1].text
    with pytest.raises(ValueError, match="unknown"):
        get_fixture("unlisted")


def test_public_task_contains_no_hidden_inputs_expected_arrays_or_canonical_fix():
    for fixture in workflow_fixtures():
        description = public_task(fixture)
        assert set(description) == {"task_id", "title", "fixture_version", "definition_sha256",
                                    "requirements", "visible_examples", "editable_paths", "files"}
        rendered = canonical_json(description)
        verifier = trusted_verifier(fixture.task_id)
        assert verifier == trusted_verifier(fixture.task_id)
        assert len(verifier.sha256) == 64
        assert any(case.visibility == "visible" for case in verifier.cases)
        assert any(case.visibility == "heldout" for case in verifier.cases)
        for case in verifier.cases:
            if case.visibility == "heldout":
                assert case.case_id not in rendered
                assert case.operations_json not in rendered
            assert canonical_json(expected_results(fixture.task_id, case)) not in rendered
        assert "canonical_fix" not in rendered
        assert "expected_results" not in rendered


def test_expiration_reference_checks_falsey_values_boundary_and_no_read_extension():
    verifier = trusted_verifier("expiry-boundary")
    assert len(verifier.cases) == 10
    first = expected_results(verifier.task_id, verifier.cases[0])
    assert first[0]["snapshot"] == {"k": {"value": False, "expires_at": 15}}
    assert first[1]["result"] == {"hit": True, "value": False}
    assert first[2]["result"] == first[3]["result"] == {"hit": False, "value": None}
    assert all(row["snapshot"] == first[0]["snapshot"] for row in first)
    for case in verifier.cases[1:8]:
        operations = json.loads(case.operations_json)
        value = operations[0]["value"]
        result = expected_results(verifier.task_id, case)
        assert result[2]["result"]["hit"] is True
        assert canonical_json(result[2]["result"]["value"]) == canonical_json(value)
        assert result[3]["result"]["hit"] is True
        assert result[4]["result"]["hit"] is False
        assert result[-1]["result"] == {"hit": True, "value": "keep"}
        assert all(row["snapshot"]["primary"]["expires_at"] == 6 for row in result)
    zero = expected_results(verifier.task_id, verifier.cases[-2])
    assert zero[1]["result"]["hit"] is False
    overwritten = expected_results(verifier.task_id, verifier.cases[-1])
    assert overwritten[2]["result"] == {"hit": True, "value": None}
    assert overwritten[3]["result"]["hit"] is False


def test_refresh_reference_checks_failure_atomicity_stale_permission_and_recovery():
    verifier = trusted_verifier("refresh-integrity")
    assert len(verifier.cases) == 14
    case = verifier.cases[0]
    initial = json.loads(case.initial_state_json)
    rows = expected_results(verifier.task_id, case)
    assert rows[0]["result"] == {"value": None, "fresh": False, "error": "offline"}
    assert rows[1]["result"] == {"value": "old", "fresh": False, "error": "offline"}
    assert rows[0]["snapshot"] == rows[1]["snapshot"] == initial
    assert rows[2]["result"] == {"value": "new", "fresh": True, "error": None}
    assert rows[2]["snapshot"]["key"] == {"value": "new", "expires_at": 15}
    assert rows[2]["snapshot"]["other"] == initial["other"]
    assert [row["loader_calls"] for row in rows] == [1, 2, 3]
    assert [row["clock_calls"] for row in rows] == [1, 2, 3]
    fresh = expected_results(verifier.task_id, verifier.cases[1])
    assert all(row["loader_calls"] == 0 and row["snapshot"] == initial for row in fresh)
    assert [row["clock_calls"] for row in fresh] == [1, 2]
    missing = expected_results(verifier.task_id, verifier.cases[2])
    assert missing[0]["snapshot"] == {}
    assert missing[0]["result"]["value"] is None
    repeated = expected_results(verifier.task_id, verifier.cases[3])
    assert all(row["snapshot"] == initial for row in repeated)
    for case in verifier.cases[4:11]:
        result = expected_results(verifier.task_id, case)
        assert result[0]["result"]["fresh"] is True
        assert result[1]["loader_calls"] == 1
        assert result[2]["loader_calls"] == 2
        assert result[2]["result"]["fresh"] is False
        assert result[0]["snapshot"] == result[2]["snapshot"]
    zero_case = next(case for case in verifier.cases if case.case_id == "zero-ttl-reload")
    zero = expected_results(verifier.task_id, zero_case)
    assert zero[0]["result"]["fresh"] is True
    assert zero[1]["result"]["fresh"] is False
    assert zero[1]["loader_calls"] == 2


def test_refresh_missing_success_commits_then_reuses_and_replaces_at_boundary():
    verifier = trusted_verifier("refresh-integrity")
    case = next(case for case in verifier.cases if case.case_id == "missing-success")
    initial = {"other": {"value": "keep", "expires_at": 100}}
    assert json.loads(case.initial_state_json) == initial
    rows = expected_results(verifier.task_id, case)
    assert rows[0]["result"] == {"value": False, "fresh": True, "error": None}
    assert rows[0]["snapshot"] == {**initial, "key": {"value": False, "expires_at": 23}}
    assert rows[1]["snapshot"] == rows[0]["snapshot"]
    assert rows[1]["result"] == rows[0]["result"]
    assert rows[2]["result"] == {"value": None, "fresh": True, "error": None}
    assert rows[2]["snapshot"] == {**initial, "key": {"value": None, "expires_at": 26}}
    assert all(row["snapshot"]["other"] == initial["other"] for row in rows)
    assert rows[3]["result"] == {"value": None, "fresh": False, "error": "offline"}
    assert rows[3]["snapshot"] == rows[2]["snapshot"]
    assert [row["loader_calls"] for row in rows] == [1, 1, 2, 3]
    assert [row["clock_calls"] for row in rows] == [1, 2, 3, 4]


def test_refresh_clock_advance_distinguishes_operation_start_from_post_load_sample():
    verifier = trusted_verifier("refresh-integrity")
    case = next(case for case in verifier.cases if case.case_id == "load-start-time")
    operations = json.loads(case.operations_json)
    rows = expected_results(verifier.task_id, case)
    assert operations[0]["loader"]["advance_time"] == 2
    assert rows[0]["snapshot"]["key"] == {"value": "later", "expires_at": 14}
    assert rows[1]["loader_calls"] == 1  # Fresh lookup must not invoke advancing loader.
    assert rows[2]["result"] == {"value": "later", "fresh": False, "error": "offline"}
    assert rows[2]["snapshot"] == rows[0]["snapshot"]
    assert rows[3]["snapshot"]["key"] == {"value": "recovered", "expires_at": 23}
    assert [row["clock_calls"] for row in rows] == [1, 2, 3, 4]
    # Same call count, wrong sampling order: no candidate source is executed.
    after_loading = json.loads(canonical_json(rows))
    after_loading[0]["snapshot"]["key"]["expires_at"] = 16
    assert not score_results(verifier.task_id, case, after_loading)["passed"]
    no_advance = [{**operation, "loader": {key: value for key, value in operation["loader"].items()
                                          if key != "advance_time"}} for operation in operations]
    assert expected_results(verifier.task_id, replace(case, operations_json=canonical_json(no_advance))) == rows


@pytest.mark.parametrize("advance", [-1, True, "2", None, float("inf"), float("nan")])
def test_refresh_rejects_invalid_loader_clock_advance(advance):
    case = trusted_verifier("refresh-integrity").cases[0]
    operation = json.loads(case.operations_json)[0]
    operation["loader"]["advance_time"] = advance
    malformed = replace(case, operations_json=json.dumps([operation]))
    with pytest.raises(ValueError, match="clock advance"):
        expected_results("refresh-integrity", malformed)


def test_refresh_loader_descriptor_remains_exact_with_optional_advance():
    case = trusted_verifier("refresh-integrity").cases[0]
    operation = json.loads(case.operations_json)[0]
    operation["loader"]["advance_time"] = 0
    assert expected_results("refresh-integrity", replace(case, operations_json=canonical_json([operation])))
    operation["loader"]["elapsed"] = 1
    with pytest.raises(ValueError, match="descriptor fields"):
        expected_results("refresh-integrity", replace(case, operations_json=canonical_json([operation])))


def test_grader_compares_actual_data_not_claimed_pass_and_does_not_confuse_false_with_zero():
    case = trusted_verifier("expiry-boundary").cases[0]
    expected = list(expected_results("expiry-boundary", case))
    assert score_results("expiry-boundary", case, expected)["passed"]
    altered = json.loads(canonical_json(expected))
    altered[1]["result"]["value"] = 0
    assert not score_results("expiry-boundary", case, altered)["passed"]
    assert not score_results("expiry-boundary", case, {"passed": True})["passed"]
    assert not score_results("expiry-boundary", case, [])["passed"]
    altered[1]["result"]["value"] = float("nan")
    assert score_results("expiry-boundary", case, altered)["error"] == "output_json"


def test_grader_accepts_equal_numeric_times_but_requires_booleans_and_exact_fields():
    case = trusted_verifier("expiry-boundary").cases[0]
    actual = json.loads(canonical_json(expected_results("expiry-boundary", case)))
    for event in actual:
        event["snapshot"]["k"]["expires_at"] = 15.0
    assert score_results("expiry-boundary", case, actual)["passed"]
    actual[1]["result"]["hit"] = 1
    assert not score_results("expiry-boundary", case, actual)["passed"]
    actual[1]["result"]["hit"] = True
    actual[1]["extra"] = "untrusted claim"
    assert not score_results("expiry-boundary", case, actual)["passed"]
    refresh = trusted_verifier("refresh-integrity").cases[1]
    actual = json.loads(canonical_json(expected_results("refresh-integrity", refresh)))
    actual[0]["clock_calls"] = 2
    assert not score_results("refresh-integrity", refresh, actual)["passed"]


def test_trusted_oracle_rejects_operations_outside_contract():
    case = trusted_verifier("expiry-boundary").cases[0]
    for change in ({"op": "shell", "key": "k", "at": 0},
                   {"op": "set", "key": "k", "at": 0, "ttl": -1, "value": 0},
                   {"op": "get", "key": "k", "at": True},
                   {"op": "get", "key": "k", "at": 0, "extra": True}):
        with pytest.raises(ValueError):
            expected_results("expiry-boundary", replace(case, operations_json=canonical_json([change])))


def test_observation_projects_current_visible_source_and_excludes_verifier_objects():
    fixture = get_fixture("expiry-boundary")
    current = FixtureFile("src/cache.py", "# revised current source\n")
    evidence = VisibleEvidence("baseline.visible", "visible fixture check", "b" * 64, "Observed one failed visible check.")
    observation = build_observation(fixture, actor_id="reviewer", stage="review", base_tree_sha256="a" * 64,
                                    visible_files=(current,), visible_evidence=(evidence,), prior_deliveries=("Actual prior message",))
    rendered = canonical_json(asdict(observation))
    assert "# revised current source" in rendered
    assert fixture.files[1].text not in rendered
    assert "files" not in observation.public_contract
    assert observation.file_sha256 == {current.path: current.sha256}
    assert json.loads(rendered)["file_sha256"] == {current.path: current.sha256}
    assert observation.observation_version == "coding-workflow-observation-2"
    assert fixture.files[1].sha256 not in observation.file_sha256.values()
    assert len(observation.sha256) == 64
    with pytest.raises(ValueError, match="model-visible"):
        build_observation(fixture, actor_id="sender", stage="inspect", base_tree_sha256="a" * 64,
                          visible_files=(), visible_evidence=(trusted_verifier(fixture.task_id),))
    with pytest.raises(ValueError, match="outside"):
        build_observation(fixture, actor_id="sender", stage="inspect", base_tree_sha256="a" * 64,
                          visible_files=(FixtureFile("hidden-tests.py", "grader"),))
    trace_fields = {field.name for field in fields(CommunicationTraceBoundary)}
    assert "sender_observation_sha256" in trace_fields
    assert not trace_fields & {"hidden_target", "receiver_private_state", "loss_vector", "expected_results"}


def test_every_stage_serializes_copyable_current_file_hashes_without_changing_fixtures():
    for fixture in workflow_fixtures():
        expected_definition = fixture.definition_sha256
        for stage in ("inspect", "propose", "implement", "review", "clarify", "repair"):
            current = tuple(FixtureFile(file.path, file.text + "\n# current revision\n")
                            for file in fixture.files)
            observation = build_observation(fixture, actor_id="sender", stage=stage,
                                              base_tree_sha256="a" * 64, visible_files=current)
            wire = json.loads(canonical_json(asdict(observation)))
            assert wire["file_sha256"] == {file.path: file.sha256 for file in current}
            assert set(wire["file_sha256"]) == {file["path"] for file in wire["visible_files"]}
            assert fixture.definition_sha256 == expected_definition


def test_observation_rejects_verifier_objects_directly_and_nested_before_projection():
    fixture = get_fixture("expiry-boundary")
    hidden = trusted_verifier(fixture.task_id)
    valid = {"actor_id": "sender", "stage": "inspect", "base_tree_sha256": "a" * 64,
             "visible_files": fixture.files}
    bad_fields = (
        {"prior_deliveries": (hidden,)},
        {"prior_deliveries": ({"nested": hidden},)},
        {"visible_evidence": (VisibleEvidence("e", "test", "b" * 64, hidden),)},
        {"visible_evidence": (VisibleEvidence("e", "test", "b" * 64, {"nested": hidden}),)},
        {"visible_evidence": (VisibleEvidence(hidden, "test", "b" * 64, "observed"),)},
        {"visible_evidence": (VisibleEvidence("e", hidden, "b" * 64, "observed"),)},
        {"visible_evidence": (VisibleEvidence("e", "test", hidden, "observed"),)},
        {"actor_id": hidden},
        {"stage": {"nested": hidden}},
        {"visible_files": (hidden,)},
        {"visible_files": (FixtureFile("src/cache.py", hidden),)},
        {"visible_files": (FixtureFile(hidden, "text"),)},
        {"acknowledgements": (AcknowledgedReference(hidden, 1, "receiver", "b" * 64),)},
        {"acknowledgements": (AcknowledgedReference("r", 1, hidden, "b" * 64),)},
    )
    for overrides in bad_fields:
        with pytest.raises(ValueError):
            build_observation(fixture, **(valid | overrides))
    for corrupted_fixture in (
        replace(fixture, requirements=(hidden,)),
        replace(fixture, visible_examples=({"nested": hidden},)),
        replace(fixture, files=(FixtureFile("src/cache.py", hidden),)),
    ):
        with pytest.raises(ValueError):
            public_task(corrupted_fixture)
        with pytest.raises(ValueError):
            build_observation(corrupted_fixture, **valid)


def test_observation_validates_primitive_ids_digests_versions_and_container_shapes():
    fixture = get_fixture("expiry-boundary")
    valid = {"actor_id": "Sender-Δ", "stage": "inspect", "base_tree_sha256": "a" * 64,
             "visible_files": fixture.files, "prior_deliveries": ("Café → src/cache.py",),
             "acknowledgements": (AcknowledgedReference("Ref-A", 1, "Receiver-Δ", "b" * 64),),
             "visible_evidence": (VisibleEvidence("Evidence-A", "visible check", "c" * 64, "Observed"),)}
    observation = build_observation(fixture, **valid)
    assert observation.actor_id == "Sender-Δ"
    assert observation.prior_deliveries == ("Café → src/cache.py",)
    assert observation.acknowledgements[0].reference_id == "Ref-A"
    for digest in (True, "a" * 63, "G" * 64, "B" * 64, ""):
        for overrides in (
            {"base_tree_sha256": digest},
            {"visible_evidence": (VisibleEvidence("e", "test", digest, "observed"),)},
            {"acknowledgements": (AcknowledgedReference("r", 1, "receiver", digest),)},
        ):
            with pytest.raises(ValueError):
                build_observation(fixture, **(valid | overrides))
    for version in (True, False, 0, -1, 1.0, "1"):
        with pytest.raises(ValueError, match="positive integer"):
            build_observation(fixture, **(valid | {
                "acknowledgements": (AcknowledgedReference("r", version, "receiver", "b" * 64),)}))
    for overrides in ({"actor_id": True}, {"actor_id": ""}, {"prior_deliveries": "not a list"},
                   {"visible_files": {}}, {"visible_evidence": None}, {"acknowledgements": "r"},
                   {"prior_deliveries": ("\ud800",)},
                   {"acknowledgements": (AcknowledgedReference("", 1, "receiver", "b" * 64),)}):
        with pytest.raises(ValueError):
            build_observation(fixture, **(valid | overrides))
