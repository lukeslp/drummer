from dataclasses import replace
import itertools
import json

import pytest

from drummer.rewrite_codec import decode_output, encode_target, prepare_input
from drummer.rewrite_corpus import (
    CLAUSES, SOURCE_FAMILIES, RewriteCorpusConfig, build_conversations, bundle_partitions,
    check_observation_sufficiency, check_source_conformance, corpus_manifest,
    normalized_meaning, render_rule, render_source, render_terse, semantic_bundle, teacher_samples,
)
from drummer.rewrite_semantics import ReferenceBinding, RewriteMeaning, parse_message


SMALL = RewriteCorpusConfig(train_size=86, validation_size=10, test_size=12)


@pytest.fixture(scope="module")
def corpus():
    # Structural fixtures, not the eventual sealed research corpus.
    return build_conversations(SMALL)


def meaning(**changes):
    return replace(RewriteMeaning("request", "inspect", "positive", "required", "always", "none",
                                  "concern", "sender", "normal", "src/Café.py", "load", "r1", 1,
                                  "src/keep.py", "keep"), **changes)


def test_all_semantic_bundles_and_source_families_round_trip_independently():
    for bundle, family in itertools.product(itertools.chain.from_iterable(bundle_partitions(9).values()), range(4)):
        move, process, modality, condition, evidence = bundle
        for polarity in ("positive", "negative"):
            if move == "request" and modality == "optional" and polarity == "negative":
                continue
            expected = meaning(move=move, process=process, modality=modality, condition=condition,
                               evidence=evidence, polarity=polarity)
            for order in (CLAUSES, CLAUSES[::-1]):
                assert parse_message(render_source(expected, family, order)).meaning == expected


def test_reproducibility_semantic_partitions_and_manifest_are_not_just_renamed_cases(corpus):
    assert build_conversations(SMALL) == corpus
    bundles = bundle_partitions(SMALL.seed)
    assert {name: len(values) for name, values in bundles.items()} == {"train": 43, "validation": 5, "test": 6}
    manifest = corpus_manifest(corpus, SMALL)
    assert not manifest["training_started"] and not manifest["test_evaluated"]
    actual = {}
    for split, rows in corpus.items():
        actual[split] = {semantic_bundle(turn.expected) for row in rows for turn in row.turns}
        assert actual[split] == set(bundles[split])
        assert {row.source_family for row in rows} == {SOURCE_FAMILIES[split]}
        assert manifest["splits"][split]["event_orders"] > 1
        assert check_source_conformance(rows)["turns"] == len(rows) * 8
    for left, right in itertools.combinations(actual, 2):
        assert actual[left].isdisjoint(actual[right])
    assert bundle_partitions(SMALL.seed + 1) != bundles


def test_contrast_variants_stay_inside_one_semantic_bundle():
    original = meaning()
    for changes in ({"polarity": "negative"}, {"affect": "satisfaction"}, {"affect_holder": "recipient"},
                    {"urgency": "urgent"}, {"path": "other.py"}, {"reference_id": "renamed"}):
        assert semantic_bundle(replace(original, **changes)) == semantic_bundle(original)


def test_event_dependencies_drops_restarts_and_counterbalanced_copy_roles(corpus):
    positions = set()
    beginnings = set()
    for row in corpus["train"]:
        names = [turn.event for turn in row.turns]
        beginnings.add(names[0])
        for prior, after in (("introduce_a", "repeat_a"), ("drop_b", "retry_b"),
                             ("repeat_a", "update_a_lost_ack"), ("update_a_lost_ack", "retry_a"),
                             ("retry_a", "restart"), ("retry_b", "restart"), ("restart", "recover")):
            assert names.index(prior) < names.index(after)
        for turn in row.turns:
            assert turn.reset_before == (turn.event == "restart")
            assert turn.payload_delivered == (turn.event != "drop_b")
            assert turn.ack_delivered == (turn.event not in ("drop_b", "update_a_lost_ack"))
            prepared = prepare_input(turn.source, "")
            def q(value):
                return json.dumps(value, ensure_ascii=False)
            positions.add(prepared.copies.index(q(turn.expected.path)) <
                          prepared.copies.index(q(turn.expected.forbidden_path)))
    assert positions == {False, True} and beginnings == {"introduce_a", "drop_b"}


@pytest.mark.parametrize("representation", ["full", "terse", "rule"])
def test_teacher_samples_have_exact_codec_targets_and_sufficient_observations(corpus, representation):
    samples = tuple(sample for row in corpus["train"][:12] for sample in teacher_samples(row, representation=representation))
    audit = check_observation_sufficiency(samples)
    assert audit["examples"] == 96 and audit["conflicts"] == 0
    for sample in samples:
        assert decode_output(sample.target_tokens, sample.prepared) == sample.target
        assert sample.parsed_source == parse_message(sample.source).meaning


def test_teacher_uses_current_ack_not_event_or_hidden_scoring_labels(corpus):
    for row in corpus["train"][:8]:
        samples = teacher_samples(row)
        for turn, sample in zip(row.turns, samples):
            has_explicit_target = "Use symbol " in sample.target
            assert has_explicit_target == (turn.event not in ("repeat_a", "recover"))
        poisoned = replace(row, conversation_id="ignored", split="test", bundle_id="ignored", source_family=3,
                           turns=tuple(replace(turn, expected=replace(turn.expected, path="hidden-wrong.py"),
                                               event="unrelated-label") for turn in row.turns))
        assert teacher_samples(poisoned) == samples
        assert teacher_samples(replace(row, turns=row.turns[:3])) == samples[:3]
        with pytest.raises(ValueError, match="scoring record"):
            check_source_conformance((poisoned,))


def test_rule_cannot_take_full_target_away_without_exact_actual_binding():
    expected = meaning()
    correct = ReferenceBinding("r1", 1, expected.path, expected.symbol, 1)
    assert render_rule(expected) == render_terse(expected)
    compact = render_rule(expected, (correct,))
    assert len(compact.encode()) < len(render_terse(expected).encode())
    assert parse_message(compact, (correct,)).meaning == expected
    for binding in (replace(correct, path="wrong.py"), replace(correct, acknowledged_version=None),
                    replace(correct, version=2), replace(correct, conflicted=True)):
        assert render_rule(expected, (binding,)) == render_terse(expected)


def test_functional_words_and_ack_status_are_visible_not_opaque(corpus):
    original = meaning()
    assert prepare_input(render_source(original), "").tokens != prepare_input(
        render_source(replace(original, affect="satisfaction")), "").tokens
    assert prepare_input(render_source(original), "").tokens != prepare_input(
        render_source(replace(original, affect_holder="recipient")), "").tokens
    sample = teacher_samples(corpus["train"][0])[0]
    bad = replace(sample, parsed_source=replace(sample.parsed_source, urgency=(
        "urgent" if sample.parsed_source.urgency == "normal" else "normal")))
    with pytest.raises(ValueError, match="conflicting"):
        check_observation_sufficiency((sample, bad))
    with pytest.raises(ValueError, match="conflicting"):
        check_observation_sufficiency((sample, replace(sample, target_tokens=encode_target("Need clarification.", sample.prepared))))


def test_literal_renaming_preserves_normalized_observations_but_unicode_roles_stay_exact():
    original = meaning()
    renamed = replace(original, path="src/Café.py", forbidden_path="src/OTHER.py", reference_id="r9")
    first, second = (prepare_input(render_source(value), "") for value in (original, renamed))
    assert first.tokens == second.tokens and first.copies != second.copies
    assert normalized_meaning(original, first) == normalized_meaning(renamed, second)
    assert parse_message(render_source(original)).meaning.path != parse_message(render_source(renamed)).meaning.path
    # The semantic parser accepts equivalent escapes; COPY does not canonicalize
    # raw source bytes. A teacher requiring a missing canonical lexeme fails.
    escaped = render_source(original).replace('"src/Café.py"', r'"src/Caf\u00e9.py"')
    assert parse_message(escaped).meaning == original
    with pytest.raises(ValueError, match="absent"):
        encode_target(render_terse(original), prepare_input(escaped, ""))


def test_manifest_rejects_wrong_assignments_gold_and_duplicates(corpus):
    row = corpus["train"][0]
    bad_turn = replace(row.turns[0], expected=replace(row.turns[0].expected, process=(
        "edit" if row.turns[0].expected.process != "edit" else "test")))
    for bad in (replace(row, turns=(bad_turn,) + row.turns[1:]), replace(row, bundle_id="wrong"),
                replace(row, source_family=2), replace(row, turns=row.turns[:-1]), corpus["train"][1]):
        poisoned = dict(corpus, train=(bad,) + corpus["train"][1:])
        with pytest.raises(ValueError):
            corpus_manifest(poisoned, SMALL)


@pytest.mark.parametrize("changes", [{"seed": True}, {"seed": -1}, {"train_size": 0},
                                      {"test_size": 16385}, {"validation_size": 2.0}])
def test_corpus_configuration_rejects_invalid_values(changes):
    with pytest.raises(ValueError):
        RewriteCorpusConfig(**changes)


def test_public_boundaries_reject_invalid_metadata_and_rendering(corpus):
    row = corpus["train"][0]
    for action in (lambda: replace(row.turns[0], ack_delivered=1), lambda: replace(row, turns=[]),
                   lambda: replace(row, source_family=True), lambda: bundle_partitions(True),
                   lambda: build_conversations({}), lambda: render_source(meaning(), family=4),
                   lambda: render_source(meaning(), clause_order=CLAUSES[:-1]),
                   lambda: teacher_samples(row, representation="mystery")):
        with pytest.raises(ValueError):
            action()
