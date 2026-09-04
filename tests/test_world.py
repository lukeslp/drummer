from __future__ import annotations

import json

import numpy as np
import pytest

from drummer.world import (
    ATTRIBUTE_CARDINALITIES,
    NUM_IDENTITIES,
    UNSEAL_CONFIRMATION,
    CorpusConfig,
    SealedTestError,
    WorldCondition,
    attributes_to_identity,
    corpus_manifest_evidence,
    generate_corpus,
    identity_to_attributes,
    load_split,
    unseal_test,
    verify_split_disjointness,
)


def test_identity_codec_is_a_complete_mixed_radix_bijection() -> None:
    observed = {identity_to_attributes(identity) for identity in range(NUM_IDENTITIES)}
    assert len(observed) == 64
    assert all(
        0 <= attributes[index] < cardinality
        for attributes in observed
        for index, cardinality in enumerate(ATTRIBUTE_CARDINALITIES)
    )
    assert [attributes_to_identity(identity_to_attributes(i)) for i in range(64)] == list(
        range(64)
    )


@pytest.fixture
def corpus_root(tmp_path):
    root = tmp_path / "corpus"
    generate_corpus(
        root,
        CorpusConfig(seed=9, train_size=320, validation_size=80, test_size=80),
    )
    return root


def test_corpus_has_exact_mixture_balanced_targets_and_disjoint_groups(corpus_root) -> None:
    manifest = json.loads((corpus_root / "corpus_manifest.json").read_text())
    expected = {
        "train": {"valid_repeat": 192, "dropped_grounding": 64, "new_reference": 64},
        "validation": {"valid_repeat": 48, "dropped_grounding": 16, "new_reference": 16},
        "test": {"valid_repeat": 48, "dropped_grounding": 16, "new_reference": 16},
    }
    for split, counts in expected.items():
        assert manifest["splits"][split]["condition_counts"] == counts
        assert (
            manifest["splits"][split]["target_count_max"]
            - manifest["splits"][split]["target_count_min"]
            <= 1
        )
    assert verify_split_disjointness(corpus_root)


def test_world_conditions_preserve_only_legitimate_history(corpus_root) -> None:
    split = load_split(corpus_root, "train")
    arrays = split.arrays
    assert np.all(np.apply_along_axis(lambda row: len(set(row)) == 4, 1, arrays["candidate_ids"]))
    assert np.all(
        arrays["candidate_ids"][np.arange(len(split)), arrays["target_index"]]
        == arrays["target_id"]
    )

    valid = arrays["condition"] == WorldCondition.VALID_REPEAT
    assert np.all(arrays["sender_previous_id"][valid] == arrays["target_id"][valid])
    assert np.all(arrays["receiver_previous_id"][valid] == arrays["target_id"][valid])
    assert np.all(arrays["history_present"][valid] & arrays["acknowledged"][valid])

    dropped = arrays["condition"] == WorldCondition.DROPPED_GROUNDING
    assert np.all(arrays["sender_previous_id"][dropped] == arrays["target_id"][dropped])
    assert np.all(arrays["receiver_previous_id"][dropped] == -1)
    assert not np.any(arrays["history_present"][dropped])
    assert not np.any(arrays["acknowledged"][dropped])

    new = arrays["condition"] == WorldCondition.NEW_REFERENCE
    assert np.all(arrays["sender_previous_id"][new] != arrays["target_id"][new])
    assert np.all(arrays["history_present"][new] & arrays["acknowledged"][new])
    for old, candidates in zip(
        arrays["receiver_previous_id"][new], arrays["candidate_ids"][new], strict=True
    ):
        assert old in candidates


def test_batch_separates_sender_intent_from_receiver_memory(corpus_root) -> None:
    split = load_split(corpus_root, "train")
    row = int(np.flatnonzero(split.arrays["condition"] == WorldCondition.DROPPED_GROUNDING)[0])
    batch = split.batch([row])
    assert batch["sender_history_present"].item() is True
    assert batch["receiver_history_present"].item() is False
    assert batch["sender_history_attrs"].tolist() == batch["target_attrs"].tolist()
    assert batch["receiver_history_attrs"].tolist() == [[0, 0, 0, 0, 0]]
    assert batch["grounding_bits"].item() == 6


def test_test_split_requires_explicit_persistent_unseal(corpus_root) -> None:
    with pytest.raises(SealedTestError):
        load_split(corpus_root, "test")
    with pytest.raises(SealedTestError):
        unseal_test(corpus_root, "yes")
    record = unseal_test(corpus_root, UNSEAL_CONFIRMATION)
    assert record.exists()
    assert len(load_split(corpus_root, "test")) == 80


def test_generation_is_idempotent_but_never_overwrites(corpus_root) -> None:
    config = CorpusConfig(seed=9, train_size=320, validation_size=80, test_size=80)
    unseal_test(corpus_root, UNSEAL_CONFIRMATION)
    before = (corpus_root / "TEST_UNSEALED.json").read_bytes()
    returned = generate_corpus(corpus_root, config)
    assert returned["config"]["seed"] == 9
    assert (corpus_root / "TEST_UNSEALED.json").read_bytes() == before
    with pytest.raises(FileExistsError):
        generate_corpus(
            corpus_root,
            CorpusConfig(seed=10, train_size=320, validation_size=80, test_size=80),
        )


def test_corpus_manifest_evidence_binds_sealed_bytes_without_unsealing(corpus_root) -> None:
    evidence = corpus_manifest_evidence(corpus_root)

    assert len(evidence["manifest_sha256"]) == 64
    assert set(evidence["splits"]) == {"train", "validation", "test"}
    assert all(len(item["logical_sha256"]) == 64 for item in evidence["splits"].values())
    assert all(len(item["file_sha256"]) == 64 for item in evidence["splits"].values())
    assert not (corpus_root / "TEST_UNSEALED.json").exists()
