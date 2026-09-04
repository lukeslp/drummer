from dataclasses import asdict, replace
import json
from pathlib import Path
import time
import unicodedata

import pytest

import drummer.phrase_induction as module
from drummer.compact_dictionary import CompactDictionary, encode_compact, negotiate_dictionary
from drummer.phrase_induction import (
    Candidate, Conversation, FrozenInventory, InductionConfig, Turn, conversation_hash,
    evaluate_heldout, generate_conversations, induce_dictionary, measure_session,
    mine_candidates, open_heldout_once, run_induction_study, select_inventory,
)
from drummer.provenance import sha256


def small_config(**kwargs):
    return InductionConfig(**{"train_conversations": 4, "validation_conversations": 2,
                              "heldout_conversations": 2, "shortlist": 4,
                              "max_entries": 4, **kwargs})


def conversation(split="train", identifier="example", text="The reviewer considers the next stage.", protected=()):
    turn = Turn("request", "inspect", "neutrality", text, protected)
    return Conversation(identifier, split, False, "0" * 64, (turn,) * 8)


def write(path, value):
    path.write_text(module._json(value) + "\n", encoding="utf-8")


def stage_corpus(tmp_path, corpus, frozen):
    manifest = {"splits": {}}
    for split, conversations in corpus.items():
        path = tmp_path / ("heldout.sealed.json" if split == "heldout" else f"{split}.json")
        write(path, [asdict(c) for c in conversations])
        manifest["splits"][split] = {"logical_sha256": conversation_hash(conversations),
                                    "file_sha256": sha256(path)}
    write(tmp_path / "corpus-manifest.json", manifest)
    write(tmp_path / "inventory-frozen.json", {"inventory": asdict(frozen), "sha256": frozen.digest})


def empty_freeze(corpus):
    return FrozenInventory((), conversation_hash(corpus["train"]),
                           conversation_hash(corpus["validation"]), None, "0" * 64)


def test_default_design_and_bounded_config():
    config = InductionConfig()
    assert (config.train_conversations, config.validation_conversations, config.heldout_conversations) == (128, 32, 32)
    assert (config.turns, config.minimum_conversations, config.shortlist, config.max_entries) == (8, 4, 64, 16)
    for changed in ({"turns": 7}, {"train_conversations": 129}, {"shortlist": 65},
                    {"max_entries": 17}, {"max_seconds": float("inf")},
                    {"max_seconds": True}, {"seed": -1}, {"seed": True}):
        with pytest.raises(ValueError):
            InductionConfig(**changed)


def test_generator_is_deterministic_disjoint_and_holds_paraphrases_out():
    corpus = generate_conversations(small_config())
    assert corpus == generate_conversations(small_config())
    assert {split: len(cs) for split, cs in corpus.items()} == {"train": 4, "validation": 2, "heldout": 2}
    all_ids, transitions, paths = set(), set(), set()
    for split, conversations in corpus.items():
        split_paths = set()
        for c in conversations:
            assert c.split == split
            assert len(c.turns) == 8
            assert c.conversation_id not in all_ids
            assert c.transition_sha256 not in transitions
            all_ids.add(c.conversation_id)
            transitions.add(c.transition_sha256)
            split_paths.update(literal for turn in c.turns for literal in turn.protected if literal.startswith("src/"))
            if split != "heldout":
                assert not c.paraphrase
                assert all("Please begin" not in turn.text for turn in c.turns)
        assert not split_paths & paths
        paths |= split_paths
    assert sum(c.paraphrase for c in corpus["heldout"]) == 1
    assert {turn.move for c in corpus["train"] for turn in c.turns} == {
        "request", "report", "acknowledge", "clarify", "repair"}
    assert len({turn.stance for c in corpus["train"] for turn in c.turns}) > 1


def test_candidates_use_only_training_conversations_and_exclude_protected_ranges():
    phrase = "The reviewer,  after careful consideration, resumes the exchange"
    target = "src/Cafe\u0301/ExactSymbol.py"
    text = f"{phrase}. The target {target} must not change before version 3."
    training = tuple(conversation(identifier=str(i), text=text,
                                  protected=(target, "must not", "3")) for i in range(4))
    config = small_config(shortlist=64)
    candidates = mine_candidates(training, config)
    assert candidates
    assert any(",  after" in candidate.phrase for candidate in candidates)
    assert all(3 <= len(list(module.WORD.finditer(c.phrase))) <= 10 for c in candidates)
    assert all(c.conversations == 4 for c in candidates)
    assert all(not any(value in c.phrase for value in (target, "must not", "3")) for c in candidates)
    assert candidates == mine_candidates(tuple(reversed(training)), config)
    with pytest.raises(ValueError, match="train"):
        mine_candidates((replace(training[0], split="heldout"),), config)
    assert not mine_candidates(training[:3], config)


def test_candidate_counts_document_frequency_not_turn_frequency():
    repeated = conversation(text="A repeated construction is useful for this exchange.")
    assert len(repeated.turns) == 8
    assert mine_candidates((repeated,), small_config()) == ()


def test_nonempty_startup_loss_is_preserved_but_later_increases_rejected(monkeypatch):
    costs = {(): 100, ("first long phrase",): 180, ("second long phrase",): 200,
             ("third long phrase",): 220,
             ("first long phrase", "second long phrase"): 140,
             ("first long phrase", "third long phrase"): 190,
             ("first long phrase", "second long phrase", "third long phrase"): 150}
    calls = []
    def cost(conversations, entries, deadline):
        assert all(c.split == "train" for c in conversations)
        calls.append(tuple(entries))
        return costs[tuple(entries)]
    monkeypatch.setattr(module, "_cost", cost)
    candidates = tuple(Candidate(phrase, 4, 32, 100) for phrase in
                       ("third long phrase", "second long phrase", "first long phrase"))
    report = induce_dictionary((conversation(),), candidates, small_config())
    assert report["entries"] == ["first long phrase", "second long phrase"]
    assert report["rounds"][0]["startup_seed"]
    assert report["rounds"][0]["delta_against_english"] == 80
    assert report["rounds"][0]["accepted"]
    assert not report["rounds"][-1]["accepted"]
    assert sum(len(row["trials"]) for row in report["rounds"]) == 6
    assert len(calls) == 7


def test_actual_greedy_cost_uses_complete_serialized_training_prompts():
    training = (conversation(text="The reviewer considers the next stage of the shared exchange."),)
    candidates = (Candidate("The reviewer considers the next stage", 1, 8, 100),)
    report = induce_dictionary(training, candidates, small_config(minimum_conversations=1))
    dictionary = CompactDictionary(entries=(candidates[0].phrase,), version=module.INDUCED_VERSION)
    assert report["rounds"][0]["selected_complete_bytes"] == measure_session(training[0].turns, dictionary)["prompt_bytes"]
    assert report["english_complete_bytes"] == measure_session(training[0].turns)["prompt_bytes"]


def test_validation_selects_only_prespecified_prefixes_and_can_choose_empty(monkeypatch):
    entries = [f"candidate phrase number {i}" for i in range(10)]
    seen = []
    def cost(conversations, chosen, deadline):
        assert all(c.split == "validation" for c in conversations)
        seen.append(len(chosen))
        return 100 + len(chosen)
    monkeypatch.setattr(module, "_cost", cost)
    frozen, report = select_inventory((conversation(split="validation"),),
                                     {"entries": entries, "training_sha256": "a" * 64})
    assert seen == [0, 4, 8, 10]
    assert frozen.entries == ()
    assert frozen.dictionary_sha256 is None
    assert report["selected_entries"] == 0
    assert frozen.selection_sha256 == module._fingerprint(report)
    with pytest.raises(ValueError, match="validation"):
        select_inventory((conversation(split="heldout"),), {"entries": entries})


def test_dictionary_roundtrip_and_unicode_protection_and_overhead():
    path = "src/Cafe\u0301/Case.py"
    text = f"The reviewer considers the next stage. Exact {path}; must not change {path}."
    turn = Turn("request", "inspect", "concern", text, (path, "must not"))
    dictionary = CompactDictionary(entries=("The reviewer considers the next stage", path, "must not"))
    measured = measure_session((turn,), dictionary)
    baseline = measure_session((turn,))
    assert measured["roundtrip_exact"] and measured["protected_exact"]
    assert measured["setup_bytes"] > 0
    assert measured["prompt_bytes"] > baseline["prompt_bytes"]
    assert measured["payload_bytes"] < measured["source_bytes"]
    assert measured["provider_tokens"] is None
    assert measured["offline_tokens"] is None
    source, protected = module._source((turn,))
    agreed = negotiate_dictionary(dictionary.capability_card(), dictionary.capability_card())
    encoded = encode_compact(source, dictionary, agreed, protected_literals=protected)
    assert encoded.local_encoding.text.count(path) == 2
    assert "must not" in encoded.local_encoding.text
    assert unicodedata.normalize("NFC", path) not in encoded.local_encoding.text
    with pytest.raises(ValueError, match="size bound"):
        measure_session((replace(turn, text="x" * 4097),))


def test_whole_prompt_tokenization_is_nonadditive_and_not_a_provider_count():
    received = []
    def tokenizer(prompt):
        received.append(prompt)
        return [prompt]
    turn = Turn("report", "inspect", "neutrality", "The report is unverified.", ("unverified",))
    result = measure_session((turn, turn), CompactDictionary(), tokenizers={"test-nonadditive": tokenizer})
    assert len(received) == 1
    assert "External policy=" in received[0]
    assert "DCD1 acknowledgement=" in received[0]
    assert result["offline_tokens"] == {"test-nonadditive": 1}
    assert result["provider_tokens"] is None


def test_heldout_requires_exact_freeze_and_opens_once(tmp_path):
    corpus = generate_conversations(small_config())
    frozen = empty_freeze(corpus)
    stage_corpus(tmp_path, corpus, frozen)
    with pytest.raises(ValueError, match="differs"):
        open_heldout_once(tmp_path, replace(frozen, selection_sha256="f" * 64))
    assert not (tmp_path / "heldout-opened.json").exists()
    assert open_heldout_once(tmp_path, frozen) == corpus["heldout"]
    with pytest.raises(FileExistsError):
        open_heldout_once(tmp_path, frozen)


def test_heldout_rejects_file_hash_change_before_marking_open(tmp_path):
    corpus = generate_conversations(small_config())
    frozen = empty_freeze(corpus)
    stage_corpus(tmp_path, corpus, frozen)
    path = tmp_path / "heldout.sealed.json"
    path.write_text(path.read_text() + " ")
    with pytest.raises(ValueError, match="changed"):
        open_heldout_once(tmp_path, frozen)
    assert not (tmp_path / "heldout-opened.json").exists()


def test_heldout_evaluation_has_paired_prefixes_resends_restarts_and_empty_fallback():
    corpus = generate_conversations(small_config())
    frozen = empty_freeze(corpus)
    observations = []
    report = evaluate_heldout(corpus["heldout"], frozen,
                              tokenizers={"whole": lambda prompt: [prompt]},
                              progress=lambda identifier, rows: observations.append((identifier, rows)))
    assert report["exactness_passed"]
    assert len(report["rows"]) == 2 * 4 * 3
    assert len(observations) == 2
    assert {row["slice"] for row in report["summaries"]} == {
        "all", "familiar-realizations", "heldout-paraphrases"}
    for row in report["rows"]:
        assert row["resend_offline_tokens"] == {"whole": row["prefix_turns"]}
        assert row["restart_offline_tokens"] == {"whole": row["prefix_turns"]}
        if row["arm"] == "induced-dictionary":
            assert row["joined_delta_bytes"] == 0
            assert row["resend_delta_bytes"] == 0
            assert row["restart_delta_bytes"] == 0
        if row["prefix_turns"] == 1:
            assert row["resend_complete_prefixes_bytes"] == row["joined"]["prompt_bytes"]
            assert row["fresh_single_turn_restarts_bytes"] == row["joined"]["prompt_bytes"]
        else:
            assert row["resend_complete_prefixes_bytes"] > row["joined"]["prompt_bytes"]


def test_small_offline_study_freezes_before_heldout_and_never_overwrites(tmp_path, monkeypatch):
    # This is a two-conversation unit fixture, not the 128/32/32 study.
    source = {"revision": "a" * 40, "dirty": False}
    monkeypatch.setattr(module, "_source_provenance", lambda: source)
    seen = []
    original_mine = module.mine_candidates
    def mine(training, config, **kwargs):
        assert not (tmp_path / "run" / "heldout-opened.json").exists()
        assert all(c.split == "train" for c in training)
        seen.append("train")
        return original_mine(training, config, **kwargs)
    original_open = module.open_heldout_once
    def opened(output, frozen):
        assert (output / "inventory-frozen.json").exists()
        assert (output / "validation-selection.json").exists()
        seen.append("heldout")
        return original_open(output, frozen)
    monkeypatch.setattr(module, "mine_candidates", mine)
    monkeypatch.setattr(module, "open_heldout_once", opened)
    config = small_config(train_conversations=2, minimum_conversations=4, max_seconds=30)
    report = run_induction_study(tmp_path / "run", config)
    assert report["status"] == "complete"
    assert report["heldout_opened"]
    assert report["source_unchanged"]
    assert report["selected_entries"] == 0
    assert report["candidate_mining_seconds"] >= 0
    assert seen == ["train", "heldout"]
    assert json.loads((tmp_path / "run" / "study.json").read_text()) == report
    with pytest.raises(ValueError, match="exists"):
        run_induction_study(tmp_path / "run", config)


def test_failure_after_opening_marker_retains_actual_heldout_exposure(tmp_path, monkeypatch):
    monkeypatch.setattr(module, "_source_provenance", lambda: {"dirty": False})

    def fail_after_marker(output, frozen):
        module._write_new(output / "heldout-opened.json", {"inventory_sha256": frozen.digest})
        raise ValueError("synthetic failure after evaluation opening")

    monkeypatch.setattr(module, "open_heldout_once", fail_after_marker)
    config = small_config(train_conversations=2, minimum_conversations=4, max_seconds=30)
    report = run_induction_study(tmp_path / "failed-open", config)
    assert report["status"] == "failed"
    assert report["heldout_opened"] is True
    assert (tmp_path / "failed-open" / "heldout-opened.json").exists()


def test_dirty_source_and_deadline_fail_closed_without_claiming_completion(tmp_path, monkeypatch):
    monkeypatch.setattr(module, "_source_provenance", lambda: {"dirty": True})
    with pytest.raises(ValueError, match="clean source"):
        run_induction_study(tmp_path / "dirty", small_config())
    assert not (tmp_path / "dirty").exists()
    monkeypatch.setattr(module, "_source_provenance", lambda: {"dirty": False})
    report = run_induction_study(tmp_path / "expired", small_config(max_seconds=0.000001))
    assert report["status"] == "deadline_reached"
    assert not report["heldout_opened"]
    assert (tmp_path / "expired" / "study.json").exists()
    with pytest.raises(module.StudyTimeout):
        module._deadline(time.monotonic() - 1)


def test_source_checkout_output_is_refused(monkeypatch):
    monkeypatch.setattr(module, "_source_provenance", lambda: {"dirty": False})
    root = Path(module.__file__).resolve().parents[2]
    with pytest.raises(ValueError, match="outside"):
        run_induction_study(root / "never-created-phrase-study", small_config())
    assert not (root / "never-created-phrase-study").exists()
