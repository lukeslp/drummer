from dataclasses import replace
import json
import os
from pathlib import Path
import stat

import pytest

from drummer.workflow_fixtures import FixtureFile, get_fixture
import drummer.workflow_patches as module
from drummer.workflow_patches import (
    PATCH_VERSION, PatchRejected, apply_patch_proposal, materialize_fixture,
    read_snapshot, validate_patch,
)


@pytest.fixture
def source(tmp_path):
    fixture = get_fixture("expiry-boundary")
    snapshot = materialize_fixture(tmp_path.resolve() / "base", fixture)
    return fixture, snapshot


def proposal(fixture, snapshot, *, old=None, new=None):
    path = fixture.editable_paths[0]
    file = next(file for file in snapshot.files if file.path == path)
    return {"version": PATCH_VERSION, "task_id": fixture.task_id,
            "base_tree_sha256": snapshot.tree_sha256,
            "files": [{"path": path, "base_sha256": file.sha256,
                       "edits": [{"old": old or file.text.splitlines()[0],
                                  "new": new or '"""Reviewed synthetic cache source."""'}]}]}


def test_copy_on_write_applies_exact_patch_and_preserves_original(source):
    fixture, base = source
    request = proposal(fixture, base)
    validated = validate_patch(base.root, fixture, request)
    assert read_snapshot(base.root, fixture) == base
    destination = base.root.parent / "revision-1"
    result = apply_patch_proposal(base.root, fixture, request, destination=destination)
    assert result.root == destination
    assert result.base_root == base.root
    assert result.changed_paths == ("src/cache.py",)
    assert result.tree_sha256 == validated.result_tree_sha256
    assert result.tree_sha256 != base.tree_sha256
    assert read_snapshot(base.root, fixture) == base
    assert read_snapshot(destination, fixture).tree_sha256 == result.tree_sha256
    assert "coordinator" in result.activation
    assert all(stat.S_IMODE((destination / file.path).stat().st_mode) == 0o644 for file in base.files)


def test_all_file_validation_precedes_any_output_mutation(source):
    fixture, base = source
    request = proposal(fixture, base)
    request["files"].append({"path": "README.md", "base_sha256": base.files[0].sha256,
                             "edits": [{"old": "Synthetic", "new": "Changed"}]})
    destination = base.root.parent / "invalid-revision"
    with pytest.raises(PatchRejected, match="unlisted"):
        apply_patch_proposal(base.root, fixture, request, destination=destination)
    assert not destination.exists()
    assert read_snapshot(base.root, fixture) == base


@pytest.mark.parametrize("target", ["../escape.py", "/tmp/escape.py", "src/../README.md",
                                    "src//cache.py", "src/./cache.py", "src\\cache.py",
                                    "src/%2e%2e/cache.py", "src/Cache.py", "src/cache.py ",
                                    ".git/config", "tests/test_cache.py", "pyproject.toml",
                                    "src/\ud800.py"])
def test_model_path_variants_never_escape_exact_allowlist(source, target):
    fixture, base = source
    request = proposal(fixture, base)
    request["files"][0]["path"] = target
    with pytest.raises(PatchRejected):
        validate_patch(base.root, fixture, request)
    assert read_snapshot(base.root, fixture) == base


def test_stale_tree_stale_file_unknown_fields_and_duplicate_json_rejected(source):
    fixture, base = source
    for change in ("tree", "file", "command", "mode", "duplicate-target", "task", "version"):
        request = proposal(fixture, base)
        if change == "tree":
            request["base_tree_sha256"] = "0" * 64
        elif change == "file":
            request["files"][0]["base_sha256"] = "0" * 64
        elif change == "command":
            request["execute"] = "shell"
        elif change == "mode":
            request["files"][0]["mode"] = "executable"
        elif change == "duplicate-target":
            request["files"].append(request["files"][0])
        elif change == "task":
            request["task_id"] = "refresh-integrity"
        else:
            request["version"] = "future"
        with pytest.raises(PatchRejected):
            validate_patch(base.root, fixture, request)
    raw = json.dumps(proposal(fixture, base)).replace('"task_id":', '"task_id":"forged","task_id":', 1)
    with pytest.raises(PatchRejected):
        validate_patch(base.root, fixture, raw)


@pytest.mark.parametrize("edits", [
    [{"old": "return", "new": "yield"}],
    [{"old": "absent old text", "new": "replacement"}],
    [{"old": "", "new": "insertion"}],
    [{"old": "class TTLCache:", "new": "class TTLCache:"}],
    [{"old": "class TTLCache:", "new": "class Other:"}, {"old": "TTLCache", "new": "Changed"}],
    [{"old": "class TTLCache:", "new": "x\x00y"}],
])
def test_ambiguous_overlapping_empty_noop_and_binary_replacements_rejected(source, edits):
    fixture, base = source
    request = proposal(fixture, base)
    request["files"][0]["edits"] = edits
    with pytest.raises(PatchRejected):
        validate_patch(base.root, fixture, request)
    assert read_snapshot(base.root, fixture) == base


def test_replacements_match_original_once_including_self_overlap_and_do_not_chain():
    with pytest.raises(PatchRejected, match="exactly once"):
        module._replace_exact("ababa", [{"old": "aba", "new": "x"}])
    with pytest.raises(PatchRejected, match="exactly once"):
        module._replace_exact("alpha", [{"old": "alpha", "new": "beta"}, {"old": "beta", "new": "gamma"}])
    assert module._replace_exact("alpha omega", [{"old": "omega", "new": "end"},
                                                  {"old": "alpha", "new": "start"}]) == "start end"
    with pytest.raises(PatchRejected, match="deletion"):
        module._replace_exact("all", [{"old": "all", "new": ""}])
    with pytest.raises(PatchRejected, match="no effect"):
        module._replace_exact("ab", [{"old": "a", "new": ""}, {"old": "b", "new": "ab"}])


def test_symlink_root_parent_directory_and_file_rejected(source):
    fixture, base = source
    linked = base.root.parent / "linked-root"
    linked.symlink_to(base.root, target_is_directory=True)
    with pytest.raises(PatchRejected, match="symlink"):
        read_snapshot(linked, fixture)
    source_file = base.root / "src/cache.py"
    outside = base.root.parent / "outside.py"
    outside.write_text(source_file.read_text())
    source_file.unlink()
    source_file.symlink_to(outside)
    with pytest.raises(PatchRejected, match="symlink"):
        read_snapshot(base.root, fixture)
    source_file.unlink()
    source_file.write_text(outside.read_text())
    source_file.chmod(0o644)
    original_directory = base.root / "src"
    relocated = base.root.parent / "relocated-source"
    original_directory.rename(relocated)
    original_directory.symlink_to(relocated, target_is_directory=True)
    with pytest.raises(PatchRejected, match="symlink"):
        read_snapshot(base.root, fixture)


def test_hardlinks_special_files_and_file_mode_changes_rejected(source):
    fixture, base = source
    file = base.root / "src/cache.py"
    link = base.root.parent / "hardlinked-source.py"
    os.link(file, link)
    with pytest.raises(PatchRejected, match="hardlinked"):
        read_snapshot(base.root, fixture)
    link.unlink()
    file.chmod(0o755)
    with pytest.raises(PatchRejected, match="mode"):
        read_snapshot(base.root, fixture)
    file.chmod(0o644)
    file.unlink()
    os.mkfifo(file)
    with pytest.raises(PatchRejected, match="special"):
        read_snapshot(base.root, fixture)


def test_unlisted_missing_and_protected_file_changes_rejected(source):
    fixture, base = source
    extra = base.root / "extra.py"
    extra.write_text("extra")
    with pytest.raises(PatchRejected, match="unlisted"):
        read_snapshot(base.root, fixture)
    extra.unlink()
    readme = base.root / "README.md"
    old = readme.read_text()
    readme.write_text(old + "changed")
    with pytest.raises(PatchRejected, match="protected"):
        read_snapshot(base.root, fixture)
    readme.unlink()
    with pytest.raises(PatchRejected, match="missing"):
        read_snapshot(base.root, fixture)


def test_existing_destination_or_nested_target_is_not_overwritten(source):
    fixture, base = source
    request = proposal(fixture, base)
    for destination in (base.root, base.root / "nested", base.root.parent / "existing"):
        if destination.name == "existing":
            destination.mkdir()
            (destination / "preserve").write_text("important")
        with pytest.raises(PatchRejected):
            apply_patch_proposal(base.root, fixture, request, destination=destination)
    assert (base.root.parent / "existing/preserve").read_text() == "important"
    assert read_snapshot(base.root, fixture) == base


def test_mid_write_failure_removes_unactivated_revision_and_preserves_base(source, monkeypatch):
    fixture, base = source
    destination = base.root.parent / "write-failure"
    original_write = module.os.write
    calls = 0
    def fail_after_first(descriptor, data):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected staging failure")
        return original_write(descriptor, data)
    monkeypatch.setattr(module.os, "write", fail_after_first)
    with pytest.raises(OSError, match="injected"):
        apply_patch_proposal(base.root, fixture, proposal(fixture, base), destination=destination)
    assert not destination.exists()
    assert read_snapshot(base.root, fixture) == base


def test_opening_new_revision_failure_cleans_owned_directory_and_preserves_base(source, monkeypatch):
    fixture, base = source
    destination = base.root.parent / "opening-failure"
    original_open = module.os.open
    def fail_open(path, *args, **kwargs):
        if path == destination.name:
            raise OSError("injected root-open failure")
        return original_open(path, *args, **kwargs)
    monkeypatch.setattr(module.os, "open", fail_open)
    with pytest.raises(OSError, match="injected"):
        apply_patch_proposal(base.root, fixture, proposal(fixture, base), destination=destination)
    assert not destination.exists()
    assert read_snapshot(base.root, fixture) == base


def test_symlink_inserted_during_staging_cannot_write_outside_revision(source, monkeypatch):
    fixture, base = source
    destination = base.root.parent / "race-revision"
    outside = base.root.parent / "outside-target"
    outside.mkdir()
    marker = outside / "cache.py"
    marker.write_text("preserve outside bytes")
    original_mkdir = module.os.mkdir
    def inject(path, *args, **kwargs):
        if path == "src" and kwargs.get("dir_fd") is not None:
            os.symlink(outside, path, dir_fd=kwargs["dir_fd"])
            raise FileExistsError("injected directory link")
        return original_mkdir(path, *args, **kwargs)
    monkeypatch.setattr(module.os, "mkdir", inject)
    with pytest.raises(OSError):
        apply_patch_proposal(base.root, fixture, proposal(fixture, base), destination=destination)
    assert marker.read_text() == "preserve outside bytes"
    assert not destination.exists()
    assert read_snapshot(base.root, fixture) == base


def test_postwrite_verification_failure_never_returns_or_activates_partial_revision(source, monkeypatch):
    fixture, base = source
    destination = base.root.parent / "verification-failure"
    original_read = module.read_snapshot
    def wrong_hash(root, spec):
        result = original_read(root, spec)
        return replace(result, tree_sha256="0" * 64) if Path(root) == destination else result
    monkeypatch.setattr(module, "read_snapshot", wrong_hash)
    with pytest.raises(PatchRejected, match="differs"):
        apply_patch_proposal(base.root, fixture, proposal(fixture, base), destination=destination)
    assert not destination.exists()
    assert original_read(base.root, fixture) == base


def test_unicode_paths_and_contents_remain_exact_and_aliases_are_rejected(tmp_path):
    initial = get_fixture("expiry-boundary")
    file = FixtureFile("src/Δelta.py", 'label = "Cafe\u0301"\n')
    fixture = replace(initial, files=(file,), editable_paths=(file.path,))
    base = materialize_fixture(tmp_path.resolve() / "unicode-base", fixture)
    request = proposal(fixture, base, old='"Cafe\u0301"', new='"Café"')
    result = apply_patch_proposal(base.root, fixture, request, destination=base.root.parent / "unicode-next")
    assert (result.root / file.path).read_text() == 'label = "Café"\n'
    assert (base.root / file.path).read_text() == 'label = "Cafe\u0301"\n'
    ambiguous = replace(fixture, files=(file, FixtureFile("src/δelta.py", "other")),
                        editable_paths=(file.path, "src/δelta.py"))
    with pytest.raises(PatchRejected, match="ambiguous"):
        materialize_fixture(tmp_path.resolve() / "rejected", ambiguous)
