"""Exact scoped patches applied as copy-on-write disposable source revisions.

The base is never modified. A coordinator may activate only the returned verified
revision, never a directory merely because it exists. This module executes no code.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import unicodedata
from typing import Mapping

from drummer.workflow_fixtures import FixtureFile, WorkflowFixture, canonical_json, fingerprint


PATCH_VERSION = "workflow-patch-1"
MAX_FILE_BYTES = 65536
MAX_TOTAL_BYTES = 262144
MAX_PROPOSAL_BYTES = 262144
MAX_FILES = 16
MAX_EDITS = 16
_SHA = re.compile(r"[a-f0-9]{64}\Z")


class PatchRejected(ValueError):
    pass


@dataclass(frozen=True)
class SnapshotFile:
    path: str
    text: str
    sha256: str
    mode: int


@dataclass(frozen=True)
class SourceSnapshot:
    root: Path
    files: tuple[SnapshotFile, ...]
    tree_sha256: str


@dataclass(frozen=True)
class ValidatedPatch:
    task_id: str
    base: SourceSnapshot
    changed_paths: tuple[str, ...]
    result_files: tuple[FixtureFile, ...]
    result_tree_sha256: str
    proposal_sha256: str


@dataclass(frozen=True)
class AppliedPatch:
    task_id: str
    root: Path
    base_root: Path
    base_tree_sha256: str
    tree_sha256: str
    changed_paths: tuple[str, ...]
    proposal_sha256: str
    activation: str = "coordinator_must_activate_returned_revision"


def _path(value: object) -> str:
    if not isinstance(value, str) or not value:
        raise PatchRejected("invalid relative path")
    try:
        if len(value.encode("utf-8")) > 512:
            raise PatchRejected("relative path exceeds byte bound")
    except UnicodeError as error:
        raise PatchRejected("invalid UTF-8 path") from error
    if (value.startswith("/") or "\\" in value or ":" in value or "%" in value
            or any(ord(character) < 32 for character in value)):
        raise PatchRejected("unsupported or unsafe path spelling")
    pieces = value.split("/")
    if any(piece in {"", ".", ".."} or piece != piece.strip() or piece.endswith(".") for piece in pieces):
        raise PatchRejected("relative path components must be exact and unambiguous")
    if any(piece.casefold() == ".git" for piece in pieces):
        raise PatchRejected("repository metadata is not a patch target")
    return value


def _scope(fixture: WorkflowFixture) -> tuple[dict[str, str], set[str]]:
    if not 1 <= len(fixture.files) <= MAX_FILES:
        raise PatchRejected("fixture file count exceeds bound")
    files = {}
    aliases = set()
    total = 0
    for file in fixture.files:
        path = _path(file.path)
        alias = unicodedata.normalize("NFC", path).casefold()
        if path in files or alias in aliases:
            raise PatchRejected("duplicate or filesystem-ambiguous fixture path")
        raw = _text_bytes(file.text)
        total += len(raw)
        files[path] = file.text
        aliases.add(alias)
    if total > MAX_TOTAL_BYTES:
        raise PatchRejected("fixture bytes exceed bound")
    editable = {_path(path) for path in fixture.editable_paths}
    if len(editable) != len(fixture.editable_paths) or not editable or not editable <= files.keys():
        raise PatchRejected("editable paths must be a unique nonempty subset of fixture files")
    if any(PurePosixPath(path).suffix != ".py" or any(
        piece.casefold() in {"test", "tests", "config", "configuration"}
        for piece in PurePosixPath(path).parts) or PurePosixPath(path).name.startswith("test_")
           for path in editable):
        raise PatchRejected("only scoped Python source, never tests or configuration, may change")
    return files, editable


def _text_bytes(text: object) -> bytes:
    if not isinstance(text, str):
        raise PatchRejected("source and replacement bodies must be text")
    try:
        raw = text.encode("utf-8")
    except UnicodeError as error:
        raise PatchRejected("invalid UTF-8 text") from error
    if len(raw) > MAX_FILE_BYTES or b"\x00" in raw:
        raise PatchRejected("binary or oversized file content is forbidden")
    return raw


def _canonical_root(root: str | Path) -> Path:
    root = Path(root).absolute()
    if root != root.resolve(strict=True):
        raise PatchRejected("snapshot root and ancestors must use their canonical non-symlink path")
    if not stat.S_ISDIR(root.lstat().st_mode):
        raise PatchRejected("snapshot root must be a real directory")
    return root


def _tree_digest(files: tuple[SnapshotFile, ...]) -> str:
    return fingerprint([{"path": file.path, "sha256": file.sha256,
                         "bytes": len(file.text.encode("utf-8")), "mode": file.mode}
                        for file in sorted(files, key=lambda item: item.path)])


def _files_snapshot(root: Path, files: tuple[FixtureFile, ...]) -> SourceSnapshot:
    records = tuple(SnapshotFile(file.path, file.text, file.sha256, 0o644)
                    for file in sorted(files, key=lambda item: item.path))
    return SourceSnapshot(root, records, _tree_digest(records))


def read_snapshot(root: str | Path, fixture: WorkflowFixture) -> SourceSnapshot:
    """Read only exact fixture files, using no-follow directory/file descriptors."""
    initial, editable = _scope(fixture)
    root = _canonical_root(root)
    expected_directories = {str(parent) for path in initial for parent in PurePosixPath(path).parents
                            if str(parent) != "."}
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    records: list[SnapshotFile] = []
    total = 0
    with _Directory(root, flags) as descriptor:
        root_device = os.fstat(descriptor).st_dev
        def walk(directory, prefix=""):
            nonlocal total
            for name in sorted(os.listdir(directory)):
                path = f"{prefix}/{name}" if prefix else name
                _path(path)
                metadata = os.stat(name, dir_fd=directory, follow_symlinks=False)
                if metadata.st_dev != root_device or stat.S_ISLNK(metadata.st_mode):
                    raise PatchRejected("symlink or cross-device snapshot entry")
                if stat.S_ISDIR(metadata.st_mode):
                    if path not in expected_directories:
                        raise PatchRejected("unlisted snapshot directory")
                    child = os.open(name, flags, dir_fd=directory)
                    try:
                        if os.fstat(child).st_ino != metadata.st_ino:
                            raise PatchRejected("directory changed during snapshot read")
                        walk(child, path)
                    finally:
                        os.close(child)
                elif stat.S_ISREG(metadata.st_mode):
                    if path not in initial or metadata.st_nlink != 1 or metadata.st_size > MAX_FILE_BYTES:
                        raise PatchRejected("unlisted, hardlinked or oversized snapshot file")
                    if stat.S_IMODE(metadata.st_mode) != 0o644:
                        raise PatchRejected("snapshot file mode changed")
                    file_descriptor = os.open(name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=directory)
                    try:
                        before = os.fstat(file_descriptor)
                        if ((before.st_ino, before.st_dev, before.st_nlink) != (metadata.st_ino, metadata.st_dev, 1)
                                or stat.S_IMODE(before.st_mode) != 0o644):
                            raise PatchRejected("file changed during snapshot read")
                        chunks = []
                        length = 0
                        while chunk := os.read(file_descriptor, 8192):
                            chunks.append(chunk)
                            length += len(chunk)
                            if length > MAX_FILE_BYTES:
                                raise PatchRejected("file grew beyond its size bound")
                        after = os.fstat(file_descriptor)
                        if (before.st_size, before.st_mtime_ns, before.st_ctime_ns, before.st_mode, before.st_nlink) != (
                            after.st_size, after.st_mtime_ns, after.st_ctime_ns, after.st_mode, after.st_nlink):
                            raise PatchRejected("file changed while reading")
                    finally:
                        os.close(file_descriptor)
                    try:
                        text = b"".join(chunks).decode("utf-8")
                    except UnicodeError as error:
                        raise PatchRejected("snapshot file is not UTF-8") from error
                    raw = _text_bytes(text)
                    if path not in editable and text != initial[path]:
                        raise PatchRejected("protected fixture file changed")
                    total += len(raw)
                    if total > MAX_TOTAL_BYTES:
                        raise PatchRejected("snapshot total bytes exceed bound")
                    records.append(SnapshotFile(path, text, hashlib.sha256(raw).hexdigest(), 0o644))
                else:
                    raise PatchRejected("special files are forbidden")
        walk(descriptor)
    if {record.path for record in records} != initial.keys():
        raise PatchRejected("snapshot has missing files")
    files = tuple(sorted(records, key=lambda record: record.path))
    return SourceSnapshot(root, files, _tree_digest(files))


class _Directory:
    def __init__(self, root: Path, flags: int):
        self.root, self.flags = root, flags

    def __enter__(self):
        self.descriptor = os.open(self.root, self.flags)
        return self.descriptor

    def __exit__(self, *unused):
        os.close(self.descriptor)


def _proposal(value: Mapping | str) -> dict:
    def unique(pairs):
        result = {}
        for key, item in pairs:
            if key in result:
                raise PatchRejected("duplicate proposal JSON key")
            result[key] = item
        return result
    try:
        raw = value if isinstance(value, str) else canonical_json(value)
        if len(raw.encode("utf-8")) > MAX_PROPOSAL_BYTES:
            raise PatchRejected("proposal exceeds size bound")
        result = json.loads(raw, object_pairs_hook=unique,
                            parse_constant=lambda value: (_ for _ in ()).throw(PatchRejected("nonfinite JSON")))
    except (ValueError, TypeError, UnicodeError, RecursionError) as error:
        raise PatchRejected("malformed patch proposal") from error
    if not isinstance(result, dict) or set(result) != {"version", "task_id", "base_tree_sha256", "files"}:
        raise PatchRejected("unexpected patch fields; authority and commands are not accepted")
    if result["version"] != PATCH_VERSION or not isinstance(result["base_tree_sha256"], str) or not _SHA.fullmatch(result["base_tree_sha256"]):
        raise PatchRejected("patch version or base digest is invalid")
    if not isinstance(result["files"], list) or not 1 <= len(result["files"]) <= MAX_FILES:
        raise PatchRejected("patch file list must be nonempty and bounded")
    return result


def _replace_exact(source: str, edits: object) -> str:
    if not isinstance(edits, list) or not 1 <= len(edits) <= MAX_EDITS:
        raise PatchRejected("edit list must be nonempty and bounded")
    located = []
    for edit in edits:
        if not isinstance(edit, dict) or set(edit) != {"old", "new"}:
            raise PatchRejected("replacement fields must be exactly old/new")
        old, new = edit["old"], edit["new"]
        _text_bytes(old)
        _text_bytes(new)
        if not old or old == new:
            raise PatchRejected("empty search text and no-op edits are forbidden")
        start = source.find(old)
        if start < 0 or source.find(old, start + 1) >= 0:
            raise PatchRejected("old text must occur exactly once, including overlapping occurrences")
        located.append((start, start + len(old), new))
    located.sort(key=lambda item: item[0])
    if any(first[1] > second[0] for first, second in zip(located, located[1:])):
        raise PatchRejected("replacement ranges overlap")
    pieces, previous = [], 0
    for start, end, replacement in located:
        pieces.extend((source[previous:start], replacement))
        previous = end
    result = "".join((*pieces, source[previous:]))
    if not result:
        raise PatchRejected("whole-file deletion is forbidden")
    if result == source:
        raise PatchRejected("combined replacements have no effect")
    _text_bytes(result)
    return result


def validate_patch(root: str | Path, fixture: WorkflowFixture, proposal: Mapping | str) -> ValidatedPatch:
    """Complete validation before creating or modifying any output revision."""
    _, editable = _scope(fixture)
    request = _proposal(proposal)
    if request["task_id"] != fixture.task_id:
        raise PatchRejected("patch belongs to another task")
    base = read_snapshot(root, fixture)
    if base.tree_sha256 != request["base_tree_sha256"]:
        raise PatchRejected("stale base tree")
    contents = {file.path: file.text for file in base.files}
    hashes = {file.path: file.sha256 for file in base.files}
    changed = set()
    for file in request["files"]:
        if not isinstance(file, dict) or set(file) != {"path", "base_sha256", "edits"}:
            raise PatchRejected("unexpected file-patch fields")
        path = _path(file["path"])
        if path not in editable or path in changed:
            raise PatchRejected("unlisted or duplicate edit target")
        if file["base_sha256"] != hashes[path]:
            raise PatchRejected("stale original file hash")
        contents[path] = _replace_exact(contents[path], file["edits"])
        changed.add(path)
    result_files = tuple(FixtureFile(path, text) for path, text in sorted(contents.items()))
    if sum(len(file.text.encode("utf-8")) for file in result_files) > MAX_TOTAL_BYTES:
        raise PatchRejected("result tree exceeds byte bound")
    expected = _files_snapshot(base.root, result_files)
    return ValidatedPatch(fixture.task_id, base, tuple(sorted(changed)), result_files,
                          expected.tree_sha256, fingerprint(request))


def _create_revision(destination: Path, files: tuple[FixtureFile, ...]) -> tuple[int, int]:
    """Create exclusively; caller must validate the complete content and destination first."""
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    parent = os.open(destination.parent, flags)
    root = None
    created_identity = None
    try:
        os.mkdir(destination.name, mode=0o700, dir_fd=parent)
        identity = os.stat(destination.name, dir_fd=parent, follow_symlinks=False)
        created_identity = (identity.st_dev, identity.st_ino)
        if not stat.S_ISDIR(identity.st_mode):
            raise PatchRejected("new revision is not a directory")
        root = os.open(destination.name, flags, dir_fd=parent)
        opened_identity = os.fstat(root)
        if (opened_identity.st_dev, opened_identity.st_ino) != created_identity:
            raise PatchRejected("new revision changed during creation")
        for file in files:
            components = PurePosixPath(file.path).parts
            current = root
            opened = []
            try:
                for component in components[:-1]:
                    try:
                        os.mkdir(component, mode=0o700, dir_fd=current)
                    except FileExistsError:
                        pass
                    child = os.open(component, flags, dir_fd=current)
                    opened.append(child)
                    if os.fstat(child).st_dev != identity.st_dev:
                        raise PatchRejected("cross-device staging directory")
                    current = child
                descriptor = os.open(components[-1], os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                                     0o600, dir_fd=current)
                try:
                    remaining = memoryview(file.text.encode("utf-8"))
                    while remaining:
                        written = os.write(descriptor, remaining)
                        if written <= 0:
                            raise OSError("incomplete revision write")
                        remaining = remaining[written:]
                    os.fchmod(descriptor, 0o644)
                    os.fsync(descriptor)
                finally:
                    os.close(descriptor)
            finally:
                for child in reversed(opened):
                    os.close(child)
        return created_identity
    except BaseException:
        if created_identity is not None:
            _discard_owned_revision(destination, created_identity)
        raise
    finally:
        if root is not None:
            os.close(root)
        os.close(parent)


def _discard_owned_revision(destination: Path, identity: tuple[int, int]) -> None:
    # Only this newly created revision can be removed; never the base or its parent.
    try:
        current = destination.lstat()
    except FileNotFoundError:
        return
    if stat.S_ISDIR(current.st_mode) and (current.st_dev, current.st_ino) == identity:
        shutil.rmtree(destination)


def materialize_fixture(destination: str | Path, fixture: WorkflowFixture) -> SourceSnapshot:
    """Write only known defective fixture bytes, not model-generated code."""
    _scope(fixture)
    destination = Path(destination).absolute()
    parent = _canonical_root(destination.parent)
    if destination.name in {"", ".", ".."} or destination.exists() or destination.is_symlink():
        raise PatchRejected("fixture destination must be a new directory")
    destination = parent / destination.name
    identity = _create_revision(destination, fixture.files)
    try:
        return read_snapshot(destination, fixture)
    except BaseException:
        _discard_owned_revision(destination, identity)
        raise


def apply_patch_proposal(root: str | Path, fixture: WorkflowFixture, proposal: Mapping | str,
                         *, destination: str | Path) -> AppliedPatch:
    validated = validate_patch(root, fixture, proposal)
    destination = Path(destination).absolute()
    parent = _canonical_root(destination.parent)
    if (parent != validated.base.root.parent or destination == validated.base.root
            or destination.exists() or destination.is_symlink()):
        raise PatchRejected("revision destination must be a new sibling of the base")
    destination = parent / destination.name
    # Recheck before output creation; the base remains read-only throughout.
    if read_snapshot(root, fixture) != validated.base:
        raise PatchRejected("base changed after validation")
    identity = _create_revision(destination, validated.result_files)
    try:
        result = read_snapshot(destination, fixture)
        if result.tree_sha256 != validated.result_tree_sha256:
            raise PatchRejected("created revision differs from validated tree")
        changed = tuple(sorted(file.path for file in result.files
                               if file.sha256 != next(original.sha256 for original in validated.base.files
                                                      if original.path == file.path)))
        if changed != validated.changed_paths:
            raise PatchRejected("actual changed-file set differs from validated scope")
        if read_snapshot(root, fixture) != validated.base:
            raise PatchRejected("base changed while creating the revision")
        return AppliedPatch(fixture.task_id, destination, validated.base.root,
                            validated.base.tree_sha256, result.tree_sha256, changed,
                            validated.proposal_sha256)
    except BaseException:
        _discard_owned_revision(destination, identity)
        raise
