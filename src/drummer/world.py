"""Deterministic one-probe reference worlds for the Milestone 1 experiment.

The arrays in a corpus contain labels needed to score the experiment.  Model
methods never receive ``target_index`` or receiver-private candidate order on
the sender side; :meth:`CorpusSplit.batch` names the permitted observations
explicitly so that boundary remains reviewable.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from enum import IntEnum
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch


ATTRIBUTE_CARDINALITIES: tuple[int, ...] = (2, 2, 2, 2, 4)
NUM_IDENTITIES = int(np.prod(ATTRIBUTE_CARDINALITIES))
NUM_CANDIDATES = 4
GROUNDING_BITS = 6
CORPUS_FORMAT_VERSION = 3
UNSEAL_CONFIRMATION = "UNSEAL DRUMMER TEST FOR FINAL EVALUATION"


class WorldCondition(IntEnum):
    """Probe condition; these values are persisted in corpus arrays."""

    VALID_REPEAT = 0
    DROPPED_GROUNDING = 1
    NEW_REFERENCE = 2


CONDITION_NAMES: dict[int, str] = {
    WorldCondition.VALID_REPEAT: "valid_repeat",
    WorldCondition.DROPPED_GROUNDING: "dropped_grounding",
    WorldCondition.NEW_REFERENCE: "new_reference",
}


class SealedTestError(PermissionError):
    """Raised when code attempts to read the sealed test split."""


@dataclass(frozen=True)
class CorpusConfig:
    """Configuration for a deterministic train/validation/sealed-test corpus."""

    seed: int = 20_260_904
    train_size: int = 100_000
    validation_size: int = 10_000
    test_size: int = 10_000

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | "CorpusConfig") -> "CorpusConfig":
        if isinstance(value, cls):
            return value
        sizes = value.get("sizes", {})
        return cls(
            seed=int(value.get("corpus_seed", value.get("seed", cls.seed))),
            train_size=int(sizes.get("train", value.get("train_size", cls.train_size))),
            validation_size=int(
                sizes.get(
                    "validation",
                    sizes.get("val", value.get("validation_size", cls.validation_size)),
                )
            ),
            test_size=int(sizes.get("test", value.get("test_size", cls.test_size))),
        )

    def __post_init__(self) -> None:
        for name, size in self.sizes.items():
            if size <= 0:
                raise ValueError(f"{name} size must be positive, got {size}")

    @property
    def sizes(self) -> dict[str, int]:
        return {
            "train": self.train_size,
            "validation": self.validation_size,
            "test": self.test_size,
        }


def identity_to_attributes(identity: int) -> tuple[int, ...]:
    """Decode the canonical 0..63 identity into five mixed-radix attributes."""

    if not 0 <= int(identity) < NUM_IDENTITIES:
        raise ValueError(f"identity must be in [0, {NUM_IDENTITIES}), got {identity}")
    remainder = int(identity)
    values: list[int] = []
    for cardinality in reversed(ATTRIBUTE_CARDINALITIES):
        values.append(remainder % cardinality)
        remainder //= cardinality
    return tuple(reversed(values))


def attributes_to_identity(attributes: Sequence[int]) -> int:
    """Encode five attributes as the canonical 0..63 identity."""

    if len(attributes) != len(ATTRIBUTE_CARDINALITIES):
        raise ValueError(f"expected {len(ATTRIBUTE_CARDINALITIES)} attributes")
    identity = 0
    for value, cardinality in zip(attributes, ATTRIBUTE_CARDINALITIES, strict=True):
        value = int(value)
        if not 0 <= value < cardinality:
            raise ValueError(f"attribute {value} is outside cardinality {cardinality}")
        identity = identity * cardinality + value
    return identity


IDENTITY_ATTRIBUTES = np.asarray(
    [identity_to_attributes(identity) for identity in range(NUM_IDENTITIES)],
    dtype=np.int64,
)


def _condition_counts(size: int) -> tuple[int, int, int]:
    """Allocate 60/20/20 deterministically, with largest-remainder rounding."""

    exact = np.asarray([0.6, 0.2, 0.2], dtype=np.float64) * size
    counts = np.floor(exact).astype(np.int64)
    remaining = size - int(counts.sum())
    order = np.argsort(-(exact - counts), kind="stable")
    counts[order[:remaining]] += 1
    return tuple(int(value) for value in counts)


def _group_key(candidates: np.ndarray, previous: int, target: int) -> tuple[Any, ...]:
    # Condition is intentionally absent.  A repeat scene cannot occur in one
    # split as VALID_REPEAT and another as DROPPED_GROUNDING.
    return (tuple(sorted(int(value) for value in candidates)), int(previous), int(target))


def _group_id(key: tuple[Any, ...]) -> int:
    digest = sha256(repr(key).encode("ascii")).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)


def _logical_hash(arrays: Mapping[str, np.ndarray]) -> str:
    digest = sha256()
    for name in sorted(arrays):
        array = np.ascontiguousarray(arrays[name])
        digest.update(name.encode("utf-8"))
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(repr(array.shape).encode("ascii"))
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _generate_split(
    size: int,
    seed: int,
    used_group_keys: set[tuple[Any, ...]],
    used_group_ids: set[int],
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    valid_count, dropped_count, new_count = _condition_counts(size)
    conditions = np.concatenate(
        [
            np.full(valid_count, WorldCondition.VALID_REPEAT, dtype=np.int8),
            np.full(dropped_count, WorldCondition.DROPPED_GROUNDING, dtype=np.int8),
            np.full(new_count, WorldCondition.NEW_REFERENCE, dtype=np.int8),
        ]
    )
    rng.shuffle(conditions)

    # Cycling before a joint shuffle keeps every identity within one occurrence
    # of every other identity, including corpora whose size is not divisible by 64.
    target_offset = int(rng.integers(NUM_IDENTITIES))
    targets = ((np.arange(size, dtype=np.int64) + target_offset) % NUM_IDENTITIES).astype(
        np.int8
    )
    joint_order = rng.permutation(size)
    targets = targets[joint_order]
    conditions = conditions[joint_order]

    sender_previous = np.empty(size, dtype=np.int8)
    receiver_previous = np.full(size, -1, dtype=np.int8)
    candidates = np.empty((size, NUM_CANDIDATES), dtype=np.int8)
    target_index = np.empty(size, dtype=np.int8)
    history_present = np.zeros(size, dtype=np.bool_)
    acknowledged = np.zeros(size, dtype=np.bool_)
    group_ids = np.empty(size, dtype=np.int64)

    all_identities = np.arange(NUM_IDENTITIES, dtype=np.int16)
    for row, (target_value, condition_value) in enumerate(zip(targets, conditions, strict=True)):
        target = int(target_value)
        condition = WorldCondition(int(condition_value))
        for _attempt in range(10_000):
            if condition is WorldCondition.NEW_REFERENCE:
                old_choices = all_identities[all_identities != target]
                old = int(rng.choice(old_choices))
                excluded = (all_identities != target) & (all_identities != old)
                decoys = rng.choice(all_identities[excluded], size=2, replace=False)
                scene = np.asarray([target, old, *decoys.tolist()], dtype=np.int8)
            else:
                old = target
                decoys = rng.choice(all_identities[all_identities != target], size=3, replace=False)
                scene = np.asarray([target, *decoys.tolist()], dtype=np.int8)
            rng.shuffle(scene)
            key = _group_key(scene, old, target)
            group_id = _group_id(key)
            if key not in used_group_keys and group_id not in used_group_ids:
                break
        else:  # pragma: no cover - impossible at the specified corpus scale
            raise RuntimeError("could not draw a fresh scene/transition group")

        used_group_keys.add(key)
        used_group_ids.add(group_id)
        sender_previous[row] = old
        candidates[row] = scene
        target_index[row] = int(np.flatnonzero(scene == target)[0])
        group_ids[row] = group_id

        if condition is not WorldCondition.DROPPED_GROUNDING:
            receiver_previous[row] = old
            history_present[row] = True
            acknowledged[row] = True

    return {
        "condition": conditions,
        "target_id": targets,
        "sender_previous_id": sender_previous,
        "receiver_previous_id": receiver_previous,
        "candidate_ids": candidates,
        "target_index": target_index,
        "history_present": history_present,
        "acknowledged": acknowledged,
        "group_id": group_ids,
    }


def _split_summary(
    filename: str,
    arrays: Mapping[str, np.ndarray],
    *,
    file_sha256: str,
) -> dict[str, Any]:
    targets = arrays["target_id"]
    target_counts = np.bincount(targets, minlength=NUM_IDENTITIES)
    conditions = arrays["condition"]
    return {
        "filename": filename,
        "file_sha256": file_sha256,
        "size": int(len(targets)),
        "logical_sha256": _logical_hash(arrays),
        "group_sha256": sha256(np.sort(arrays["group_id"]).tobytes()).hexdigest(),
        "target_count_min": int(target_counts.min()),
        "target_count_max": int(target_counts.max()),
        "condition_counts": {
            name: int(np.count_nonzero(conditions == value))
            for value, name in CONDITION_NAMES.items()
        },
    }


def generate_corpus(
    root: str | Path,
    config: CorpusConfig | Mapping[str, Any] = CorpusConfig(),
) -> dict[str, Any]:
    """Generate all three splits and return their deterministic manifest.

    Test bytes are deliberately named ``test.sealed.npz``.  The supported load
    path refuses to expose them until :func:`unseal_test` records an explicit
    acknowledgement.
    """

    config = CorpusConfig.from_mapping(config)
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "corpus_manifest.json"
    corpus_artifacts = {
        root / "train.npz",
        root / "validation.npz",
        root / "test.sealed.npz",
        root / "TEST_UNSEALED.json",
    }
    if manifest_path.exists():
        existing = _read_manifest(root)
        if existing.get("config") != asdict(config):
            raise FileExistsError(
                "corpus root already contains a different configuration; use a new root"
            )
        corpus_manifest_evidence(root)
        for split, summary in existing["splits"].items():
            if split == "test":
                # The physical sealed artifact was verified above.  Do not load
                # its label arrays merely to reuse an existing corpus.
                continue
            path = root / summary["filename"]
            with np.load(path, allow_pickle=False) as archive:
                arrays = {name: archive[name] for name in archive.files}
            if _logical_hash(arrays) != summary["logical_sha256"]:
                raise ValueError(f"existing {split} corpus failed logical hash verification")
        if not verify_split_disjointness(root):
            raise ValueError("existing corpus has overlapping scene/transition groups")
        return existing
    unexpected = sorted(str(path) for path in corpus_artifacts if path.exists())
    if unexpected:
        raise FileExistsError(
            "refusing to overwrite corpus artifacts without their manifest: " + ", ".join(unexpected)
        )
    used_group_keys: set[tuple[Any, ...]] = set()
    used_group_ids: set[int] = set()
    summaries: dict[str, dict[str, Any]] = {}

    for split_index, (split, size) in enumerate(config.sizes.items()):
        arrays = _generate_split(
            size=size,
            seed=int(np.random.SeedSequence([config.seed, split_index]).generate_state(1)[0]),
            used_group_keys=used_group_keys,
            used_group_ids=used_group_ids,
        )
        filename = "test.sealed.npz" if split == "test" else f"{split}.npz"
        split_path = root / filename
        np.savez_compressed(split_path, **arrays)
        summaries[split] = _split_summary(
            filename,
            arrays,
            file_sha256=_sha256_file(split_path),
        )

    manifest: dict[str, Any] = {
        "format_version": CORPUS_FORMAT_VERSION,
        "config": asdict(config),
        "attribute_cardinalities": list(ATTRIBUTE_CARDINALITIES),
        "num_identities": NUM_IDENTITIES,
        "num_candidates": NUM_CANDIDATES,
        "mixture": {"valid_repeat": 0.6, "dropped_grounding": 0.2, "new_reference": 0.2},
        "grounding": {
            "encoding": "canonical_fixed_width",
            "emitted_bits_per_episode": GROUNDING_BITS,
            "dropped_condition_is_charged": True,
        },
        "test_sealed": True,
        "splits": summaries,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


@dataclass
class CorpusSplit:
    """Loaded corpus arrays with an explicit observation-building boundary."""

    name: str
    arrays: dict[str, np.ndarray]
    logical_sha256: str

    def __len__(self) -> int:
        return int(len(self.arrays["target_id"]))

    def batch(
        self,
        indices: Sequence[int] | np.ndarray | torch.Tensor,
        device: torch.device | str = "cpu",
    ) -> dict[str, torch.Tensor]:
        if isinstance(indices, torch.Tensor):
            index = indices.detach().cpu().numpy()
        else:
            index = np.asarray(indices, dtype=np.int64)
        target_ids_np = self.arrays["target_id"][index].astype(np.int64)
        sender_previous_ids_np = self.arrays["sender_previous_id"][index].astype(np.int64)
        receiver_previous_ids_np = self.arrays["receiver_previous_id"][index].astype(np.int64)
        candidate_ids_np = self.arrays["candidate_ids"][index].astype(np.int64)
        history_np = self.arrays["history_present"][index]

        sender_previous_attrs = IDENTITY_ATTRIBUTES[sender_previous_ids_np].copy()
        safe_receiver_previous = np.maximum(receiver_previous_ids_np, 0)
        receiver_previous_attrs = IDENTITY_ATTRIBUTES[safe_receiver_previous].copy()
        receiver_previous_attrs[~history_np] = 0

        def tensor(value: np.ndarray, dtype: torch.dtype) -> torch.Tensor:
            return torch.as_tensor(value, dtype=dtype, device=device)

        sender_history = tensor(sender_previous_attrs, torch.long)
        receiver_history = tensor(receiver_previous_attrs, torch.long)
        batch = {
            "world_condition": tensor(self.arrays["condition"][index], torch.long),
            "target_id": tensor(target_ids_np, torch.long),
            "target_attrs": tensor(IDENTITY_ATTRIBUTES[target_ids_np], torch.long),
            "sender_history_attrs": sender_history,
            "receiver_history_attrs": receiver_history,
            "candidate_ids": tensor(candidate_ids_np, torch.long),
            "candidate_attrs": tensor(IDENTITY_ATTRIBUTES[candidate_ids_np], torch.long),
            "target_index": tensor(self.arrays["target_index"][index], torch.long),
            # Sender-owned prior intent exists even when its grounding packet was
            # dropped.  Only the receiver's actual memory follows delivery.
            "sender_history_present": torch.ones(
                len(index), dtype=torch.bool, device=device
            ),
            "receiver_history_present": tensor(history_np, torch.bool),
            "acknowledged": tensor(self.arrays["acknowledged"][index], torch.bool),
            "group_id": tensor(self.arrays["group_id"][index], torch.long),
            "grounding_bits": torch.full(
                (len(index),), float(GROUNDING_BITS), dtype=torch.float32, device=device
            ),
        }
        return batch


def _read_manifest(root: Path) -> dict[str, Any]:
    path = root / "corpus_manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"corpus manifest not found: {path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("format_version") != CORPUS_FORMAT_VERSION:
        raise ValueError(f"unsupported corpus format: {manifest.get('format_version')}")
    return manifest


def corpus_manifest_evidence(root: str | Path) -> dict[str, Any]:
    """Return exact corpus identity without opening any split arrays.

    The manifest binds logical array hashes to the physical compressed files.
    Reading and hashing those files does not deserialize the sealed test labels,
    so this evidence can be frozen before the one-way unseal event.
    """

    root = Path(root)
    manifest_path = root / "corpus_manifest.json"
    manifest = _read_manifest(root)
    splits: dict[str, dict[str, Any]] = {}
    for split in ("train", "validation", "test"):
        summary = manifest.get("splits", {}).get(split)
        if not isinstance(summary, Mapping):
            raise ValueError(f"corpus manifest lacks {split} split metadata")
        path = root / str(summary.get("filename", ""))
        if not path.is_file():
            raise FileNotFoundError(f"existing corpus is incomplete: {path}")
        expected_file_hash = summary.get("file_sha256")
        actual_file_hash = _sha256_file(path)
        if expected_file_hash != actual_file_hash:
            raise ValueError(f"physical hash mismatch for {split}: {path}")
        logical_hash = summary.get("logical_sha256")
        if not isinstance(logical_hash, str) or not _is_sha256(logical_hash):
            raise ValueError(f"corpus manifest has an invalid logical hash for {split}")
        splits[split] = {
            "logical_sha256": logical_hash,
            "file_sha256": actual_file_hash,
            "size": int(summary["size"]),
        }
    return {
        "format_version": int(manifest["format_version"]),
        "manifest_sha256": _sha256_file(manifest_path),
        "splits": splits,
    }


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def load_split(root: str | Path, split: str, *, verify: bool = True) -> CorpusSplit:
    """Load a split, enforcing the persistent sealed-test acknowledgement."""

    root = Path(root)
    split = "validation" if split == "val" else split
    if split not in {"train", "validation", "test"}:
        raise ValueError(f"unknown split: {split}")
    manifest = _read_manifest(root)
    if split == "test" and not (root / "TEST_UNSEALED.json").exists():
        raise SealedTestError(
            "test is sealed; run the explicit unseal command before final evaluation"
        )
    summary = manifest["splits"][split]
    path = root / summary["filename"]
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    logical_hash = _logical_hash(arrays)
    if verify and logical_hash != summary["logical_sha256"]:
        raise ValueError(f"logical hash mismatch for {split}: {path}")
    return CorpusSplit(name=split, arrays=arrays, logical_sha256=logical_hash)


def unseal_test(root: str | Path, acknowledgement: str) -> Path:
    """Record the one-way procedural transition to final test evaluation."""

    if acknowledgement != UNSEAL_CONFIRMATION:
        raise SealedTestError(f"exact acknowledgement required: {UNSEAL_CONFIRMATION!r}")
    root = Path(root)
    manifest = _read_manifest(root)
    record = {
        "unsealed_at_utc": datetime.now(UTC).isoformat(),
        "test_logical_sha256": manifest["splits"]["test"]["logical_sha256"],
        "acknowledgement": acknowledgement,
    }
    path = root / "TEST_UNSEALED.json"
    path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def verify_split_disjointness(root: str | Path) -> bool:
    """Verify stored group IDs are pairwise disjoint without reading test labels."""

    root = Path(root)
    manifest = _read_manifest(root)
    groups: dict[str, set[int]] = {}
    for split, summary in manifest["splits"].items():
        with np.load(root / summary["filename"], allow_pickle=False) as archive:
            groups[split] = set(int(value) for value in archive["group_id"])
    names = tuple(groups)
    return all(not (groups[a] & groups[b]) for i, a in enumerate(names) for b in names[i + 1 :])
