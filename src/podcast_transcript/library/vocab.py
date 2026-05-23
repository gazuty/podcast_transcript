"""Controlled vocabulary for topics and speakers.

Two JSON files live under ``podcast-library/index/vocab/``:

- ``topics.json`` — every topic that may appear on an episode record.
- ``speakers.json`` — every host/guest/speaker name.

Both share the same shape::

    {
      "canonical": {"<Canonical Name>": {"added": "YYYY-MM-DD"}},
      "aliases":   {"<Variant>":        "<Canonical Name>"}
    }

Resolution rules (see :func:`Vocab.resolve`):

1. Exact match against ``canonical`` → return as-is, *pending=False*.
2. Exact match against ``aliases`` → return the canonical it points to,
   *pending=False*.
3. Otherwise → return the input unchanged with *pending=True*. The caller
   is expected to record the pending entry on the episode (so
   ``pending-vocab.md`` can surface it) and to call :meth:`add_pending`
   so the vocab file remembers we've seen the term.

We deliberately do **not** match case-insensitively or normalise
whitespace at resolution time — every alias must be spelled out. This
keeps the vocab file a complete audit log of the rewrites we're
performing.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

__all__ = [
    "Vocab",
    "VocabError",
    "load_vocab",
    "save_vocab",
]


class VocabError(ValueError):
    """Raised when a vocab file is structurally invalid."""


@dataclass
class Vocab:
    """In-memory view of one vocab file.

    *canonical* maps a canonical name to metadata (today: just ``added``).
    *aliases* maps a variant spelling to the canonical it should resolve
    to — alias targets must exist in *canonical*.
    """

    canonical: dict[str, dict[str, Any]] = field(default_factory=dict)
    aliases: dict[str, str] = field(default_factory=dict)

    def resolve(self, name: str) -> tuple[str, bool]:
        """Resolve *name* against the vocab.

        Returns ``(resolved_name, pending)``. *pending* is True when
        *name* matched neither table — the caller still gets a usable
        name back so ingest doesn't block.
        """
        stripped = name.strip()
        if not stripped:
            raise VocabError("cannot resolve empty name")
        if stripped in self.canonical:
            return stripped, False
        target = self.aliases.get(stripped)
        if target is not None:
            if target not in self.canonical:
                raise VocabError(
                    f"alias {stripped!r} points at non-canonical {target!r}",
                )
            return target, False
        return stripped, True

    def add_pending(self, name: str, *, today: str | None = None) -> bool:
        """Promote a previously-unseen *name* into the canonical table.

        Returns True if a new entry was added, False if *name* was already
        canonical (in which case this is a no-op). Aliases are never
        auto-added — those require a deliberate human edit.

        *today* lets tests inject a deterministic date; otherwise we use
        ``datetime.now(UTC)``.
        """
        stripped = name.strip()
        if not stripped:
            raise VocabError("cannot add empty name to vocab")
        if stripped in self.canonical:
            return False
        if stripped in self.aliases:
            raise VocabError(
                f"{stripped!r} is already an alias for "
                f"{self.aliases[stripped]!r}; promote that instead",
            )
        added = today or datetime.now(UTC).date().isoformat()
        self.canonical[stripped] = {"added": added, "pending": True}
        return True

    def validate(self) -> None:
        """Raise if any alias points at a non-canonical name."""
        for variant, target in self.aliases.items():
            if target not in self.canonical:
                raise VocabError(
                    f"alias {variant!r} points at non-canonical {target!r}",
                )

    def to_dict(self) -> dict[str, Any]:
        return {"canonical": dict(self.canonical), "aliases": dict(self.aliases)}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Vocab:
        unknown = set(data) - {"canonical", "aliases"}
        if unknown:
            raise VocabError(f"unknown vocab fields: {sorted(unknown)}")
        canonical_raw = data.get("canonical", {})
        aliases_raw = data.get("aliases", {})
        if not isinstance(canonical_raw, dict):
            raise VocabError("'canonical' must be an object")
        if not isinstance(aliases_raw, dict):
            raise VocabError("'aliases' must be an object")
        canonical = {
            str(k): (v if isinstance(v, dict) else {"added": str(v)})
            for k, v in canonical_raw.items()
        }
        aliases = {str(k): str(v) for k, v in aliases_raw.items()}
        vocab = cls(canonical=canonical, aliases=aliases)
        vocab.validate()
        return vocab


def load_vocab(path: Path) -> Vocab:
    """Load a vocab file. Missing file → empty Vocab."""
    if not path.is_file():
        return Vocab()
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise VocabError(f"{path}: expected a JSON object at the root")
    return Vocab.from_dict(data)


def save_vocab(path: Path, vocab: Vocab) -> None:
    """Atomically rewrite *path* with *vocab*.

    Keys are sorted at write time so the file diffs stably.
    """
    vocab.validate()
    payload = {
        "canonical": dict(sorted(vocab.canonical.items())),
        "aliases": dict(sorted(vocab.aliases.items())),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    part_path = path.with_suffix(path.suffix + ".part")
    with part_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")
    part_path.replace(path)
