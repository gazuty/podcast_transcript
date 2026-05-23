"""Tests for :mod:`podcast_transcript.library.vocab`."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from podcast_transcript.library.vocab import Vocab, VocabError, load_vocab, save_vocab

if TYPE_CHECKING:
    from pathlib import Path


def test_resolve_canonical_passes_through() -> None:
    vocab = Vocab(canonical={"Hillary Lin": {"added": "2026-04-17"}})
    resolved, pending = vocab.resolve("Hillary Lin")
    assert resolved == "Hillary Lin"
    assert pending is False


def test_resolve_alias_rewrites_to_canonical() -> None:
    vocab = Vocab(
        canonical={"Hillary Lin": {"added": "2026-04-17"}},
        aliases={"Dr. Hillary Lin": "Hillary Lin", "Hillary Lin, MD": "Hillary Lin"},
    )
    for variant in ("Dr. Hillary Lin", "Hillary Lin, MD"):
        resolved, pending = vocab.resolve(variant)
        assert resolved == "Hillary Lin"
        assert pending is False


def test_resolve_unknown_returns_pending() -> None:
    vocab = Vocab(canonical={"Hillary Lin": {"added": "2026-04-17"}})
    resolved, pending = vocab.resolve("Andrej Karpathy")
    assert resolved == "Andrej Karpathy"
    assert pending is True


def test_resolve_rejects_empty() -> None:
    vocab = Vocab()
    with pytest.raises(VocabError):
        vocab.resolve("")


def test_resolve_alias_pointing_at_missing_canonical_raises() -> None:
    vocab = Vocab(
        canonical={"Hillary Lin": {"added": "2026-04-17"}},
        aliases={"Dr. Lin": "Hillary L."},  # canonical target is wrong
    )
    with pytest.raises(VocabError, match="non-canonical"):
        vocab.resolve("Dr. Lin")


def test_validate_catches_orphan_alias() -> None:
    with pytest.raises(VocabError):
        Vocab.from_dict(
            {
                "canonical": {"A": {"added": "2026-04-17"}},
                "aliases": {"variant": "B"},  # B isn't canonical
            },
        )


def test_add_pending_inserts_with_today() -> None:
    vocab = Vocab()
    assert vocab.add_pending("ApoB", today="2026-05-23") is True
    assert vocab.canonical["ApoB"] == {"added": "2026-05-23", "pending": True}


def test_add_pending_is_idempotent() -> None:
    vocab = Vocab(canonical={"ApoB": {"added": "2026-04-17"}})
    assert vocab.add_pending("ApoB", today="2026-05-23") is False
    # Doesn't overwrite existing entry
    assert vocab.canonical["ApoB"] == {"added": "2026-04-17"}


def test_add_pending_rejects_collision_with_alias() -> None:
    vocab = Vocab(
        canonical={"Hillary Lin": {"added": "2026-04-17"}},
        aliases={"Dr. Lin": "Hillary Lin"},
    )
    with pytest.raises(VocabError, match="already an alias"):
        vocab.add_pending("Dr. Lin")


def test_load_save_round_trip_sorts_keys(tmp_path: Path) -> None:
    path = tmp_path / "vocab.json"
    vocab = Vocab(
        canonical={
            "Zebra": {"added": "2026-04-17"},
            "Alpha": {"added": "2026-04-17"},
        },
        aliases={"zee": "Zebra"},
    )
    save_vocab(path, vocab)
    content = path.read_text(encoding="utf-8")
    # Alpha sorts before Zebra in the rendered file
    assert content.index('"Alpha"') < content.index('"Zebra"')
    reloaded = load_vocab(path)
    assert reloaded.canonical == vocab.canonical
    assert reloaded.aliases == vocab.aliases


def test_load_missing_returns_empty(tmp_path: Path) -> None:
    vocab = load_vocab(tmp_path / "nope.json")
    assert vocab.canonical == {}
    assert vocab.aliases == {}


def test_load_rejects_root_array(tmp_path: Path) -> None:
    path = tmp_path / "v.json"
    path.write_text(json.dumps([]), encoding="utf-8")
    with pytest.raises(VocabError, match="JSON object"):
        load_vocab(path)
