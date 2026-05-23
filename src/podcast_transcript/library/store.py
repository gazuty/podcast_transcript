"""JSONL store for ``podcast-library/index/episodes.jsonl``.

One JSON object per line, keyed on :attr:`Episode.id`. The store is small
(thousands of episodes at most) so we keep it as a flat file — no SQLite,
no locking. Writes are atomic via ``.part`` rename so a crashed ingest can
never corrupt the spine.

All public functions accept a :class:`pathlib.Path` to the JSONL file so
tests can point at a tmp file.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from .episode import Episode, EpisodeValidationError

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

__all__ = [
    "load_all",
    "load_index",
    "save_all",
    "upsert",
]


def load_all(path: Path) -> list[Episode]:
    """Read every line of *path* into an :class:`Episode`.

    Blank lines are tolerated (people occasionally leave a trailing
    newline). An empty file returns ``[]``. Each line is validated; a
    malformed row aborts the whole load so callers don't silently lose
    records.
    """
    if not path.is_file():
        return []
    episodes: list[Episode] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError as exc:
                raise EpisodeValidationError(
                    f"{path}:{line_no}: invalid JSON: {exc.msg}",
                ) from exc
            episode = Episode.from_dict(data)
            episode.validate()
            episodes.append(episode)
    return episodes


def load_index(path: Path) -> dict[str, Episode]:
    """Same as :func:`load_all` but keyed on :attr:`Episode.id`.

    Raises if two rows share an id — that's a corruption we want to
    notice immediately.
    """
    index: dict[str, Episode] = {}
    for episode in load_all(path):
        if episode.id in index:
            raise EpisodeValidationError(
                f"duplicate episode id in {path}: {episode.id!r}",
            )
        index[episode.id] = episode
    return index


def save_all(path: Path, episodes: Iterable[Episode]) -> None:
    """Atomically rewrite *path* with one row per *episode*.

    Each episode is validated before writing — the JSONL is supposed to be
    the source of truth, so we never let an invalid record reach disk.
    Sorted by id for stable diffs.
    """
    rows = sorted(episodes, key=lambda e: e.id)
    for episode in rows:
        episode.validate()
    path.parent.mkdir(parents=True, exist_ok=True)
    part_path = path.with_suffix(path.suffix + ".part")
    with part_path.open("w", encoding="utf-8") as f:
        for episode in rows:
            json.dump(episode.to_dict(), f, ensure_ascii=False, sort_keys=False)
            f.write("\n")
    part_path.replace(path)


def upsert(path: Path, episode: Episode) -> bool:
    """Insert *episode* or replace the existing row with the same id.

    Returns ``True`` if a previous row was replaced, ``False`` if this was
    a fresh insert. The whole file is rewritten — fine at the scale of a
    personal podcast library, and avoids any concurrent-access subtleties.
    """
    episode.validate()
    index = load_index(path)
    replaced = episode.id in index
    index[episode.id] = episode
    save_all(path, index.values())
    return replaced
