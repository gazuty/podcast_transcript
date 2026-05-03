"""User-managed corrections file plus bundled correction packs.

The ``clean`` and ``run`` subcommands layer corrections in this order:

1. Bundled defaults (``data/corrections.toml``).
2. Bundled packs requested via ``--corrections-pack`` (e.g. ``razib_khan``
   resolves to ``data/corrections.razib_khan.toml``).
3. User file at :data:`USER_CORRECTIONS_PATH` if it exists.
4. Explicit ``--corrections`` paths.

Later layers override earlier ones on key conflicts.

The user file is the destination for ``podcast-transcript add-correction``,
so it gets rewritten whenever the user discovers a new mistranscription.
:func:`upsert_correction` does an in-place write that preserves existing
entries but does not preserve hand-written comments — comments belong in
the bundled packs, not the user file.
"""

from __future__ import annotations

import os
from importlib import resources
from pathlib import Path

from .clean import CorrectionsFile, load_corrections_file

__all__ = [
    "USER_CORRECTIONS_PATH",
    "PackNotFoundError",
    "load_corrections_pack",
    "load_user_corrections",
    "upsert_correction",
    "user_corrections_path",
]


class PackNotFoundError(Exception):
    """Raised when ``--corrections-pack NAME`` cannot be resolved to a bundled file."""


def user_corrections_path() -> Path:
    """Resolve the per-user corrections file under ``$XDG_CONFIG_HOME``.

    Falls back to ``~/.config`` when ``XDG_CONFIG_HOME`` is unset, matching
    the XDG Base Directory spec.
    """
    base = os.environ.get("XDG_CONFIG_HOME") or str(Path.home() / ".config")
    return Path(base) / "podcast_transcript" / "corrections.toml"


# Cached at import; tests that need to override it should patch the
# attribute rather than mutate it.
USER_CORRECTIONS_PATH: Path = user_corrections_path()


def load_corrections_pack(name: str) -> CorrectionsFile:
    """Load a bundled corrections pack by short name.

    ``load_corrections_pack("razib_khan")`` resolves to
    ``podcast_transcript/data/corrections.razib_khan.toml``.
    """
    if not name or "/" in name or "\\" in name or name.startswith("."):
        raise PackNotFoundError(f"invalid pack name: {name!r}")
    filename = f"corrections.{name}.toml"
    ref = resources.files("podcast_transcript.data").joinpath(filename)
    if not ref.is_file():
        raise PackNotFoundError(f"no bundled corrections pack named {name!r}")
    with resources.as_file(ref) as concrete_path:
        return load_corrections_file(concrete_path)


def load_user_corrections(path: Path | None = None) -> CorrectionsFile:
    """Load the user file if it exists, else return an empty CorrectionsFile."""
    target = path if path is not None else USER_CORRECTIONS_PATH
    if not target.is_file():
        return CorrectionsFile()
    return load_corrections_file(target)


def _quote_toml_string(value: str) -> str:
    """Escape *value* for a TOML basic string literal.

    TOML basic strings allow ``\\`` and ``"`` escapes; everything else can
    pass through. We do not attempt to emit literal strings (single quotes)
    because we cannot represent values containing single quotes that way.
    """
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _serialize(file_data: CorrectionsFile) -> str:
    parts: list[str] = ["[corrections]\n"]
    for k in sorted(file_data.corrections):
        parts.append(f"{_quote_toml_string(k)} = {_quote_toml_string(file_data.corrections[k])}\n")
    parts.append("\n[uncertain]\n")
    for k in sorted(file_data.uncertain):
        parts.append(f"{_quote_toml_string(k)} = {_quote_toml_string(file_data.uncertain[k])}\n")
    return "".join(parts)


def upsert_correction(
    wrong: str,
    right: str,
    *,
    uncertain: bool = False,
    path: Path | None = None,
) -> Path:
    """Add or update an entry in the user corrections file.

    Creates the parent directory and the file if missing. If *wrong* already
    exists in the chosen table, its value is overwritten. Returns the
    resolved path.

    ``uncertain=True`` writes to the ``[uncertain]`` table and accepts an
    empty *right* (flag-only entry). With ``uncertain=False`` an empty
    *right* is rejected — a confident replacement must be non-empty.
    """
    if not wrong:
        raise ValueError("wrong term must be non-empty")
    if not uncertain and not right:
        raise ValueError("right term must be non-empty unless --uncertain is set")

    target = path if path is not None else USER_CORRECTIONS_PATH
    target.parent.mkdir(parents=True, exist_ok=True)

    existing = load_user_corrections(target) if target.is_file() else CorrectionsFile()
    if uncertain:
        existing.uncertain[wrong] = right
        # If a confident entry shadowed it before, remove that — the term is
        # being demoted to uncertain on purpose.
        existing.corrections.pop(wrong, None)
    else:
        existing.corrections[wrong] = right
        existing.uncertain.pop(wrong, None)

    target.write_text(_serialize(existing), encoding="utf-8")
    return target
