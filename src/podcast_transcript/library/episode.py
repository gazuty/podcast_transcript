"""Schema for a podcast library episode record.

Each ``Episode`` is one line in ``podcast-library/index/episodes.jsonl`` and
serves as the source of truth — the four ``by-*.md`` indexes are derived
from it, and the summary/transcript files on disk are pointed to by it.

Validation rules are intentionally strict (we'd rather catch a typo at
ingest time than discover it weeks later via a missing index entry):

- ``id`` must follow ``<podcast-slug>__<YYYY-MM-DD>__<title-slug>``.
- ``pub_date`` parses as ISO ``YYYY-MM-DD``.
- ``transcript.source`` is one of a known set.
- ``summary.qc_status`` is one of ``passed`` / ``flagged`` / ``failed``.
- ``checksum`` is a SHA-256 hex digest (64 lowercase hex chars).

The dataclass round-trips losslessly through JSON via :meth:`to_dict` /
:meth:`from_dict`; we don't use :mod:`dataclasses.asdict` directly because
we want to drop fields with their default value to keep the JSONL line
compact and diff-friendly.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

__all__ = [
    "ID_PATTERN",
    "SLUG_PATTERN",
    "VALID_QC_STATUSES",
    "VALID_TRANSCRIPT_SOURCES",
    "Episode",
    "EpisodeValidationError",
    "SourceUrls",
    "SummaryRef",
    "TranscriptRef",
    "compute_transcript_checksum",
    "make_episode_id",
    "slugify",
]


# ---------------------------------------------------------------------------
# Regex / enum constants
# ---------------------------------------------------------------------------


SLUG_PATTERN = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
# <podcast-slug>__<YYYY-MM-DD>__<title-slug>
ID_PATTERN = re.compile(
    r"^[a-z0-9]+(?:-[a-z0-9]+)*__\d{4}-\d{2}-\d{2}__[a-z0-9]+(?:-[a-z0-9]+)*$",
)
_CHECKSUM_PATTERN = re.compile(r"^[0-9a-f]{64}$")
VALID_TRANSCRIPT_SOURCES: frozenset[str] = frozenset(
    {"whisper", "official", "youtube_captions", "rss", "page"},
)
VALID_QC_STATUSES: frozenset[str] = frozenset({"passed", "flagged", "failed"})


class EpisodeValidationError(ValueError):
    """Raised when an :class:`Episode` fails its schema invariants."""


# ---------------------------------------------------------------------------
# Slug / id helpers
# ---------------------------------------------------------------------------


_SLUG_STRIP = re.compile(r"[^a-z0-9]+")


def slugify(value: str) -> str:
    """Lowercase + collapse non-alphanumerics into single hyphens.

    Aggressively conservative: anything that isn't ``[a-z0-9]`` is treated
    as a separator. Empty input raises (we never want a degenerate slug
    sneaking into an ``id``).
    """
    if not value or not value.strip():
        raise EpisodeValidationError("cannot slugify empty string")
    lowered = value.lower()
    collapsed = _SLUG_STRIP.sub("-", lowered).strip("-")
    if not collapsed:
        raise EpisodeValidationError(f"slug for {value!r} reduced to empty string")
    return collapsed


def make_episode_id(*, podcast_slug: str, pub_date: str, title: str) -> str:
    """Build the canonical ``<podcast>__<date>__<title>`` id.

    Each component is independently slugified except *pub_date*, which is
    validated as ISO ``YYYY-MM-DD`` and inserted verbatim — using the
    actual date in the id makes by-date sorting a string-sort.
    """
    _parse_iso_date(pub_date)  # raises on bad format
    return f"{slugify(podcast_slug)}__{pub_date}__{slugify(title)}"


def compute_transcript_checksum(path: Path) -> str:
    """SHA-256 hex digest of *path* in 64 KiB chunks."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(64 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Nested record types
# ---------------------------------------------------------------------------


@dataclass
class SourceUrls:
    """URLs that were combined to fetch this episode.

    All fields are optional — most episodes won't have all four. We keep
    them on the record so the provenance is auditable later.
    """

    rss_item: str | None = None
    audio: str | None = None
    episode_page: str | None = None
    youtube: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if v is not None}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SourceUrls:
        allowed = {"rss_item", "audio", "episode_page", "youtube"}
        unknown = set(data) - allowed
        if unknown:
            raise EpisodeValidationError(f"unknown source_urls fields: {sorted(unknown)}")
        return cls(**{k: data.get(k) for k in allowed})


@dataclass
class TranscriptRef:
    """Pointer to the transcript file plus provenance metadata."""

    path: str
    source: str  # one of VALID_TRANSCRIPT_SOURCES
    model: str | None = None  # only meaningful when source == "whisper"
    has_timestamps: bool = False

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "path": self.path,
            "source": self.source,
            "has_timestamps": self.has_timestamps,
        }
        if self.model is not None:
            out["model"] = self.model
        return out

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TranscriptRef:
        return cls(
            path=data["path"],
            source=data["source"],
            model=data.get("model"),
            has_timestamps=bool(data.get("has_timestamps", False)),
        )


@dataclass
class SummaryRef:
    """Pointer to the summary file plus QC outcome."""

    path: str
    generated_at: str  # ISO 8601 UTC timestamp
    model: str
    qc_status: str  # one of VALID_QC_STATUSES
    qc_notes_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "path": self.path,
            "generated_at": self.generated_at,
            "model": self.model,
            "qc_status": self.qc_status,
        }
        if self.qc_notes_path is not None:
            out["qc_notes_path"] = self.qc_notes_path
        return out

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SummaryRef:
        return cls(
            path=data["path"],
            generated_at=data["generated_at"],
            model=data["model"],
            qc_status=data["qc_status"],
            qc_notes_path=data.get("qc_notes_path"),
        )


# ---------------------------------------------------------------------------
# Episode
# ---------------------------------------------------------------------------


@dataclass
class Episode:
    """One row of ``podcast-library/index/episodes.jsonl``."""

    id: str
    podcast: str
    podcast_slug: str
    episode_title: str
    pub_date: str
    transcript: TranscriptRef
    summary: SummaryRef
    ingested_at: str
    checksum: str

    # Optional / list-valued fields after the required block.
    episode_number: int | None = None
    duration_seconds: int | None = None
    host: str | None = None
    guests: list[str] = field(default_factory=list)
    speakers: list[str] = field(default_factory=list)
    topics: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    series: str | None = None
    series_part: int | None = None
    source_urls: SourceUrls = field(default_factory=SourceUrls)

    # Pending-vocab tracking. When the summariser proposes a topic or
    # speaker that isn't canonical (and has no alias), it lands here so
    # `pending-vocab.md` can surface it for review.
    pending_topics: list[str] = field(default_factory=list)
    pending_speakers: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-safe dict, dropping empty optionals."""
        out: dict[str, Any] = {
            "id": self.id,
            "podcast": self.podcast,
            "podcast_slug": self.podcast_slug,
            "episode_title": self.episode_title,
            "pub_date": self.pub_date,
        }
        if self.episode_number is not None:
            out["episode_number"] = self.episode_number
        if self.duration_seconds is not None:
            out["duration_seconds"] = self.duration_seconds
        if self.host is not None:
            out["host"] = self.host
        if self.guests:
            out["guests"] = list(self.guests)
        if self.speakers:
            out["speakers"] = list(self.speakers)
        if self.topics:
            out["topics"] = list(self.topics)
        if self.tags:
            out["tags"] = list(self.tags)
        if self.series is not None:
            out["series"] = self.series
        if self.series_part is not None:
            out["series_part"] = self.series_part
        source_urls = self.source_urls.to_dict()
        if source_urls:
            out["source_urls"] = source_urls
        out["transcript"] = self.transcript.to_dict()
        out["summary"] = self.summary.to_dict()
        out["ingested_at"] = self.ingested_at
        out["checksum"] = self.checksum
        if self.pending_topics:
            out["pending_topics"] = list(self.pending_topics)
        if self.pending_speakers:
            out["pending_speakers"] = list(self.pending_speakers)
        return out

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Episode:
        allowed = {
            "id",
            "podcast",
            "podcast_slug",
            "episode_title",
            "episode_number",
            "pub_date",
            "duration_seconds",
            "host",
            "guests",
            "speakers",
            "topics",
            "tags",
            "series",
            "series_part",
            "source_urls",
            "transcript",
            "summary",
            "ingested_at",
            "checksum",
            "pending_topics",
            "pending_speakers",
        }
        unknown = set(data) - allowed
        if unknown:
            raise EpisodeValidationError(f"unknown episode fields: {sorted(unknown)}")
        return cls(
            id=data["id"],
            podcast=data["podcast"],
            podcast_slug=data["podcast_slug"],
            episode_title=data["episode_title"],
            episode_number=data.get("episode_number"),
            pub_date=data["pub_date"],
            duration_seconds=data.get("duration_seconds"),
            host=data.get("host"),
            guests=list(data.get("guests", [])),
            speakers=list(data.get("speakers", [])),
            topics=list(data.get("topics", [])),
            tags=list(data.get("tags", [])),
            series=data.get("series"),
            series_part=data.get("series_part"),
            source_urls=SourceUrls.from_dict(data.get("source_urls", {})),
            transcript=TranscriptRef.from_dict(data["transcript"]),
            summary=SummaryRef.from_dict(data["summary"]),
            ingested_at=data["ingested_at"],
            checksum=data["checksum"],
            pending_topics=list(data.get("pending_topics", [])),
            pending_speakers=list(data.get("pending_speakers", [])),
        )

    def validate(self) -> None:
        """Raise :class:`EpisodeValidationError` if invariants are violated."""
        if not ID_PATTERN.match(self.id):
            raise EpisodeValidationError(
                f"id {self.id!r} must be <slug>__YYYY-MM-DD__<slug>",
            )
        if not SLUG_PATTERN.match(self.podcast_slug):
            raise EpisodeValidationError(
                f"podcast_slug {self.podcast_slug!r} must be lowercase kebab-case",
            )
        if not self.podcast.strip():
            raise EpisodeValidationError("podcast must be non-empty")
        if not self.episode_title.strip():
            raise EpisodeValidationError("episode_title must be non-empty")
        _parse_iso_date(self.pub_date)
        _parse_iso_timestamp(self.ingested_at)
        _parse_iso_timestamp(self.summary.generated_at)
        if self.transcript.source not in VALID_TRANSCRIPT_SOURCES:
            raise EpisodeValidationError(
                f"transcript.source {self.transcript.source!r} not in "
                f"{sorted(VALID_TRANSCRIPT_SOURCES)}",
            )
        if self.summary.qc_status not in VALID_QC_STATUSES:
            raise EpisodeValidationError(
                f"summary.qc_status {self.summary.qc_status!r} not in {sorted(VALID_QC_STATUSES)}",
            )
        if not _CHECKSUM_PATTERN.match(self.checksum):
            raise EpisodeValidationError(
                f"checksum {self.checksum!r} must be a 64-char lowercase hex SHA-256",
            )
        if self.episode_number is not None and self.episode_number < 0:
            raise EpisodeValidationError(
                f"episode_number must be >= 0, got {self.episode_number}",
            )
        if self.duration_seconds is not None and self.duration_seconds < 0:
            raise EpisodeValidationError(
                f"duration_seconds must be >= 0, got {self.duration_seconds}",
            )
        if self.series_part is not None and self.series_part < 1:
            raise EpisodeValidationError(
                f"series_part must be >= 1, got {self.series_part}",
            )


# ---------------------------------------------------------------------------
# Internal parse helpers
# ---------------------------------------------------------------------------


def _parse_iso_date(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except (TypeError, ValueError) as exc:
        raise EpisodeValidationError(
            f"pub_date {value!r} must be ISO 8601 YYYY-MM-DD",
        ) from exc


def _parse_iso_timestamp(value: str) -> datetime:
    # ``datetime.fromisoformat`` accepts trailing ``Z`` only on 3.11+; we
    # normalise just in case some caller passed a Z-suffixed string.
    candidate = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        parsed = datetime.fromisoformat(candidate)
    except (TypeError, ValueError) as exc:
        raise EpisodeValidationError(
            f"timestamp {value!r} must be ISO 8601 (e.g. 2026-05-23T17:00:00+00:00)",
        ) from exc
    # The fields these feed (generated_at, ingested_at) are documented as
    # UTC; a naive timestamp would silently shift meaning across machines.
    if parsed.tzinfo is None:
        raise EpisodeValidationError(
            f"timestamp {value!r} must include a UTC offset (e.g. +00:00)",
        )
    return parsed
