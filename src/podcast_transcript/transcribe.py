"""Run OpenAI Whisper on a local audio file and write transcripts to disk.

The :mod:`whisper` package is imported lazily because:

1. It pulls in :mod:`torch`, which is large and slow to install. Keeping
   ``openai-whisper`` as an *optional* extra (``pip install -e '.[whisper]'``)
   means CI can lint, type-check, and unit-test this package without
   downloading torch.
2. Tests can monkey-patch :data:`sys.modules` to inject a fake whisper module
   without ever touching the real one.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

if TYPE_CHECKING:
    from collections.abc import Iterable

__all__ = [
    "DEFAULT_LANGUAGE",
    "DEFAULT_MODEL",
    "OUTPUT_FORMATS",
    "TranscriptionError",
    "TranscriptionResult",
    "transcribe_audio",
]

DEFAULT_MODEL: Final[str] = "large-v3"
DEFAULT_LANGUAGE: Final[str] = "en"

# The set of output formats the whisper CLI's ``--output_format all`` produces.
OUTPUT_FORMATS: Final[tuple[str, ...]] = ("txt", "srt", "vtt", "tsv", "json")

# Whisper returns a dict with a heterogeneous shape. We keep this as an alias
# of ``dict[str, Any]`` rather than a TypedDict because the upstream schema
# evolves and we don't want to fight mypy for every new field.
TranscriptionResult = dict[str, Any]


class TranscriptionError(Exception):
    """Raised when transcription fails."""


def _import_whisper() -> Any:
    """Import the optional :mod:`whisper` dependency, with a useful error."""
    try:
        import whisper
    except ImportError as exc:  # pragma: no cover -- exercised via fake module in tests
        raise TranscriptionError(
            "openai-whisper is not installed. Install with: pip install -e '.[whisper]'",
        ) from exc
    return whisper


def transcribe_audio(
    audio_path: Path | str,
    *,
    model_name: str = DEFAULT_MODEL,
    language: str = DEFAULT_LANGUAGE,
    output_dir: Path | str = Path("transcripts"),
    output_formats: Iterable[str] = OUTPUT_FORMATS,
) -> TranscriptionResult:
    """Transcribe *audio_path* with Whisper and write outputs to *output_dir*.

    Args:
        audio_path: Path to a local audio file (``.mp3``, ``.m4a``, ``.wav``, ...).
        model_name: Whisper model name (e.g. ``large-v3``, ``turbo``, ``base``).
        language: ISO-639-1 language code, or ``None``-equivalent for autodetect
            (whisper accepts an empty string or a real code).
        output_dir: Directory to write transcript files into. Created if missing.
        output_formats: Iterable of formats to write. Each must be one of
            :data:`OUTPUT_FORMATS`.

    Returns:
        The raw whisper result dict (text + segments + language detection info).

    Raises:
        FileNotFoundError: If *audio_path* does not exist.
        ValueError: If any value in *output_formats* is not in
            :data:`OUTPUT_FORMATS`.
        TranscriptionError: If the optional ``openai-whisper`` dependency is not
            installed.
    """
    audio_path = Path(audio_path)
    if not audio_path.is_file():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    requested_formats = tuple(output_formats)
    invalid = [f for f in requested_formats if f not in OUTPUT_FORMATS]
    if invalid:
        raise ValueError(
            f"Unsupported output format(s): {invalid!r}. Supported: {list(OUTPUT_FORMATS)!r}",
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    whisper = _import_whisper()
    model = whisper.load_model(model_name)
    transcribe_kwargs: dict[str, Any] = {}
    if language:
        transcribe_kwargs["language"] = language
    result: TranscriptionResult = model.transcribe(str(audio_path), **transcribe_kwargs)

    # ``whisper.utils.get_writer`` returns a ResultWriter callable that takes
    # ``(result, audio_path)`` and writes to ``output_dir``. Calling it once
    # per format mirrors what ``--output_format all`` does in the CLI.
    from whisper.utils import get_writer

    for fmt in requested_formats:
        writer = get_writer(fmt, str(output_dir))
        writer(result, str(audio_path))

    return result
