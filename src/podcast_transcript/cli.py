"""Command-line interface for ``podcast-transcript``.

The CLI exposes two subcommands:

- ``download`` — fetch a podcast MP3 from a direct URL.
- ``transcribe`` — run Whisper on a local audio file.

It is wired up via ``[project.scripts]`` in ``pyproject.toml`` so installing
the package puts a ``podcast-transcript`` executable on ``$PATH``.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from . import __version__
from .download import (
    DEFAULT_TIMEOUT_SECONDS,
    DownloadError,
    download_podcast,
)
from .transcribe import (
    DEFAULT_LANGUAGE,
    DEFAULT_MODEL,
    TranscriptionError,
    transcribe_audio,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

logger = logging.getLogger("podcast_transcript")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="podcast-transcript",
        description=("Download podcast audio and transcribe it locally with OpenAI Whisper."),
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )

    subparsers = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")

    download_parser = subparsers.add_parser(
        "download",
        help="Download a podcast MP3 from a direct URL.",
    )
    download_parser.add_argument("url", help="Direct http(s) URL to the audio file.")
    download_parser.add_argument(
        "stem",
        help="Output filename stem (the .mp3 extension is appended).",
    )
    download_parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(),
        help="Directory to save the file in (default: current directory).",
    )
    download_parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT_SECONDS,
        help=f"Request timeout in seconds (default: {DEFAULT_TIMEOUT_SECONDS}).",
    )

    transcribe_parser = subparsers.add_parser(
        "transcribe",
        help="Transcribe a local audio file with Whisper.",
    )
    transcribe_parser.add_argument(
        "audio_file",
        type=Path,
        help="Path to a local audio file.",
    )
    transcribe_parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Whisper model name (default: {DEFAULT_MODEL}).",
    )
    transcribe_parser.add_argument(
        "--language",
        default=DEFAULT_LANGUAGE,
        help=(
            f"Language code (default: {DEFAULT_LANGUAGE}). "
            "Pass an empty string to let Whisper autodetect."
        ),
    )
    transcribe_parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("transcripts"),
        help="Directory to write transcript outputs (default: ./transcripts).",
    )

    return parser


def _configure_logging(*, verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def _run_download(args: argparse.Namespace) -> int:
    output_path = args.output_dir / f"{args.stem}.mp3"
    try:
        result_path = download_podcast(
            args.url,
            output_path,
            timeout=args.timeout,
        )
    except (DownloadError, ValueError) as exc:
        logger.error("download failed: %s", exc)
        return 2
    logger.info("downloaded to %s", result_path)
    return 0


def _run_transcribe(args: argparse.Namespace) -> int:
    try:
        transcribe_audio(
            args.audio_file,
            model_name=args.model,
            language=args.language,
            output_dir=args.output_dir,
        )
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        return 2
    except (TranscriptionError, ValueError) as exc:
        logger.error("transcription failed: %s", exc)
        return 2
    logger.info("transcripts written to %s", args.output_dir)
    return 0


_COMMANDS: dict[str, Callable[[argparse.Namespace], int]] = {
    "download": _run_download,
    "transcribe": _run_transcribe,
}


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for the ``podcast-transcript`` console script."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    _configure_logging(verbose=args.verbose)
    # ``required=True`` on the subparsers makes argparse reject missing/unknown
    # commands before we get here, so the lookup is total.
    return _COMMANDS[args.command](args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
