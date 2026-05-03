"""Command-line interface for ``podcast-transcript``.

The CLI exposes:

- ``download`` — fetch a podcast MP3 from a direct URL.
- ``transcribe`` — run Whisper on a local audio file.
- ``clean`` — apply rule-based cleanup to a Whisper transcript.
- ``add-correction`` — append/update an entry in the per-user corrections file.
- ``run`` — end-to-end: download (or pick from RSS) → transcribe → clean.

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
from .clean import (
    DEFAULT_REFLOW_SENTENCES,
    CleanStats,
    CorrectionsFile,
    clean_transcript,
    load_corrections_file,
    load_default_corrections_file,
    merge_corrections_files,
)
from .corrections_user import (
    USER_CORRECTIONS_PATH,
    PackNotFoundError,
    load_corrections_pack,
    upsert_correction,
)
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

    clean_parser = subparsers.add_parser(
        "clean",
        help="Apply rule-based cleanup to a Whisper transcript.",
    )
    clean_parser.add_argument(
        "input_file",
        type=Path,
        help="Path to a transcript .txt file produced by `transcribe`.",
    )
    output_group = clean_parser.add_mutually_exclusive_group()
    output_group.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Where to write the cleaned transcript. Defaults to "
            "`<input>.clean.txt` next to the input file."
        ),
    )
    output_group.add_argument(
        "--in-place",
        action="store_true",
        help="Overwrite the input file with the cleaned output.",
    )
    _add_corrections_args(clean_parser)
    clean_parser.add_argument(
        "--reflow",
        action="store_true",
        help="Reflow per-segment lines into prose paragraphs.",
    )
    clean_parser.add_argument(
        "--sentences-per-paragraph",
        type=int,
        default=DEFAULT_REFLOW_SENTENCES,
        help=(f"With --reflow, sentences per paragraph (default: {DEFAULT_REFLOW_SENTENCES})."),
    )
    clean_parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress the cleanup summary line.",
    )

    add_corr_parser = subparsers.add_parser(
        "add-correction",
        help="Append or update an entry in the per-user corrections file.",
    )
    add_corr_parser.add_argument(
        "wrong",
        help="The mistranscribed term (matched word-bounded, case-sensitive).",
    )
    add_corr_parser.add_argument(
        "right",
        nargs="?",
        default="",
        help=(
            "The correct term. Required for confident corrections. Optional "
            "with --uncertain (an empty value flags the term without a "
            "suggestion)."
        ),
    )
    add_corr_parser.add_argument(
        "--uncertain",
        action="store_true",
        help="Write to the [uncertain] table — wraps matches as [?: x → y] in output.",
    )
    add_corr_parser.add_argument(
        "--dict",
        dest="dict_path",
        type=Path,
        default=None,
        help=(
            f"Destination TOML file (default: {USER_CORRECTIONS_PATH}). "
            "The chosen file is created if it does not exist."
        ),
    )

    run_parser = subparsers.add_parser(
        "run",
        help="End-to-end: download (or pick from RSS) → transcribe → clean.",
    )
    source_group = run_parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--url", help="Direct http(s) URL to a podcast MP3.")
    source_group.add_argument("--rss", help="RSS feed URL.")
    run_parser.add_argument(
        "--episode-regex",
        help="With --rss, regex matched against episode <title> (first match wins).",
    )
    run_parser.add_argument(
        "--episode-index",
        type=int,
        help="With --rss, 0-based index into the feed (newest=0).",
    )
    run_parser.add_argument(
        "--slug",
        required=True,
        help="Output filename stem (.mp3 / .txt / _clean.txt suffixes are appended).",
    )
    run_parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("transcripts"),
        help="Directory to write transcript outputs (default: ./transcripts).",
    )
    run_parser.add_argument(
        "--audio-dir",
        type=Path,
        default=Path(),
        help="Directory to write the downloaded MP3 (default: current directory).",
    )
    run_parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Whisper model name (default: {DEFAULT_MODEL}).",
    )
    run_parser.add_argument(
        "--language",
        default=DEFAULT_LANGUAGE,
        help=f"Language code (default: {DEFAULT_LANGUAGE}). Empty string = autodetect.",
    )
    run_parser.add_argument(
        "--strip-before",
        action="append",
        default=[],
        metavar="REGEX",
        help="Drop everything up to and including the first matching line. Repeatable.",
    )
    run_parser.add_argument(
        "--strip-after",
        action="append",
        default=[],
        metavar="REGEX",
        help=(
            "Drop everything from the last matching line onward (matched only in the "
            "tail half of the transcript to avoid mid-conversation false positives). "
            "Repeatable."
        ),
    )
    run_parser.add_argument(
        "--reflow",
        action="store_true",
        help="Reflow per-segment lines into prose paragraphs.",
    )
    run_parser.add_argument(
        "--sentences-per-paragraph",
        type=int,
        default=DEFAULT_REFLOW_SENTENCES,
        help=f"With --reflow, sentences per paragraph (default: {DEFAULT_REFLOW_SENTENCES}).",
    )
    _add_corrections_args(run_parser)
    run_parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT_SECONDS,
        help=f"HTTP timeout in seconds (default: {DEFAULT_TIMEOUT_SECONDS}).",
    )

    return parser


def _add_corrections_args(parser: argparse.ArgumentParser) -> None:
    """Shared corrections-layering flags for ``clean`` and ``run``."""
    parser.add_argument(
        "--corrections",
        action="append",
        type=Path,
        default=[],
        metavar="PATH",
        help=(
            "Additional corrections TOML file. Repeatable; later files override "
            "earlier ones on key conflicts."
        ),
    )
    parser.add_argument(
        "--corrections-pack",
        action="append",
        default=[],
        metavar="NAME",
        help=(
            "Bundled corrections pack name (e.g. `razib_khan` → "
            "data/corrections.razib_khan.toml). Repeatable."
        ),
    )
    parser.add_argument(
        "--no-default-corrections",
        action="store_true",
        help="Skip the bundled `corrections.toml` defaults.",
    )
    parser.add_argument(
        "--no-user-corrections",
        action="store_true",
        help=f"Skip the per-user corrections file ({USER_CORRECTIONS_PATH}).",
    )


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


def _resolve_corrections(args: argparse.Namespace) -> CorrectionsFile:
    """Build the merged corrections + uncertain dictionaries from CLI flags.

    Layering order (later wins on key conflicts):
        defaults → bundled packs → user file → explicit --corrections paths.
    """
    files: list[CorrectionsFile] = []
    if not args.no_default_corrections:
        files.append(load_default_corrections_file())
    for pack_name in args.corrections_pack:
        files.append(load_corrections_pack(pack_name))
    if not args.no_user_corrections and USER_CORRECTIONS_PATH.is_file():
        files.append(load_corrections_file(USER_CORRECTIONS_PATH))
    for explicit in args.corrections:
        files.append(load_corrections_file(explicit))
    return merge_corrections_files(files)


def _resolve_clean_output(args: argparse.Namespace) -> Path:
    input_path: Path = args.input_file
    if args.in_place:
        return input_path
    explicit: Path | None = args.output
    if explicit is not None:
        return explicit
    return input_path.with_suffix(input_path.suffix + ".clean")


def _log_clean_summary(
    *,
    src: Path,
    dst: Path,
    stats: CleanStats,
) -> None:
    logger.info(
        "cleaned %s → %s (lines %d → %d, loops collapsed: %d, outro lines stripped: %d, "
        "corrections: %d, uncertain: %d%s)",
        src,
        dst,
        stats.lines_in,
        stats.lines_out,
        stats.loops_collapsed,
        stats.outro_lines_stripped,
        stats.corrections_applied,
        stats.uncertain_applied,
        ", reflowed" if stats.reflowed else "",
    )
    if stats.preview_cut_reason is not None:
        logger.warning(
            "source MP3 may be a preview cut — tail contains %s",
            stats.preview_cut_reason,
        )


def _run_clean(args: argparse.Namespace) -> int:
    input_path: Path = args.input_file
    if not input_path.is_file():
        logger.error("input file not found: %s", input_path)
        return 2

    try:
        corrections_file = _resolve_corrections(args)
    except (OSError, ValueError, PackNotFoundError) as exc:
        logger.error("could not load corrections: %s", exc)
        return 2

    text = input_path.read_text(encoding="utf-8")
    cleaned, stats = clean_transcript(
        text,
        corrections=corrections_file.corrections,
        uncertain=corrections_file.uncertain,
        reflow=args.reflow,
        sentences_per_paragraph=args.sentences_per_paragraph,
    )

    output_path = _resolve_clean_output(args)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(cleaned, encoding="utf-8")

    if not args.quiet:
        _log_clean_summary(src=input_path, dst=output_path, stats=stats)
    return 0


def _run_add_correction(args: argparse.Namespace) -> int:
    try:
        target = upsert_correction(
            args.wrong,
            args.right,
            uncertain=args.uncertain,
            path=args.dict_path,
        )
    except ValueError as exc:
        logger.error("%s", exc)
        return 2
    table = "uncertain" if args.uncertain else "corrections"
    if args.uncertain and not args.right:
        logger.info("flagged %r in [%s] of %s", args.wrong, table, target)
    else:
        logger.info("set %r → %r in [%s] of %s", args.wrong, args.right, table, target)
    return 0


def _run_run(args: argparse.Namespace) -> int:
    # Imported here so the heavy pipeline module (which imports feed.py and
    # transcribe.py's deps lazily) is only paid for when actually invoked.
    from .pipeline import PipelineError, run_pipeline

    try:
        corrections_file = _resolve_corrections(args)
    except (OSError, ValueError, PackNotFoundError) as exc:
        logger.error("could not load corrections: %s", exc)
        return 2

    try:
        run_pipeline(
            url=args.url,
            rss_url=args.rss,
            episode_regex=args.episode_regex,
            episode_index=args.episode_index,
            slug=args.slug,
            audio_dir=args.audio_dir,
            transcripts_dir=args.output_dir,
            model=args.model,
            language=args.language,
            corrections=corrections_file,
            strip_before=args.strip_before,
            strip_after=args.strip_after,
            reflow=args.reflow,
            sentences_per_paragraph=args.sentences_per_paragraph,
            timeout=args.timeout,
        )
    except PipelineError as exc:
        logger.error("%s", exc)
        return 2
    except (DownloadError, TranscriptionError, FileNotFoundError, ValueError) as exc:
        logger.error("%s", exc)
        return 2
    return 0


_COMMANDS: dict[str, Callable[[argparse.Namespace], int]] = {
    "download": _run_download,
    "transcribe": _run_transcribe,
    "clean": _run_clean,
    "add-correction": _run_add_correction,
    "run": _run_run,
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
