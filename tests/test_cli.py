"""Tests for :mod:`podcast_transcript.cli`."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from podcast_transcript import __version__
from podcast_transcript.cli import main
from podcast_transcript.download import DownloadError

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path
    from unittest.mock import MagicMock

    from .conftest import Responder


AUDIO_BODY = b"ID3\x04\x00" + b"\x00" * 1024


def _audio_responder() -> Responder:
    def respond(_path: str) -> tuple[int, dict[str, str], bytes]:
        return (
            200,
            {"Content-Type": "audio/mpeg", "Content-Length": str(len(AUDIO_BODY))},
            AUDIO_BODY,
        )

    return respond


def test_cli_version_flag(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc_info:
        main(["--version"])
    assert exc_info.value.code == 0
    captured = capsys.readouterr()
    assert __version__ in captured.out


def test_cli_no_command_exits_nonzero() -> None:
    with pytest.raises(SystemExit) as exc_info:
        main([])
    assert exc_info.value.code != 0


def test_cli_download_happy_path(tmp_path: Path, http_server: Callable[[Responder], str]) -> None:
    base_url = http_server(_audio_responder())

    rc = main(
        [
            "download",
            f"{base_url}/episode.mp3",
            "show1",
            "--output-dir",
            str(tmp_path),
        ]
    )

    assert rc == 0
    assert (tmp_path / "show1.mp3").read_bytes() == AUDIO_BODY


def test_cli_download_returns_2_on_download_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(*_args: object, **_kwargs: object) -> Path:
        raise DownloadError("simulated failure")

    monkeypatch.setattr("podcast_transcript.cli.download_podcast", boom)

    rc = main(
        [
            "download",
            "https://example.com/x.mp3",
            "show1",
            "--output-dir",
            str(tmp_path),
        ]
    )

    assert rc == 2


def test_cli_transcribe_happy_path(tmp_path: Path, fake_whisper: MagicMock) -> None:
    audio = tmp_path / "show.mp3"
    audio.write_bytes(b"ID3\x04\x00" + b"\x00" * 64)
    fake_whisper.load_model.return_value.transcribe.return_value = {
        "text": "hi",
        "segments": [],
    }
    output_dir = tmp_path / "transcripts"

    rc = main(
        [
            "transcribe",
            str(audio),
            "--model",
            "turbo",
            "--output-dir",
            str(output_dir),
        ]
    )

    assert rc == 0
    fake_whisper.load_model.assert_called_once_with("turbo")
    assert output_dir.is_dir()


def test_cli_transcribe_missing_audio_returns_2(tmp_path: Path, fake_whisper: MagicMock) -> None:
    rc = main(["transcribe", str(tmp_path / "missing.mp3")])
    assert rc == 2
    fake_whisper.load_model.assert_not_called()


def test_cli_clean_writes_default_output(tmp_path: Path) -> None:
    src = tmp_path / "in.txt"
    src.write_text(
        "We use Tehima's D.\nbody line.\nbody line.\nbody line.\nbody line.\n",
        encoding="utf-8",
    )

    rc = main(["clean", str(src)])

    assert rc == 0
    out = tmp_path / "in.txt.clean"
    assert out.is_file()
    cleaned = out.read_text(encoding="utf-8")
    assert "Tajima's D" in cleaned
    assert cleaned.count("body line.") == 1


def test_cli_clean_in_place(tmp_path: Path) -> None:
    src = tmp_path / "in.txt"
    src.write_text("Tehima's D rules.\n", encoding="utf-8")

    rc = main(["clean", str(src), "--in-place"])

    assert rc == 0
    assert src.read_text(encoding="utf-8").startswith("Tajima's D")


def test_cli_clean_missing_input_returns_2(tmp_path: Path) -> None:
    rc = main(["clean", str(tmp_path / "nope.txt")])
    assert rc == 2
