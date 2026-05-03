"""Tests for :mod:`podcast_transcript.cli`."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from podcast_transcript import __version__
from podcast_transcript.cli import main
from podcast_transcript.download import DownloadError

if TYPE_CHECKING:
    from collections.abc import Callable
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


def test_cli_clean_with_corrections_pack(tmp_path: Path) -> None:
    src = tmp_path / "in.txt"
    src.write_text("Razeeb said hello.\nStephen Ghazal joined the team.\n", encoding="utf-8")

    rc = main(
        [
            "clean",
            str(src),
            "--corrections-pack",
            "razib_khan",
            "--no-default-corrections",
        ]
    )
    assert rc == 0
    cleaned = (tmp_path / "in.txt.clean").read_text(encoding="utf-8")
    assert "Razib said hello." in cleaned
    assert "[?: Stephen Ghazal → Stephen Gazal]" in cleaned


def test_cli_clean_unknown_pack_returns_2(tmp_path: Path) -> None:
    src = tmp_path / "in.txt"
    src.write_text("hello\n", encoding="utf-8")
    rc = main(["clean", str(src), "--corrections-pack", "no-such-pack"])
    assert rc == 2


def test_cli_add_correction_writes_user_file(user_corrections_path: Path) -> None:
    rc = main(["add-correction", "Razeeb", "Razib"])
    assert rc == 0
    assert user_corrections_path.is_file()
    content = user_corrections_path.read_text(encoding="utf-8")
    assert '"Razeeb" = "Razib"' in content
    assert "[uncertain]" in content


def test_cli_add_correction_uncertain_with_blank(user_corrections_path: Path) -> None:
    rc = main(["add-correction", "benorephora", "--uncertain"])
    assert rc == 0
    content = user_corrections_path.read_text(encoding="utf-8")
    assert '"benorephora" = ""' in content


def test_cli_add_correction_rejects_blank_confident(user_corrections_path: Path) -> None:
    rc = main(["add-correction", "foo"])
    assert rc == 2


def test_cli_add_correction_demotes_then_promotes(user_corrections_path: Path) -> None:
    main(["add-correction", "foo", "bar"])
    main(["add-correction", "foo", "baz", "--uncertain"])
    content = user_corrections_path.read_text(encoding="utf-8")
    assert '"foo" = "baz"' in content
    # Confident table should no longer carry the entry.
    assert content.count('"foo"') == 1


def test_cli_clean_picks_up_user_file(
    tmp_path: Path,
    user_corrections_path: Path,
) -> None:
    main(["add-correction", "fnord", "FNORD"])
    src = tmp_path / "in.txt"
    src.write_text("the fnord is everywhere.\n", encoding="utf-8")
    rc = main(["clean", str(src)])
    assert rc == 0
    cleaned = (tmp_path / "in.txt.clean").read_text(encoding="utf-8")
    assert "FNORD" in cleaned


def test_cli_run_url_mode(
    tmp_path: Path,
    http_server: Callable[[Responder], str],
    fake_whisper: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base_url = http_server(_audio_responder())
    transcripts_dir = tmp_path / "transcripts"
    transcripts_dir.mkdir()

    def writer_factory(fmt: str, _out_dir: str) -> object:
        def write(_result: object, audio_path: str) -> None:
            stem = Path(audio_path).stem
            (transcripts_dir / f"{stem}.{fmt}").write_text(
                "Razeeb here.\nbody.\n", encoding="utf-8"
            )

        return write

    fake_whisper.utils.get_writer.side_effect = writer_factory
    fake_whisper.load_model.return_value.transcribe.return_value = {"text": "x", "segments": []}

    rc = main(
        [
            "run",
            "--url",
            f"{base_url}/show.mp3",
            "--slug",
            "show1",
            "--audio-dir",
            str(tmp_path),
            "--output-dir",
            str(transcripts_dir),
            "--corrections-pack",
            "razib_khan",
        ]
    )
    assert rc == 0
    cleaned = (transcripts_dir / "show1_clean.txt").read_text(encoding="utf-8")
    assert "Razib here." in cleaned
