"""Tests for :mod:`podcast_transcript.transcribe`."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from podcast_transcript.transcribe import (
    DEFAULT_LANGUAGE,
    DEFAULT_MODEL,
    OUTPUT_FORMATS,
    transcribe_audio,
)

if TYPE_CHECKING:
    from pathlib import Path
    from unittest.mock import MagicMock


def _make_audio_file(tmp_path: Path) -> Path:
    audio = tmp_path / "episode.mp3"
    audio.write_bytes(b"ID3\x04\x00" + b"\x00" * 32)
    return audio


def test_transcribe_invokes_whisper_with_defaults(tmp_path: Path, fake_whisper: MagicMock) -> None:
    audio = _make_audio_file(tmp_path)
    output_dir = tmp_path / "out"
    fake_result = {"text": "hello world", "segments": []}
    fake_whisper.load_model.return_value.transcribe.return_value = fake_result

    result = transcribe_audio(audio, output_dir=output_dir)

    assert result is fake_result
    fake_whisper.load_model.assert_called_once_with(DEFAULT_MODEL)
    fake_whisper.load_model.return_value.transcribe.assert_called_once_with(
        str(audio), language=DEFAULT_LANGUAGE
    )
    assert output_dir.is_dir()


def test_transcribe_calls_writer_for_each_format(tmp_path: Path, fake_whisper: MagicMock) -> None:
    audio = _make_audio_file(tmp_path)
    output_dir = tmp_path / "out"
    fake_whisper.load_model.return_value.transcribe.return_value = {"text": "x", "segments": []}

    transcribe_audio(audio, output_dir=output_dir)

    formats_called = [call.args[0] for call in fake_whisper.utils.get_writer.call_args_list]
    assert formats_called == list(OUTPUT_FORMATS)


def test_transcribe_passes_custom_model_and_language(
    tmp_path: Path, fake_whisper: MagicMock
) -> None:
    audio = _make_audio_file(tmp_path)
    fake_whisper.load_model.return_value.transcribe.return_value = {"text": "x", "segments": []}

    transcribe_audio(
        audio,
        model_name="turbo",
        language="fr",
        output_dir=tmp_path / "out",
    )

    fake_whisper.load_model.assert_called_once_with("turbo")
    fake_whisper.load_model.return_value.transcribe.assert_called_once_with(
        str(audio), language="fr"
    )


def test_transcribe_omits_language_when_empty(tmp_path: Path, fake_whisper: MagicMock) -> None:
    audio = _make_audio_file(tmp_path)
    fake_whisper.load_model.return_value.transcribe.return_value = {"text": "x", "segments": []}

    transcribe_audio(audio, language="", output_dir=tmp_path / "out")

    # Empty language should be treated as autodetect: don't forward it.
    fake_whisper.load_model.return_value.transcribe.assert_called_once_with(str(audio))


def test_transcribe_missing_audio_file_raises(tmp_path: Path, fake_whisper: MagicMock) -> None:
    with pytest.raises(FileNotFoundError):
        transcribe_audio(tmp_path / "nope.mp3", output_dir=tmp_path / "out")
    fake_whisper.load_model.assert_not_called()


def test_transcribe_unsupported_format_raises(tmp_path: Path, fake_whisper: MagicMock) -> None:
    audio = _make_audio_file(tmp_path)

    with pytest.raises(ValueError, match="Unsupported"):
        transcribe_audio(audio, output_dir=tmp_path / "out", output_formats=["pdf"])
    fake_whisper.load_model.assert_not_called()
