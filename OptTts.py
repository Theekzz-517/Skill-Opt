
import logging
import os
import re
import subprocess
from pathlib import Path

import numpy as np

import config
from skills.stt import sanitize_text
from utils.wav_io import load_wav

log = logging.getLogger("tts")

_SINHALA = re.compile(r"[\u0D80-\u0DFF]")


_RE_STARS = re.compile(r"\*+")
_RE_PARENS = re.compile(r"\([^)]*\)")
_RE_SPACES = re.compile(r"\s+")


class TTSError(Exception):
    """Raised when speech synthesis fails."""


def clean_for_speech(text):
    """Removes things that sound bad if read aloud: *emphasis*, (asides), extra spaces."""
    text = _RE_STARS.sub("", text)
    text = _RE_PARENS.sub("", text)
    text = _RE_SPACES.sub(" ", text).strip()
    return text


def pick_language(text):
    return "si" if _SINHALA.search(text) else "en"


def _need_pair(model_path, config_path):
    for p in (model_path, config_path):
        if not Path(p).is_file():
            raise FileNotFoundError(f"Piper voice file not found: {p}\nPut it in the models/piper/ folder.")


class Synthesizer:
    def __init__(self, fake=None):
        self.fake = config.TTS_FAKE if fake is None else fake

    def voice_for(self, language):
        if language == "si":
            return config.TTS_VOICE_SI_MODEL, config.TTS_VOICE_SI_CONFIG
        return config.TTS_VOICE_EN_MODEL, config.TTS_VOICE_EN_CONFIG

    def synthesize(self, text, language=None, out_path="tts_output.wav"):
        """
        Returns (audio int16 array, sample_rate), or (None, None) if there was
        nothing sayable left after cleaning.
        """
        text = sanitize_text(clean_for_speech(text)).replace("\n", " ").strip()
        if not text:
            log.warning("Nothing left to say after cleaning the text")
            return None, None

        language = language or pick_language(text)

        if self.fake:
            sr = config.TTS_SAMPLE_RATE
            t = np.arange(int(0.5 * sr), dtype=np.float32) / sr
            tone = (3000.0 * np.sin(2.0 * np.pi * 300.0 * t)).astype(np.int16)
            log.info("TTS fake mode: beep instead of '%s' (%s)", text[:40], language)
            return tone, sr

        model_path, config_path = self.voice_for(language)
        _need_pair(model_path, config_path)

        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONUTF8"] = "1"
        try:
            result = subprocess.run(
                ["piper", "--model", str(model_path), "--config", str(config_path),
                 "--output_file", str(out_path)],
                input=text.encode("utf-8"), stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE, env=env, timeout=config.TTS_TIMEOUT_S,
            )
        except FileNotFoundError as e:
            raise TTSError("The 'piper' program was not found. Is it installed and on PATH?") from e
        except subprocess.TimeoutExpired as e:
            raise TTSError(f"Piper took longer than {config.TTS_TIMEOUT_S}s") from e

        if result.returncode != 0:
            raise TTSError(f"Piper failed: {result.stderr.decode(errors='ignore')}")
        if not Path(out_path).is_file() or Path(out_path).stat().st_size <= 44:
            raise TTSError("Piper produced an empty audio file")

        return load_wav(out_path)