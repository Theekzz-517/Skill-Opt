
import logging
import re
from pathlib import Path

import numpy as np

import config

log = logging.getLogger("stt")

_SURROGATES = re.compile(r"[\ud800-\udfff]")


def sanitize_text(text):
    if not isinstance(text, str):
        return text
    return _SURROGATES.sub("", text)


class Transcriber:
    def __init__(self, fake=None, model_path=None):
        self.fake = config.STT_FAKE if fake is None else fake
        self.fake_text = ""            
        self._model = None
        self._model_path = model_path or config.STT_MODEL_PATH

    def load(self):
        if self.fake:
            log.info("STT fake mode: no model loaded")
            return
        if not Path(self._model_path).is_file():
            raise FileNotFoundError(
                f"Whisper model not found: {self._model_path}\nPut your ggml model file there.")
        from pywhispercpp.model import Model
        self._model = Model(str(self._model_path), n_threads=config.STT_THREADS)
        log.info("Whisper model loaded: %s", self._model_path)

    def transcribe(self, audio, sample_rate=None):
        """audio: int16 or float numpy array. Returns clean text, or '' if nothing usable was said."""
        if self.fake:
            text = sanitize_text(self.fake_text)
            return "" if text in config.STT_SILENCE_LABELS else text
        if self._model is None:
            raise RuntimeError("Whisper model is not loaded; call load() first.")

        samples = np.asarray(audio)
        if samples.size == 0:
            return ""
        if samples.dtype == np.int16:
            samples = samples.astype(np.float32) / 32768.0
        else:
            samples = samples.astype(np.float32, copy=False)

        segments = self._model.transcribe(samples)
        text = sanitize_text(" ".join(segment.text for segment in segments)).strip()
        return "" if text in config.STT_SILENCE_LABELS else text