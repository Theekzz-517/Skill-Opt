
import numpy as np

import config


def mouth_envelope(audio, sample_rate, fps=None):
    """
    audio: int16 or float numpy array (mono).
    Returns a uint8 array: one mouth-opening value (0-255) per video-style frame.
    frame[i] is the mouth opening at time i / fps seconds into the audio.
    """
    fps = config.MOUTH_FPS if fps is None else fps
    
    # Check original type before casting to float32 to properly normalize integer audio (e.g., int16)
    audio_arr = np.asarray(audio)
    if np.issubdtype(audio_arr.dtype, np.integer):
        audio = audio_arr.astype(np.float32) / 32768.0
    else:
        audio = audio_arr.astype(np.float32)

    hop = max(1, int(round(sample_rate / fps)))
    n = len(audio) // hop
    if n == 0:
        return np.zeros(1, dtype=np.uint8)

    frames = audio[: n * hop].reshape(n, hop)
    loudness = np.sqrt((frames ** 2).mean(axis=1))         # RMS loudness per frame

    peak = np.percentile(loudness, 95)
    loudness = loudness / (peak + 1e-6)
    
    # Clip and apply noise gate efficiently using boolean masking/clip
    np.clip(loudness, 0.0, 1.0, out=loudness)
    loudness[loudness < config.MOUTH_NOISE_GATE] = 0.0     # ignore background hiss

    w = config.MOUTH_SMOOTH_WINDOW
    if w > 1:
        kernel = np.ones(w, dtype=np.float32) / w
        loudness = np.convolve(loudness, kernel, mode="same")

    return (loudness * 255).astype(np.uint8)


def frame_at(envelope, elapsed_s, fps=None):
    """The mouth value to send right now, given how many seconds of playback have passed."""
    fps = config.MOUTH_FPS if fps is None else fps
    idx = int(elapsed_s * fps)
    if idx < 0 or idx >= len(envelope):
        return 0
    return int(envelope[idx])