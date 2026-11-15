
import collections
import logging
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

import config
import protocol as P

log = logging.getLogger("emotion")

# FER+ model output order (fixed by the model). 'contempt' is not used.
_FERPLUS = ["neutral", "happiness", "surprise", "sadness", "anger", "disgust", "fear", "contempt"]
_TO_PROJECT = {
    "neutral": "neutral", "happiness": "happy", "sadness": "sad", "anger": "angry",
    "surprise": "surprise", "fear": "fear", "disgust": "disgust"
}

# Pre-compute index lookup once for performance O(1) matching during classification
_FERPLUS_MAP = {k: i for i, k in enumerate(_FERPLUS)}
_IDX = np.array([_FERPLUS_MAP[k] for e in P.EMOTIONS for k, v in _TO_PROJECT.items() if v == e], dtype=int)


def _need_file(path):
    if not Path(path).is_file():
        raise FileNotFoundError(f"Model file not found: {path}\nPut it in the models/ folder.")


class FaceFinder:
    """Finds the biggest (= nearest) face. Returns (x, y, w, h) or None."""

    def __init__(self, model_path=None):
        model_path = model_path or config.FACE_MODEL_PATH
        _need_file(model_path)
        self._det = cv2.FaceDetectorYN.create(
            str(model_path), "", (320, 320),
            config.FACE_SCORE_MIN, 0.3, 5000
        )

    def find(self, frame):
        h, w = frame.shape[:2]
        self._det.setInputSize((w, h))
        _, faces = self._det.detect(frame)
        if faces is None or len(faces) == 0:
            return None
        
        # Avoid lambda overhead by using an itemgetter or fast generator expression
        best = max(faces, key=lambda f: f[2] * f[3])
        if best[2] < config.MIN_FACE_PX:
            return None
        return tuple(int(v) for v in best[:4])


def face_offset(box, frame_width):
    """
    box: (x, y, w, h) from FaceFinder.find(). frame_width: the frame's width in pixels.
    Returns a float from -1.0 (face at the far left) to +1.0 (face at the far right),
    0.0 = centered. This is the raw signal the eyes/neck follow.
    """
    x, y, w, h = box
    center_x = x + (w * 0.5)
    half_width = frame_width * 0.5
    offset = (center_x - half_width) / half_width
    return -1.0 if offset < -1.0 else (1.0 if offset > 1.0 else offset)


class HeadTracker:
    """Smooths the face offset and turns it into eye/neck CAN byte values (0-255, 128=center)."""

    def __init__(self, alpha=None, neck_gain=None):
        self.alpha = config.HEAD_EMA_ALPHA if alpha is None else alpha
        self.neck_gain = config.HEAD_NECK_GAIN if neck_gain is None else neck_gain
        self._offset = 0.0

    def reset(self):
        """Call this when the face is lost - moves the head back to center."""
        self._offset = 0.0

    def update(self, offset):
        """offset: -1.0..1.0 from face_offset(). Returns (eye_x, neck), both 0-255 bytes."""
        self._offset = (1.0 - self.alpha) * self._offset + self.alpha * offset
        
        center = P.HEAD_CENTER
        eye_x = int(center + self._offset * center)
        neck = int(center + self._offset * self.neck_gain * center)
        
        return max(0, min(255, eye_x)), max(0, min(255, neck))

    def center_bytes(self):
        return P.HEAD_CENTER, P.HEAD_CENTER


class EmotionClassifier:
    """Face crop -> probabilities for the 7 project emotions (same order as protocol.EMOTIONS)."""

    def __init__(self, model_path=None):
        model_path = model_path or config.EMOTION_MODEL_PATH
        _need_file(model_path)
        providers = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider")
                     if p in ort.get_available_providers()]
        self._sess = ort.InferenceSession(str(model_path), providers=providers)
        self._in = self._sess.get_inputs()[0].name
        log.info("Emotion model loaded, providers: %s", self._sess.get_providers())

    def classify(self, frame, box):
        x, y, w, h = box
        m = int(0.1 * w)  # 10% margin around the face
        
        # Optimize boundary clips
        h_max, w_max = frame.shape[:2]
        x0, y0 = max(0, x - m), max(0, y - m)
        x1, y1 = min(w_max, x + w + m), min(h_max, y + h + m)
        
        crop = frame[y0:y1, x0:x1]
        if crop.size == 0:
            return None
            
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        # Optimized resize and direct casting without allocating redundant intermediates
        inp = cv2.resize(gray, (64, 64), interpolation=cv2.INTER_LINEAR).astype(np.float32)[None, None]
        
        logits = self._sess.run(None, {self._in: inp})[0][0]
        
        # Log-sum-exp numerical stabilization
        logits_max = logits.max()
        e = np.exp(logits - logits_max)
        e_sum = e.sum()
        if e_sum == 0:
            return None
            
        probs = (e / e_sum)[_IDX]
        probs_sum = probs.sum()
        if probs_sum == 0:
            return None
            
        return probs / probs_sum


class EmotionSmoother:
    """Averages the results over time and stops the face from flickering."""

    def __init__(self, alpha=None, min_hold_s=None):
        self.alpha = config.EMOTION_EMA_ALPHA if alpha is None else alpha
        self.min_hold_s = config.EMOTION_MIN_HOLD_S if min_hold_s is None else min_hold_s
        self.reset()

    def reset(self):
        n = len(P.EMOTIONS)
        self._p = np.full(n, 1.0 / n, dtype=float)
        self.current = "neutral"
        self._changed_at = -float('inf')

    def update(self, probs, now):
        """Returns (emotion, confidence 0-1)."""
        p = np.asarray(probs, dtype=float)
        p_sum = p.sum()
        if p_sum > 0:
            p /= p_sum
            
        self._p = (1.0 - self.alpha) * self._p + self.alpha * p
        best = P.EMOTIONS[int(np.argmax(self._p))]
        
        if best != self.current and (now - self._changed_at) >= self.min_hold_s:
            self.current = best
            self._changed_at = now
            
        return self.current, float(self._p[P.EMOTION_ID[self.current]])


class EmotionHistory:
    """Remembers emotions for a short time. Answers: what was the main emotion since time t0?"""

    def __init__(self, keep_s=None):
        self.keep_s = config.EMOTION_HISTORY_S if keep_s is None else keep_s
        self._items = collections.deque()  # (time, emotion)

    def add(self, now, emotion):
        self._items.append((now, emotion))
        while self._items and (now - self._items[0][0]) > self.keep_s:
            self._items.popleft()

    def dominant_since(self, t0, default="neutral"):
        labels = [e for (t, e) in self._items if t >= t0]
        if not labels:
            return default
            
        counts = collections.Counter(e for e in labels if e != "neutral")
        if counts:
            best, n = counts.most_common(1)[0]
            if (n / len(labels)) >= 0.30:  # a real emotion in 30%+ of the time wins
                return best
        return "neutral"