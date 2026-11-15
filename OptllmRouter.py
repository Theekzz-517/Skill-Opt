
import logging
import re
import socket
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from datetime import date

import requests

import config
import protocol as P
from skills.stt import sanitize_text
from skills.web_search import search_context

log = logging.getLogger("llm_router")

_SINHALA = re.compile(r"[\u0D80-\u0DFF]")
_SINGLISH_WORDS = {
    "sinhala", "hondai", "kohomada", "moko", "ane", "ekath", "eka",
    "hari", "denna", "karanna", "ennako", "balanna", "mokada",
    "mokakda", "gaththa", "kiyanne", "thiyenawa", "nethuwa", "subha",
}
_UNCERTAIN_PHRASES = (
    "don't know", "dont know", "do not know", "not sure",
    "cannot understand", "can't understand", "i'm sorry",
    "unknown", "invalid",
)
_SMALL_TALK = {"hi", "hello", "thanks", "thank you", "bye", "ok", "okay"}
_EMO_TAG = re.compile(r"\[?\s*emotion\s*[:=]\s*(\w+)\s*\]?", re.I)
_BARE_TAG = re.compile(r"\[\s*(\w+)\s*\]\s*$")     # catches a lone "[fear]" at the end

# Spoken when nothing can answer (no internet and no local model for this language)
_OFFLINE_EN = "Sorry, I cannot reach the internet right now, so I cannot answer that."
_OFFLINE_SI = "සමාවෙන්න, මට දැන් අන්තර්ජාලයට සම්බන්ධ විය නොහැක. කරුණාකර පසුව නැවත උත්සාහ කරන්න."

# Time (epoch seconds) until which we do not call Gemini, set after an HTTP 429
_gemini_blocked_until = 0.0
_net_cache = {"ok": True, "t": 0.0}

# Module-level shared thread pool for web search to avoid recreation overhead per query
_search_pool = ThreadPoolExecutor(max_workers=1)


class LLMError(Exception):
    """Raised when no model (local or cloud) could answer."""


def _gemini_available():
    return time.time() >= _gemini_blocked_until


def internet_ok(cache_s=10):
    """Quick internet check (no DNS). The result is remembered for a few seconds."""
    now = time.time()
    if now - _net_cache["t"] < cache_s:
        return _net_cache["ok"]
    
    ok = False
    for host in (("8.8.8.8", 53), ("1.1.1.1", 53)):
        try:
            socket.create_connection(host, timeout=1.5).close()
            ok = True
            break
        except OSError:
            continue
            
    _net_cache["ok"] = ok
    _net_cache["t"] = now
    return ok


def looks_sinhala(text):
    if _SINHALA.search(text):
        return True
    words = re.sub(r"[^\w\s]", "", text.lower()).split()
    return any(w in _SINGLISH_WORDS for w in words)


def looks_uncertain(text):
    if not text or len(text.strip()) < 2:
        return True
    low = text.lower()
    return any(p in low for p in _UNCERTAIN_PHRASES) or "unknown" in low.upper().lower()


def strip_emotion_tag(reply):
    """Returns (clean_text, emotion_or_None). Understands [emotion: X] and a lone [X]."""
    m = _EMO_TAG.search(reply)
    clean = _EMO_TAG.sub("", reply).strip()
    emo = m.group(1).lower() if m else None
    if emo is None:
        b = _BARE_TAG.search(clean)
        if b and b.group(1).lower() in P.EMOTIONS:
            emo = b.group(1).lower()
            clean = _BARE_TAG.sub("", clean).strip()
    return clean, emo if emo in P.EMOTIONS else None


class ConversationHistory:
    """Keeps the last few turns in RAM. Resets when the program restarts."""

    def __init__(self, max_turns=None):
        self.max_turns = config.MAX_HISTORY_TURNS if max_turns is None else max_turns
        self._turns = []               # list of {"role": "user"/"model", "text": ...}

    def add(self, user_text, model_text):
        self._turns.append({"role": "user", "text": sanitize_text(user_text)})
        self._turns.append({"role": "model", "text": sanitize_text(model_text)})

    def recent(self):
        return self._turns[-2 * self.max_turns:]

    def clear(self):
        self._turns = []

    def as_ollama_prompt(self):
        lines = []
        for t in self.recent():
            speaker = "User" if t["role"] == "user" else "Assistant"
            lines.append(f"{speaker}: {t['text']}")
        return "\n".join(lines)

    def as_gemini_contents(self):
        return [{"role": t["role"], "parts": [{"text": t["text"]}]} for t in self.recent()]


_GROUNDING_NOTE_EN = (
    " You may use Google Search to check current facts. Even so, your reply must still be "
    "1-2 short spoken sentences, no markdown, no citations, no source names, and no URLs or links - "
    "this text is read aloud by a voice, so only plain spoken words are allowed. Still end with the "
    "[emotion: X] tag exactly as instructed above."
)
_GROUNDING_NOTE_SI = (
    " අවශ්‍ය නම් Google Search මගින් නවතම කරුණු පරීක්ෂා කළ හැක. එසේ වුවද ඔබේ පිළිතුර තවමත් කෙටි "
    "වාක්‍ය 1-2ක් විය යුතු අතර, markdown, උපුටා දැක්වීම්, මූලාශ්‍ර නම් හෝ URL/links නොතිබිය යුතුය - "
    "මෙම පෙළ හඬකින් කියවනු ලැබේ. ඉහත පරිදි [emotion: X] ටැගය අගයෙහි තබන්න."
)


def _build_instruction(language, grounded=False):
    tag = (config.EMOTION_TAG_INSTRUCTION_SI if language == "si"
           else config.EMOTION_TAG_INSTRUCTION_EN).format(emotions=", ".join(P.EMOTIONS))
    if language == "si":
        text = (
            "ඔබ පිළිතුරු දෙන්නේ සිංහල භාෂාවෙන් පමණි. English වචන හෝ markdown "
            "භාවිතා නොකරන්න. කෙටි වාක්‍ය 1-2ක් පමණක් යොදන්න. " + tag
        )
        return text + (_GROUNDING_NOTE_SI if grounded else "")
    text = (
        "You are a friendly robot companion talking out loud with a person. "
        "Speak warmly and naturally, like a kind friend, not like a software tool. "
        "For small talk such as 'how are you', answer simply and kindly. "
        "Only talk about earlier topics if the person asks about them. "
        "Reply in 1-2 short sentences. Do not use markdown formatting or "
        "parenthetical translations. " + tag
    )
    return text + (_GROUNDING_NOTE_EN if grounded else "")


def query_ollama(prompt, history, post=requests.post):
    convo = history.as_ollama_prompt()
    full_prompt = f"{_build_instruction('en')}\n\n"
    if convo:
        full_prompt += f"{convo}\n"
    full_prompt += f"User: {prompt}\nAssistant:"
    payload = {"model": config.OLLAMA_MODEL, "prompt": full_prompt, "stream": False}
    response = post(config.OLLAMA_URL, json=payload, timeout=config.OLLAMA_TIMEOUT_S)
    if response.status_code != 200:
        raise LLMError(f"Ollama HTTP {response.status_code}")
    return sanitize_text(response.json().get("response", "").strip())


def _gemini_has_grounding(data):
    """True if Gemini's response shows it actually used Google Search."""
    try:
        return bool(data["candidates"][0].get("groundingMetadata"))
    except (KeyError, IndexError, TypeError):
        return False


def _call_gemini_once(url, payload, post, max_retries, deadline=None):
    global _gemini_blocked_until
    last_error = None
    for attempt in range(max_retries + 1):
        timeout = config.GEMINI_TIMEOUT_S
        if deadline is not None:
            remaining = deadline - time.time()
            if remaining < 1:
                raise LLMError("Gemini took too long (time limit reached)")
            timeout = min(timeout, remaining)

        try:
            response = post(url, json=payload, timeout=timeout)
        except requests.exceptions.RequestException as e:
            last_error = f"Network error: {e}"
            log.warning("Gemini attempt %d/%d failed: %s", attempt + 1, max_retries + 1, last_error)
            time.sleep(1.5 * (attempt + 1))
            continue

        if response.status_code == 429:
            cooldown = getattr(config, "GEMINI_COOLDOWN_S", 60)
            _gemini_blocked_until = time.time() + cooldown
            log.warning("Gemini rate limit (HTTP 429). Pausing Gemini for %ds.", cooldown)
            raise LLMError("Gemini rate limit (HTTP 429): too many requests or quota used up")

        if response.status_code in (500, 502, 503):
            last_error = f"HTTP {response.status_code}"
            log.warning("Gemini transient error, retrying (%d/%d)", attempt + 1, max_retries + 1)
            time.sleep(1.5 * (attempt + 1))
            continue
        if response.status_code != 200:
            raise LLMError(f"Gemini HTTP {response.status_code}: {response.text}")

        data = response.json()
        candidates = data.get("candidates", [])
        if not candidates:
            raise LLMError(f"Gemini returned no candidates: {data.get('promptFeedback', {})}")
        try:
            parts = candidates[0]["content"]["parts"]
            text = sanitize_text("".join(p.get("text", "") for p in parts).strip())
            if not text:
                raise KeyError("empty text")
            return text, _gemini_has_grounding(data)
        except (KeyError, IndexError) as e:
            raise LLMError(f"Could not parse Gemini response: {data}") from e