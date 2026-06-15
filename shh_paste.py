#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "mlx-whisper>=0.3.0",
#   "sounddevice>=0.4.0",
#   "numpy>=1.24.0",
#   "pynput>=1.7.0",
# ]
# ///
"""
shh-paste — Local voice transcription for macOS
All processing is on-device. No network calls during normal use.

Edit the CONFIG block below to change behavior.
Run with:  uv run shh_paste.py

Modes:
  (default)   Daemon — hotkey-triggered, pastes into active app
  --once      Pipe — record one utterance, print to stdout, exit
              Usage: uv run shh_paste.py --once
              Compose: uv run shh_paste.py --once >> ~/notes.md
"""

import argparse
import os
import re
import signal
import sqlite3
import sys
import subprocess
import threading
import time
import wave
from datetime import datetime
from pathlib import Path
import tomllib
import numpy as np
import sounddevice as sd
from pynput import keyboard
import mlx_whisper

from recorder import AudioRecorder


# ========== CONFIG ==========

MODEL         = "mlx-community/whisper-large-v3-turbo"  # runs on Apple Silicon GPU via MLX
                              # alternatives: "mlx-community/whisper-small-mlx" (faster, less accurate)
TRIGGER_MODE  = "smart"       # "hold"   = hold key to record, release to transcribe
                              # "toggle" = press once to start, press again to stop+transcribe
                              # "smart"  = tap to toggle, long-press to hold-to-record
LONG_PRESS_THRESHOLD = 0.5    # Seconds to distinguish a tap (toggle) from a long-press (hold)
HOTKEY        = "right_alt"   # Right Option (⌥) key — paste mode
                              # Other options: "left_alt" | "right_ctrl" | "right_shift"
FILLER_REMOVE = False         # True = strip "um", "uh", "you know", "like" from output
AUTO_PASTE    = True          # Paste into active app after transcription (needs Accessibility permission)
NOTIFICATION  = True          # Show macOS notification with transcription preview
NOTIFY_START  = True          # Show notification when recording starts
MAX_RECORD    = 600           # Max seconds before auto-stopping (safety cutoff)
WARN_BEFORE   = 30            # Seconds before MAX_RECORD to show a warning notification
AUDIO_DEVICE  = None          # None = system default mic
                              # Or set to a name string, e.g. "MacBook Air Microphone"
                              # Run: python3 -c "import sounddevice; print(sounddevice.query_devices())"
SAMPLE_RATE   = 16000         # Whisper requires 16kHz — do not change
INITIAL_PROMPT = ""           # String passed to Whisper to bias vocabulary/spelling toward
                              # specific proper nouns and jargon. Keep short (< ~200 tokens).
                              # Example: "Kubernetes, PostgreSQL, GraphQL, nginx, Redis."
SUBSTITUTIONS = {}            # Post-transcription regex fixes for stubborn mishears.
                              # Keys are literal phrases (case-insensitive, word-bounded);
                              # values are the exact replacement text.
                              # Example: { "my sequel" = "MySQL" }

# Voice memo mode — Cmd + HOTKEY appends timestamped transcription to a file
MEMO_FILE     = os.path.expanduser("~/voice-notes.md")  # Where memos are saved
MEMO_MODE     = "smart"       # Same options as TRIGGER_MODE: "hold" | "toggle" | "smart"

# ============================

# ── Load optional config overrides ────────────────────────────────────────────
_CONFIG_KEYS = {
    "model", "trigger_mode", "long_press_threshold", "hotkey",
    "filler_remove", "auto_paste", "notification", "notify_start",
    "max_record", "warn_before", "audio_device", "sample_rate",
    "memo_file", "memo_mode", "initial_prompt", "substitutions",
}
for _cfg_path in (Path.home() / ".config" / "shh-paste" / "config.toml", Path("shh-paste.toml")):
    if _cfg_path.is_file():
        with open(_cfg_path, "rb") as _f:
            _cfg = tomllib.load(_f)
        for _k, _v in _cfg.items():
            if _k in _CONFIG_KEYS:
                if _k == "memo_file":
                    _v = os.path.expanduser(_v)
                globals()[_k.upper()] = _v
        break

HOTKEY_MAP = {
    "left_alt":    keyboard.Key.alt_l,
    "right_alt":   keyboard.Key.alt_r,
    "right_ctrl":  keyboard.Key.ctrl_r,
    "right_shift": keyboard.Key.shift_r,
    "f13":         keyboard.Key.f13,
    "f14":         keyboard.Key.f14,
    "f15":         keyboard.Key.f15,
}

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "shh_paste.db")

FILLER_RE = re.compile(r'\b(um+|uh+|you know|like)\b\s*', re.IGNORECASE)
_transcribe_lock = threading.Lock()

# Compile substitution patterns — word-bounded, case-insensitive literal phrase match.
_SUBSTITUTION_PATTERNS = [
    (re.compile(r'\b' + re.escape(_k) + r'\b', re.IGNORECASE), _v)
    for _k, _v in SUBSTITUTIONS.items()
]


def _init_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    con = sqlite3.connect(DB_PATH)
    con.execute("""
        CREATE TABLE IF NOT EXISTS transcriptions (
            id INTEGER PRIMARY KEY,
            timestamp TEXT NOT NULL,
            text TEXT NOT NULL,
            mode TEXT NOT NULL,
            audio_seconds REAL,
            word_count INTEGER,
            model TEXT,
            audio_device TEXT,
            trigger_mode TEXT,
            active_app TEXT
        )
    """)
    con.commit()
    con.close()


def _get_active_app() -> str | None:
    try:
        result = subprocess.run(
            ['osascript', '-e', 'tell application "System Events" to get name of first application process whose frontmost is true'],
            capture_output=True, text=True, timeout=2,
        )
        return result.stdout.strip() or None
    except Exception:
        return None


def _log_transcription(text: str, mode: str, audio_seconds: float | None = None, trigger: str | None = None):
    try:
        con = sqlite3.connect(DB_PATH)
        con.execute(
            "INSERT INTO transcriptions (timestamp, text, mode, audio_seconds, word_count, model, audio_device, trigger_mode, active_app) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                datetime.now().isoformat(),
                text, mode, audio_seconds,
                len(text.split()),
                MODEL,
                str(AUDIO_DEVICE) if AUDIO_DEVICE else "system_default",
                trigger,
                _get_active_app(),
            ),
        )
        con.commit()
        con.close()
    except Exception as e:
        log(f"⚠  DB log failed: {e}")


def log(msg: str):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


# ── Silence-hallucination blocklist ────────────────────────────────────────────
# Whisper outputs these phrases on silence/near-silence even when no_speech_threshold
# is set. Strip punctuation and lowercasebefore comparing.
_SILENCE_HALLUCINATIONS = {
    "thank you",
    "thanks for watching",
    "thank you for watching",
    "you",
    ".",
    "",
    "bye",
    "bye bye",
    "please subscribe",
    "subscribe",
}


def _is_silence_hallucination(text: str) -> bool:
    normalized = text.strip().rstrip(".!?,").strip().lower()
    return normalized in _SILENCE_HALLUCINATIONS


# ── Repetition-hallucination guard ─────────────────────────────────────────────
# whisper-large-v3-turbo occasionally falls into a decode loop and emits one token
# hundreds of times ("page page page…"). Collapse runs; if a run is pathological,
# the caller discards the whole transcription so it never gets pasted.
REPEAT_HALLUCINATION = 8   # consecutive identical tokens => treat as decode failure


def _collapse_repeats(text: str, max_keep: int = 3) -> tuple[str, int]:
    """Collapse runs of an identical token. Returns (collapsed_text, longest_run)."""
    out: list[str] = []
    longest = 0
    run_key = None
    run = 0
    for w in text.split():
        key = w.lower().strip(".,!?-")
        if key and key == run_key:
            run += 1
        else:
            run_key = key
            run = 1
        longest = max(longest, run)
        if run <= max_keep:
            out.append(w)
    return " ".join(out), longest


def _has_concatenated_repeat(text: str, min_unit: int = 2, threshold: int = 8) -> bool:
    """Detect runaway repetition with no spaces, e.g. "нологнологнолог…".

    Scans the longest whitespace-free token; if any short substring repeats
    back-to-back >= threshold times, it's a decode-loop hallucination.
    """
    for token in text.split():
        n = len(token)
        if n < min_unit * threshold:
            continue
        for unit in range(min_unit, n // threshold + 1):
            seg = token[:unit]
            if seg * threshold in token:
                return True
    return False


def clean(text: str) -> str:
    text = text.strip()
    if FILLER_REMOVE:
        text = FILLER_RE.sub('', text).strip()
        text = re.sub(r'  +', ' ', text)
    for pattern, replacement in _SUBSTITUTION_PATTERNS:
        text = pattern.sub(replacement, text)
    return text


def notify(message: str, title: str = "shh-paste"):
    """Send a macOS notification. Callers decide whether to gate on config flags."""
    message = message[:80].replace('"', '\\"').replace("'", "\\'")
    subprocess.run(
        ['osascript', '-e',
         f'display notification "{message}" with title "{title}"'],
        capture_output=True
    )


def deliver(text: str, audio_seconds: float | None = None):
    if not text:
        log("⚠  Nothing transcribed")
        return

    # Always copy to clipboard
    subprocess.run(['pbcopy'], input=text.encode('utf-8'), check=True)

    # Auto-paste into active app by typing directly
    if AUTO_PASTE:
        time.sleep(0.15)
        controller = keyboard.Controller()
        controller.type(text)

    if NOTIFICATION:
        notify(text)

    _log_transcription(text, "paste", audio_seconds, trigger=TRIGGER_MODE)
    log(f"✓  {text}")


def deliver_memo(text: str, audio_seconds: float | None = None):
    """Append timestamped transcription to MEMO_FILE."""
    if not text:
        log("⚠  Nothing transcribed (memo)")
        return

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    entry = f"- [{timestamp}] {text}\n"

    # Create file with header if it doesn't exist
    if not os.path.exists(MEMO_FILE):
        os.makedirs(os.path.dirname(MEMO_FILE), exist_ok=True)
        with open(MEMO_FILE, 'w') as f:
            f.write("# Voice Notes\n\n")

    with open(MEMO_FILE, 'a') as f:
        f.write(entry)

    if NOTIFICATION:
        notify(f"📝 {text}", title="shh-paste memo")

    _log_transcription(text, "memo", audio_seconds, trigger=MEMO_MODE)
    log(f"📝  Saved to {MEMO_FILE}: {text}")


def transcribe(audio: np.ndarray) -> str:
    log("⟳  Transcribing...")
    audio_flat = audio.flatten().astype(np.float32)
    kwargs = {
        "path_or_hf_repo": MODEL,
        "language": "en",
        "condition_on_previous_text": False,
        # Decode-time repetition/silence guards. compression_ratio_threshold is
        # the robust catch — it works on raw decoder output regardless of spacing.
        "compression_ratio_threshold": 2.4,
        "no_speech_threshold": 0.6,
        "temperature": (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
    }
    if INITIAL_PROMPT:
        kwargs["initial_prompt"] = INITIAL_PROMPT
    result = mlx_whisper.transcribe(audio_flat, **kwargs)
    text, max_run = _collapse_repeats(result["text"])
    if max_run >= REPEAT_HALLUCINATION or _has_concatenated_repeat(result["text"]):
        log(f"⚠  Repetition hallucination (run x{max_run}) — discarding garbage output")
        return ""
    cleaned = clean(text)
    if _is_silence_hallucination(cleaned):
        log(f"⚠  Silence hallucination ({repr(cleaned)}) — discarding")
        return ""
    return cleaned


# A normal decode is a few seconds. If one runs past this it is wedged — abandon it
# so it can never hold _transcribe_lock forever and freeze every later clip.
TRANSCRIBE_TIMEOUT = 90   # seconds


def _transcribe_with_timeout(audio: np.ndarray) -> str:
    """Run transcribe() in a worker thread; raise TimeoutError if it wedges."""
    box: dict = {}

    def _run():
        try:
            box["text"] = transcribe(audio)
        except Exception as e:  # noqa: BLE001
            box["error"] = e

    worker = threading.Thread(target=_run, daemon=True)
    worker.start()
    worker.join(TRANSCRIBE_TIMEOUT)
    if worker.is_alive():
        raise TimeoutError(f"transcription exceeded {TRANSCRIBE_TIMEOUT}s")
    if "error" in box:
        raise box["error"]
    return box.get("text", "")


def warm_up_model():
    """Load + JIT the model once at startup with a short silent clip, so the first
    real dictation isn't slowed by a cold model load. Runs in a daemon thread."""
    try:
        t0 = time.time()
        silence = np.zeros(int(SAMPLE_RATE * 0.5), dtype=np.float32)
        with _transcribe_lock:
            transcribe(silence)
        log(f"✓  Model warm ({time.time() - t0:.1f}s) — first dictation will be fast")
    except Exception as e:  # noqa: BLE001
        log(f"⚠  Warm-up skipped: {e}")


def process_audio(audio: np.ndarray):
    """Transcribe and deliver audio. Serialized; a wedged decode is abandoned, not blocking."""
    audio_seconds = len(audio) / SAMPLE_RATE
    if not _transcribe_lock.acquire(timeout=TRANSCRIBE_TIMEOUT + 10):
        log("⚠  Previous transcription still busy — skipping this clip")
        return
    try:
        text = _transcribe_with_timeout(audio)
        deliver(text, audio_seconds)
    except TimeoutError as e:
        log(f"✗  {e} — abandoned so the hotkey stays responsive")
    except Exception as e:
        log(f"✗  Error during transcription: {e}")
    finally:
        _transcribe_lock.release()


def process_memo(audio: np.ndarray):
    """Transcribe and save as voice memo. Serialized; a wedged decode is abandoned."""
    audio_seconds = len(audio) / SAMPLE_RATE
    if not _transcribe_lock.acquire(timeout=TRANSCRIBE_TIMEOUT + 10):
        log("⚠  Previous transcription still busy — skipping this memo")
        return
    try:
        text = _transcribe_with_timeout(audio)
        deliver_memo(text, audio_seconds)
    except TimeoutError as e:
        log(f"✗  {e} — abandoned so the hotkey stays responsive")
    except Exception as e:
        log(f"✗  Error during memo transcription: {e}")
    finally:
        _transcribe_lock.release()



# ── Audio engine + listeners ────────────────────────────────────────────────────
# The recorder (recorder.py) drives the mic on ONE dedicated, serialized thread, so a
# CoreAudio open/stop can never run on the keyboard thread or overlap another — the two
# conditions behind the deadlock that used to freeze the hotkey. Key callbacks here only
# post non-blocking start/stop intents; finished audio comes back via on_audio.

RECORDING_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "last_recording.wav")
OP_TIMEOUT = 8.0   # an audio op slower than this is treated as wedged → self-restart


def _offload(process_fn):
    """Wrap a processor so transcription runs OFF the audio-control thread."""
    def _cb(audio, seconds):
        threading.Thread(target=process_fn, args=(audio,), daemon=True).start()
    return _cb


def _on_wedged(op, secs):
    """Last-ditch safety: if an audio op ever hangs, restart so it can't stick forever.
    The in-flight clip survives on disk (RECORDING_PATH) and is recoverable."""
    log(f"✗  Audio op '{op}' wedged > {secs:.0f}s — self-restarting to recover")
    try:
        notify("Audio stuck — restarting (clip saved, use: shh recover)")
    except Exception:
        pass
    os._exit(75)   # non-zero → the services watchdog reloads a clean process


class _BaseListener:
    """Shared plumbing: key events become non-blocking recorder intents."""

    def __init__(self, target_key, recorder):
        self._key = target_key
        self._rec = recorder
        self._lock = threading.Lock()

    def _reset(self):
        """Clear recording state. Called under self._lock. Overridden where needed."""

    def force_finish(self) -> bool:
        """External stop (SIGUSR1 rescue), bypassing the keyboard."""
        if not self._rec.active:
            return False
        with self._lock:
            self._reset()
        self._rec.stop()
        return True


class HoldListener(_BaseListener):
    """Hold HOTKEY to record, release to transcribe."""

    def on_press(self, key):
        if key == self._key and not self._rec.active:
            with self._lock:
                self._rec.start()
                if NOTIFY_START:
                    notify("Recording…")

    def on_release(self, key):
        if key == self._key:
            with self._lock:
                self._rec.stop()


class ToggleListener(_BaseListener):
    """Press HOTKEY to start, press again to stop + transcribe."""

    def on_press(self, key):
        if key != self._key:
            return
        with self._lock:
            if not self._rec.active:
                self._rec.start()
                if NOTIFY_START:
                    notify("Recording…")
            else:
                self._rec.stop()

    def on_release(self, key):
        pass


class SmartListener(_BaseListener):
    """Tap to toggle, long-press to hold-to-record (default)."""

    _IDLE, _DECIDING, _HOLD, _TOGGLE = "idle", "deciding", "hold", "toggle"

    def __init__(self, target_key, recorder):
        super().__init__(target_key, recorder)
        self._state = self._IDLE
        self._decide_timer: threading.Timer | None = None

    def _reset(self):
        self._state = self._IDLE
        if self._decide_timer:
            self._decide_timer.cancel()
            self._decide_timer = None

    def _on_decide(self):
        with self._lock:
            if self._state == self._DECIDING:
                self._state = self._HOLD

    def on_press(self, key):
        if key != self._key:
            return
        with self._lock:
            if self._state == self._IDLE:
                self._rec.start()
                self._state = self._DECIDING
                self._decide_timer = threading.Timer(LONG_PRESS_THRESHOLD, self._on_decide)
                self._decide_timer.start()
            elif self._state == self._TOGGLE:
                self._state = self._IDLE
                self._rec.stop()

    def on_release(self, key):
        if key != self._key:
            return
        with self._lock:
            if self._state == self._DECIDING:
                if self._decide_timer:
                    self._decide_timer.cancel()
                    self._decide_timer = None
                self._state = self._TOGGLE
                if NOTIFY_START:
                    notify("Recording… tap to stop")
            elif self._state == self._HOLD:
                self._state = self._IDLE
                self._rec.stop()


def _make_handler(mode, target_key, recorder):
    return {"hold": HoldListener, "toggle": ToggleListener, "smart": SmartListener}[mode](target_key, recorder)


def _load_wav(path: str) -> np.ndarray | None:
    if not os.path.exists(path) or os.path.getsize(path) < 1024:
        return None
    try:
        with wave.open(path, "rb") as w:
            frames = w.readframes(w.getnframes())
    except (wave.Error, EOFError):
        return None
    if not frames:
        return None
    return np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32767.0


def run_recover():
    """Re-transcribe the last recording from disk → clipboard."""
    audio = _load_wav(RECORDING_PATH)
    if audio is None or len(audio) < SAMPLE_RATE * 0.5:
        log(f"⚠  No recoverable audio at {RECORDING_PATH}")
        sys.exit(1)
    secs = len(audio) / SAMPLE_RATE
    log(f"⟳  Recovering {secs:.1f}s …")
    text = transcribe(audio)
    if not text:
        log("⚠  Nothing transcribed from recovery audio")
        sys.exit(1)
    subprocess.run(["pbcopy"], input=text.encode("utf-8"), check=True)
    _log_transcription(text, "recover", secs, trigger="recover")
    log(f"✓  Recovered → clipboard: {text}")
    print(text)


def run_pipe_mode():
    """Record one utterance, print to stdout, exit. No hotkey listener."""
    box: dict = {}
    done = threading.Event()

    def _cb(audio, secs):
        box["text"] = transcribe(audio)
        done.set()

    rec = AudioRecorder(sample_rate=SAMPLE_RATE, device=AUDIO_DEVICE, on_audio=_cb)
    log("●  Recording... (press Enter to stop)")
    rec.start()
    try:
        input()
    except (KeyboardInterrupt, EOFError):
        pass
    rec.stop()
    if not done.wait(TRANSCRIBE_TIMEOUT + 5):
        sys.exit(1)
    text = box.get("text", "")
    if not text:
        sys.exit(1)
    _log_transcription(text, "pipe", None, trigger="once")
    print(text)


def main():
    parser = argparse.ArgumentParser(description="shh-paste — local voice transcription")
    parser.add_argument("--once", action="store_true",
                        help="Pipe mode: record one utterance, print to stdout, exit")
    parser.add_argument("--recover", action="store_true",
                        help="Re-transcribe the last recording from disk → clipboard, exit")
    args = parser.parse_args()

    _init_db()
    if args.once:
        run_pipe_mode()
        return
    if args.recover:
        run_recover()
        return

    threading.Thread(target=warm_up_model, daemon=True).start()

    target_key = HOTKEY_MAP.get(HOTKEY)
    if target_key is None:
        log(f"✗  Unknown HOTKEY '{HOTKEY}'. Valid options: {', '.join(HOTKEY_MAP)}")
        sys.exit(1)

    valid_modes = ("hold", "toggle", "smart")
    if TRIGGER_MODE not in valid_modes:
        log(f"✗  Unknown TRIGGER_MODE '{TRIGGER_MODE}'. Use {', '.join(valid_modes)}.")
        sys.exit(1)
    if MEMO_MODE not in valid_modes:
        log(f"✗  Unknown MEMO_MODE '{MEMO_MODE}'. Use {', '.join(valid_modes)}.")
        sys.exit(1)

    # Two recorders so paste vs memo deliver to different processors; only one is ever
    # active at a time (shared key). The paste one streams to disk for recovery.
    paste_rec = AudioRecorder(
        sample_rate=SAMPLE_RATE, device=AUDIO_DEVICE,
        on_audio=_offload(process_audio), wav_path=RECORDING_PATH,
        op_timeout=OP_TIMEOUT, on_wedged=_on_wedged,
    )
    memo_rec = AudioRecorder(
        sample_rate=SAMPLE_RATE, device=AUDIO_DEVICE,
        on_audio=_offload(process_memo),
        op_timeout=OP_TIMEOUT, on_wedged=_on_wedged,
    )
    paste_handler = _make_handler(TRIGGER_MODE, target_key, paste_rec)
    memo_handler = _make_handler(MEMO_MODE, target_key, memo_rec)

    # SIGUSR1 = rescue: force-stop + transcribe whatever is recording, bypass the keyboard.
    def _on_rescue(signum, frame):
        def _do():
            if not (paste_handler.force_finish() or memo_handler.force_finish()):
                log("⛑  Rescue signal, but nothing was recording")
            else:
                log("⛑  Rescue: stopping + transcribing")
        threading.Thread(target=_do, daemon=True).start()
    signal.signal(signal.SIGUSR1, _on_rescue)

    log(f"✓  Model: {MODEL} (downloads on first use if not cached)")
    log(f"   Paste mode: {TRIGGER_MODE} | Key: {HOTKEY} | Auto-paste: {AUTO_PASTE}")
    log(f"   Memo mode:  Cmd+{HOTKEY} | File: {MEMO_FILE}")
    log(f"   Audio: dedicated serialized thread | self-heal > {OP_TIMEOUT:.0f}s | recovery: {RECORDING_PATH}")
    if TRIGGER_MODE == "hold":
        log(f"   Hold {HOTKEY} to record, release to transcribe.")
    elif TRIGGER_MODE == "toggle":
        log(f"   Press {HOTKEY} to start, press again to stop and transcribe.")
    else:
        log(f"   Tap {HOTKEY} to toggle. Long-press ({LONG_PRESS_THRESHOLD}s) to hold-to-record.")
    log(f"   Rescue a stuck recording: pkill -USR1 -f shh_paste.py  (or: shh rescue)")
    log("")
    log("   If the hotkey does not respond, grant Accessibility access:")
    log("   System Settings → Privacy & Security → Accessibility → enable Terminal")
    log("")
    log("Ready — waiting for hotkey (Ctrl+C to quit)...")

    _cmd_held = {"value": False}

    def _dispatch_press(key):
        if key in (keyboard.Key.cmd, keyboard.Key.cmd_l, keyboard.Key.cmd_r):
            _cmd_held["value"] = True
            return
        (memo_handler if _cmd_held["value"] else paste_handler).on_press(key)

    def _dispatch_release(key):
        if key in (keyboard.Key.cmd, keyboard.Key.cmd_l, keyboard.Key.cmd_r):
            _cmd_held["value"] = False
            return
        (memo_handler if _cmd_held["value"] else paste_handler).on_release(key)

    with keyboard.Listener(on_press=_dispatch_press, on_release=_dispatch_release) as listener:
        try:
            listener.join()
        except KeyboardInterrupt:
            log("Shutting down.")


if __name__ == "__main__":
    main()
