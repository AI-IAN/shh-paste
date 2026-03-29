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
import sqlite3
import sys
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path
import tomllib
import numpy as np
import sounddevice as sd
from pynput import keyboard
import mlx_whisper


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

# Voice memo mode — Cmd + HOTKEY appends timestamped transcription to a file
MEMO_FILE     = os.path.expanduser("~/voice-notes.md")  # Where memos are saved
MEMO_MODE     = "smart"       # Same options as TRIGGER_MODE: "hold" | "toggle" | "smart"

# ============================

# ── Load optional config overrides ────────────────────────────────────────────
_CONFIG_KEYS = {
    "model", "trigger_mode", "long_press_threshold", "hotkey",
    "filler_remove", "auto_paste", "notification", "notify_start",
    "max_record", "warn_before", "audio_device", "sample_rate",
    "memo_file", "memo_mode",
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
    print(msg, flush=True)


def clean(text: str) -> str:
    text = text.strip()
    if FILLER_REMOVE:
        text = FILLER_RE.sub('', text).strip()
        text = re.sub(r'  +', ' ', text)
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
    result = mlx_whisper.transcribe(audio_flat, path_or_hf_repo=MODEL, language="en", condition_on_previous_text=False)
    return clean(result["text"])


def process_audio(audio: np.ndarray):
    """Transcribe and deliver audio. Serialized — no parallel transcriptions."""
    audio_seconds = len(audio) / SAMPLE_RATE
    with _transcribe_lock:
        try:
            text = transcribe(audio)
            deliver(text, audio_seconds)
        except Exception as e:
            log(f"✗  Error during transcription: {e}")


def process_memo(audio: np.ndarray):
    """Transcribe and save as voice memo. Serialized."""
    audio_seconds = len(audio) / SAMPLE_RATE
    with _transcribe_lock:
        try:
            text = transcribe(audio)
            deliver_memo(text, audio_seconds)
        except Exception as e:
            log(f"✗  Error during memo transcription: {e}")


class Recorder:
    """Handles audio capture via sounddevice."""

    def __init__(self):
        self._chunks: list[np.ndarray] = []
        self._stream: sd.InputStream | None = None
        self._lock = threading.Lock()

    @property
    def active(self) -> bool:
        return self._stream is not None

    def start(self):
        with self._lock:
            if self._stream is not None:
                return
            self._chunks = []
            log("●  Recording...")

            def _callback(indata, frames, t, status):
                self._chunks.append(indata.copy())

            self._stream = sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype='float32',
                callback=_callback,
                device=AUDIO_DEVICE,
            )
            self._stream.start()

    def stop(self) -> np.ndarray | None:
        """Stop recording and return audio array, or None if too short."""
        with self._lock:
            if self._stream is None:
                return None
            self._stream.stop()
            self._stream.close()
            self._stream = None

        if not self._chunks:
            return None
        audio = np.concatenate(self._chunks, axis=0)
        # Require at least 0.5s to avoid accidental hotkey triggers
        if len(audio) < SAMPLE_RATE * 0.5:
            log("⚠  Recording too short, ignoring")
            return None
        return audio


class HoldListener:
    """
    Hold-to-record: hold HOTKEY to record, release to transcribe.
    Recordings longer than MAX_RECORD seconds are auto-stopped.
    """

    def __init__(self, target_key, process_fn=None):
        self._key = target_key
        self._recorder = Recorder()
        self._timer: threading.Timer | None = None
        self._warn_timer: threading.Timer | None = None
        self._lock = threading.Lock()
        self._process_fn = process_fn or process_audio

    def _warn(self):
        log(f"⚠  {WARN_BEFORE}s of recording time remaining")
        if NOTIFICATION:
            notify(f"Recording — {WARN_BEFORE}s remaining")

    def _auto_stop(self):
        log(f"⚠  Max record time ({MAX_RECORD}s) reached, stopping")
        if NOTIFICATION:
            notify(f"Max {MAX_RECORD}s reached — transcribing…")
        audio = self._recorder.stop()
        if audio is not None:
            threading.Thread(target=self._process_fn, args=(audio,), daemon=True).start()

    def on_press(self, key):
        if key != self._key:
            return
        with self._lock:
            if not self._recorder.active:
                self._recorder.start()
                if NOTIFY_START:
                    notify("Recording…")
                warn_at = MAX_RECORD - WARN_BEFORE
                if warn_at > 0:
                    self._warn_timer = threading.Timer(warn_at, self._warn)
                    self._warn_timer.start()
                self._timer = threading.Timer(MAX_RECORD, self._auto_stop)
                self._timer.start()

    def on_release(self, key):
        if key != self._key:
            return
        with self._lock:
            if self._timer:
                self._timer.cancel()
                self._timer = None
            if self._warn_timer:
                self._warn_timer.cancel()
                self._warn_timer = None
            if self._recorder.active:
                audio = self._recorder.stop()
                if audio is not None:
                    threading.Thread(target=self._process_fn, args=(audio,), daemon=True).start()


class ToggleListener:
    """
    Toggle mode: press HOTKEY to start recording, press again to stop + transcribe.
    """

    def __init__(self, target_key, process_fn=None):
        self._key = target_key
        self._recorder = Recorder()
        self._lock = threading.Lock()
        self._timer: threading.Timer | None = None
        self._warn_timer: threading.Timer | None = None
        self._process_fn = process_fn or process_audio

    def _warn(self):
        log(f"⚠  {WARN_BEFORE}s of recording time remaining")
        if NOTIFICATION:
            notify(f"Recording — {WARN_BEFORE}s remaining")

    def _auto_stop(self):
        log(f"⚠  Max record time ({MAX_RECORD}s) reached, stopping")
        if NOTIFICATION:
            notify(f"Max {MAX_RECORD}s reached — transcribing…")
        audio = self._recorder.stop()
        if audio is not None:
            threading.Thread(target=self._process_fn, args=(audio,), daemon=True).start()

    def _cancel_timers(self):
        if self._timer:
            self._timer.cancel()
            self._timer = None
        if self._warn_timer:
            self._warn_timer.cancel()
            self._warn_timer = None

    def on_press(self, key):
        if key != self._key:
            return
        with self._lock:
            if not self._recorder.active:
                self._recorder.start()
                if NOTIFY_START:
                    notify("Recording…")
                warn_at = MAX_RECORD - WARN_BEFORE
                if warn_at > 0:
                    self._warn_timer = threading.Timer(warn_at, self._warn)
                    self._warn_timer.start()
                self._timer = threading.Timer(MAX_RECORD, self._auto_stop)
                self._timer.start()
            else:
                self._cancel_timers()
                audio = self._recorder.stop()
                if audio is not None:
                    threading.Thread(target=self._process_fn, args=(audio,), daemon=True).start()

    def on_release(self, key):
        pass  # toggle mode ignores key release


class SmartListener:
    """
    Smart mode: tap to toggle, long-press to hold-to-record.
    Tap (release before LONG_PRESS_THRESHOLD) → press again to stop and transcribe.
    Long-press (still held after LONG_PRESS_THRESHOLD) → release to transcribe.
    Audio starts immediately on press so nothing is lost during the decision window.
    """

    _IDLE             = "idle"
    _DECIDING         = "deciding"
    _HOLD_RECORDING   = "hold_recording"
    _TOGGLE_RECORDING = "toggle_recording"

    def __init__(self, target_key, process_fn=None):
        self._key = target_key
        self._recorder = Recorder()
        self._lock = threading.Lock()
        self._state = self._IDLE
        self._decide_timer: threading.Timer | None = None
        self._warn_timer: threading.Timer | None = None
        self._max_timer: threading.Timer | None = None
        self._process_fn = process_fn or process_audio

    def _start_limit_timers(self):
        warn_at = MAX_RECORD - WARN_BEFORE
        if warn_at > 0:
            self._warn_timer = threading.Timer(warn_at, self._on_warn)
            self._warn_timer.start()
        self._max_timer = threading.Timer(MAX_RECORD, self._on_max)
        self._max_timer.start()

    def _cancel_limit_timers(self):
        if self._warn_timer:
            self._warn_timer.cancel()
            self._warn_timer = None
        if self._max_timer:
            self._max_timer.cancel()
            self._max_timer = None

    def _on_warn(self):
        log(f"⚠  {WARN_BEFORE}s of recording time remaining")
        if NOTIFICATION:
            notify(f"Recording — {WARN_BEFORE}s remaining")

    def _on_max(self):
        log(f"⚠  Max record time ({MAX_RECORD}s) reached, stopping")
        if NOTIFICATION:
            notify(f"Max {MAX_RECORD}s reached — transcribing…")
        with self._lock:
            self._state = self._IDLE
            self._warn_timer = None
            self._max_timer = None
        audio = self._recorder.stop()
        if audio is not None:
            threading.Thread(target=self._process_fn, args=(audio,), daemon=True).start()

    def _on_decide(self):
        """Decision timer fired — long press confirmed, switch to hold mode."""
        with self._lock:
            if self._state != self._DECIDING:
                return
            self._state = self._HOLD_RECORDING
            self._start_limit_timers()
        log("●  Hold mode (release to transcribe)")
        if NOTIFY_START:
            notify("Recording… release to stop")

    def on_press(self, key):
        if key != self._key:
            return
        with self._lock:
            if self._state == self._IDLE:
                self._recorder.start()
                self._state = self._DECIDING
                self._decide_timer = threading.Timer(LONG_PRESS_THRESHOLD, self._on_decide)
                self._decide_timer.start()
            elif self._state == self._TOGGLE_RECORDING:
                # Second tap — stop and transcribe
                self._cancel_limit_timers()
                self._state = self._IDLE
                audio = self._recorder.stop()
                if audio is not None:
                    threading.Thread(target=self._process_fn, args=(audio,), daemon=True).start()

    def on_release(self, key):
        if key != self._key:
            return
        with self._lock:
            if self._state == self._DECIDING:
                # Released before threshold — tap confirmed, switch to toggle mode
                if self._decide_timer:
                    self._decide_timer.cancel()
                    self._decide_timer = None
                self._state = self._TOGGLE_RECORDING
                self._start_limit_timers()
                log("●  Toggle mode (press again to stop)")
                if NOTIFY_START:
                    notify("Recording… tap to stop")
            elif self._state == self._HOLD_RECORDING:
                # Released during hold — stop and transcribe
                self._cancel_limit_timers()
                self._state = self._IDLE
                audio = self._recorder.stop()
                if audio is not None:
                    threading.Thread(target=self._process_fn, args=(audio,), daemon=True).start()


def run_pipe_mode():
    """Record one utterance, print to stdout, exit. No hotkey listener."""
    recorder = Recorder()
    log("●  Recording... (press Enter to stop)")
    recorder.start()
    try:
        input()  # block until Enter
    except (KeyboardInterrupt, EOFError):
        pass
    audio = recorder.stop()
    if audio is None:
        sys.exit(1)
    text = transcribe(audio)
    if not text:
        sys.exit(1)
    _log_transcription(text, "pipe", len(audio) / SAMPLE_RATE, trigger="once")
    print(text)  # stdout only — no clipboard, no paste, no notification


def _make_handler(mode, target_key, process_fn):
    """Create a listener handler for the given mode and processor function."""
    listeners = {"hold": HoldListener, "toggle": ToggleListener, "smart": SmartListener}
    return listeners[mode](target_key, process_fn=process_fn)


def main():
    parser = argparse.ArgumentParser(description="shh-paste — local voice transcription")
    parser.add_argument("--once", action="store_true",
                        help="Pipe mode: record one utterance, print to stdout, exit")
    args = parser.parse_args()

    _init_db()

    if args.once:
        run_pipe_mode()
        return

    # Validate config
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

    log(f"✓  Model: {MODEL} (downloads on first use if not cached)")
    log(f"   Paste mode: {TRIGGER_MODE} | Key: {HOTKEY} | Auto-paste: {AUTO_PASTE}")
    log(f"   Memo mode:  Cmd+{HOTKEY} | File: {MEMO_FILE}")
    log(f"   Filler removal: {FILLER_REMOVE} | Notification: {NOTIFICATION}")
    if TRIGGER_MODE == "hold":
        log(f"   Hold {HOTKEY} to record, release to transcribe.")
    elif TRIGGER_MODE == "toggle":
        log(f"   Press {HOTKEY} to start, press again to stop and transcribe.")
    else:
        log(f"   Tap {HOTKEY} to toggle. Long-press ({LONG_PRESS_THRESHOLD}s) to hold-to-record.")
    log(f"   Hold Cmd + tap {HOTKEY} for voice memo → {MEMO_FILE}")
    log("")
    log("   If the hotkey does not respond, grant Accessibility access:")
    log("   System Settings → Privacy & Security → Accessibility → enable Terminal")
    log("")
    log("Ready — waiting for hotkey (Ctrl+C to quit)...")

    # Build handlers — both use the same key, routed by Cmd modifier state
    paste_handler = _make_handler(TRIGGER_MODE, target_key, process_audio)
    memo_handler = _make_handler(MEMO_MODE, target_key, process_memo)

    # Track Cmd modifier state
    _cmd_held = {"value": False}

    def _dispatch_press(key):
        if key in (keyboard.Key.cmd, keyboard.Key.cmd_l, keyboard.Key.cmd_r):
            _cmd_held["value"] = True
            return
        if _cmd_held["value"]:
            memo_handler.on_press(key)
        else:
            paste_handler.on_press(key)

    def _dispatch_release(key):
        if key in (keyboard.Key.cmd, keyboard.Key.cmd_l, keyboard.Key.cmd_r):
            _cmd_held["value"] = False
            return
        if _cmd_held["value"]:
            memo_handler.on_release(key)
        else:
            paste_handler.on_release(key)

    with keyboard.Listener(
        on_press=_dispatch_press,
        on_release=_dispatch_release,
    ) as listener:
        try:
            listener.join()
        except KeyboardInterrupt:
            log("Shutting down.")


if __name__ == "__main__":
    main()
