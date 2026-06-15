#!/usr/bin/env python3
"""
recorder_v2.py — dedicated-audio-thread recorder for shh-paste.

Root-cause fix for the CoreAudio deadlock that froze the hotkey.

Two live stack traces showed the freeze is a deadlock on CoreAudio's internal HAL
mutex, hit when a PortAudio stream is opened or stopped. It becomes likely when:
  (a) an open and a stop run concurrently (overlap), or
  (b) a control op runs on the pynput event-tap thread (a CFRunLoop thread), whose run
      loop CoreAudio itself needs — a self-deadlock.

This recorder removes both conditions:
  * ALL PortAudio control ops (open/start/abort/close) run on ONE dedicated worker
    thread, pulled from a queue and executed strictly one at a time. Open and stop can
    never overlap.
  * That thread is a plain pthread, never the keyboard/run-loop thread.
  * stop uses abort() (Pa_AbortStream) — immediate, no buffer drain.

The keyboard callback only calls start()/stop(), which just enqueue an intent and
return instantly — they never touch CoreAudio and never block. Finished audio is
delivered via the on_audio callback (fired from the audio thread).
"""

from __future__ import annotations

import os
import queue
import threading
import time
import wave

import numpy as np
import sounddevice as sd


class _Cmd:
    __slots__ = ("kind", "done", "elapsed", "error")

    def __init__(self, kind: str):
        self.kind = kind            # 'start' | 'stop' | 'shutdown'
        self.done = threading.Event()
        self.elapsed = 0.0
        self.error: Exception | None = None


class AudioRecorder:
    def __init__(
        self,
        sample_rate: int = 16000,
        device=None,
        on_audio=None,                 # on_audio(audio_np: np.ndarray, seconds: float)
        wav_path: str | None = None,   # stream a recovery copy here if set
        min_seconds: float = 0.5,      # discard shorter clips
        silence_stop: float = 0.0,     # auto-stop after this many seconds of silence (0=off)
        silence_rms: float = 0.01,
        min_auto_stop: float = 2.0,
        op_timeout: float = 8.0,       # an op slower than this is treated as wedged
        on_wedged=None,                # on_wedged(op: str, seconds: float) — escalation hook
    ):
        self._sr = sample_rate
        self._device = device
        self._on_audio = on_audio
        self._wav_path = wav_path
        self._min_seconds = min_seconds
        self._silence_stop = silence_stop
        self._silence_rms = silence_rms
        self._min_auto_stop = min_auto_stop
        self._op_timeout = op_timeout
        self._on_wedged = on_wedged

        # Intent state (flips synchronously so the keyboard layer sees it immediately).
        self._intent_lock = threading.Lock()
        self._intended = False

        # Audio-thread-owned state (touched only by the audio thread + PortAudio cb).
        self._stream: sd.InputStream | None = None
        self._chunks: list[np.ndarray] = []
        self._wav: wave.Wave_write | None = None
        self._start_ts = 0.0
        self._last_voice_ts = 0.0

        self._q: "queue.Queue[_Cmd]" = queue.Queue()
        self._wedged = False
        self._thread = threading.Thread(target=self._run, name="audio-ctl", daemon=True)
        self._thread.start()
        self._monitor = threading.Thread(target=self._silence_monitor, name="audio-mon", daemon=True)
        self._monitor.start()

    # ── public, non-blocking API ────────────────────────────────────────────────
    @property
    def active(self) -> bool:
        return self._intended

    @property
    def wedged(self) -> bool:
        return self._wedged

    def start(self) -> _Cmd | None:
        with self._intent_lock:
            if self._intended:
                return None
            self._intended = True
        return self._enqueue("start")

    def stop(self) -> _Cmd | None:
        with self._intent_lock:
            if not self._intended:
                return None
            self._intended = False
        return self._enqueue("stop")

    def shutdown(self):
        self._enqueue("shutdown")

    # ── internals ───────────────────────────────────────────────────────────────
    def _enqueue(self, kind: str) -> _Cmd:
        cmd = _Cmd(kind)
        self._q.put(cmd)
        # Watchdog: if this op doesn't complete in time, the audio device is wedged.
        if kind in ("start", "stop"):
            threading.Thread(target=self._watch, args=(cmd,), daemon=True).start()
        return cmd

    def _watch(self, cmd: _Cmd):
        if not cmd.done.wait(self._op_timeout) and not self._wedged:
            self._wedged = True
            if self._on_wedged:
                try:
                    self._on_wedged(cmd.kind, self._op_timeout)
                except Exception:
                    pass

    def _run(self):
        while True:
            cmd = self._q.get()
            t0 = time.monotonic()
            try:
                if cmd.kind == "start":
                    self._do_start()
                elif cmd.kind == "stop":
                    self._do_stop()
                elif cmd.kind == "shutdown":
                    self._do_stop()
                    cmd.elapsed = time.monotonic() - t0
                    cmd.done.set()
                    return
            except Exception as e:  # noqa: BLE001
                cmd.error = e
            cmd.elapsed = time.monotonic() - t0
            cmd.done.set()

    def _pa_callback(self, indata, frames, t, status):
        self._chunks.append(indata.copy())
        if self._wav is not None:
            try:
                self._wav.writeframes((np.clip(indata, -1.0, 1.0) * 32767).astype(np.int16).tobytes())
            except Exception:
                pass
        if self._silence_stop > 0:
            rms = float(np.sqrt(np.mean(np.square(indata.astype(np.float64)))))
            if rms >= self._silence_rms:
                self._last_voice_ts = time.monotonic()

    def _do_start(self):
        if self._stream is not None:
            return
        self._chunks = []
        self._start_ts = time.monotonic()
        self._last_voice_ts = self._start_ts
        if self._wav_path:
            try:
                os.makedirs(os.path.dirname(self._wav_path), exist_ok=True)
                self._wav = wave.open(self._wav_path, "wb")
                self._wav.setnchannels(1)
                self._wav.setsampwidth(2)
                self._wav.setframerate(self._sr)
            except Exception:
                self._wav = None
        self._stream = sd.InputStream(
            samplerate=self._sr,
            channels=1,
            dtype="float32",
            callback=self._pa_callback,
            device=self._device,
        )
        self._stream.start()

    def _do_stop(self):
        stream = self._stream
        self._stream = None
        if self._wav is not None:
            try:
                self._wav.close()
            except Exception:
                pass
            self._wav = None
        if stream is not None:
            try:
                stream.abort()      # immediate stop, no drain
            except Exception:
                pass
            try:
                stream.close()
            except Exception:
                pass
        chunks, self._chunks = self._chunks, []
        if not chunks:
            return
        audio = np.concatenate(chunks, axis=0)
        if len(audio) < self._sr * self._min_seconds:
            return
        if self._on_audio:
            self._on_audio(audio, len(audio) / self._sr)

    def _silence_monitor(self):
        if self._silence_stop <= 0:
            return
        while True:
            time.sleep(0.3)
            if not self._intended or self._stream is None:
                continue
            now = time.monotonic()
            if (now - self._start_ts) >= self._min_auto_stop and (now - self._last_voice_ts) >= self._silence_stop:
                self.stop()
