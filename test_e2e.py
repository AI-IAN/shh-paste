#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "mlx-whisper>=0.3.0", "sounddevice>=0.4.0", "numpy>=1.24.0", "pynput>=1.7.0",
# ]
# ///
"""
End-to-end plumbing test: recorder.AudioRecorder → transcriber.
Records ~2s from the mic and runs it through shh_paste.transcribe (the real
transcription path) to prove the audio engine feeds it correctly. No keyboard, no paste.
"""
import threading
import time

import shh_paste as base
from recorder import AudioRecorder

box = {}
done = threading.Event()


def on_audio(audio, secs):
    try:
        box["secs"] = secs
        box["text"] = base.transcribe(audio)   # real transcription path
    except Exception as e:
        box["error"] = repr(e)
    finally:
        done.set()


rec = AudioRecorder(sample_rate=base.SAMPLE_RATE, device=base.AUDIO_DEVICE, on_audio=on_audio)
print("recording 2s (silence is fine — testing the pipeline, not the words)…", flush=True)
rec.start()
time.sleep(2.0)
rec.stop()

if not done.wait(60):
    print("RESULT: FAIL ❌ — transcription never returned", flush=True)
    raise SystemExit(1)

if "error" in box:
    print(f"RESULT: FAIL ❌ — {box['error']}", flush=True)
    raise SystemExit(1)

print(f"captured {box.get('secs', 0):.1f}s, transcript = {box.get('text', '')!r}", flush=True)
print("RESULT: PASS ✅ — recorder_v2 audio flows through the real transcriber cleanly", flush=True)
