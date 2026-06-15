#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "sounddevice>=0.4.0",
#   "numpy>=1.24.0",
# ]
# ///
"""
Headless stress test for recorder.AudioRecorder.

Reproduces the open/stop churn that deadlocked CoreAudio — especially immediate
stop→start back-to-back — without needing the keyboard or any speech. Every control
op has a hard timeout; if one hangs, the test reports HANG and exits non-zero instead
of freezing. Uses the real default mic in short bursts (the orange dot will blink).

Run:  uv run test_recorder.py
"""

import sys
import time

import numpy as np

from recorder import AudioRecorder

OP_TIMEOUT = 8.0          # an op slower than this = wedged
RESULTS = {"delivered": 0, "secs": 0.0}
LATENCIES: list[tuple[str, float]] = []
HANGS: list[str] = []
ERRORS: list[str] = []


def on_audio(audio: np.ndarray, seconds: float):
    RESULTS["delivered"] += 1
    RESULTS["secs"] += seconds


def wedged(op, secs):
    print(f"  !! WEDGED on {op} (> {secs:.0f}s)", flush=True)


def await_op(cmd, label):
    """Wait for one control op; record latency or a hang."""
    if cmd is None:
        return
    ok = cmd.done.wait(OP_TIMEOUT)
    if not ok:
        HANGS.append(label)
        print(f"  HANG: {label} did not complete in {OP_TIMEOUT:.0f}s", flush=True)
        return
    if cmd.error:
        ERRORS.append(f"{label}: {cmd.error!r}")
        print(f"  ERROR: {label}: {cmd.error!r}", flush=True)
    LATENCIES.append((label, cmd.elapsed))


def phase(name):
    print(f"\n=== {name} ===", flush=True)


def main():
    rec = AudioRecorder(sample_rate=16000, on_audio=on_audio, on_wedged=wedged,
                        op_timeout=OP_TIMEOUT, wav_path=None)
    t_start = time.monotonic()

    phase("Phase 1 — normal cycles (start, record ~0.7s, stop) x10")
    for i in range(10):
        await_op(rec.start(), f"start#{i}")
        time.sleep(0.7)
        await_op(rec.stop(), f"stop#{i}")
        if HANGS:
            break
        print(f"  cycle {i}: ok", flush=True)

    phase("Phase 2 — immediate stop→start back-to-back (the deadlock pattern) x20")
    for i in range(20):
        await_op(rec.start(), f"r2.start#{i}")
        time.sleep(0.25)
        # stop and immediately re-start with no gap — max HAL stress
        await_op(rec.stop(), f"r2.stop#{i}")
        if HANGS:
            break
    print(f"  completed {min(20, i+1)} back-to-back cycles", flush=True)

    phase("Phase 3 — torture: queue overlapping intents without waiting x30")
    # Fire start/stop pairs fast so several commands sit in the queue at once.
    cmds = []
    for i in range(30):
        cmds.append((f"t.start#{i}", rec.start()))
        time.sleep(0.05)
        cmds.append((f"t.stop#{i}", rec.stop()))
        time.sleep(0.05)
    for label, c in cmds:
        await_op(c, label)
        if HANGS:
            break
    print(f"  drained {len(cmds)} queued ops", flush=True)

    rec.shutdown()
    elapsed = time.monotonic() - t_start

    # ── summary ──
    print("\n" + "=" * 48)
    lat = [e for _, e in LATENCIES]
    print(f"wall time:        {elapsed:.1f}s")
    print(f"control ops:      {len(LATENCIES)}")
    if lat:
        lat_sorted = sorted(lat)
        p50 = lat_sorted[len(lat_sorted) // 2]
        print(f"op latency p50:   {p50*1000:.0f} ms")
        print(f"op latency max:   {max(lat)*1000:.0f} ms")
        slow = [(l, e) for l, e in LATENCIES if e > 1.0]
        if slow:
            print(f"slow ops (>1s):   {slow}")
    print(f"clips delivered:  {RESULTS['delivered']}  ({RESULTS['secs']:.1f}s audio)")
    print(f"hangs:            {len(HANGS)}  {HANGS}")
    print(f"errors:           {len(ERRORS)}  {ERRORS}")

    ok = not HANGS and not ERRORS and not rec.wedged
    print("\nRESULT:", "PASS ✅" if ok else "FAIL ❌")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
