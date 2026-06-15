# Changelog

All notable changes to shh-paste are documented here.

## [2.0.0] - 2026-06-14 - Reliability rebuild

The hotkey could freeze mid-recording: the orange mic dot stuck on, taps ignored, and the
clip lost. Two live stack traces traced it to a CoreAudio HAL-mutex deadlock, hit when a
PortAudio stream was opened or stopped **on the keyboard's event-tap thread**, or when an
open overlapped a still-closing stop. This release reworks the audio layer so neither can
happen, and makes a recording recoverable even if something does go wrong.

### Added
- **Dedicated, serialized audio engine** (`recorder.py`): all microphone open/stop runs on
  one worker thread, one operation at a time, never on the keyboard thread. Key callbacks
  only post non-blocking start/stop intents.
- **Crash-proof recovery:** audio is streamed to `data/last_recording.wav` as you speak;
  `uv run shh_paste.py --recover` re-transcribes the last clip to the clipboard.
- **SIGUSR1 rescue:** `pkill -USR1 -f shh_paste.py` force-stops and transcribes a stuck
  recording without the keyboard.
- **Self-heal:** if an audio operation wedges for more than 8 seconds, the daemon restarts
  itself; the in-flight clip is still recoverable from disk.
- **`shh`** control helper: one-word `start` / `stop` / `restart` / `status` / `logs` /
  `rescue` / `recover` / `stuck`.
- Headless tests: `test_recorder.py` (open/stop churn stress, incl. the deadlock pattern)
  and `test_e2e.py` (audio engine to transcriber).

### Changed
- Listeners now translate key events into intents for the audio engine and deliver audio
  asynchronously. The smart / hold / toggle and paste / memo UX is unchanged.

### Fixed
- Hotkey freeze / stuck microphone ("orange dot stays on") under rapid start-stop and on
  longer recordings.

### Unchanged
- Transcription (`whisper-large-v3-turbo` via MLX), the silence/repetition hallucination
  guards, delivery, configuration, and all trigger/memo modes carry over as-is.
