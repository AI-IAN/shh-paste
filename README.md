# shh-paste

Local voice transcription. Speak, and it types. All processing on-device — no cloud, no subscription, no data leaves your machine.

Ships ready to run on macOS (Apple Silicon) via [MLX Whisper](https://github.com/ml-explore/mlx-examples/tree/main/whisper). Adaptable to Linux/Windows — see [Other platforms](#other-platforms).

## Requirements

- macOS on Apple Silicon (M1+) — or see [Other platforms](#other-platforms)
- Python 3.11+
- [uv](https://github.com/astral-sh/uv)

## Quick start

```bash
git clone https://github.com/AI-IAN/shh-paste.git
cd shh-paste
uv run shh_paste.py
```

Or use the installer for auto-start on login:
```bash
bash install.sh
```

## How it works

Press a hotkey → speak → release (or press again) → transcribed text is typed into the active app and copied to clipboard.

Three trigger modes:

| Mode | Behavior |
|------|----------|
| **smart** (default) | Tap to toggle recording, long-press for hold-to-record |
| **hold** | Hold the key to record, release to transcribe |
| **toggle** | Press once to start, press again to stop |

Two output modes, same hotkey:

| Combo | What happens |
|-------|-------------|
| Hotkey alone | Transcribes → pastes into active app |
| Cmd/Ctrl + Hotkey | Transcribes → appends timestamped entry to a memo file |

Pipe mode for scripting:
```bash
uv run shh_paste.py --once >> ~/notes.md
```

## Configuration

Edit the `CONFIG` block at the top of `shh_paste.py`, or create a TOML config file:

```bash
mkdir -p ~/.config/shh-paste
cat > ~/.config/shh-paste/config.toml << 'EOF'
memo_file = "~/my-voice-notes.md"
hotkey = "right_alt"
trigger_mode = "smart"
EOF
```

A `shh-paste.toml` in the current directory also works (local config takes lower priority).

Available options:

| Key | Default | Description |
|-----|---------|-------------|
| `model` | `mlx-community/whisper-large-v3-turbo` | MLX Whisper model (HuggingFace repo) |
| `hotkey` | `right_alt` | Trigger key: `right_alt`, `left_alt`, `right_ctrl`, `right_shift`, `f13`-`f15` |
| `trigger_mode` | `smart` | `smart`, `hold`, or `toggle` |
| `memo_file` | `~/voice-notes.md` | Where modifier+hotkey memos are saved |
| `memo_mode` | `smart` | Trigger mode for memo recording |
| `auto_paste` | `true` | Type text into active app after transcription |
| `filler_remove` | `false` | Strip "um", "uh", "like", "you know" |
| `max_record` | `600` | Max recording length in seconds |
| `notification` | `true` | Show desktop notifications |

## Permissions (macOS)

The first run will prompt for **Microphone** access. You also need to manually enable:

- **Accessibility** (System Settings → Privacy & Security → Accessibility → enable your terminal app) — required for auto-paste
- Without Accessibility, transcriptions still go to clipboard

## Scripts

| Script | What |
|--------|------|
| `start.sh` | Start shh-paste |
| `stop.sh` | Stop shh-paste |
| `install.sh` | Install deps, download model, optionally set up auto-start on login |
| `uninstall.sh` | Stop and remove LaunchAgent |

## Transcription log

All transcriptions are logged to `data/shh_paste.db` (SQLite) with timestamp, text, duration, word count, model, trigger mode, and which app was active. Useful for analyzing your dictation habits.

## Other platforms

Built for macOS (Apple Silicon) out of the box. The core logic — audio capture, transcription, hotkey listener, auto-paste — uses cross-platform libraries (`sounddevice`, `pynput`). To run on Linux or Windows, you'd swap:

- `mlx-whisper` → [`faster-whisper`](https://github.com/SYSTRAN/faster-whisper) (CPU/CUDA)
- `pbcopy` → `xclip`/`xsel` (Linux) or `clip.exe` (Windows)
- `osascript` notifications → `notify-send` (Linux)

## License

MIT
