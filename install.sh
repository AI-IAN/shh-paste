#!/usr/bin/env bash
# shh-paste installer
# Installs dependencies, downloads the Whisper model, and wires up helpers.
# Run once from this directory: bash install.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SHH_PASTE="$SCRIPT_DIR/shh_paste.py"
PLIST_DEST="$HOME/Library/LaunchAgents/com.shh-paste.plist"

echo ""
echo "╔════════════════════════════════════╗"
echo "║       shh-paste Installer          ║"
echo "╚════════════════════════════════════╝"
echo ""

# ── Homebrew ────────────────────────────────────────────────────────────────
if ! command -v brew &>/dev/null; then
    echo "→ Installing Homebrew..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    # Apple Silicon path
    eval "$(/opt/homebrew/bin/brew shellenv)" 2>/dev/null || true
else
    echo "✓ Homebrew $(brew --version | head -1)"
fi

# ── uv ──────────────────────────────────────────────────────────────────────
if ! command -v uv &>/dev/null; then
    echo "→ Installing uv..."
    brew install uv
else
    echo "✓ uv $(uv --version)"
fi

# ── Download Whisper model ───────────────────────────────────────────────────
echo "→ Pre-downloading MLX Whisper model (cached in ~/.cache/huggingface)..."
uv run --with "mlx-whisper>=0.3.0" --with "huggingface_hub" python3 - <<'PYEOF'
from huggingface_hub import snapshot_download
print("  Downloading mlx-community/whisper-large-v3-turbo...", flush=True)
snapshot_download("mlx-community/whisper-large-v3-turbo")
print("  Model ready.", flush=True)
PYEOF

# ── Make main script executable ─────────────────────────────────────────────
chmod +x "$SHH_PASTE"

# ── Generate helper scripts ──────────────────────────────────────────────────
cat > "$SCRIPT_DIR/start.sh" <<'STARTEOF'
#!/usr/bin/env bash
# Start shh-paste
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec uv run "$SCRIPT_DIR/shh_paste.py"
STARTEOF
chmod +x "$SCRIPT_DIR/start.sh"

cat > "$SCRIPT_DIR/stop.sh" <<STOPEOF
#!/usr/bin/env bash
# Stop shh-paste
if pkill -f "shh_paste.py"; then
    echo "✓ shh-paste stopped"
else
    echo "shh-paste is not running"
fi
STOPEOF
chmod +x "$SCRIPT_DIR/stop.sh"

echo "✓ Created start.sh and stop.sh"

# ── LaunchAgent (auto-start on login) ────────────────────────────────────────
echo ""
read -r -p "Auto-start shh-paste on login? [y/N] " REPLY
echo ""

if [[ "${REPLY:-N}" =~ ^[Yy]$ ]]; then
    UV_BIN="$(which uv)"
    mkdir -p "$HOME/Library/LaunchAgents"

    # Write plist with resolved paths
    python3 - <<PYEOF
import plistlib, pathlib

plist = {
    "Label": "com.shh-paste",
    "ProgramArguments": ["$UV_BIN", "run", "$SHH_PASTE"],
    "RunAtLoad": True,
    "KeepAlive": True,
    "StandardOutPath": "/tmp/shh-paste.log",
    "StandardErrorPath": "/tmp/shh-paste.error.log",
    "EnvironmentVariables": {
        "PATH": "/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin"
    },
}
dest = pathlib.Path("$PLIST_DEST")
dest.parent.mkdir(parents=True, exist_ok=True)
with open(dest, "wb") as f:
    plistlib.dump(plist, f)
print(f"  Written: {dest}")
PYEOF

    launchctl load "$PLIST_DEST" && echo "✓ LaunchAgent loaded — shh-paste will start on login"

    # generate uninstall that knows about the plist
    cat > "$SCRIPT_DIR/uninstall.sh" <<UNEOF
#!/usr/bin/env bash
pkill -f "shh_paste.py" 2>/dev/null && echo "✓ Stopped shh-paste" || true
if [ -f "$PLIST_DEST" ]; then
    launchctl unload "$PLIST_DEST" 2>/dev/null || true
    rm "$PLIST_DEST"
    echo "✓ Removed LaunchAgent"
fi
echo "✓ shh-paste uninstalled (model cache kept at ~/.cache/huggingface)"
UNEOF
else
    cat > "$SCRIPT_DIR/uninstall.sh" <<UNEOF
#!/usr/bin/env bash
pkill -f "shh_paste.py" 2>/dev/null && echo "✓ Stopped shh-paste" || true
PLIST="$HOME/Library/LaunchAgents/com.shh-paste.plist"
if [ -f "\$PLIST" ]; then
    launchctl unload "\$PLIST" 2>/dev/null || true
    rm "\$PLIST"
    echo "✓ Removed LaunchAgent"
fi
echo "✓ shh-paste uninstalled"
UNEOF
fi
chmod +x "$SCRIPT_DIR/uninstall.sh"

# ── Permissions reminder ─────────────────────────────────────────────────────
echo ""
echo "╔════════════════════════════════════════════════════════════╗"
echo "║             Required: macOS Permissions                   ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""
echo "  Open System Settings → Privacy & Security and enable:"
echo ""
echo "  1. Microphone      → Terminal.app"
echo "     (macOS will prompt automatically on first use)"
echo ""
echo "  2. Accessibility   → Terminal.app"
echo "     Required for auto-paste. Add Terminal.app manually."
echo "     Without this, transcribed text still goes to clipboard."
echo ""
echo "  3. Notifications   → will be prompted on first use"
echo ""
echo "  TIP: Run ./start.sh from Terminal first to trigger the"
echo "       permission dialogs before relying on auto-start."
echo ""
echo "╔════════════════════════════════════════════════════════════╗"
echo "║                        Done!                              ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""
echo "  Start:       ./start.sh"
echo "  Stop:        ./stop.sh"
echo "  Logs:        tail -f /tmp/shh-paste.log"
echo "  Uninstall:   ./uninstall.sh"
echo ""
echo "  Config:      edit the CONFIG block at the top of shh_paste.py"
echo "               TRIGGER_MODE  = 'hold', 'toggle', or 'smart'"
echo "               MODEL         = 'base' | 'small' | 'medium'"
echo "               HOTKEY        = 'right_alt' | 'f13' | 'f14' ..."
echo ""
