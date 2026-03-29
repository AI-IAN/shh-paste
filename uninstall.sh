#!/usr/bin/env bash
pkill -f "shh_paste.py" 2>/dev/null && echo "✓ Stopped shh-paste" || true
PLIST="$HOME/Library/LaunchAgents/com.shh-paste.plist"
if [ -f "$PLIST" ]; then
    launchctl unload "$PLIST" 2>/dev/null || true
    rm "$PLIST"
    echo "✓ Removed LaunchAgent"
fi
echo "✓ shh-paste uninstalled (model cache kept at ~/.cache/huggingface)"
