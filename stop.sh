#!/usr/bin/env bash
# Stop shh-paste
if pkill -f "shh_paste.py"; then
    echo "✓ shh-paste stopped"
else
    echo "shh-paste is not running"
fi
