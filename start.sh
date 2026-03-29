#!/usr/bin/env bash
# Start shh-paste
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec uv run "$SCRIPT_DIR/shh_paste.py"
