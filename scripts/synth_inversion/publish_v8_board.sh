#!/usr/bin/env bash
# Owner: ANTIGRAVITY <antigravity@aavepyora.online>
# publish_v8_board.sh — publish staged V8 Dynamic Synth Inversion board from ~/evals_aac/ to aavepyora.online
# NOTE: Executed strictly by Kim's team per deployment policy ("only my team can publish them").
set -euo pipefail

CONF="$HOME/.config/aavepyora.conf"
if [ ! -f "$CONF" ]; then
    echo "Error: missing $CONF"
    exit 1
fi

KEY=$(grep -E '^KEY=' "$CONF" | cut -d= -f2-)
HOST=$(grep -E '^HOST=' "$CONF" | cut -d= -f2-)
DEST_DIR=$(grep -E '^DEST_DIR=' "$CONF" | cut -d= -f2-)
DEST="${DEST_DIR}/evals"

BOARD_DIR="/home/kim/evals_aac/synth_inversion_v8"
REDIRECT_HTML="/home/kim/evals_aac/synth_inversion_v8.html"
INDEX_HTML="/home/kim/evals_aac/index.html"

if [ ! -d "$BOARD_DIR" ] || [ ! -f "$BOARD_DIR/index.html" ]; then
    echo "Error: Board directory or index.html missing at $BOARD_DIR"
    exit 1
fi

echo "=== Running leak scan on HTML boards ==="
if grep -rnoE '/home/|/run/media|\.ckpt|\.safetensors|\.pt"|[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}|password|token=' "$BOARD_DIR"/*.html "$REDIRECT_HTML"; then
    echo "LEAK SCAN FAILED: private local paths or credentials found. Aborting publish."
    exit 2
fi
echo "Leak scan passed cleanly."

echo "=== Syncing V8 Audition Board Assets to https://aavepyora.online/files/evals/synth_inversion_v8/ ==="
rsync -avz --chmod=F644,D755 -e "ssh -i $KEY -o BatchMode=yes" \
    "$BOARD_DIR/" \
    "$HOST:$DEST/synth_inversion_v8/"

echo "=== Syncing Redirect & Catalog Index ==="
rsync -avz --chmod=F644,D755 -e "ssh -i $KEY -o BatchMode=yes" \
    "$REDIRECT_HTML" \
    "$INDEX_HTML" \
    "$HOST:$DEST/"

echo "=== PUBLISH COMPLETE ==="
echo "Live URLs:"
echo "  - https://aavepyora.online/files/evals/synth_inversion_v8/index.html"
echo "  - https://aavepyora.online/files/evals/synth_inversion_v8.html"
echo "  - https://aavepyora.online/files/evals/index.html"
