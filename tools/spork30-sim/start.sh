#!/bin/sh
# Simulation only: strip every live-trading credential and gate, whatever the service env says.
unset POLYMARKET_PRIVATE_KEY POLYMARKET_FUNDER POLYMARKET_SIGNATURE_TYPE LIVE_TRADING_CONFIRM POLYGON_RPC
export TRADING_MODE=SIMULATION
export BOT_DATA_DIR="${BOT_DATA_DIR:-/app/data/runtime}"
# One shared simulation in tmux; every browser attaches read-only to the same screen.
tmux new-session -d -s sim -x "${COLS:-150}" -y "${ROWS:-46}" \
  "while true; do python -m spork30.dashboard; echo restarting...; sleep 3; done"
tmux set -g window-size manual
tmux set -g status off
exec ttyd -p "${PORT:-8080}" -t titleFixed="spork30 · 5-min maker · SIMULATION" -t fontSize=13 \
  -t 'theme={"background":"#050505"}' tmux attach -r -t sim
