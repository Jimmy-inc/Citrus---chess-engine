#!/bin/bash
# Launcher for lichess-bot and chess GUIs, for the version two net.
#
# They start the engine as a bare subprocess, so the virtualenv has to be
# activated here rather than in whatever shell you happen to be using.
#
# Make it executable once:   chmod +x run_engine_v2.sh

cd "$HOME/Desktop/chessbot" || exit 1
source .venv/bin/activate
exec python uci_v2.py "$@"
