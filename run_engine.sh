#!/bin/bash
# Launcher for lichess-bot and chess GUIs.
#
# They start the engine as a bare subprocess, so the virtualenv has to be
# activated here rather than in whatever shell you happen to be using.
#
# Make it executable once:   chmod +x run_engine.sh

cd "$HOME/Desktop/chessbot" || exit 1
source .venv/bin/activate
exec python uci.py "$@"
