#!/bin/bash
# status.sh - where everything stands, in one command.
#
#   ./status.sh          a snapshot
#   ./status.sh watch    refreshes every 30 seconds, Ctrl-C to stop
#
# Make it executable once:  chmod +x status.sh

cd "$HOME/chessbot" || exit 1

snapshot() {
    echo "=============================================================="
    date "+  %a %d %b  %H:%M:%S"
    echo "=============================================================="

    echo
    echo "JOBS"
    if squeue -u "$USER" -h | grep -q .; then
        squeue -u "$USER" -o "  %.6i  %.9T  %.11M elapsed  %.11l limit  %R"
    else
        echo "  nothing queued or running"
    fi

    echo
    echo "DISK"
    quota -s 2>/dev/null | tail -1 | awk '{print "  " $2 " used of " $3}'
    echo "  data/lc0:  $(du -sh data/lc0 2>/dev/null | cut -f1 || echo 0)"
    echo "  scratch:   $(du -sh data/_work 2>/dev/null | cut -f1 || echo 0)"

    echo
    echo "LEELA DATA"
    finished=$(ls data/lc0/*.bin 2>/dev/null | grep -vc partial)
    partial=$(ls data/lc0/*.partial 2>/dev/null | wc -l)
    echo "  archives converted : ${finished:-0}"
    [ "${partial:-0}" -gt 0 ] && echo "  currently converting: $(basename $(ls -t data/lc0/*.partial 2>/dev/null | head -1) .bin.partial)"
    if [ "${finished:-0}" -gt 0 ]; then
        # 192 bytes a record, so the byte count gives the position count.
        bytes=$(du -sb data/lc0 2>/dev/null | cut -f1)
        echo "  positions          : $(( bytes / 192 ))" | \
            sed ':a;s/\B[0-9]\{3\}\>/,&/;ta'
    fi
    if [ -f logs/fetch.out ]; then
        echo "  last from the log  :"
        grep -E "^---|running total|budget reached|stopping:" logs/fetch.out \
            2>/dev/null | tail -3 | sed 's/^/    /'
    fi

    echo
    echo "TRAINING"
    for prefix in v2 lc0; do
        latest="checkpoints/${prefix}_latest.pt"
        [ -f "$latest" ] || continue
        step=$(./.venv/bin/python -c "
import torch,sys
try:
    c = torch.load('$latest', map_location='cpu')
    print(f\"step {c.get('step',0):,}\")
except Exception as e:
    print('unreadable')
" 2>/dev/null)
        best="checkpoints/${prefix}_best.pt"
        age=""
        if [ -f "$best" ]; then
            mins=$(( ( $(date +%s) - $(stat -c %Y "$best") ) / 60 ))
            age=", best saved ${mins}m ago"
        fi
        echo "  ${prefix}: ${step}${age}"
    done
    for log in logs/v2.out logs/lc0train.out; do
        [ -f "$log" ] || continue
        line=$(grep validation "$log" 2>/dev/null | tail -1)
        [ -n "$line" ] && echo "  $(basename $log .out): $(echo $line | sed 's/^ *//')"
    done
}

if [ "$1" = "watch" ]; then
    while true; do
        clear
        snapshot
        echo
        echo "  refreshing every 30s, Ctrl-C to stop"
        sleep 30
    done
else
    snapshot
fi
