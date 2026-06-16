#!/usr/bin/env bash
# Stop EPP and Envoy started by start-llmd-stack.sh.

set -euo pipefail

EPP_PID_FILE="${EPP_PID_FILE:-/tmp/epp.pid}"
ENVOY_PID_FILE="${ENVOY_PID_FILE:-/tmp/envoy.pid}"

stopped=0

for pidfile in "$EPP_PID_FILE" "$ENVOY_PID_FILE"; do
    if [[ -f "$pidfile" ]]; then
        pid=$(cat "$pidfile")
        if kill -0 "$pid" 2>/dev/null; then
            echo "[llmd-stack] stopping PID ${pid} ($(basename "$pidfile" .pid))"
            kill "$pid"
        else
            echo "[llmd-stack] PID ${pid} already gone ($(basename "$pidfile" .pid))"
        fi
        rm -f "$pidfile"
        (( stopped++ )) || true
    else
        echo "[llmd-stack] no pid file: ${pidfile}"
    fi
done

echo "[llmd-stack] done (${stopped} process(es) signalled)"
