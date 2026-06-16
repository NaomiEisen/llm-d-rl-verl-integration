#!/usr/bin/env bash
# Start EPP and Envoy as background processes on the head pod.
#
# Usage:
#   ./start-llmd-stack.sh              # use bundled configs
#   EPP_CONFIG=/my/config.yaml ./start-llmd-stack.sh
#
# Env overrides:
#   EPP_BINARY        path to EPP binary          (default: /usr/local/bin/epp)
#   ENVOY_BINARY      path to Envoy binary         (default: /usr/local/bin/envoy)
#   EPP_CONFIG        path to EPP config YAML      (default: <script-dir>/config.yaml)
#   ENVOY_CONFIG      path to Envoy config YAML    (default: <script-dir>/envoy.yaml)
#   EPP_ENDPOINTS     path to endpoints file        (default: /tmp/epp-endpoints.yaml)
#   EPP_GRPC_PORT     EPP gRPC port                (default: 9002)
#   ENVOY_PORT        Envoy listener port           (default: 8081)
#   EPP_PID_FILE      where to write EPP PID       (default: /tmp/epp.pid)
#   ENVOY_PID_FILE    where to write Envoy PID     (default: /tmp/envoy.pid)
#   EPP_LOG           EPP stdout/stderr log         (default: /tmp/epp.log)
#   ENVOY_LOG         Envoy stdout/stderr log       (default: /tmp/envoy.log)
#   READY_TIMEOUT     seconds to wait for ports    (default: 120)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

EPP_BINARY="${EPP_BINARY:-/usr/local/bin/epp}"
ENVOY_BINARY="${ENVOY_BINARY:-/usr/local/bin/envoy}"
EPP_CONFIG="${EPP_CONFIG:-${SCRIPT_DIR}/config.yaml}"
ENVOY_CONFIG="${ENVOY_CONFIG:-${SCRIPT_DIR}/envoy.yaml}"
EPP_ENDPOINTS="${EPP_ENDPOINTS:-/tmp/epp-endpoints.yaml}"
EPP_GRPC_PORT="${EPP_GRPC_PORT:-9002}"
EPP_HEALTH_PORT="${EPP_HEALTH_PORT:-9003}"
ENVOY_PORT="${ENVOY_PORT:-8081}"
EPP_PID_FILE="${EPP_PID_FILE:-/tmp/epp.pid}"
ENVOY_PID_FILE="${ENVOY_PID_FILE:-/tmp/envoy.pid}"
EPP_LOG="${EPP_LOG:-/tmp/epp.log}"
ENVOY_LOG="${ENVOY_LOG:-/tmp/envoy.log}"
READY_TIMEOUT="${READY_TIMEOUT:-120}"

# ── helpers ──────────────────────────────────────────────────────────────────

die() { echo "[llmd-stack] ERROR: $*" >&2; exit 1; }

wait_tcp() {
    local name="$1" port="$2"
    local deadline=$(( $(date +%s) + READY_TIMEOUT ))
    echo "[llmd-stack] waiting for ${name} on :${port} ..."
    while ! bash -c ">/dev/tcp/127.0.0.1/${port}" 2>/dev/null; do
        if [[ $(date +%s) -ge $deadline ]]; then
            die "timed out after ${READY_TIMEOUT}s waiting for ${name} on :${port}"
        fi
        sleep 0.5
    done
    echo "[llmd-stack] ${name} ready on :${port}"
}

wait_epp_health() {
    local health_port="$1"
    local deadline=$(( $(date +%s) + READY_TIMEOUT ))
    echo "[llmd-stack] waiting for EPP health on :${health_port} ..."
    while ! python3 - <<EOF 2>/dev/null
import grpc, sys
ch = grpc.insecure_channel("127.0.0.1:${health_port}")
m = ch.unary_unary("/grpc.health.v1.Health/Check", request_serializer=lambda x:x, response_deserializer=lambda x:x)
try:
    r = m(b"", timeout=3.0)
    sys.exit(0 if r[:2] == b"\x08\x01" else 1)
except Exception:
    sys.exit(1)
EOF
    do
        if [[ $(date +%s) -ge $deadline ]]; then
            die "timed out after ${READY_TIMEOUT}s waiting for EPP health on :${health_port}"
        fi
        sleep 0.5
    done
    echo "[llmd-stack] EPP ready (SERVING) on :${health_port}"
}

# ── preflight checks ─────────────────────────────────────────────────────────

[[ -x "$EPP_BINARY" ]]   || die "EPP binary not found or not executable: ${EPP_BINARY}"
[[ -x "$ENVOY_BINARY" ]] || die "Envoy binary not found or not executable: ${ENVOY_BINARY}"
[[ -f "$EPP_CONFIG" ]]   || die "EPP config not found: ${EPP_CONFIG}"
[[ -f "$ENVOY_CONFIG" ]] || die "Envoy config not found: ${ENVOY_CONFIG}"

# ── create an empty endpoints file if one doesn't exist ──────────────────────
# EPP starts with an empty pool and hot-reloads once verl writes real endpoints.

if [[ ! -f "$EPP_ENDPOINTS" ]]; then
    echo "[llmd-stack] creating empty endpoints file at ${EPP_ENDPOINTS}"
    cat >"$EPP_ENDPOINTS" <<'EOF'
endpoints: []
EOF
fi

# ── stop any previous instances ──────────────────────────────────────────────

for pidfile in "$EPP_PID_FILE" "$ENVOY_PID_FILE"; do
    if [[ -f "$pidfile" ]]; then
        pid=$(cat "$pidfile")
        if kill -0 "$pid" 2>/dev/null; then
            echo "[llmd-stack] stopping existing process PID ${pid}"
            kill "$pid" || true
            sleep 1
        fi
        rm -f "$pidfile"
    fi
done

# ── start EPP ────────────────────────────────────────────────────────────────

echo "[llmd-stack] starting EPP ..."
setsid "$EPP_BINARY" \
    --config-file="${EPP_CONFIG}" \
    --pool-name=file-discovery \
    --pool-namespace=default \
    --grpc-port="${EPP_GRPC_PORT}" \
    --grpc-health-port="${EPP_HEALTH_PORT}" \
    --metrics-port=9090 \
    --secure-serving=false \
    --tracing=false \
    -v=5 \
    >"$EPP_LOG" 2>&1 &

echo $! >"$EPP_PID_FILE"
disown "$(cat "$EPP_PID_FILE")"
echo "[llmd-stack] EPP PID $(cat "$EPP_PID_FILE") — logs: ${EPP_LOG}"

wait_epp_health "$EPP_HEALTH_PORT"

# ── start Envoy ──────────────────────────────────────────────────────────────

echo "[llmd-stack] starting Envoy ..."
setsid "$ENVOY_BINARY" \
    --service-node envoy-proxy \
    --log-level info \
    --concurrency 8 \
    --drain-strategy immediate \
    --drain-time-s 60 \
    --disable-hot-restart \
    -c "${ENVOY_CONFIG}" \
    >"$ENVOY_LOG" 2>&1 &

echo $! >"$ENVOY_PID_FILE"
disown "$(cat "$ENVOY_PID_FILE")"
echo "[llmd-stack] Envoy PID $(cat "$ENVOY_PID_FILE") — logs: ${ENVOY_LOG}"

wait_tcp "Envoy" "$ENVOY_PORT"

# ── done ─────────────────────────────────────────────────────────────────────

echo ""
echo "[llmd-stack] stack is up"
echo "  Envoy URL : http://localhost:${ENVOY_PORT}"
echo "  EPP gRPC  : localhost:${EPP_GRPC_PORT}"
echo "  EPP metrics: http://localhost:9090/metrics"
echo "  Envoy admin: http://localhost:19000/ready"
echo ""
echo "Use these in your verl config:"
echo "  rollout:"
echo "    custom:"
echo "      envoy_address: \"localhost:${ENVOY_PORT}\""
echo "      epp_endpoints_file: \"${EPP_ENDPOINTS}\""
