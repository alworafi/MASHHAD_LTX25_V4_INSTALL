#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="${MASHHAD_LTX_ROOT:-/workspace/LTX25-DIRECT}"
STATE="$ROOT/.mashhad"
PYTHON="$ROOT/.venv/bin/python"
DIRECT_PORT="${MASHHAD_DIRECT_PORT:-8189}"
WORKER_PORT="${MASHHAD_WORKER_PORT:-8000}"
export PATH="$ROOT/.runtime/bin:$ROOT/bin:$ROOT/.venv/bin:$PATH"

mkdir -p "$STATE" "$ROOT/logs" "$ROOT/inputs" "$ROOT/outputs"

if [[ ! -x "$PYTHON" || ! -f "$ROOT/direct_api.py" ]]; then
  echo "LTX25-DIRECT is incomplete; run install.sh once before start.sh." >&2
  exit 42
fi

stop_pid_file() {
  local file="$1"
  if [[ -f "$file" ]]; then
    local pid
    pid="$(cat "$file" 2>/dev/null || true)"
    if [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null; then
      kill "$pid" 2>/dev/null || true
      for _ in {1..20}; do
        kill -0 "$pid" 2>/dev/null || break
        sleep 0.25
      done
    fi
  fi
}

stop_pid_file "$STATE/direct-api.pid"
stop_pid_file "$STATE/worker.pid"

nohup env \
  MASHHAD_LTX_ROOT="$ROOT" \
  LTX_MODELS_DIR="$ROOT/models" \
  WORKER_INPUT_DIR="$ROOT/inputs" \
  WORKER_OUTPUT_DIR="$ROOT/outputs" \
  MASHHAD_DIRECT_OFFLOAD="${MASHHAD_DIRECT_OFFLOAD:-auto}" \
  MASHHAD_DIRECT_PRELOAD="${MASHHAD_DIRECT_PRELOAD:-1}" \
  "$PYTHON" -m uvicorn direct_api:app --app-dir "$ROOT" --host 127.0.0.1 --port "$DIRECT_PORT" \
  > "$ROOT/logs/direct-api.log" 2>&1 &
echo $! > "$STATE/direct-api.pid"

for _ in {1..120}; do
  if curl -fsS --max-time 2 "http://127.0.0.1:$DIRECT_PORT/health" >/dev/null 2>&1; then
    break
  fi
  if ! kill -0 "$(cat "$STATE/direct-api.pid")" 2>/dev/null; then
    tail -n 100 "$ROOT/logs/direct-api.log" >&2 || true
    exit 1
  fi
  sleep 0.5
done
curl -fsS --max-time 3 "http://127.0.0.1:$DIRECT_PORT/health" >/dev/null

if [[ ! -f "$ROOT/mashhad-worker/worker/main.py" || -z "${WORKER_SHARED_SECRET:-}" ]]; then
  echo "Direct API is running. Waiting for the Mashhad site to upload/start its Worker once."
  echo "Future Pods using this Volume will reuse that persistent Worker automatically."
  exec tail -f /dev/null
fi

nohup env \
  PYTHONPATH="$ROOT/mashhad-worker" \
  WORKER_SHARED_SECRET="$WORKER_SHARED_SECRET" \
  MASHHAD_LTX_ROOT="$ROOT" \
  LTX_MODELS_DIR="$ROOT/models" \
  WORKER_INPUT_DIR="$ROOT/inputs" \
  WORKER_OUTPUT_DIR="$ROOT/outputs" \
  DIRECT_API_URL="http://127.0.0.1:$DIRECT_PORT" \
  MASHHAD_DEFAULT_ENGINE="direct" \
  "$PYTHON" -m uvicorn worker.main:app --host 0.0.0.0 --port "$WORKER_PORT" \
  > "$ROOT/logs/worker.log" 2>&1 &
echo $! > "$STATE/worker.pid"

echo "Mashhad Direct service ready on 127.0.0.1:$DIRECT_PORT; worker starting on :$WORKER_PORT"
echo "Service readiness and model readiness are reported separately by /health."
wait "$(cat "$STATE/worker.pid")"
