#!/usr/bin/env bash
# One-shot start of the full live_cam stack on the Windows laptop (run from Git Bash):
#   Docker container ugv-run (ROS) + rosbridge port forward + Windows webcam bridge + UI dev server.
#
#   bash run.sh
#
# Env: CAM_INDEX (DirectShow camera index, default 0).
# Ctrl+C stops the UI and the webcam bridge (camera LED off). The ROS stack keeps running in ugv-run and holds
# (no camera -> zero /cmd_vel); rerunning this script restarts it.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONTAINER=ugv-run
FORWARD=ugv-rosbridge-port
LOGS="$REPO/.logs"
mkdir -p "$LOGS"

log() { printf '\n== %s\n' "$*"; }

log "Model weights"
# Each live model needs either the HuggingFace safetensors folder (CUDA) or the OpenVINO IR pair (Intel).
W="$REPO/turing/weights"
missing=()
for m in rugd-segformer da3metric-large; do
  if [ -s "$W/$m/model.safetensors" ] || { [ -s "$W/$m.xml" ] && [ -s "$W/$m.bin" ]; }; then
    echo "$m ok"
  else
    missing+=("$m")
  fi
done
if [ ${#missing[@]} -gt 0 ]; then
  echo "Missing weights: ${missing[*]}"
  echo "Git does not ship the model weights. Follow user_manual.md (section 1: download; section 2 for Intel/OpenVINO)."
  exit 1
fi

log "Docker"
if ! docker info >/dev/null 2>&1; then
  echo "Docker Desktop is not running; starting it"
  "/c/Program Files/Docker/Docker/Docker Desktop.exe" >/dev/null 2>&1 &
  for _ in $(seq 60); do docker info >/dev/null 2>&1 && break; sleep 2; done
  docker info >/dev/null 2>&1 || { echo "Docker did not come up"; exit 1; }
fi
docker start "$CONTAINER" >/dev/null
echo "$CONTAINER up"

log "rosbridge port forward (host :9090 -> $CONTAINER:9090)"
# The container IP can change across Docker restarts, so recreate the forwarder against the current one.
IP="$(docker inspect -f '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' "$CONTAINER")"
docker rm -f "$FORWARD" >/dev/null 2>&1 || true
docker run -d --name "$FORWARD" -p 9090:9090 alpine/socat \
  tcp-listen:9090,fork,reuseaddr "tcp-connect:$IP:9090" >/dev/null
echo "$FORWARD -> $IP:9090"

log "Webcam bridge (:8090)"
WEBCAM_PID=""
if netstat -ano | grep -qE '[:.]8090 +[^ ]+ +LISTENING'; then
  echo "already serving on :8090, reusing it"
else
  python "$REPO/ugv_nav/ugv_bringup/scripts/webcam_stream.py" --index "${CAM_INDEX:-0}" \
    >"$LOGS/webcam.log" 2>&1 &
  WEBCAM_PID=$!
  sleep 3
  kill -0 "$WEBCAM_PID" 2>/dev/null || { echo "webcam bridge died:"; cat "$LOGS/webcam.log"; exit 1; }
  echo "started (log: .logs/webcam.log)"
fi

cleanup() {
  [ -n "$WEBCAM_PID" ] && kill "$WEBCAM_PID" 2>/dev/null && echo "webcam bridge stopped"
}
trap cleanup EXIT

log "ROS stack (sync + colcon build + live_cam launch + rosbridge)"
MSYS_NO_PATHCONV=1 docker exec "$CONTAINER" bash /ws/restart_live.sh  # keep Git Bash from rewriting /ws/...

# First free port from 5173 up, so the summary below shows the exact URL (an older dev server may hold 5173).
UI_PORT=5173
while netstat -ano | grep -qE "[:.]$UI_PORT +[^ ]+ +LISTENING"; do UI_PORT=$((UI_PORT + 1)); done

log "Running"
printf '  %-22s %s\n' \
  "UI"                    "http://localhost:$UI_PORT  (opens in your browser)" \
  "API gateway"           "http://localhost:8080" \
  "rosbridge (websocket)" "ws://localhost:9090" \
  "Webcam stream"         "http://localhost:8090/cam.mjpg" \
  "ROS log"               "docker exec $CONTAINER tail -f /tmp/live.log" \
  "Stop"                  "Ctrl+C (stops UI + webcam; ROS holds in $CONTAINER)"

log "UI dev server"
cd "$REPO/ui"
[ -d node_modules ] || npm install
npm run dev -- --port "$UI_PORT" --strictPort --open
