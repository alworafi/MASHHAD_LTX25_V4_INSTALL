#!/usr/bin/env bash
set -Eeuo pipefail

VERSION="LTX25_DIRECT_PARALLEL_FAST_V2_20261009_1"
ROOT="${MASHHAD_LTX_ROOT:-/workspace/LTX25-DIRECT}"
STATE="$ROOT/.mashhad"
RUNTIME="$ROOT/.runtime"
VENV="$ROOT/.venv"
LTX_REPO="$ROOT/LTX-2"
MODEL_DIR="$ROOT/models"
BASE_PYTHON="${MASHHAD_BASE_PYTHON:-/usr/local/bin/python3.12}"
LTX_COMMIT="${LTX_GIT_COMMIT:-2d6e71c88be37b55a2dd698c2dff447edfbe5898}"
INSTALLER_REPO="${MASHHAD_INSTALLER_REPO:-alworafi/MASHHAD_LTX25_V4_INSTALL}"
INSTALLER_REF="${MASHHAD_INSTALLER_REF:-codex/direct-parallel-fast-v2}"
START_TS="${MASHHAD_START_TS:-$(date +%s)}"
MODEL_PID=""
RUNTIME_PID=""

elapsed() {
  local seconds=$(( $(date +%s) - START_TS ))
  printf '%02d:%02d:%02d' "$((seconds/3600))" "$(((seconds%3600)/60))" "$((seconds%60))"
}

fail() {
  echo "ERROR: $*" >&2
  exit 2
}

cleanup_jobs() {
  local rc=$?
  if [[ -n "$MODEL_PID" ]] && kill -0 "$MODEL_PID" 2>/dev/null; then kill "$MODEL_PID" 2>/dev/null || true; fi
  if [[ -n "$RUNTIME_PID" ]] && kill -0 "$RUNTIME_PID" 2>/dev/null; then kill "$RUNTIME_PID" 2>/dev/null || true; fi
  if (( rc != 0 )); then
    echo "INSTALL FAILED | elapsed=$(elapsed) | log=$LOG" >&2
  fi
  exit "$rc"
}

[[ "$EUID" == 0 ]] || fail "Run this installer as root inside the RunPod container."
[[ "${MASHHAD_PREPARE_ENGINES:-direct}" == "direct" ]] || fail "This template is Direct-only."
command -v findmnt >/dev/null 2>&1 || fail "findmnt is unavailable in the selected base image."
MOUNT_SOURCE="$(findmnt -T /workspace -n -o SOURCE 2>/dev/null || true)"
MOUNT_TYPE="$(findmnt -T /workspace -n -o FSTYPE 2>/dev/null || true)"
[[ -n "$MOUNT_SOURCE" && "$MOUNT_TYPE" != "overlay" && "$MOUNT_TYPE" != "tmpfs" ]] \
  || fail "/workspace must be backed by an attached Network Volume."

mkdir -p "$ROOT/logs" "$MODEL_DIR" "$ROOT/inputs" "$ROOT/outputs" "$STATE" "$RUNTIME/bin"
exec 9>"$ROOT/.install.lock"
flock -n 9 || fail "Another install/start process is already using this Volume."
LOG="$ROOT/logs/install_$(date -u +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1
printf '%s\n' "$LOG" > "$ROOT/logs/current_install_log"
printf '%s\n' "$START_TS" > "$ROOT/logs/current_start_ts"
trap cleanup_jobs EXIT

echo "=== $VERSION ==="
echo "Started: $(date -Is)"
echo "Persistent root: $ROOT ($MOUNT_SOURCE, $MOUNT_TYPE)"
echo "Models: $MODEL_DIR (one flat directory, five BF16 files, no duplicate model cache)"
echo "Base image expected: runpod/pytorch:1.0.3-cu1281-torch291-ubuntu2404"

export MASHHAD_LTX_ROOT="$ROOT"
export LTX_MODELS_DIR="$MODEL_DIR"
export WORKER_INPUT_DIR="$ROOT/inputs"
export WORKER_OUTPUT_DIR="$ROOT/outputs"
export PATH="$RUNTIME/bin:$ROOT/bin:$PATH"
export UV_CACHE_DIR="$RUNTIME/uv-cache"
# The cache and venv share the persistent Volume, so linking avoids a second
# network-filesystem copy of packages while keeping resume behavior intact.
export UV_LINK_MODE=hardlink
export HF_HUB_DISABLE_XET=0
export HF_XET_HIGH_PERFORMANCE=1
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-600}"
export HF_HUB_ETAG_TIMEOUT="${HF_HUB_ETAG_TIMEOUT:-60}"
export HF_HOME="$ROOT/.downloads/.hf"
export HF_XET_CACHE="$ROOT/.downloads/.xet"

if [[ "$(cat "$ROOT/.DIRECT_PARALLEL_FAST_V2_READY" 2>/dev/null || true)" == "$VERSION" ]]; then
  [[ -x "$VENV/bin/python" && -f "$ROOT/start.sh" && -f "$ROOT/direct_api.py" ]] \
    || fail "Persistent Direct V2 marker exists, but runtime files are incomplete."
  echo "[FAST RESUME] Persistent environment found. No Python, Torch, dependency, or model download will run."
  "$VENV/bin/python" "$ROOT/validate_runtime.py" \
    || fail "Persistent environment is intact but incompatible with this Pod/GPU; nothing was reinstalled or downloaded."
  echo "[FAST RESUME] GPU/runtime compatibility passed in $(elapsed)."
  trap - EXIT
  exec "$ROOT/start.sh"
fi

rm -f -- "$ROOT/.DIRECT_PARALLEL_FAST_V2_READY" "$ROOT/.MASHHAD_READY_V4"

echo "[1/6] Installing small OS tools needed only for the first installation..."
packages=(git git-lfs curl ca-certificates build-essential)
missing=()
for package in "${packages[@]}"; do
  [[ "$(dpkg-query -W -f='${Status}' "$package" 2>/dev/null || true)" == "install ok installed" ]] || missing+=("$package")
done
if ((${#missing[@]})); then
  apt-get update
  DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends "${missing[@]}"
  apt-get clean
fi

echo "[2/6] Fetching the versioned Direct V2 helpers..."
raw_base="https://raw.githubusercontent.com/$INSTALLER_REPO/$INSTALLER_REF/direct-parallel-fast-v2"
for helper in download_models.py validate_runtime.py direct_api.py start.sh; do
  temporary="$ROOT/.${helper}.download"
  curl -fLsS --retry 5 --connect-timeout 20 "$raw_base/$helper" -o "$temporary"
  mv -f -- "$temporary" "$ROOT/$helper"
done
chmod +x "$ROOT/start.sh"

if [[ ! -x "$RUNTIME/bin/uv" ]]; then
  curl -fLsS --retry 5 https://astral.sh/uv/install.sh -o /tmp/mashhad-direct-uv.sh
  env UV_UNMANAGED_INSTALL="$RUNTIME/bin" sh /tmp/mashhad-direct-uv.sh
fi

[[ -x "$BASE_PYTHON" ]] || fail "The RunPod base-image Python 3.12 was not found at $BASE_PYTHON."
base_probe="$($BASE_PYTHON - <<'PY'
import torch
print(f"{torch.__version__}|{torch.version.cuda}")
PY
)"
[[ "$base_probe" == 2.9.1*"|12.8"* ]] || fail "Base Torch/CUDA mismatch: $base_probe"
echo "[REUSE] Base runtime accepted: Python=$($BASE_PYTHON -V 2>&1), Torch/CUDA=$base_probe"

echo "[3/6] Preparing the lightweight downloader, then starting all five files in parallel..."
rm -rf -- "$ROOT/.download-libs"
uv pip install --target "$ROOT/.download-libs" \
  "huggingface_hub>=0.34,<2" "hf-xet>=1.1,<2" "safetensors>=0.5,<1"
(
  export PYTHONPATH="$ROOT/.download-libs"
  "$BASE_PYTHON" "$ROOT/download_models.py"
) > >(tee -a "$ROOT/logs/models.log") 2>&1 &
MODEL_PID=$!

echo "[4/6] Installing the official pinned Direct runtime in parallel with model downloads..."
(
  if [[ ! -d "$LTX_REPO/.git" ]]; then
    rm -rf -- "$LTX_REPO"
    git clone --filter=blob:none --no-checkout https://github.com/Lightricks/LTX-2.git "$LTX_REPO"
  fi
  git -C "$LTX_REPO" fetch --depth 1 origin "$LTX_COMMIT"
  git -C "$LTX_REPO" checkout --detach --force "$LTX_COMMIT"

  if [[ ! -x "$VENV/bin/python" ]]; then
    "$BASE_PYTHON" -m venv --system-site-packages "$VENV"
  fi
  uv pip install --python "$VENV/bin/python" \
    -e "$LTX_REPO/packages/ltx-core" \
    -e "$LTX_REPO/packages/ltx-pipelines" \
    "fastapi==0.118.0" "uvicorn[standard]==0.37.0" "httpx==0.28.1" \
    "pydantic==2.11.9" "python-multipart==0.0.20" "imageio-ffmpeg>=0.6,<1"
  uv pip install --python "$VENV/bin/python" --no-deps \
    "natten==0.21.5+torch290cu128" -f https://whl.natten.org

  runtime_probe="$($VENV/bin/python - <<'PY'
import torch
assert torch.__version__.startswith("2.9.1"), torch.__version__
assert (torch.version.cuda or "").startswith("12.8"), torch.version.cuda
print(f"{torch.__version__}|{torch.version.cuda}")
PY
)"
  echo "[REUSE VERIFIED] Persistent venv sees the base Torch/CUDA: $runtime_probe"

  mkdir -p "$ROOT/bin"
  ffmpeg_path="$($VENV/bin/python - <<'PY'
import imageio_ffmpeg
print(imageio_ffmpeg.get_ffmpeg_exe())
PY
)"
  ln -sfn "$ffmpeg_path" "$ROOT/bin/ffmpeg"
) > >(tee -a "$ROOT/logs/runtime.log") 2>&1 &
RUNTIME_PID=$!

model_rc=0
runtime_rc=0
wait "$MODEL_PID" || model_rc=$?
MODEL_PID=""
wait "$RUNTIME_PID" || runtime_rc=$?
RUNTIME_PID=""
(( model_rc == 0 )) || fail "Model download failed with exit code $model_rc; see $ROOT/logs/models.log"
(( runtime_rc == 0 )) || fail "Direct runtime installation failed with exit code $runtime_rc; see $ROOT/logs/runtime.log"
rm -rf -- "$ROOT/.download-libs" "$ROOT/.downloads"
echo "[PARALLEL COMPLETE] Runtime and all five models finished in $(elapsed)."

echo "[5/6] Validating GPU, BF16, packages, and the single flat model directory..."
"$VENV/bin/python" "$ROOT/validate_runtime.py"

extra_weights="$(find "$MODEL_DIR" -maxdepth 1 -type f \( -name '*.safetensors' -o -name '*.ckpt' -o -name '*.pt' -o -name '*.bin' \) | wc -l)"
[[ "$extra_weights" == 5 ]] || fail "Expected exactly five model weight files in $MODEL_DIR, found $extra_weights."
[[ ! -d "$ROOT/.downloads" ]] || fail "Temporary model download cache was not removed."

echo "[6/6] Writing persistent readiness state..."
mkdir -p "$STATE/engines"
printf '%s\n' direct > "$STATE/prepared_engines"
printf '%s\n' direct > "$STATE/default_engine"
printf '%s\n' direct > "$STATE/active_engine"
printf '%s\n' "$VERSION" > "$STATE/engines/direct.ready"
printf '%s\n' "$VERSION" > "$ROOT/.DIRECT_PARALLEL_FAST_V2_READY"
printf '%s\n' "$VERSION" > "$ROOT/.MASHHAD_READY_V4"

echo "INSTALL VERIFIED in $(elapsed)"
echo "Five BF16 model files, one directory: $MODEL_DIR"
echo "INT8 ConvRot: not tested and not enabled."
echo "Next Pod on this Volume takes the FAST RESUME path without reinstalling or redownloading."

trap - EXIT
exec "$ROOT/start.sh"
