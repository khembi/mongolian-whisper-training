#!/usr/bin/env bash
# One-shot RunPod bootstrap: clone repo → install deps → train → merge → eval → (optional) ggml.
#
# Paste into RunPod "Start Command" (PyTorch 2.8 template, 50 GB+ disk):
#   bash -c 'curl -fsSL https://raw.githubusercontent.com/khembi/mongolian-whisper-training/master/scripts/runpod_start.sh | bash'
#
# Or after SSH:
#   cd /workspace/mongolian-whisper-training && bash scripts/runpod_start.sh
#
# Environment variables (all optional):
#   REPO_URL          Git clone URL (default: this repo)
#   BRANCH            Git branch (default: master)
#   WORKSPACE         Base dir (default: /workspace)
#   SKIP_CLONE        1 = use existing PROJECT_DIR, do not git pull
#   BATCH_SIZE        Training batch size (auto if unset)
#   EPOCHS            Training epochs (default: 3)
#   CONVERT_GGML      1 = convert merged model to ggml after training (default: 1)
#   HF_TOKEN          Hugging Face token for hub push / gated models
#   PUSH_TO_HUB       1 = push merged model (requires HF_TOKEN + HUB_MODEL_ID)
#   HUB_MODEL_ID      e.g. your-user/whisper-large-v3-mn
#
set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/khembi/mongolian-whisper-training.git}"
BRANCH="${BRANCH:-master}"
WORKSPACE="${WORKSPACE:-/workspace}"
PROJECT_DIR="${PROJECT_DIR:-$WORKSPACE/mongolian-whisper-training}"
EPOCHS="${EPOCHS:-3}"
CONVERT_GGML="${CONVERT_GGML:-1}"
SKIP_CLONE="${SKIP_CLONE:-0}"

log() { echo ""; echo "==> $*"; }

# --- Clone or update repo ---
if [ "$SKIP_CLONE" = "1" ]; then
  log "SKIP_CLONE=1 — using $PROJECT_DIR"
  [ -d "$PROJECT_DIR" ] || { echo "ERROR: $PROJECT_DIR not found"; exit 1; }
else
  log "Fetching project from $REPO_URL ($BRANCH)..."
  if [ -d "$PROJECT_DIR/.git" ]; then
    git -C "$PROJECT_DIR" fetch origin
    git -C "$PROJECT_DIR" checkout "$BRANCH"
    git -C "$PROJECT_DIR" pull --ff-only origin "$BRANCH"
  else
    git clone --depth 1 --branch "$BRANCH" "$REPO_URL" "$PROJECT_DIR"
  fi
fi

cd "$PROJECT_DIR"

# --- Hugging Face login (optional) ---
if [ -n "${HF_TOKEN:-}" ]; then
  log "Logging into Hugging Face..."
  pip install -q huggingface_hub
  huggingface-cli login --token "$HF_TOKEN" --add-to-git-credential
fi

# --- GPU detection → correct PyTorch wheels ---
log "Detecting GPU and installing dependencies..."
python3 <<'PY'
import subprocess, sys

def sm_version():
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.get_device_capability(0)
    except Exception:
        pass
    # nvidia-smi fallback before torch is installed
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
            text=True,
        ).strip()
        major, minor = out.split(".")[:2]
        return int(major), int(minor)
    except Exception:
        return None

cap = sm_version()
if cap and cap[0] >= 12:
    script = "scripts/setup_cuda_blackwell.sh"
    print(f"Blackwell GPU detected (CC {cap[0]}.{cap[1]}) → {script}")
elif cap:
    script = "scripts/setup_cuda_ampere.sh"
    print(f"Ampere/Hopper GPU detected (CC {cap[0]}.{cap[1]}) → {script}")
else:
    script = "scripts/setup.sh"
    print("Could not detect GPU arch → scripts/setup.sh")

with open("/tmp/runpod_setup_script.txt", "w") as f:
    f.write(script)
PY

SETUP_SCRIPT="$(cat /tmp/runpod_setup_script.txt)"
bash "$SETUP_SCRIPT"

# --- Auto batch size from VRAM if not set ---
if [ -z "${BATCH_SIZE:-}" ]; then
  BATCH_SIZE="$(python3 -c "
import torch
gb = torch.cuda.get_device_properties(0).total_memory / 1e9
if gb >= 70: print(16)
elif gb >= 35: print(8)
else: print(4)
")"
  VRAM_GB="$(python3 -c "import torch; print(f'{torch.cuda.get_device_properties(0).total_memory/1e9:.0f}')")"
  log "Auto BATCH_SIZE=$BATCH_SIZE (${VRAM_GB} GB VRAM)"
else
  log "Using BATCH_SIZE=$BATCH_SIZE"
fi

export BATCH_SIZE EPOCHS

# --- Training pipeline ---
log "Starting training pipeline (epochs=$EPOCHS, batch=$BATCH_SIZE)..."
bash scripts/run_training.sh

# --- Optional Hub push ---
if [ "${PUSH_TO_HUB:-0}" = "1" ]; then
  [ -n "${HUB_MODEL_ID:-}" ] || { echo "ERROR: PUSH_TO_HUB=1 requires HUB_MODEL_ID"; exit 1; }
  log "Pushing merged model to $HUB_MODEL_ID..."
  python3 -c "
from transformers import WhisperForConditionalGeneration, WhisperProcessor
mid = '$HUB_MODEL_ID'
WhisperProcessor.from_pretrained('./merged-model').push_to_hub(mid)
WhisperForConditionalGeneration.from_pretrained('./merged-model').push_to_hub(mid)
print('Pushed to', mid)
"
fi

# --- GGML conversion for Remotion ---
if [ "$CONVERT_GGML" = "1" ]; then
  log "Converting merged model to whisper.cpp ggml..."
  apt-get update -qq && apt-get install -y -qq git >/dev/null 2>&1 || true
  bash scripts/convert_to_ggml.sh ./merged-model ./ggml-large-v3-mn.bin
fi

log "All done. Artifacts in $PROJECT_DIR:"
echo "  LoRA:        $PROJECT_DIR/output/best"
echo "  Merged HF:   $PROJECT_DIR/merged-model"
[ "$CONVERT_GGML" = "1" ] && echo "  Remotion:    $PROJECT_DIR/ggml-large-v3-mn.bin"
echo ""
echo "Download via RunPod file browser or:"
echo "  scp root@POD_IP:$PROJECT_DIR/ggml-large-v3-mn.bin ."
