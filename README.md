# Mongolian Whisper Training for Remotion

Fine-tune **openai/whisper-large-v3** on the [Mongolian Common Voice dataset](https://huggingface.co/datasets/Ganaa0614/mongolian-commonvoice-stt-translated-full) using LoRA, then convert the model for **@remotion/install-whisper-cpp** word-level captions.

## What you get

| Output | Purpose |
|---|---|
| `output/best/` | LoRA adapter checkpoint |
| `merged-model/` | Full HF model (after merge) |
| `ggml-large-v3-mn.bin` | whisper.cpp model for Remotion |
| `output/logs/` | TensorBoard training logs |

---

## Part 0 — Local smoke test (before RunPod)

Validate the pipeline on your PC for free before renting a GPU.

### Quick test (dataset + inference, ~2 min on CPU)

```powershell
cd c:\Projects\mongolian-whisper-training
pip install -r requirements.txt
python smoke_test.py --skip-training
```

Expected output:
```
[PASS] imports
[PASS] dataset
[PASS] inference
All local checks passed. Safe to run on RunPod.
```

### PyTorch version by GPU

| GPU | Architecture | PyTorch (pinned) | Install index |
|---|---|---|---|
| **RTX PRO 6000** (96 GB) | Blackwell `sm_120` | **`torch==2.7.1+cu128`** + `torchaudio==2.7.1+cu128` | `cu128` |
| RTX 3060 / A100 / L40S | Ampere–Hopper | `torch==2.6.0+cu124` + `torchaudio==2.6.0+cu124` | `cu124` |

**RTX PRO 6000:** Blackwell needs **CUDA 12.8 wheels** (`cu128`). Older `cu124` builds will fail with `sm_120 is not compatible` or similar. Use:

```powershell
.\scripts\setup_cuda_blackwell.ps1
# or Linux: bash scripts/setup_cuda_blackwell.sh
```

**RTX 3060 (your current PC):**

```powershell
.\scripts\setup_cuda_windows.ps1
```

Verify:

```powershell
python -c "import torch; print(torch.__version__, torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))"
# RTX PRO 6000 should show: 2.7.1+cu128 ... (12, 0)
```

NVIDIA driver **570+** recommended for Blackwell. Newer stable PyTorch (`2.10.0+cu128`, etc.) also works if you prefer latest; **2.7.1+cu128** is the minimum with official Blackwell support.

### Mini training test (needs GPU + ~6 GB free VRAM)

```powershell
python smoke_test.py
```

This runs 5 training steps on `whisper-small` with 16 samples — enough to verify LoRA training + merge work.

> **RTX PRO 6000 (96 GB):** You can run **full `whisper-large-v3` LoRA training locally** — no RunPod required. Use `--bf16` and batch size 16+ in `train.py`.
>
> **RTX 3060 (12 GB):** Mini smoke test only; use RunPod or RTX PRO 6000 for full training.

### What the smoke test checks

| Step | What it validates |
|---|---|
| imports | All Python packages installed |
| dataset | HF dataset downloads, audio decodes, preprocessing works |
| inference | Whisper transcribes Mongolian audio |
| mini-train | LoRA training loop runs (GPU only, if enough VRAM) |
| merge | LoRA weights merge into full model |

---

## Part 1 — RunPod setup

### 1. Create a pod

1. Go to [runpod.io](https://www.runpod.io) → **Pods** → **Deploy**
2. Choose GPU: **RTX PRO 6000 96GB** (if available), **A100 80GB PCIe**, or **A100 40GB**
3. Choose template: **PyTorch 2.8** (see table below). Then reinstall the correct CUDA wheel — RunPod templates often ship `cu118`/`cu124`, which **do not work** on RTX PRO 6000.
4. Disk: **50 GB** minimum (model + dataset cache)
5. Deploy the pod

#### RunPod PyTorch template picker

| Template | RTX PRO 6000 (Blackwell) | A100 / L40S |
|---|---|---|
| **PyTorch 2.8** | **Use this** — then run `bash scripts/setup_cuda_blackwell.sh` | OK — run `bash scripts/setup.sh` or blackwell script |
| PyTorch 2.4 | No — no `sm_120` support | OK |
| PyTorch 2.2 | No | OK (older) |
| PyTorch 2.1 | No | OK (older) |

**RTX PRO 6000:** Only **2.8** is viable among RunPod’s listed templates. After the pod starts, **always** run:

```bash
cd /workspace/mongolian-whisper-training
bash scripts/setup_cuda_blackwell.sh   # installs torch==2.7.1+cu128 (or bump to 2.8.0+cu128)
python -c "import torch; print(torch.__version__, torch.cuda.get_device_capability(0))"
# expect: ...cu128 ... (12, 0)
```

Do **not** rely on the template’s preinstalled torch for Blackwell — pick **2.8** for a recent Python/CUDA base, then overwrite torch with `cu128`.

### 2. Upload this project

**Option A — Git (recommended)**

If you pushed this folder to GitHub:

```bash
cd /workspace
git clone https://github.com/YOUR_USER/mongolian-whisper-training.git
cd mongolian-whisper-training
```

**Option B — Upload via RunPod file browser**

Upload the `mongolian-whisper-training` folder to `/workspace/`.

**Option C — Copy from your PC with scp**

```bash
scp -r mongolian-whisper-training root@YOUR_POD_IP:/workspace/
```

### 3. Install dependencies

```bash
cd /workspace/mongolian-whisper-training
bash scripts/setup.sh
```

### 4. (Optional) Hugging Face login

Only needed if you want to push the model to the Hub:

```bash
pip install huggingface_hub
huggingface-cli login
```

---

## Part 2 — Train

### Quick start (full pipeline)

Runs train → merge → evaluate in one command:

```bash
cd /workspace/mongolian-whisper-training
bash scripts/run_training.sh
```

Expected runtime on **A100 80GB**: ~2–4 hours.

### Manual step-by-step

```bash
# 1. Train LoRA
python train.py \
  --output-dir ./output \
  --batch-size 16 \
  --grad-accum 2 \
  --epochs 3 \
  --bf16

# 2. Merge LoRA into base weights
python merge_lora.py \
  --adapter-dir ./output/best \
  --output-dir ./merged-model

# 3. Evaluate on test split
python run_eval.py --model-dir ./merged-model
```

### GPU memory tuning

| GPU | Suggested `--batch-size` |
|---|---|
| A100 80GB | 16 (default) |
| A100 40GB | 8 |
| RTX 4090 | 4–8 |

If you hit OOM, reduce batch size and increase grad accum to keep effective batch ≈ 32:

```bash
python train.py --batch-size 8 --grad-accum 4 --bf16
```

### Monitor training

In a second terminal on the pod:

```bash
tensorboard --logdir /workspace/mongolian-whisper-training/output/logs --bind_all --port 6006
```

Open the TensorBoard URL from RunPod's port forwarding UI. Watch **eval_wer** — lower is better. Training stops at the best WER checkpoint automatically.

### Push to Hugging Face Hub (optional)

```bash
python train.py \
  --push-to-hub \
  --hub-model-id YOUR_USERNAME/whisper-large-v3-mn \
  --bf16
```

---

## Part 3 — Convert for Remotion

After training, convert the merged model to whisper.cpp format:

```bash
bash scripts/convert_to_ggml.sh ./merged-model ./ggml-large-v3-mn.bin
```

Download `ggml-large-v3-mn.bin` to your Windows machine (RunPod file browser or `scp`).

---

## Part 4 — Use in Remotion (Windows)

### 1. Install Remotion whisper.cpp

In your Remotion project:

```bash
npm install @remotion/install-whisper-cpp @remotion/captions
```

### 2. Copy the model

Place your converted model in the whisper.cpp models folder:

```
your-remotion-project/
  whisper.cpp/
    models/
      ggml-large-v3-mn.bin    <-- your file
```

### 3. Transcribe with word-level timing

Create `scripts/transcribe-mongolian.ts`:

```ts
import path from 'path';
import fs from 'fs';
import {execSync} from 'child_process';
import {
  installWhisperCpp,
  transcribe,
  toCaptions,
} from '@remotion/install-whisper-cpp';

const WHISPER_VERSION = '1.5.5';
const whisperPath = path.join(process.cwd(), 'whisper.cpp');
const videoPath = process.argv[2];
const wavPath = videoPath.replace(/\.[^.]+$/, '.wav');

if (!videoPath) {
  console.error('Usage: npx tsx scripts/transcribe-mongolian.ts <video-or-audio-file>');
  process.exit(1);
}

await installWhisperCpp({to: whisperPath, version: WHISPER_VERSION});

// Convert to 16kHz mono WAV
execSync(`ffmpeg -i "${videoPath}" -ar 16000 -ac 1 "${wavPath}" -y`, {stdio: 'inherit'});

const output = await transcribe({
  inputPath: wavPath,
  whisperPath,
  whisperCppVersion: WHISPER_VERSION,
  model: 'large-v3',
  modelFolder: path.join(whisperPath, 'models'),
  language: 'mn',
  tokenLevelTimestamps: true,
  splitOnWord: true,
});

const {captions} = toCaptions({whisperCppOutput: output});

const outPath = wavPath.replace('.wav', '.captions.json');
fs.writeFileSync(outPath, JSON.stringify(captions, null, 2));
console.log(`Captions saved to ${outPath}`);
```

Run:

```bash
npx tsx scripts/transcribe-mongolian.ts ./my-mongolian-video.mp4
```

Output `captions.json` format (ready for `@remotion/captions`):

```json
[
  {
    "text": "цаг",
    "startMs": 120,
    "endMs": 480,
    "timestampMs": 300,
    "confidence": 0.95
  }
]
```

> **Note:** If Remotion's `model` param doesn't pick up a custom filename, pass the exact `.bin` filename via whisper.cpp's model path. You may need to rename to `ggml-large-v3.bin` or use `additionalArgs` — check [Remotion transcribe docs](https://www.remotion.dev/docs/install-whisper-cpp/transcribe).

---

## Part 5 — When you already have a transcript

For videos where you already know the script, **forced alignment** gives better word timing than ASR:

```bash
pip install stable-ts
```

```python
import stable_whisper
model = stable_whisper.load_model("merged-model")  # or path to merged HF model
result = model.align("audio.wav", "your mongolian transcript here", language="mn")
result.to_srt_vtt("aligned.vtt", word_level=True)
```

Convert the aligned word timestamps into Remotion `Caption` objects for animation.

---

## Project structure

```
mongolian-whisper-training/
├── README.md                 # This guide
├── requirements.txt
├── train.py                  # LoRA fine-tuning
├── merge_lora.py             # Merge LoRA → full model
├── run_eval.py               # Test WER evaluation
└── scripts/
    ├── setup.sh              # Install deps on RunPod
    ├── run_training.sh       # Full pipeline
    └── convert_to_ggml.sh    # HF → whisper.cpp conversion
```

---

## Training configuration (defaults)

| Setting | Value |
|---|---|
| Base model | `openai/whisper-large-v3` |
| Method | LoRA r=64, alpha=128 |
| Learning rate | 1e-5 |
| Epochs | 3 |
| Effective batch size | 32 (16 × grad_accum 2) |
| Label column | `sentence` (Mongolian Cyrillic) |
| Metric | WER (word error rate) |

---

## Troubleshooting

### CUDA out of memory

```bash
python train.py --batch-size 4 --grad-accum 8 --bf16
```

### Training loss drops but WER rises

Overfitting. Stop early — the script already saves the best WER checkpoint. Try fewer epochs:

```bash
EPOCHS=2 bash scripts/run_training.sh
```

### `convert-h5-to-ggml.py` fails

Ensure `git` is installed on the pod and both repos cloned:

```bash
apt-get update && apt-get install -y git
bash scripts/convert_to_ggml.sh ./merged-model ./ggml-large-v3-mn.bin
```

### Dataset download slow

The dataset is ~1 GB. First run caches to `~/.cache/huggingface/`. Subsequent runs are fast.

### Mongolian text looks wrong

The dataset uses **Cyrillic Mongolian** (`сайн байна уу`), not traditional script. Your video audio should match.

---

## Cost estimate (RunPod)

| GPU | Hours | Approx cost |
|---|---|---|
| A100 80GB @ ~$1.50/hr | 3 | ~$4.50 |
| A100 40GB @ ~$1.10/hr | 4 | ~$4.40 |

Stop the pod when done to avoid idle charges.

---

## Next steps

1. Train on RunPod with `bash scripts/run_training.sh`
2. Download `ggml-large-v3-mn.bin`
3. Wire into your Remotion project for word-by-word captions
4. For scripted videos, add stable-ts alignment
