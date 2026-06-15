# Local prep + RunPod GPU training

Prepare mel features on your Windows PC (free, overnight), then rent a GPU pod for training only (~$8 vs ~$19 all-in-one).

## Part A — Prepare locally (Windows)

```powershell
cd c:\Projects\mongolian-whisper-training
.\scripts\prepare_data_local.ps1
```

Or manually:

```powershell
pip install -r requirements.txt
python prepare_data.py --output-dir ./prepared-dataset --splits train,validation
```

| Split | Samples | CPU time |
|---|---|---|
| train | ~29,000 | ~4 h |
| validation | ~3,600 | ~30 min |
| test (optional) | ~3,600 | ~30 min — add later with `--splits test` |

Output:

```
prepared-dataset/
├── train/
└── validation/
```

**Disk:** ~10–12 GB for train + validation.

Resume: if interrupted, rerun the same command — completed splits are skipped automatically.

---

## Part B — Upload to RunPod

```powershell
scp -r prepared-dataset root@YOUR_POD_IP:/workspace/mongolian-whisper-training/
```

~10–20 min depending on upload speed.

---

## Part C — Train on RunPod (GPU only)

Pod: **PyTorch 2.8**, **A100 80GB** or **RTX PRO 6000**, **50 GB disk** (prepared data already uploaded).

```bash
cd /workspace/mongolian-whisper-training
git clone https://github.com/khembi/mongolian-whisper-training.git .  # if fresh pod
git pull

bash scripts/setup_cuda_blackwell.sh   # or setup_cuda_ampere.sh

bash scripts/runpod_train_only.sh
```

Training starts within **minutes** — no multi-hour preprocessing.

**Credit needed:** ~$10–15 (3–5 hours GPU @ $2.10/hr).

---

## Part D — Download + stop pod

```powershell
scp root@YOUR_POD_IP:/workspace/mongolian-whisper-training/ggml-large-v3-mn.bin .
```

Stop the pod in RunPod console when done.

---

## Optional: prepare test split (for WER eval)

Locally before upload, or on pod after training:

```powershell
python prepare_data.py --output-dir ./prepared-dataset --splits test
```

---

## Cost comparison

| Path | RunPod billable time | Approx. cost |
|---|---|---|
| All-in-one on RunPod | ~9 h | ~$19 |
| **Local prep + GPU train** | **~4 h** | **~$8** |

---

## Troubleshooting

**`No prepared splits found`** — `prepared-dataset/train/` must exist on the pod. Re-run `scp`.

**Upload too slow** — compress first: `tar -czf prepared-dataset.tar.gz prepared-dataset`, upload, `tar -xzf` on pod.

**Local prep slow on Windows** — normal (~2 examples/sec). Run overnight; GPU not required.
