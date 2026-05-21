# Piper Training

Local version of the workflow from `piper_demo.ipynb`.

## Setup

```bash
cd /home/tyuukau/work/exp/tts/train/piper
uv sync --python 3.11
uv run python train_piper.py setup
uv run python train_piper.py doctor
```

The environment pins PyTorch `2.8.0+cu126` from the CUDA 12.6 wheel index. On
this machine, `doctor` currently sees the NVIDIA GPU and required command-line
tools.

## Dataset

The default training set is the filtered ThuaThienHue v2 set:

```bash
uv run python train_piper.py dataset-summary
uv run python train_piper.py prepare-metadata
```

Current summary:

- source metadata: `/home/tyuukau/datasets/thuathienhue/metadata_monos_cleared_v2.csv`
- Piper metadata: `data/thuathienhue_v2_piper_metadata.csv`
- audio root: `/home/tyuukau/datasets/thuathienhue`
- valid samples: `1417`
- missing audio: `0`
- total duration: `72.65` minutes / `1.211` hours

## Train

```bash
uv run python train_piper.py train --dry-run
uv run python train_piper.py train
```

Defaults:

- dataset: `/home/tyuukau/datasets/thuathienhue`
- metadata: `data/thuathienhue_v2_piper_metadata.csv`
- voice name: `thuathienhue_v2`
- espeak voice: `vi`
- sample rate: `22050`
- batch size: `16`
- max epochs: `4000`
- accelerator: `auto`
- cleaned checkpoint: `checkpoints/hfc_male_medium.cleaned.ckpt`
- Lightning log dir: `lightning_logs_v2`

When the NVIDIA driver is visible to PyTorch, `auto` uses `gpu`. Otherwise, it
falls back to `cpu`. The helper patches Piper's Lightning module to log
`train/loss_g`, `train/loss_d`, and `train/loss_total` every step for live loss
monitoring.

Extra debug signals are also available in the patched Lightning module:

- generator/discriminator sub-losses: mel, KL, duration, feature matching, adversarial, discriminator
- learning rates for both optimizers
- gradient norm, max absolute gradient, and non-finite gradient counts
- optional parameter histograms
- validation audio samples and target/predicted mel images

For a one-batch smoke test without restoring the high-epoch pretrained
checkpoint:

```bash
uv run python train_piper.py train \
  --fast-dev-run \
  --no-restore-checkpoint \
  --checkpoint-every-n-epochs 0
```

## Live Loss

Start TensorBoard in another terminal:

```bash
uv run python train_piper.py tensorboard --host 127.0.0.1 --port 6006
```

For access from another machine on the LAN, use `--host 0.0.0.0`.

## W&B

TensorBoard remains the default logger. To log the same run to both TensorBoard
and Weights & Biases:

```bash
uv run python train_piper.py train \
  --wandb \
  --wandb-project nghitts-piper \
  --wandb-run-name thuathienhue_v2_debug
```

For machines without W&B login/network access, use offline mode:

```bash
uv run python train_piper.py train --wandb --wandb-mode offline
```

Useful debug toggles:

- `--log-grad-every-n-steps 50`: gradient stats logging cadence; use `0` to disable
- `--log-param-hist-every-n-epochs 1`: TensorBoard parameter histograms; default `0` disables them
- `--log-mel-every-n-epochs 1`: validation mel image cadence
- `--max-debug-audio-examples 5`: generated validation audio examples
- `--detect-anomaly`: enable PyTorch autograd anomaly detection
- `--wandb-log-model`: upload checkpoints as W&B artifacts

## Checkpoints

Training saves all periodic checkpoints plus `last.ckpt`:

- directory: `training_checkpoints/thuathienhue_v2`
- filename pattern: `epoch={epoch:04d}-step={step}`
- interval: every `200` epochs by default
- retention: `save_top_k=-1`, so periodic checkpoints are kept

## Export

```bash
uv run python train_piper.py export \
  --export-checkpoint training_checkpoints/thuathienhue_v2/last.ckpt
```

## Test Inference

```bash
uv run python train_piper.py synthesize \
  --text "Xin chao tat ca moi nguoi co mat trong ngay hom nay"
```
