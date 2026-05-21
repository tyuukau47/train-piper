# Piper Training

Local version of the workflow from `piper_demo.ipynb`.

## Setup

```bash
cd /home/tyuukau/work/exp/tts/train/train-piper
uv sync --python 3.11
uv run python train_piper.py setup
uv run python train_piper.py doctor
```

The environment pins PyTorch `2.8.0+cu126` from the CUDA 12.6 wheel index. On
this machine, `doctor` currently sees the NVIDIA GPU and required command-line
tools.

## Dataset

Input schema required by `--source-csv`:

- format: UTF-8 CSV using `|` as the delimiter
- required header columns: `filename` and `text`
- `filename`: audio file path relative to `--audio-dir`
- `text`: normalized transcript to train on
- rows with an empty `filename` or `text` are skipped
- rows whose audio file does not exist under `--audio-dir` are skipped with a warning

Example source CSV:

```csv
filename|text
wavs/0001.wav|Xin chao tat ca moi nguoi
wavs/0002.wav|Hom nay troi dep
```

`prepare-metadata` converts the source CSV into Piper metadata at
`--metadata-csv`. Piper metadata has no header and contains exactly two
pipe-delimited columns:

```csv
wavs/0001.wav|Xin chao tat ca moi nguoi
wavs/0002.wav|Hom nay troi dep
```

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

## Arguments

Global:

- `--debug`: enable debug-level logging for the helper script

Common dataset/checkpoint arguments used by `setup`, `prepare-metadata`,
`dataset-summary`, `download-checkpoint`, `clean-checkpoint`, `train`,
`tensorboard`, and `export`:

- `--source-csv PATH`: source metadata CSV with `filename|text` columns
- `--audio-dir PATH`: directory containing audio files referenced by `filename`
- `--metadata-csv PATH`: generated Piper metadata CSV
- `--checkpoint PATH`: downloaded pretrained checkpoint
- `--clean-checkpoint PATH`: checkpoint rewritten for the local Piper version
- `--config-path PATH`: generated Piper voice config JSON
- `--cache-dir PATH`: Piper preprocessing cache directory
- `--log-dir PATH`: Lightning/TensorBoard/W&B log directory

Setup and checkpoint commands:

- `setup --force`: redownload the pretrained checkpoint before cleaning it
- `download-checkpoint --force`: redownload even if the checkpoint already exists

Training arguments:

- `--voice-name NAME`: Piper voice name used for config/checkpoint paths
- `--espeak-voice VOICE`: eSpeak voice code, default `vi`
- `--sample-rate HZ`: target audio sample rate, default `22050`
- `--batch-size N`: training batch size, default `16`
- `--num-workers N`: data loader worker count, default `2`
- `--max-epochs N`: maximum training epochs, default `4000`
- `--accelerator auto|gpu|cpu`: trainer accelerator, default `auto`
- `--devices VALUE`: Lightning devices value, default `1`
- `--log-every-n-steps N`: Lightning logging cadence, default `1`
- `--checkpoint-every-n-epochs N`: save interval; use `0` to disable periodic saves
- `--checkpoint-dir PATH`: root directory for training checkpoints
- `--no-restore-checkpoint`: start without loading the pretrained checkpoint
- `--fast-dev-run`: run Lightning's one-batch smoke test
- `--dry-run`: print the training command without starting training
- `--detect-anomaly`: enable PyTorch autograd anomaly detection

Debug logging and W&B arguments:

- `--log-grad-every-n-steps N`: gradient stats cadence; use `0` to disable
- `--log-param-hist-every-n-epochs N`: parameter histogram cadence; default `0`
- `--log-mel-every-n-epochs N`: validation mel image cadence
- `--max-debug-audio-examples N`: validation audio examples to log
- `--wandb`: enable W&B alongside TensorBoard
- `--wandb-project NAME`: W&B project, default `nghitts-piper`
- `--wandb-entity NAME`: optional W&B entity/team
- `--wandb-run-name NAME`: optional W&B run name
- `--wandb-tags TAGS`: comma-separated W&B tags, default `piper,tts`
- `--wandb-mode online|offline`: W&B mode, default `online`
- `--wandb-log-model`: upload checkpoints as W&B artifacts
- `--wandb-watch-log-freq N`: W&B watch logging frequency

TensorBoard arguments:

- `--host HOST`: bind address, default `127.0.0.1`
- `--port PORT`: bind port, default `6006`

Export arguments:

- `--export-checkpoint PATH`: required checkpoint to export
- `--output-onnx PATH`: output ONNX model path

Synthesis arguments:

- `--output-onnx PATH`: ONNX model path to load
- `--wav-out PATH`: output WAV file
- `--text TEXT`: text to synthesize
- `--use-cuda`: use CUDA for inference when available

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

The pretrained warm-start checkpoints come from the Rhasspy Piper checkpoints
dataset on Hugging Face:

- male: `https://huggingface.co/datasets/rhasspy/piper-checkpoints/resolve/main/en/en_US/hfc_male/medium/epoch%3D2785-step%3D2128064.ckpt`
- female: `https://huggingface.co/datasets/rhasspy/piper-checkpoints/resolve/main/en/en_US/hfc_female/medium/epoch%3D2868-step%3D1575188.ckpt`

The helper downloads the male checkpoint by default to
`checkpoints/hfc_male_medium.ckpt`. To start from the female checkpoint,
download it manually and pass both checkpoint paths through setup/training:

```bash
mkdir -p checkpoints
wget -O checkpoints/hfc_female_medium.ckpt \
  "https://huggingface.co/datasets/rhasspy/piper-checkpoints/resolve/main/en/en_US/hfc_female/medium/epoch%3D2868-step%3D1575188.ckpt"

uv run python train_piper.py clean-checkpoint \
  --checkpoint checkpoints/hfc_female_medium.ckpt \
  --clean-checkpoint checkpoints/hfc_female_medium.cleaned.ckpt

uv run python train_piper.py train \
  --checkpoint checkpoints/hfc_female_medium.ckpt \
  --clean-checkpoint checkpoints/hfc_female_medium.cleaned.ckpt
```

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
