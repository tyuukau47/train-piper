#!/usr/bin/env python3
"""Local Piper training helper.

This script turns the Colab notebook in this directory into a repeatable local
workflow for the ThuaThienHue dataset.
"""

from __future__ import annotations

import argparse
import csv
import inspect
import json
import logging
import shutil
import subprocess
import sys
import wave
from pathlib import Path, PosixPath, WindowsPath
from urllib.request import urlretrieve

LOG = logging.getLogger("train_piper")

ROOT = Path(__file__).resolve().parent
PIPER_REPO = ROOT / "piper1-gpl"
DEFAULT_DATASET = Path("/home/tyuukau/datasets/thuathienhue")
DEFAULT_SOURCE_CSV = DEFAULT_DATASET / "metadata_monos_cleared_v2.csv"
DEFAULT_METADATA = ROOT / "data" / "thuathienhue_v2_piper_metadata.csv"
DEFAULT_CACHE = ROOT / "CACHE_DIR_v2"
DEFAULT_CONFIG = ROOT / "outputs" / "thuathienhue_v2.onnx.json"
DEFAULT_CKPT = ROOT / "checkpoints" / "hfc_male_medium.ckpt"
DEFAULT_CLEAN_CKPT = ROOT / "checkpoints" / "hfc_male_medium.cleaned.ckpt"
DEFAULT_ONNX = ROOT / "outputs" / "thuathienhue_v2.onnx"
DEFAULT_TRAIN_CHECKPOINTS = ROOT / "training_checkpoints"
DEFAULT_LOG_DIR = ROOT / "lightning_logs_v2"
DEFAULT_WANDB_PROJECT = "nghitts-piper"
PIPER_LIGHTNING = PIPER_REPO / "src" / "piper" / "train" / "vits" / "lightning.py"

CHECKPOINT_URL = (
    "https://huggingface.co/datasets/rhasspy/piper-checkpoints/resolve/main/"
    "en/en_US/hfc_male/medium/epoch%3D2785-step%3D2128064.ckpt"
)


def run(cmd: list[str], cwd: Path | None = None) -> None:
    LOG.info("$ %s", " ".join(cmd))
    subprocess.run(cmd, cwd=cwd, check=True)


def cuda_summary() -> tuple[bool, str]:
    try:
        import torch
    except Exception as exc:  # pragma: no cover - diagnostic path
        return False, f"torch import failed: {exc}"

    if not torch.cuda.is_available():
        return False, f"CUDA unavailable to torch {torch.__version__}"

    device = torch.cuda.get_device_name(0)
    torch_cuda = torch.version.cuda or "unknown"
    capability = ".".join(str(x) for x in torch.cuda.get_device_capability(0))
    return True, f"CUDA available: {device}, torch CUDA {torch_cuda}, capability {capability}"


def doctor(_: argparse.Namespace) -> None:
    has_cuda, summary = cuda_summary()
    LOG.info(summary)
    for exe in ("cmake", "ninja", "pkg-config", "espeak-ng", "ffmpeg"):
        found = shutil.which(exe)
        LOG.info("%s: %s", exe, found or "missing")
    if not has_cuda:
        LOG.warning("Training will default to CPU unless you pass --accelerator gpu after fixing the driver.")


def prepare_metadata(args: argparse.Namespace) -> None:
    source_csv = args.source_csv
    output_csv = args.metadata_csv
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    rows = 0
    with source_csv.open("r", encoding="utf-8", newline="") as in_file, output_csv.open(
        "w", encoding="utf-8", newline=""
    ) as out_file:
        reader = csv.DictReader(in_file, delimiter="|")
        writer = csv.writer(out_file, delimiter="|", lineterminator="\n")
        for row in reader:
            filename = (row.get("filename") or "").strip()
            text = (row.get("text") or "").strip()
            if not filename or not text:
                continue
            audio_path = args.audio_dir / filename
            if not audio_path.exists():
                LOG.warning("Skipping missing audio: %s", audio_path)
                continue
            writer.writerow([filename, text])
            rows += 1

    LOG.info("Wrote %s Piper metadata rows to %s", rows, output_csv)


def dataset_summary(args: argparse.Namespace) -> None:
    source_csv = args.source_csv
    dataset_dir = args.audio_dir
    durations = []
    rows = 0
    missing = 0

    with source_csv.open("r", encoding="utf-8", newline="") as in_file:
        reader = csv.DictReader(in_file, delimiter="|")
        for row in reader:
            filename = (row.get("filename") or "").strip()
            if not filename:
                continue
            rows += 1
            audio_path = dataset_dir / filename
            if not audio_path.exists():
                missing += 1
                continue
            with wave.open(str(audio_path), "rb") as wav_file:
                durations.append(wav_file.getnframes() / wav_file.getframerate())

    total_seconds = sum(durations)
    LOG.info("Source CSV: %s", source_csv)
    LOG.info("Audio root: %s", dataset_dir)
    LOG.info("Rows: %s", rows)
    LOG.info("Existing audio: %s", len(durations))
    LOG.info("Missing audio: %s", missing)
    LOG.info("Total duration: %.2f minutes (%.3f hours)", total_seconds / 60, total_seconds / 3600)
    if durations:
        LOG.info("Duration min/mean/max: %.3fs / %.3fs / %.3fs", min(durations), total_seconds / len(durations), max(durations))


def download_checkpoint(args: argparse.Namespace) -> None:
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    if args.checkpoint.exists() and not args.force:
        LOG.info("Checkpoint already exists: %s", args.checkpoint)
        return
    LOG.info("Downloading checkpoint to %s", args.checkpoint)
    urlretrieve(CHECKPOINT_URL, args.checkpoint)


def convert_paths_to_strings(value):
    if isinstance(value, dict):
        return {key: convert_paths_to_strings(item) for key, item in value.items()}
    if isinstance(value, list):
        return [convert_paths_to_strings(item) for item in value]
    if isinstance(value, (PosixPath, WindowsPath)):
        return str(value)
    return value


def clean_checkpoint(args: argparse.Namespace) -> None:
    import torch
    from piper.train.vits.lightning import VitsModel

    args.clean_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    checkpoint = convert_paths_to_strings(checkpoint)

    if "hyper_parameters" in checkpoint:
        valid_params = set(inspect.signature(VitsModel.__init__).parameters)
        invalid_params = set(checkpoint["hyper_parameters"]) - valid_params
        for param in sorted(invalid_params):
            LOG.info("Removing incompatible checkpoint hyperparameter: %s", param)
            del checkpoint["hyper_parameters"][param]

    torch.save(checkpoint, args.clean_checkpoint)
    LOG.info("Saved cleaned checkpoint: %s", args.clean_checkpoint)


def build_alignment(_: argparse.Namespace) -> None:
    run(["bash", "build_monotonic_align.sh"], cwd=PIPER_REPO)
    run([sys.executable, "setup.py", "build_ext", "--inplace", "-v"], cwd=PIPER_REPO)


def ensure_debug_logging() -> None:
    """Make Piper's Lightning module expose richer debug signals.

    This script carries a local Piper checkout. Keep its Lightning module patched
    with scalar sub-losses, gradients, learning rates, histograms, and sample
    media hooks that are useful in TensorBoard and W&B.
    """

    source = PIPER_LIGHTNING.read_text(encoding="utf-8")
    if "_log_grad_stats" in source and "debug_wandb_watch" in source:
        LOG.info("Debug logging patch already present: %s", PIPER_LIGHTNING)
        return

    old = '''        self.log("loss_g", loss_g, batch_size=self.batch_size)
        opt_g.zero_grad()
        self.manual_backward(loss_g, retain_graph=True)
        opt_g.step()

        self.log("loss_d", loss_d, batch_size=self.batch_size)
        opt_d.zero_grad()
        self.manual_backward(loss_d)
        opt_d.step()
'''
    new = '''        self.log(
            "train/loss_g",
            loss_g,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=self.batch_size,
        )
        opt_g.zero_grad()
        self.manual_backward(loss_g, retain_graph=True)
        opt_g.step()

        self.log(
            "train/loss_d",
            loss_d,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=self.batch_size,
        )
        self.log(
            "train/loss_total",
            loss_g + loss_d,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=self.batch_size,
        )
        opt_d.zero_grad()
        self.manual_backward(loss_d)
        opt_d.step()
'''
    if old not in source:
        LOG.warning(
            "Could not apply legacy loss-only patch because %s has changed. "
            "Assuming the richer debug patch is maintained in the local file.",
            PIPER_LIGHTNING,
        )
        return

    PIPER_LIGHTNING.write_text(source.replace(old, new), encoding="utf-8")
    LOG.info("Patched explicit loss logging in %s", PIPER_LIGHTNING)


def split_csv(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def train(args: argparse.Namespace) -> None:
    if not args.metadata_csv.exists():
        prepare_metadata(args)

    if args.accelerator == "auto":
        has_cuda, summary = cuda_summary()
        accelerator = "gpu" if has_cuda else "cpu"
        LOG.info("%s; using trainer accelerator=%s", summary, accelerator)
    else:
        accelerator = args.accelerator

    args.config_path.parent.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    ensure_debug_logging()

    cmd = [
        sys.executable,
        "-m",
        "piper.train",
        "fit",
        "--data.voice_name",
        args.voice_name,
        "--data.csv_path",
        str(args.metadata_csv),
        "--data.audio_dir",
        str(args.audio_dir),
        "--model.sample_rate",
        str(args.sample_rate),
        "--data.espeak_voice",
        args.espeak_voice,
        "--data.cache_dir",
        str(args.cache_dir),
        "--data.config_path",
        str(args.config_path),
        "--data.batch_size",
        str(args.batch_size),
        "--data.num_workers",
        str(args.num_workers),
        "--trainer.max_epochs",
        str(args.max_epochs),
        "--trainer.accelerator",
        accelerator,
        "--trainer.devices",
        str(args.devices),
        "--trainer.log_every_n_steps",
        str(args.log_every_n_steps),
        "--trainer.default_root_dir",
        str(args.log_dir),
        "--trainer.detect_anomaly",
        str(args.detect_anomaly).lower(),
        "--trainer.fast_dev_run",
        str(args.fast_dev_run).lower(),
        "--model.debug_log_grad_every_n_steps",
        str(args.log_grad_every_n_steps),
        "--model.debug_log_param_hist_every_n_epochs",
        str(args.log_param_hist_every_n_epochs),
        "--model.debug_log_mel_every_n_epochs",
        str(args.log_mel_every_n_epochs),
        "--model.debug_max_audio_examples",
        str(args.max_debug_audio_examples),
        "--model.debug_wandb_watch",
        str(args.wandb).lower(),
        "--model.debug_wandb_log_freq",
        str(args.wandb_watch_log_freq),
    ]
    if args.wandb:
        run_name = args.wandb_run_name or args.voice_name
        wandb_tags = split_csv(args.wandb_tags)
        tensorboard_logger = {
            "class_path": "lightning.pytorch.loggers.TensorBoardLogger",
            "init_args": {
                "save_dir": str(args.log_dir),
                "name": "tensorboard",
            },
        }
        wandb_init_args = {
            "project": args.wandb_project,
            "name": run_name,
            "save_dir": str(args.log_dir),
            "offline": args.wandb_mode == "offline",
            "log_model": "all" if args.wandb_log_model else False,
        }
        if args.wandb_entity:
            wandb_init_args["entity"] = args.wandb_entity
        if wandb_tags:
            wandb_init_args["tags"] = wandb_tags
        wandb_logger = {
            "class_path": "lightning.pytorch.loggers.WandbLogger",
            "init_args": wandb_init_args,
        }
        cmd.extend(
            [
                "--trainer.logger+=" + json.dumps(tensorboard_logger),
                "--trainer.logger+=" + json.dumps(wandb_logger),
            ]
        )
    else:
        cmd.extend(
            [
                "--trainer.logger",
                "lightning.pytorch.loggers.TensorBoardLogger",
                "--trainer.logger.save_dir",
                str(args.log_dir),
                "--trainer.logger.name",
                "tensorboard",
            ]
        )
    if args.checkpoint_every_n_epochs > 0:
        checkpoint_dir = args.checkpoint_dir / args.voice_name
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        cmd.extend(
            [
                "--trainer.callbacks",
                "lightning.pytorch.callbacks.ModelCheckpoint",
                "--trainer.callbacks.dirpath",
                str(checkpoint_dir),
                "--trainer.callbacks.filename",
                "epoch={epoch:04d}-step={step}",
                "--trainer.callbacks.every_n_epochs",
                str(args.checkpoint_every_n_epochs),
                "--trainer.callbacks.save_top_k",
                "-1",
                "--trainer.callbacks.save_last",
                "true",
                "--trainer.callbacks.save_on_train_epoch_end",
                "true",
            ]
        )
    if args.no_restore_checkpoint:
        LOG.warning("Checkpoint restore disabled; training will start from scratch.")
    elif args.clean_checkpoint.exists():
        cmd.extend(["--ckpt_path", str(args.clean_checkpoint)])
    elif args.checkpoint.exists():
        cmd.extend(["--ckpt_path", str(args.checkpoint)])
    else:
        LOG.warning("No checkpoint found; training from scratch.")

    LOG.info("Training command:")
    LOG.info("%s", " ".join(cmd))
    if args.dry_run:
        LOG.info("Dry run requested; not starting training.")
        return

    run(cmd, cwd=ROOT)


def tensorboard(args: argparse.Namespace) -> None:
    run(
        [
            sys.executable,
            "-m",
            "tensorboard.main",
            "--logdir",
            str(args.log_dir),
            "--host",
            args.host,
            "--port",
            str(args.port),
        ],
        cwd=ROOT,
    )


def export_onnx(args: argparse.Namespace) -> None:
    args.output_onnx.parent.mkdir(parents=True, exist_ok=True)
    run(
        [
            sys.executable,
            "-m",
            "piper.train.export_onnx",
            "--checkpoint",
            str(args.export_checkpoint),
            "--output-file",
            str(args.output_onnx),
        ],
        cwd=ROOT,
    )
    if args.config_path.exists():
        output_config = args.output_onnx.with_suffix(".onnx.json")
        if args.config_path.resolve() != output_config.resolve():
            shutil.copyfile(args.config_path, output_config)
            LOG.info("Copied config next to ONNX: %s", output_config)
        else:
            LOG.info("Config already next to ONNX: %s", output_config)


def synthesize(args: argparse.Namespace) -> None:
    from piper import PiperVoice, SynthesisConfig

    has_cuda, _summary = cuda_summary()
    voice = PiperVoice.load(
        model_path=args.output_onnx,
        config_path=args.output_onnx.with_suffix(".onnx.json"),
        use_cuda=has_cuda and args.use_cuda,
    )
    syn_config = SynthesisConfig(
        volume=1.0,
        length_scale=1.0,
        noise_scale=1.0,
        noise_w_scale=1.0,
        normalize_audio=False,
    )
    args.wav_out.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(args.wav_out), "wb") as wav_file:
        voice.synthesize_wav(args.text, wav_file, syn_config=syn_config)
    LOG.info("Wrote %s", args.wav_out)


def setup_all(args: argparse.Namespace) -> None:
    prepare_metadata(args)
    download_checkpoint(args)
    clean_checkpoint(args)
    build_alignment(args)


def add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--source-csv", type=Path, default=DEFAULT_SOURCE_CSV)
    parser.add_argument("--audio-dir", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--metadata-csv", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CKPT)
    parser.add_argument("--clean-checkpoint", type=Path, default=DEFAULT_CLEAN_CKPT)
    parser.add_argument("--config-path", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--debug", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("doctor").set_defaults(func=doctor)

    p = sub.add_parser("prepare-metadata")
    add_common(p)
    p.set_defaults(func=prepare_metadata)

    p = sub.add_parser("dataset-summary")
    add_common(p)
    p.set_defaults(func=dataset_summary)

    p = sub.add_parser("download-checkpoint")
    add_common(p)
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=download_checkpoint)

    p = sub.add_parser("clean-checkpoint")
    add_common(p)
    p.set_defaults(func=clean_checkpoint)

    p = sub.add_parser("build-alignment")
    p.set_defaults(func=build_alignment)

    p = sub.add_parser("setup")
    add_common(p)
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=setup_all)

    p = sub.add_parser("train")
    add_common(p)
    p.add_argument("--voice-name", default="thuathienhue_v2")
    p.add_argument("--espeak-voice", default="vi")
    p.add_argument("--sample-rate", type=int, default=22050)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--max-epochs", type=int, default=4000)
    p.add_argument("--accelerator", choices=("auto", "gpu", "cpu"), default="auto")
    p.add_argument("--devices", default="1")
    p.add_argument("--log-every-n-steps", type=int, default=1)
    p.add_argument("--log-grad-every-n-steps", type=int, default=50)
    p.add_argument("--log-param-hist-every-n-epochs", type=int, default=0)
    p.add_argument("--log-mel-every-n-epochs", type=int, default=1)
    p.add_argument("--max-debug-audio-examples", type=int, default=5)
    p.add_argument("--detect-anomaly", action="store_true")
    p.add_argument("--fast-dev-run", action="store_true")
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--wandb-project", default=DEFAULT_WANDB_PROJECT)
    p.add_argument("--wandb-entity")
    p.add_argument("--wandb-run-name")
    p.add_argument("--wandb-tags", default="piper,tts")
    p.add_argument("--wandb-mode", choices=("online", "offline"), default="online")
    p.add_argument("--wandb-log-model", action="store_true")
    p.add_argument("--wandb-watch-log-freq", type=int, default=100)
    p.add_argument("--checkpoint-every-n-epochs", type=int, default=200)
    p.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_TRAIN_CHECKPOINTS)
    p.add_argument("--no-restore-checkpoint", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(func=train)

    p = sub.add_parser("tensorboard")
    add_common(p)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=6006)
    p.set_defaults(func=tensorboard)

    p = sub.add_parser("export")
    add_common(p)
    p.add_argument("--export-checkpoint", type=Path, required=True)
    p.add_argument("--output-onnx", type=Path, default=DEFAULT_ONNX)
    p.set_defaults(func=export_onnx)

    p = sub.add_parser("synthesize")
    p.add_argument("--output-onnx", type=Path, default=DEFAULT_ONNX)
    p.add_argument("--wav-out", type=Path, default=ROOT / "outputs" / "test.wav")
    p.add_argument("--text", default="Xin chao tat ca moi nguoi co mat trong ngay hom nay")
    p.add_argument("--use-cuda", action="store_true")
    p.set_defaults(func=synthesize)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    args.func(args)


if __name__ == "__main__":
    main()
