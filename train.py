#!/usr/bin/env python3
"""
Authoritative training entrypoint for Goat Detector (YOLO)

ROLE:
- Configuration orchestration
- Environment validation
- Dataset validation
- Run directory management
- Delegates training & logging to Ultralytics
"""

from __future__ import annotations

import argparse
import os
import sys
import socket
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

import yaml
from ultralytics import YOLO

from utils.env_checks import run_env_checks
from utils.dataset_checks import run_dataset_checks
from utils.ui_logger import banner, section, info, success, error, warn
from utils.exceptions import classify_exception

from contextlib import contextmanager


@contextmanager
def tee_stdout(log_path: Path):
    import sys

    class Tee:
        def __init__(self, *files):
            self.files = files

        def write(self, data):
            for f in self.files:
                f.write(data)
                f.flush()

        def flush(self):
            for f in self.files:
                f.flush()

    with log_path.open("w", encoding="utf-8", buffering=1) as f:
        old_stdout, old_stderr = sys.stdout, sys.stderr
        sys.stdout = sys.stderr = Tee(old_stdout, f)
        try:
            yield
        finally:
            sys.stdout, sys.stderr = old_stdout, old_stderr


# ---------------------------------------------------------
# PATHS
# ---------------------------------------------------------

ROOT = Path(__file__).resolve().parent
CONFIGS_DIR = ROOT / "configs"
RUNS_DIR = ROOT / "runs"

EPOCH_STATE = {}


# ---------------------------------------------------------
# CONFIG
# ---------------------------------------------------------


def load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing config: {path}")
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            base[k] = deep_merge(base[k], v)
        else:
            base[k] = v
    return base


# ---------------------------------------------------------
# RUN DIR
# ---------------------------------------------------------


def get_project_dir() -> Path:
    RUNS_DIR.mkdir(exist_ok=True)
    project_dir = RUNS_DIR / "goat"
    project_dir.mkdir(parents=True, exist_ok=True)
    return project_dir


def snapshot_config(cfg: Dict[str, Any], run_dir: Path):
    with (run_dir / "config_snapshot.yaml").open("w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


def snapshot_system(run_dir: Path):
    info = {
        "hostname": socket.gethostname(),
        "python": sys.version,
        "cwd": str(Path.cwd()),
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    with (run_dir / "system_info.yaml").open("w") as f:
        yaml.safe_dump(info, f, sort_keys=False)


def on_train_epoch_end(trainer):
    stopper = trainer.stopper
    if stopper is None:
        return

    epoch = trainer.epoch

    if stopper.best_epoch is None:
        patience_used = 0
    else:
        patience_used = epoch - stopper.best_epoch

    EPOCH_STATE["epoch"] = epoch
    EPOCH_STATE["best_epoch"] = stopper.best_epoch
    EPOCH_STATE["best_fitness"] = stopper.best_fitness
    EPOCH_STATE["patience_used"] = patience_used
    EPOCH_STATE["patience_limit"] = stopper.patience
    EPOCH_STATE["possible_stop"] = stopper.possible_stop


def on_val_end(validator):
    m = validator.metrics
    if m is None:
        return

    # ---- metrics (authoritative) ----
    fitness = m.fitness
    mp, mr, map50, map5095 = m.mean_results()

    # ---- training-side info ----
    epoch = EPOCH_STATE.get("epoch")
    best_epoch = EPOCH_STATE.get("best_epoch")
    best_fitness = EPOCH_STATE.get("best_fitness")
    patience_used = EPOCH_STATE.get("patience_used", 0)
    patience_limit = EPOCH_STATE.get("patience_limit", 0)
    possible_stop = EPOCH_STATE.get("possible_stop", False)

    if epoch is None:
        return

    human_epoch = epoch + 1
    human_best_epoch = best_epoch + 1 if best_epoch is not None else "N/A"

    def fmt(x):
        return f"{x:.4f}" if isinstance(x, (int, float)) else "N/A"

    is_improvement = patience_used == 0
    near_stop = possible_stop or (patience_limit - patience_used) <= 3

    if is_improvement:
        info(
            f"[EPOCH {human_epoch}] 🚀 NEW BEST | "
            f"fitness={fmt(fitness)} | "  # in this i changed from best_fitness to fitness if something goes wrong make sure undo this changes
            f"best@{human_best_epoch} | "
            f"P={fmt(mp)} R={fmt(mr)} | "
            f"mAP50={fmt(map50)} mAP50-95={fmt(map5095)} | "
            f"patience reset"
        )
    else:
        status = "⚠ plateau" if near_stop else "⏳ training"
        info(
            f"[EPOCH {human_epoch}] {status} | "
            f"fitness={fmt(fitness)} | "
            f"best={fmt(best_fitness)}@{human_best_epoch} | "
            f"P={fmt(mp)} R={fmt(mr)} | "
            f"mAP50={fmt(map50)} mAP50-95={fmt(map5095)} | "
            f"patience {patience_used}/{patience_limit}"
        )


def on_train_end(trainer):
    """
    Final training summary.
    Runs once, clean and deterministic.
    """

    stopper = trainer.stopper
    total_epochs = trainer.args.epochs
    final_epoch = trainer.epoch + 1

    best_fitness = stopper.best_fitness if stopper else None
    best_epoch = (
        stopper.best_epoch + 1 if stopper and stopper.best_epoch is not None else "N/A"
    )

    stopped_early = stopper.possible_stop if stopper else False

    def fmt(x):
        return f"{x:.4f}" if isinstance(x, (int, float)) else "N/A"

    section("Training Summary")

    if stopped_early:
        success(
            f"Early stopping triggered at epoch {final_epoch}/{total_epochs} | "
            f"Best fitness={fmt(best_fitness)} @ epoch {best_epoch}"
        )
    else:
        success(
            f"Training completed ({final_epoch}/{total_epochs} epochs) | "
            f"Best fitness={fmt(best_fitness)} @ epoch {best_epoch}"
        )


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------


def main():
    banner("GOAT DETECTOR – YOLO TRAINING", "Production Training Pipeline")

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-config", required=True)
    args = parser.parse_args()

    section("Configuration")

    cfg = deep_merge(
        load_yaml(CONFIGS_DIR / "base.yaml"),
        load_yaml(CONFIGS_DIR / args.model_config),
    )

    success("Configuration loaded")

    run_env_checks(cfg)
    run_dataset_checks(cfg)

    section("Run Initialization")

    project_dir = get_project_dir()
    success(f"YOLO project directory: {project_dir}")

    run_dir: Path | None = None
    tmp_log = project_dir / "_bootstrap_train.log"

    try:
        section("Training")

        info(f"Model weights: {cfg['model']['weights']}")
        info("Delegating logging & progress to Ultralytics")

        model = YOLO(cfg["model"]["weights"])

        model.add_callback("on_train_epoch_end", on_train_epoch_end)
        model.add_callback("on_val_end", on_val_end)
        model.add_callback("on_train_end", on_train_end)

        with tee_stdout(tmp_log):

            model.train(
                # ---------------- DATA ----------------
                data=cfg["dataset"]["data_yaml"],
                imgsz=cfg["training"]["imgsz"],
                epochs=cfg["training"]["epochs"],
                batch=cfg["training"]["batch"],
                device=cfg["training"]["device"],
                workers=cfg["training"]["workers"],
                cache=cfg["dataset"]["cache"],
                # ---------------- OPTIMIZATION ----------------
                optimizer=cfg["optimizer"]["type"],  # SGD (good for large datasets)
                lr0=cfg["training"]["lr0"],  # initial LR
                momentum=cfg["optimizer"]["momentum"],
                weight_decay=cfg["optimizer"]["weight_decay"],
                amp=cfg["training"]["amp"],  # mixed precision (KEEP THIS)
                # ---------------- MODEL BEHAVIOR ----------------
                pretrained=cfg["training"]["pretrained"],
                single_cls=True,  # enforced single class
                patience=cfg["training"]["patience"],  # early stopping
                # ---------------- LOGGING & VISUALS ----------------
                verbose=cfg["logging"]["verbose"],  # console logs (KEEP ON)
                plots=cfg["logging"]["save_plots"],  # loss/mAP curves (KEEP ON)
                visualize=cfg["logging"]["visualize"],  # feature maps (DEBUG USE)
                save=True,  # save best + last weights
                # Save checkpoints every N epochs
                save_period=10,  # ⭐ RECOMMENDED (not -1)
                # ---------------- OUTPUT ----------------
                project=project_dir,
                name=f"{cfg['model']['name']}_img{cfg['training']['imgsz']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                exist_ok=False,
                # ---------------- AUGMENTATION ----------------
                **cfg["augmentation"],
            )

        run_dir = Path(model.trainer.save_dir)
        success(f"Unified run directory resolved: {run_dir}")

        # -------------------------------------------------
        # Finalize logs + metadata
        # -------------------------------------------------
        logs_dir = run_dir / "logs"
        logs_dir.mkdir(exist_ok=True)

        final_log = logs_dir / "train.log"
        final_log.write_text(tmp_log.read_text(encoding="utf-8"))

        snapshot_config(cfg, run_dir)
        snapshot_system(run_dir)
        (run_dir / "status.txt").write_text("COMPLETED\n")

        success("Training completed successfully")

    # ⭐ CLEAN MANUAL STOP
    except KeyboardInterrupt:
        section("Manual Stop Requested")
        warn("Ctrl+C detected — waiting for YOLO to stop cleanly at epoch boundary")
        warn("Do NOT force kill the process")

        if run_dir is not None:
            (run_dir / "status.txt").write_text("STOPPED_MANUAL\n")

        success("Training stopped safely. All artifacts preserved.")

    except Exception as e:
        exc = classify_exception(e)
        error(str(exc))
        if run_dir is not None:
            (run_dir / "status.txt").write_text(exc.status + "\n")
        traceback.print_exc()
        raise

    finally:
        # cleanup bootstrap log
        if tmp_log.exists():
            tmp_log.unlink()


if __name__ == "__main__":
    main()
