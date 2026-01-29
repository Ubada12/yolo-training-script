"""
Environment sanity checks for YOLO training.

Cross-platform, production-safe.
Fails FAST for fatal issues.
Warns & prompts for risky-but-allowed conditions.
NOW INCLUDES:
- Cache safety & visibility
- Dataset size estimation
- Disk/RAM feasibility checks
"""

import os
import shutil
import torch
from pathlib import Path
from platform import system
from typing import Tuple

from utils.ui_logger import (
    section,
    checklist,
    warn,
    success,
    console,
)


class EnvironmentError(RuntimeError):
    """Fatal environment error."""


# -------------------------------------------------
# PROMPT HELPER
# -------------------------------------------------


def confirm(prompt: str) -> bool:
    """
    Ask user for Y/N confirmation in terminal.
    """
    while True:
        choice = (
            console.input(f"[bold yellow]{prompt} [y/N]: [/bold yellow]")
            .strip()
            .lower()
        )
        if choice in ("y", "yes"):
            return True
        if choice in ("n", "no", ""):
            return False


# -------------------------------------------------
# CACHE SIZE ESTIMATION  ✅ ADDED
# -------------------------------------------------


def estimate_dataset_size(root: Path) -> Tuple[int, float]:
    """
    Estimate dataset cache size.

    Returns:
        (image_count, estimated_size_gb)

    NOTE:
    - YOLO cache includes decoded images + augmentation overhead
    - We conservatively multiply raw image size
    """
    image_exts = {".jpg", ".jpeg", ".png", ".bmp"}
    total_bytes = 0
    count = 0

    for split in ("train", "val"):
        img_dir = root / "images" / split
        if not img_dir.exists():
            continue

        for p in img_dir.rglob("*"):
            if p.suffix.lower() in image_exts:
                count += 1
                total_bytes += p.stat().st_size

    # 🔒 SAFETY FACTOR:
    # decoded images + augmentation buffers
    estimated_bytes = total_bytes * 3.0
    estimated_gb = estimated_bytes / (1024**3)

    return count, estimated_gb


# -------------------------------------------------
# MAIN ENTRY
# -------------------------------------------------


def run_env_checks(cfg: dict):
    section("Environment Checks")

    results = []

    os_name = system()
    results.append(("Operating System", True, os_name))

    # -------------------------------------------------
    # CUDA
    # -------------------------------------------------

    cuda_ok = torch.cuda.is_available()
    results.append(("CUDA available", cuda_ok, ""))

    if not cuda_ok:
        checklist("Environment Status", results)
        raise EnvironmentError("CUDA is not available")

    gpu_count = torch.cuda.device_count()
    results.append(("GPU detected", gpu_count > 0, f"{gpu_count} GPU(s)"))

    # -------------------------------------------------
    # VRAM (SOFT FAIL)
    # -------------------------------------------------

    props = torch.cuda.get_device_properties(0)
    total_gb = props.total_memory / (1024**3)
    min_required = cfg.get("training", {}).get("min_vram_gb", 8)

    vram_ok = total_gb >= min_required
    results.append(
        ("VRAM", vram_ok, f"{total_gb:.1f} GB (recommended ≥ {min_required} GB)")
    )

    # -------------------------------------------------
    # CPU WORKERS
    # -------------------------------------------------

    cpu_count = os.cpu_count() or 1
    requested_workers = cfg["training"]["workers"]
    workers_ok = requested_workers <= cpu_count

    results.append(("CPU workers", workers_ok, f"{requested_workers} / {cpu_count}"))

    if not workers_ok:
        checklist("Environment Status", results)
        raise EnvironmentError(
            f"workers={requested_workers} exceeds CPU cores={cpu_count}"
        )

    # -------------------------------------------------
    # SHARED MEMORY (LINUX ONLY)
    # -------------------------------------------------

    shm_free_gb = None

    if os_name == "Linux":
        shm = Path("/dev/shm")
        if shm.exists():
            shm_free_gb = shutil.disk_usage(shm).free / (1024**3)
            shm_ok = shm_free_gb >= 2
            results.append(("/dev/shm", shm_ok, f"{shm_free_gb:.1f} GB free"))
        else:
            shm_ok = False
            results.append(("/dev/shm", False, "not found"))

        if not shm_ok:
            checklist("Environment Status", results)
            raise EnvironmentError(
                "Insufficient /dev/shm space (need ≥ 2GB). "
                "Use cache='disk' or reduce workers."
            )
    else:
        results.append(("/dev/shm", True, "skipped (non-Linux OS)"))

    # -------------------------------------------------
    # data.yaml
    # -------------------------------------------------

    data_yaml = Path(cfg["dataset"]["data_yaml"])
    data_ok = data_yaml.exists()
    results.append(("data.yaml", data_ok, str(data_yaml)))

    if not data_ok:
        checklist("Environment Status", results)
        raise EnvironmentError("data.yaml not found")

    # -------------------------------------------------
    # CACHE SAFETY CHECK  ⭐ ADDED
    # -------------------------------------------------

    section("Cache Safety Checks")

    cache_mode = cfg.get("dataset", {}).get("cache", False)
    data_root = data_yaml.parent

    img_count, est_cache_gb = estimate_dataset_size(data_root)

    cache_results = []
    cache_results.append(("Cache enabled", bool(cache_mode), str(cache_mode)))
    cache_results.append(("Images detected", img_count > 0, f"{img_count} images"))
    cache_results.append(("Estimated cache size", True, f"~{est_cache_gb:.2f} GB"))

    # Determine cache target
    if cache_mode and os_name == "Linux" and shm_free_gb is not None:
        cache_target = "/dev/shm"
        available_gb = shm_free_gb
    else:
        cache_target = "disk"
        disk = shutil.disk_usage(data_root)
        available_gb = disk.free / (1024**3)

    cache_ok = available_gb >= est_cache_gb

    cache_results.append(("Cache target", True, cache_target))
    cache_results.append(("Available space", cache_ok, f"{available_gb:.2f} GB free"))

    checklist("Cache Status", cache_results)

    if cache_mode and not cache_ok:
        warn("Cache space is LOWER than estimated requirement.")
        warn("Training may stall, slow down, or be killed by OS.")

        proceed = confirm("Proceed with current cache settings anyway?")
        if not proceed:
            raise EnvironmentError("Aborted due to insufficient cache space")

        success("User accepted cache risk — continuing")

    # -------------------------------------------------
    # FINAL ENV STATUS
    # -------------------------------------------------

    checklist("Environment Status", results)

    # -------------------------------------------------
    # VRAM PROMPT (AFTER DISPLAY)
    # -------------------------------------------------

    if not vram_ok:
        warn(f"Detected VRAM {total_gb:.1f} GB < recommended {min_required} GB")
        warn("Training MAY fail with OOM, instability, or degraded performance.")

        proceed = confirm("Do you want to continue anyway?")
        if not proceed:
            raise EnvironmentError("Aborted by user due to low VRAM")

        success("User accepted VRAM risk — continuing training")
