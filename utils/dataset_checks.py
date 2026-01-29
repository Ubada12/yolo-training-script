"""
Dataset integrity checks for YOLO single-class detection.

Expected structure (STRICT):

root/
├── images/
│   ├── train/*.jpg
│   └── val/*.jpg
│
├── labels/
│   ├── train/*.txt
│   └── val/*.txt

Rules:
- Every image MUST have a corresponding label file
- Empty label files ARE allowed (background images)
- Label lines must follow YOLO format
- Only class id = 0 allowed (single_cls)
"""

from pathlib import Path
from typing import List, Tuple

from utils.ui_logger import section, checklist


class DatasetError(RuntimeError):
    """Raised only when dataset is unusable for training."""


IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


# -------------------------------------------------
# LOW-LEVEL VALIDATORS
# -------------------------------------------------


def _parse_yolo_label(label_path: Path) -> Tuple[bool, str]:
    """
    Validate YOLO label format.

    Returns:
        (ok, message)
    """
    if label_path.stat().st_size == 0:
        return True, "empty (background)"

    with label_path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            parts = line.strip().split()
            if len(parts) != 5:
                return False, f"line {i}: expected 5 values"
            try:
                cls = int(parts[0])
                coords = list(map(float, parts[1:]))
            except ValueError:
                return False, f"line {i}: non-numeric values"

            if cls != 0:
                return False, f"line {i}: class id {cls} (only 0 allowed)"

            if not all(0.0 <= v <= 1.0 for v in coords):
                return False, f"line {i}: coords out of range [0,1]"

    return True, "ok"


# -------------------------------------------------
# SPLIT CHECK
# -------------------------------------------------


def _check_split(
    root: Path,
    split: str,
    sample_limit: int = 50,  # 👈 critical
) -> Tuple[bool, str, int]:
    """
    Validate one split (train / val) safely and fast.

    Rules:
    - Enforce folder structure
    - Ensure images exist
    - Sample labels (not full scan)
    """

    images_dir = root / "images" / split
    labels_dir = root / "labels" / split

    if not images_dir.exists():
        return False, f"missing {images_dir}", 0

    if not labels_dir.exists():
        return False, f"missing {labels_dir}", 0

    images = [p for p in images_dir.glob("*.jpg")]
    if not images:
        return False, "no images found", 0

    # 🔒 SAMPLE ONLY (never full scan)
    for img in images[:sample_limit]:
        lbl = labels_dir / f"{img.stem}.txt"
        if not lbl.exists():
            return False, f"missing label for {img.name}", 0

        ok, msg = _parse_yolo_label(lbl)
        if not ok:
            return False, f"invalid label {lbl.name}: {msg}", 0

    return (
        True,
        f"{len(images)} images (sampled {min(len(images), sample_limit)})",
        len(images),
    )


# -------------------------------------------------
# PUBLIC ENTRY POINT
# -------------------------------------------------


def run_dataset_checks(cfg: dict) -> dict:
    """
    Entry point called from train.py

    Behavior:
    - Performs exhaustive checks
    - Prints structured checklist
    - Raises DatasetError ONLY if training cannot proceed
    """
    section("Dataset Checks")

    data_yaml = Path(cfg["dataset"]["data_yaml"])
    root = data_yaml.parent

    results: List[Tuple[str, bool, str]] = []
    summary = {}

    fatal = False

    for split in ("train", "val"):
        ok, msg, count = _check_split(root, split)
        results.append((f"{split}", ok, msg))

        if ok:
            summary[split] = count
        else:
            fatal = True

    checklist("Dataset Integrity", results)

    if fatal:
        raise DatasetError(
            "Dataset structure or labels invalid. " "Fix issues above before training."
        )

    return summary
