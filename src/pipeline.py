"""
Energy meter reading pipeline – main entry point.

Usage (production on Pi):
    python src/pipeline.py

Usage (local testing with an image file):
    python src/pipeline.py --image energy_reading.png

Usage (debug mode, saves intermediate images):
    python src/pipeline.py --image energy_reading.png --debug

Usage (calibrate: shows detected display crop and exits):
    python src/pipeline.py --image energy_reading.png --calibrate

Export readings to CSV:
    python src/pipeline.py --export readings.csv

Verify OCR against a labelled dataset:
    python src/pipeline.py --verify-dataset dataset/
"""

import argparse
import logging
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml


# ── Bootstrap logging before importing project modules ───────────────────────

def _setup_logging(config: dict) -> None:
    pipe_cfg = config.get("pipeline", {})
    level_name = pipe_cfg.get("log_level", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)

    handlers = [logging.StreamHandler(sys.stdout)]

    log_file = pipe_cfg.get("log_file")
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=level,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
    )


logger = logging.getLogger(__name__)


# ── Config loading ───────────────────────────────────────────────────────────

def _load_config(config_path: str = "config.yaml") -> dict:
    path = Path(config_path)
    if not path.exists():
        logger.warning("Config file %s not found; using defaults.", config_path)
        return {}
    with open(path) as f:
        return yaml.safe_load(f) or {}


# ── Pipeline ─────────────────────────────────────────────────────────────────

def run(config: dict, test_image_path: str = None, debug: bool = False) -> dict:
    """
    Execute a single capture→preprocess→OCR→store cycle.

    Args:
        config: Loaded config dict.
        test_image_path: Path to a local image for testing (bypasses camera).
        debug: If True, save preprocessing images to debug_dir.

    Returns:
        Dict with keys: value, unit, confidence, raw_text, sane, row_id.
    """
    from src.capture import capture_image, delete_image
    from src.preprocess import extract_display
    from src.ocr import read_display, sanity_check, OCRError
    from src import database as db

    pipe_cfg = config.get("pipeline", {})
    db_path = config.get("database", {}).get("path", "data/readings.db")
    debug_dir = pipe_cfg.get("debug_dir", "debug") if debug else None
    delete_after = pipe_cfg.get("delete_images", True) and test_image_path is None

    # ── 1. Capture ────────────────────────────────────────────────────────────
    logger.info("=== Energy meter reading pipeline started ===")
    timestamp = datetime.now(timezone.utc)

    image, image_path = capture_image(config, test_image_path=test_image_path)
    logger.info("Image: %s", image_path)

    # ── 2. Preprocess ─────────────────────────────────────────────────────────
    preprocessed = extract_display(image, config, debug_dir=debug_dir)

    # ── 3. OCR ────────────────────────────────────────────────────────────────
    try:
        result = read_display(preprocessed, config)
    except OCRError as exc:
        logger.error("OCR failed: %s", exc)
        # Store a failed/flagged row so we keep a complete audit trail
        db.init_db(db_path)
        row_id = db.insert_reading(
            db_path,
            value=0.0,
            raw_text=str(exc),
            confidence=0.0,
            image_path=image_path if not delete_after else None,
            sane=False,
            timestamp=timestamp,
        )
        if delete_after:
            delete_image(image_path)
        return {"error": str(exc), "sane": False, "row_id": row_id}

    # ── 4. Sanity check ───────────────────────────────────────────────────────
    db.init_db(db_path)
    last = db.get_last_reading(db_path)
    last_value = last["value"] if last else None
    sane = sanity_check(result.value, last_value, config)

    if not sane:
        logger.warning(
            "Reading %.3f %s flagged as insane (last=%.3f). "
            "Stored with sane=False.",
            result.value,
            result.unit,
            last_value if last_value is not None else float("nan"),
        )

    # ── 5. Store ──────────────────────────────────────────────────────────────
    stored_path = None if delete_after else image_path

    row_id = db.insert_reading(
        db_path,
        value=result.value,
        unit=result.unit,
        raw_text=result.raw_text,
        confidence=result.confidence,
        image_path=stored_path,
        sane=sane,
        timestamp=timestamp,
    )

    # ── 6. Dataset collection ─────────────────────────────────────────────────
    dataset_dir = pipe_cfg.get("dataset_dir")
    if dataset_dir and image_path:
        _save_dataset_image(image_path, result.value, timestamp, dataset_dir, pipe_cfg)

    # ── 7. Cleanup ────────────────────────────────────────────────────────────
    if delete_after:
        delete_image(image_path)

    logger.info("=== Done: %.3f %s (id=%d) ===", result.value, result.unit, row_id)

    return {
        "value": result.value,
        "unit": result.unit,
        "confidence": result.confidence,
        "raw_text": result.raw_text,
        "sane": sane,
        "row_id": row_id,
    }


# ── Dataset helpers ───────────────────────────────────────────────────────────

def _save_dataset_image(
    image_path: str,
    value: float,
    timestamp: datetime,
    dataset_dir: str,
    pipe_cfg: dict,
) -> None:
    """
    Copy the captured image into dataset_dir with the OCR reading in the filename.

    Filename format: <value>_<timestamp>.png
    Example:         35.768_2026-02-21T14-30-00.png

    After collecting a full run you can rename any mislabelled files (where
    the OCR was wrong) to the correct reading, then run --verify-dataset to
    measure accuracy on the ground-truth names.
    """
    dest_dir = Path(dataset_dir)
    max_images = int(pipe_cfg.get("dataset_hours", 24))

    # Check limit
    if max_images > 0:
        existing = sorted(dest_dir.glob("*.png")) if dest_dir.exists() else []
        if len(existing) >= max_images:
            logger.info(
                "Dataset limit reached (%d images). Not saving %s.",
                max_images, image_path,
            )
            return

    dest_dir.mkdir(parents=True, exist_ok=True)
    ts_str = timestamp.strftime("%Y-%m-%dT%H-%M-%S")
    dest = dest_dir / f"{value:.3f}_{ts_str}.png"
    shutil.copy2(image_path, dest)
    logger.info("Dataset image saved: %s", dest)


def verify_dataset(dataset_dir: str, config: dict) -> None:
    """
    Run OCR on every image in dataset_dir and compare to the ground-truth
    reading encoded in the filename.

    Expected filename format: <reading>_<anything>.png
    Example: 35.768_2026-02-21T14-30-00.png  →  ground truth = 35.768

    Prints a per-image summary and an overall accuracy score.
    """
    from src.capture import _load_image
    from src.preprocess import extract_display
    from src.ocr import read_display, OCRError

    images = sorted(Path(dataset_dir).glob("*.png"))
    if not images:
        print(f"No PNG images found in {dataset_dir}")
        return

    correct = 0
    total = 0
    errors = []

    print(f"{'File':<45}  {'Ground truth':>13}  {'OCR':>13}  {'Conf':>6}  Match")
    print("-" * 90)

    for img_path in images:
        # Parse ground truth from filename prefix
        try:
            gt_str = img_path.stem.split("_")[0]
            gt = float(gt_str)
        except (ValueError, IndexError):
            logger.warning("Cannot parse ground truth from filename: %s", img_path.name)
            continue

        total += 1
        try:
            image = _load_image(str(img_path))
            preprocessed = extract_display(image, config)
            result = read_display(preprocessed, config)
            match = abs(result.value - gt) < 0.0005   # within half a display unit
            if match:
                correct += 1
            mark = "✓" if match else "✗"
            print(
                f"{img_path.name:<45}  {gt:>13.3f}  {result.value:>13.3f}"
                f"  {result.confidence:>6.1f}  {mark}"
            )
            if not match:
                errors.append((img_path.name, gt, result.value))
        except OCRError as exc:
            print(f"{img_path.name:<45}  {gt:>13.3f}  {'ERROR':>13}  {'':>6}  ✗")
            errors.append((img_path.name, gt, f"OCR error: {exc}"))
            continue

    print("-" * 90)
    pct = 100 * correct / total if total else 0
    print(f"Accuracy: {correct}/{total} ({pct:.1f}%)")

    if errors:
        print("\nMismatches:")
        for name, gt, got in errors:
            print(f"  {name}: expected {gt}, got {got}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Energy meter OCR pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    p.add_argument("--image", default=None, help="Use a local image instead of camera")
    p.add_argument("--debug", action="store_true", help="Save preprocessing debug images")
    p.add_argument("--calibrate", action="store_true",
                   help="Show detected display region and exit (requires --image)")
    p.add_argument("--export", default=None, metavar="FILE",
                   help="Export all readings to a CSV file and exit")
    p.add_argument("--list", action="store_true",
                   help="Print the last 20 readings and exit")
    p.add_argument("--verify-dataset", default=None, metavar="DIR",
                   help="Run OCR on every image in DIR and compare to ground-truth "
                        "readings encoded in filenames (<reading>_<timestamp>.png)")
    return p


def main() -> None:
    args = _build_parser().parse_args()
    config = _load_config(args.config)
    _setup_logging(config)

    # ── Verify-dataset mode ───────────────────────────────────────────────────
    if args.verify_dataset:
        verify_dataset(args.verify_dataset, config)
        return

    # ── Export mode ───────────────────────────────────────────────────────────
    if args.export:
        from src import database as db
        db_path = config.get("database", {}).get("path", "data/readings.db")
        db.init_db(db_path)
        n = db.export_csv(db_path, args.export)
        print(f"Exported {n} readings to {args.export}")
        return

    # ── List mode ─────────────────────────────────────────────────────────────
    if args.list:
        from src import database as db
        db_path = config.get("database", {}).get("path", "data/readings.db")
        db.init_db(db_path)
        rows = db.get_readings(db_path, limit=20, sane_only=False)
        if not rows:
            print("No readings in database.")
            return
        print(f"{'ID':>5}  {'Timestamp':<22}  {'Value':>10}  {'Unit':<5}  {'Conf':>5}  Sane")
        print("-" * 62)
        for r in reversed(rows):
            sane_str = "✓" if r["sane"] else "✗"
            print(
                f"{r['id']:>5}  {r['timestamp']:<22}  "
                f"{r['value']:>10.3f}  {r['unit']:<5}  "
                f"{(r['confidence'] or 0):>5.1f}  {sane_str}"
            )
        return

    # ── Calibrate mode ────────────────────────────────────────────────────────
    if args.calibrate:
        if not args.image:
            print("--calibrate requires --image <path>", file=sys.stderr)
            sys.exit(1)
        _run_calibrate(args.image, config)
        return

    # ── Normal run ────────────────────────────────────────────────────────────
    result = run(config, test_image_path=args.image, debug=args.debug)

    if "error" in result:
        print(f"ERROR: {result['error']}", file=sys.stderr)
        sys.exit(1)

    print(f"{result['value']:.3f} {result['unit']}")


def _run_calibrate(image_path: str, config: dict) -> None:
    """Show the auto-detected display region and suggested crop coordinates."""
    import cv2
    from src.capture import _load_image
    from src.preprocess import _detect_display

    image = _load_image(image_path)
    display_cfg = config.get("display", {})
    crop, bbox = _detect_display(image, display_cfg, debug_dir=None)

    if crop is None:
        print("No display detected. Try adjusting detection parameters or set crop_region manually.")
        return

    x, y, w, h = bbox
    print(f"\nDetected display region:")
    print(f"  x={x}, y={y}, width={w}, height={h}")
    print(f"\nTo hard-code this region in config.yaml:")
    print(f"  display:")
    print(f"    crop_region: [{x}, {y}, {w}, {h}]")

    # Draw rectangle on image and save
    debug_path = "debug/calibration.png"
    Path("debug").mkdir(exist_ok=True)
    annotated = image.copy()
    cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 3)
    cv2.imwrite(debug_path, annotated)
    print(f"\nAnnotated image saved to: {debug_path}")
    print(f"Cropped display saved to: debug/calibration_crop.png")
    cv2.imwrite("debug/calibration_crop.png", crop)


if __name__ == "__main__":
    main()
