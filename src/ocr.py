"""
OCR module for reading 7-segment LCD digits on energy meters.

Uses ssocr (Seven Segment OCR), a purpose-built tool for 7-segment displays.

Approach:
  1. CLAHE + binary threshold to get a clean black-on-white binary image.
  2. Column projection to locate each digit blob (separated by white gaps).
  3. Call ssocr -d 1 on each blob individually, bypassing ssocr's own
     (unreliable) digit-separation step.
  4. Collect the valid single-digit results and assemble the reading.

ssocr installation (Raspberry Pi OS):
    sudo apt install ssocr
"""

import logging
import os
import re
import subprocess
import tempfile
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class OCRResult:
    value: float          # Parsed numeric reading
    raw_text: str         # Raw digit string from ssocr
    confidence: float     # Fraction of digits read confidently (0–100)
    unit: str             # From config (e.g. "MWh")


class OCRError(Exception):
    """Raised when no valid reading can be extracted."""


def read_display(preprocessed: np.ndarray, config: dict) -> OCRResult:
    """
    Run OCR on a preprocessed display image and return a structured result.

    Args:
        preprocessed: Grayscale image from preprocess.extract_display().
        config: Full config dict (uses 'ocr' and 'database' sections).

    Returns:
        OCRResult with the parsed meter value.

    Raises:
        OCRError: If fewer than (decimal_places + 1) digits can be extracted.
    """
    ocr_cfg = config.get("ocr", {})
    db_cfg  = config.get("database", {})
    unit    = db_cfg.get("unit", "MWh")

    raw_text, confidence = _run_ssocr(preprocessed, ocr_cfg)
    logger.debug("ssocr raw: %r  confidence: %.1f", raw_text, confidence)

    value = _parse_reading(raw_text, ocr_cfg)
    logger.info("Parsed reading: %.3f %s  (confidence %.1f)", value, unit, confidence)

    return OCRResult(value=value, raw_text=raw_text, confidence=confidence, unit=unit)


# ── ssocr ─────────────────────────────────────────────────────────────────────

def _run_ssocr(image: np.ndarray, ocr_cfg: dict) -> tuple:
    """
    Run ssocr digit-by-digit on the preprocessed display image.

    Pipeline:
      1. Scale up and CLAHE-enhance for better local contrast.
      2. Binary threshold to produce a clean black-on-white image.
      3. Trim the top portion (label text) using the ssocr_y_start fraction.
      4. Trim the right portion (unit label) using the ssocr_x_end fraction.
      5. Column projection: find runs of columns with dark pixels (= digits).
      6. For each candidate segment wider than min_segment_width pixels,
         call ``ssocr -d 1`` and keep results that are a single digit [0-9].

    Returns:
        (raw_text, confidence) where confidence = 100 × (valid / total_digits).

    Raises:
        OCRError: If ssocr is not installed, or if fewer than
                  (decimal_places + 1) valid digits are found.
    """
    scale              = int(ocr_cfg.get("scale_factor", 4))
    binarize_threshold = int(ocr_cfg.get("binarize_threshold", 80))
    ssocr_threshold    = int(ocr_cfg.get("ssocr_threshold", 50))
    total_digits       = int(ocr_cfg.get("total_digits", 7))
    min_digits         = int(ocr_cfg.get("decimal_places", 3)) + 1
    # Additional crop applied to the binary image to remove label text / unit.
    y_start = float(ocr_cfg.get("ssocr_y_start", 0.25))
    x_end   = float(ocr_cfg.get("ssocr_x_end",   0.82))

    gray = image if len(image.shape) == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    scaled = cv2.resize(
        gray,
        (gray.shape[1] * scale, gray.shape[0] * scale),
        interpolation=cv2.INTER_LANCZOS4,
    )

    # CLAHE + binary threshold → black digits on white background
    clahe   = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(6, 3))
    enhanced = clahe.apply(scaled)
    _, binary = cv2.threshold(enhanced, binarize_threshold, 255, cv2.THRESH_BINARY)

    h, w = binary.shape
    digit_area = binary[int(h * y_start):int(h * 0.92), 0:int(w * x_end)]
    dh, dw = digit_area.shape

    # Column projection: dark-pixel count per column
    dark_col = (digit_area == 0).sum(axis=0)
    gap_thresh = dh * 0.05        # < 5 % of rows dark  →  inter-digit gap
    in_gap = dark_col < gap_thresh

    transitions = np.diff(in_gap.astype(int))
    starts = list(np.where(transitions == -1)[0] + 1)
    ends   = list(np.where(transitions == 1)[0])
    if not in_gap[0]:
        starts.insert(0, 0)
    if not in_gap[-1]:
        ends.append(dw - 1)

    # Minimum segment width: ~40 % of expected per-digit width, at least 10 px
    expected_digit_w = dw / max(total_digits, 1)
    min_seg_w = max(10, int(expected_digit_w * 0.40))

    segs = [(s, e) for s, e in zip(starts, ends) if e - s >= min_seg_w]
    logger.debug("Column projection: %d qualifying segments (min_w=%d)", len(segs), min_seg_w)

    # Read each segment with ssocr -d 1; keep only valid single digits 0-9
    valid_digits: list[str] = []
    for s, e in segs:
        strip = digit_area[:, s:e]
        padded = cv2.copyMakeBorder(strip, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=255)

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            tmp_path = f.name
        try:
            cv2.imwrite(tmp_path, padded)
            try:
                proc = subprocess.run(
                    ["ssocr", "-d", "1", "-t", str(ssocr_threshold), tmp_path],
                    capture_output=True, text=True, timeout=10,
                )
            except FileNotFoundError:
                raise OCRError(
                    "ssocr binary not found. Install: sudo apt install ssocr"
                )
            digit = proc.stdout.strip()
            if re.fullmatch(r"[0-9]", digit):
                valid_digits.append(digit)
                logger.debug("ssocr cols %d-%d → %r", s, e, digit)
            else:
                logger.debug("ssocr cols %d-%d → invalid %r (skipped)", s, e, digit)
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    logger.debug(
        "ssocr per-digit: %d/%d valid digits → %r",
        len(valid_digits), total_digits, "".join(valid_digits),
    )

    if len(valid_digits) < min_digits:
        raise OCRError(
            f"ssocr found only {len(valid_digits)} valid digit(s) "
            f"(need at least {min_digits})"
        )

    raw_text   = "".join(valid_digits)
    confidence = len(valid_digits) / total_digits * 100.0
    return raw_text, confidence


# ── Parsing ──────────────────────────────────────────────────────────────────

def _parse_reading(raw_text: str, ocr_cfg: dict) -> float:
    """
    Convert a raw digit string to a float meter reading.

    Strategy:
      1. Strip everything that isn't a digit or '.'.
      2. If ssocr already placed a '.', trust it.
      3. Otherwise insert the decimal point based on config decimal_places.
    """
    decimal_places = int(ocr_cfg.get("decimal_places", 3))

    cleaned = re.sub(r"[^0-9.]", "", raw_text)

    if not cleaned:
        raise OCRError(f"No digits found in OCR output: {raw_text!r}")

    # Multiple dots → noise; keep only the last one
    if cleaned.count(".") > 1:
        parts = cleaned.split(".")
        cleaned = "".join(parts[:-1]) + "." + parts[-1]

    if "." in cleaned:
        integer_part, frac_part = cleaned.split(".", 1)
        digits_only = integer_part + frac_part
        if len(frac_part) == decimal_places:
            try:
                return float(cleaned)
            except ValueError:
                pass
        cleaned = digits_only  # dot position looks wrong; rebuild from digits

    digits_only = re.sub(r"\D", "", cleaned)
    if not digits_only:
        raise OCRError(f"No digits after cleanup: {raw_text!r}")

    if decimal_places > 0 and len(digits_only) > decimal_places:
        integer_digits = digits_only[:-decimal_places]
        frac_digits    = digits_only[-decimal_places:]
        value_str = f"{integer_digits}.{frac_digits}"
    else:
        value_str = digits_only

    try:
        return float(value_str)
    except ValueError as exc:
        raise OCRError(f"Could not convert {value_str!r} to float") from exc


# ── Sanity check ─────────────────────────────────────────────────────────────

def sanity_check(new_value: float, last_value: Optional[float], config: dict) -> bool:
    """
    Return True if the new reading is plausible compared to the last one.

    Checks:
      - Value is non-negative.
      - Value does not decrease (energy meters are monotonically increasing).
      - Jump since last reading is within max_jump limit.
    """
    if new_value < 0:
        logger.warning("Sanity check failed: negative value %.3f", new_value)
        return False

    if last_value is None:
        return True

    if new_value < last_value:
        logger.warning(
            "Sanity check failed: new reading %.3f < last reading %.3f",
            new_value,
            last_value,
        )
        return False

    max_jump = config.get("ocr", {}).get("max_jump")
    if max_jump is not None:
        jump = new_value - last_value
        if jump > float(max_jump):
            logger.warning(
                "Sanity check failed: jump of %.3f exceeds max_jump=%.3f",
                jump,
                max_jump,
            )
            return False

    return True
