"""
OCR module for reading 7-segment LCD digits on energy meters.

Uses Tesseract with settings tuned for 7-segment displays, then applies
regex post-processing and a decimal-point insertion rule to produce a
clean floating-point reading.
"""

import logging
import re
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class OCRResult:
    value: float          # Parsed numeric reading
    raw_text: str         # Raw string from Tesseract
    confidence: float     # Tesseract mean confidence (0–100)
    unit: str             # From config (e.g. "MWh")


class OCRError(Exception):
    """Raised when no valid reading can be extracted."""


def read_display(preprocessed: np.ndarray, config: dict) -> OCRResult:
    """
    Run OCR on a preprocessed display image and return a structured result.

    Args:
        preprocessed: Grayscale/binary image from preprocess.extract_display().
        config: Full config dict (uses 'ocr' and 'database' sections).

    Returns:
        OCRResult with the parsed meter value.

    Raises:
        OCRError: If no digits can be extracted or parsed.
    """
    ocr_cfg = config.get("ocr", {})
    db_cfg = config.get("database", {})
    unit = db_cfg.get("unit", "MWh")

    raw_text, confidence = _run_tesseract(preprocessed, ocr_cfg)
    logger.debug("Tesseract raw: %r  confidence: %.1f", raw_text, confidence)

    value = _parse_reading(raw_text, ocr_cfg)
    logger.info("Parsed reading: %.3f %s  (confidence %.1f)", value, unit, confidence)

    return OCRResult(value=value, raw_text=raw_text, confidence=confidence, unit=unit)


# ── Tesseract ────────────────────────────────────────────────────────────────

def _run_tesseract(image: np.ndarray, ocr_cfg: dict) -> tuple:
    """
    Run Tesseract with multiple preprocessing configs and return the
    best result via per-digit voting.

    We try several (scale, CLAHE, threshold, PSM) combinations because
    7-segment LCD displays have low contrast and no single set of
    parameters is universally optimal.  The "winner" at each digit
    position is the character that appears most often across all passes.

    Returns:
        (raw_text, confidence_0_to_100)
    """
    try:
        import pytesseract
    except ImportError as exc:
        raise OCRError(
            "pytesseract is not installed. Run: pip install pytesseract"
        ) from exc

    decimal_places = int(ocr_cfg.get("decimal_places", 3))
    total_digits = int(ocr_cfg.get("total_digits", 7))   # full display width
    # Accept partial results with at least (decimal_places + 1) digits.
    # e.g. for 3 decimal places, accept ≥ 4 digits (right-aligned by zfill).
    min_digits = decimal_places + 1

    # ── Parameter sets ────────────────────────────────────────────────────────
    # Each entry: (scale, clahe_clip, clahe_grid, threshold, psm)
    #
    # PSM 13 (raw line, no layout analysis) consistently outperforms PSM 7
    # on 7-segment LCD displays because it bypasses Tesseract's page layout
    # stage, which can misidentify digit groups.
    #
    # Key findings from calibration on Landis+Gyr meter images:
    #   • scale=3-4, threshold=65 captures faint top bars (avoids "7"→"1")
    #   • clip=3.0-4.0 avoids over-normalising which merges segment noise
    #   • PSM 13 is far more reliable than PSM 7 for this display type
    param_sets = [
        # (scale, clahe_clip, clahe_grid, threshold, psm)
        #
        # Calibrated against two Landis+Gyr meter images with different
        # lighting.  All use PSM 13 (raw line) which is superior to PSM 7
        # for 7-segment LCD displays.
        #
        # The voting + tiebreakers below handle the remaining ambiguities:
        #   clip=3.0 / thresh=65  → captures faint segment bars well
        #   clip=4.0 / thresh=65  → slightly more contrast, fewer noise pixels
        #   thresh=70              → cleaner on well-lit / high-contrast images
        (4, 3.0, (6, 3), 65, 13),   # primary
        (3, 3.0, (6, 3), 65, 13),   # 3× variant (catches different scale artifacts)
        (4, 4.0, (6, 3), 65, 13),   # higher clip (fills in faint segments)
        (4, 3.0, (6, 3), 70, 13),   # higher threshold (less noise)
        (4, 4.0, (6, 3), 70, 13),   # both higher clip + threshold
    ]

    all_digit_strings = []  # each entry is a string of exactly `total_digits` chars

    for scale, clip, grid, thresh, psm in param_sets:
        try:
            binary = _preprocess_for_pass(image, scale, clip, grid, thresh)
            padded = cv2.copyMakeBorder(binary, 15, 15, 15, 15,
                                        cv2.BORDER_CONSTANT, value=255)
            cfg_str = f"--psm {psm} --oem 1 -c tessedit_char_whitelist=0123456789."
            text = pytesseract.image_to_string(padded, config=cfg_str).strip()
            digits = re.sub(r"[^0-9]", "", text)

            # Accept results with enough digits; right-align shorter ones
            # with leading zeros (the leading digits are most likely to be
            # dropped by Tesseract on a cluttered or noisy crop boundary).
            if len(digits) == total_digits:
                all_digit_strings.append(digits)
            elif min_digits <= len(digits) < total_digits:
                all_digit_strings.append(digits.zfill(total_digits))

            logger.debug("Pass scale=%d clip=%.1f thresh=%d psm=%d → %r",
                         scale, clip, thresh, psm, text)
        except Exception as exc:
            logger.debug("OCR pass failed: %s", exc)

    if not all_digit_strings:
        # Last-resort: single pass with default config
        try:
            cfg_str = "--psm 13 --oem 1 -c tessedit_char_whitelist=0123456789."
            text = pytesseract.image_to_string(
                cv2.copyMakeBorder(image, 15, 15, 15, 15, cv2.BORDER_CONSTANT, value=255),
                config=cfg_str,
            ).strip()
            return text, 0.0
        except pytesseract.TesseractNotFoundError as exc:
            raise OCRError(
                "Tesseract binary not found. Install: sudo apt install tesseract-ocr"
            ) from exc

    # ── Per-digit voting ──────────────────────────────────────────────────────
    voted = []
    per_digit_conf = []
    for pos in range(total_digits):
        counts: dict = {}
        for s in all_digit_strings:
            ch = s[pos]
            counts[ch] = counts.get(ch, 0) + 1

        winner = max(counts, key=counts.get)

        # 7-segment tiebreakers for common faint-bar misreads.
        # In each pair the simpler digit (fewer segments) wins ties because
        # a faint "extra" segment is more likely to be noise than a genuine
        # segment that was missed.
        #
        #   7 vs 1  – top horizontal bar of "7" is the faintest segment
        #   0 vs 8  – middle bar absent in "0" can appear as faint noise
        #   0 vs 9  – middle bar absent in "0"; top-left absent in "9"
        #   3 vs 9  – "9" = "3" + top-left bar (faint = noise)
        tiebreaks = [("1", "7"), ("8", "0"), ("9", "0"), ("9", "3")]
        for bad, good in tiebreaks:
            if winner == bad and good in counts and counts[good] >= counts[bad]:
                winner = good
                break

        votes_for_winner = counts[winner]
        voted.append(winner)
        per_digit_conf.append(votes_for_winner / len(all_digit_strings) * 100)

    voted_text = "".join(voted)
    mean_conf   = float(sum(per_digit_conf) / len(per_digit_conf))

    logger.debug(
        "Multi-pass voted: %r  (conf=%.1f, from %d valid pass(es))",
        voted_text, mean_conf, len(all_digit_strings),
    )
    return voted_text, mean_conf


def _preprocess_for_pass(
    image: np.ndarray,
    scale: int,
    clip: float,
    grid: tuple,
    thresh: int,
) -> np.ndarray:
    """Scale, CLAHE-enhance, and threshold a grayscale/binary image for one OCR pass."""
    # Expect a grayscale image (already extracted by preprocess.py)
    gray = image if len(image.shape) == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    scaled = cv2.resize(gray, (gray.shape[1] * scale, gray.shape[0] * scale),
                        interpolation=cv2.INTER_LANCZOS4)
    clahe   = cv2.createCLAHE(clipLimit=clip, tileGridSize=grid)
    enhanced = clahe.apply(scaled)
    _, binary = cv2.threshold(enhanced, thresh, 255, cv2.THRESH_BINARY)

    kernel  = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 2))
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    return cleaned


# ── Parsing ──────────────────────────────────────────────────────────────────

def _parse_reading(raw_text: str, ocr_cfg: dict) -> float:
    """
    Convert raw OCR text to a float meter reading.

    Strategy:
      1. Strip everything that isn't a digit or '.'.
      2. If Tesseract already placed a '.', trust it.
      3. Otherwise insert the decimal point based on config decimal_places.
    """
    decimal_places = int(ocr_cfg.get("decimal_places", 3))

    # Keep only digits and dots
    cleaned = re.sub(r"[^0-9.]", "", raw_text)

    if not cleaned:
        raise OCRError(f"No digits found in OCR output: {raw_text!r}")

    # Multiple dots → Tesseract noise; keep only the last one
    if cleaned.count(".") > 1:
        parts = cleaned.split(".")
        cleaned = "".join(parts[:-1]) + "." + parts[-1]

    if "." in cleaned:
        # Tesseract detected a decimal point; validate the fractional part length
        integer_part, frac_part = cleaned.split(".", 1)
        digits_only = integer_part + frac_part

        if len(frac_part) == decimal_places:
            try:
                return float(cleaned)
            except ValueError:
                pass  # Fall through to digit-only path

        # Dot position looks wrong; rebuild from all digits
        cleaned = digits_only

    # No reliable decimal point → insert from the right
    digits_only = re.sub(r"\D", "", cleaned)
    if not digits_only:
        raise OCRError(f"No digits after cleanup: {raw_text!r}")

    if decimal_places > 0 and len(digits_only) > decimal_places:
        integer_digits = digits_only[:-decimal_places]
        frac_digits = digits_only[-decimal_places:]
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
        return True  # First reading, nothing to compare

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
