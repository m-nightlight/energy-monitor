"""
Image preprocessing module for energy meter reading.

Pipeline:
  1. Find the LCD display rectangle using perspective detection
     (handles camera angle / display tilt automatically).
  2. Apply a perspective warp to produce a straight, level display crop.
  3. Crop to the digit-only region (configurable fractions).
  4. Convert to grayscale, scale up, CLAHE, threshold, morphological close.
  5. Return a binary image ready for Tesseract.
"""

import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def extract_display(image: np.ndarray, config: dict, debug_dir: Optional[str] = None) -> np.ndarray:
    """
    Locate the LCD display and return a cropped, contrast-enhanced
    **grayscale** image ready for OCR multi-pass processing.

    The OCR module (`ocr.py`) applies its own per-pass thresholding and
    morphological operations, so this function stops at grayscale.

    Strategy (tried in order):
      1. Manual crop_region from config (most reliable once calibrated).
      2. Automatic perspective detection via minAreaRect.
      3. Full image fallback with a warning.

    Args:
        image: BGR image (from camera or file).
        config: Full config dict.
        debug_dir: If set, intermediate images are saved here.

    Returns:
        Grayscale numpy array for multi-pass Tesseract OCR.
    """
    display_cfg = config.get("display", {})

    # ── 1. Manual crop override ──────────────────────────────────────────────
    crop_region = display_cfg.get("crop_region")
    if crop_region:
        x, y, w, h = crop_region
        logger.debug("Using manual crop_region: x=%d y=%d w=%d h=%d", x, y, w, h)
        crop = image[y : y + h, x : x + w]
        _save_debug(crop, debug_dir, "01_manual_crop.png")
        return _to_grayscale(crop, debug_dir)

    # ── 2. Perspective detection ─────────────────────────────────────────────
    warped = _detect_and_warp(image, display_cfg, debug_dir)

    if warped is not None:
        _save_debug(warped, debug_dir, "02_warped_display.png")
        digit_crop = _crop_digit_region(warped, display_cfg)
        _save_debug(digit_crop, debug_dir, "03_digit_crop.png")
        return _to_grayscale(digit_crop, debug_dir)

    # ── 3. Full-image fallback ───────────────────────────────────────────────
    logger.warning(
        "Display not detected; using full image. "
        "Run with --calibrate after mounting the camera to set crop_region "
        "or adjust the ROI fractions in config.yaml."
    )
    _save_debug(image, debug_dir, "02_full_image_fallback.png")
    return _to_grayscale(image, debug_dir)


# ── Perspective detection ────────────────────────────────────────────────────

def _detect_display(
    image: np.ndarray, cfg: dict, debug_dir: Optional[str] = None
) -> tuple:
    """
    Locate the LCD display and return ``(warped_crop, (x, y, w, h))`` in
    full-image coordinates, or ``(None, None)`` if detection fails.

    Intended for the ``--calibrate`` workflow so callers can report the
    bounding box to the user and draw an annotation on the original image.
    """
    result = _find_and_warp(image, cfg, debug_dir)
    if result is None:
        return None, None
    warped, best_box = result
    x, y, w, h = cv2.boundingRect(best_box)
    return warped, (x, y, w, h)


def _detect_and_warp(image: np.ndarray, cfg: dict, debug_dir: Optional[str]) -> Optional[np.ndarray]:
    """
    Locate the LCD display rectangle (which may be tilted relative to the
    camera) using the following approach:

      • Convert to HSV value channel (best contrast for the dark LCD body).
      • CLAHE equalize a configurable ROI sub-region.
      • Threshold to isolate the dark LCD background.
      • Morphological close to produce a solid rectangle.
      • minAreaRect on the largest qualifying contour.
      • getPerspectiveTransform → warpPerspective.

    Returns the warped (level, rectangular) display image, or None.
    """
    result = _find_and_warp(image, cfg, debug_dir)
    return result[0] if result is not None else None


def _find_and_warp(
    image: np.ndarray, cfg: dict, debug_dir: Optional[str]
) -> Optional[tuple]:
    """
    Core detection routine.  Returns ``(warped, best_box)`` where
    ``best_box`` is a (4, 2) int array of corner points in full-image
    coordinates, or ``None`` if no display was found.
    """
    h_img, w_img = image.shape[:2]

    # ROI fractions: narrow down to the area where the display lives
    roi_y0 = int(h_img * cfg.get("roi_y_start", 0.35))
    roi_y1 = int(h_img * cfg.get("roi_y_end",   0.70))
    roi_x0 = int(w_img * cfg.get("roi_x_start", 0.25))
    roi_x1 = int(w_img * cfg.get("roi_x_end",   0.75))

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    val = hsv[:, :, 2]
    roi = val[roi_y0:roi_y1, roi_x0:roi_x1]

    _save_debug(roi, debug_dir, "00a_roi_value.png")

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    roi_eq = clahe.apply(roi)

    dark_thresh = int(cfg.get("display_dark_threshold", 80))
    _, dark = cv2.threshold(roi_eq, dark_thresh, 255, cv2.THRESH_BINARY_INV)
    _save_debug(dark, debug_dir, "00b_dark_mask.png")

    # Dilate horizontally to merge segments within the same display row
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 10))
    closed = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, kernel)
    _save_debug(closed, debug_dir, "00c_closed.png")

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    roi_area = roi.shape[0] * roi.shape[1]
    min_area = roi_area * cfg.get("min_area_fraction", 0.05)
    min_ar   = cfg.get("min_aspect_ratio", 2.0)
    max_ar   = cfg.get("max_aspect_ratio", 10.0)

    best_area = 0
    best_box  = None

    for cnt in sorted(contours, key=cv2.contourArea, reverse=True)[:10]:
        area = cv2.contourArea(cnt)
        if area < min_area:
            break

        rect = cv2.minAreaRect(cnt)
        rw, rh = rect[1]
        if rh == 0 or rw == 0:
            continue
        ar = max(rw, rh) / min(rw, rh)

        if min_ar <= ar <= max_ar and area > best_area:
            best_area = area
            best_box  = np.intp(cv2.boxPoints(rect))

    if best_box is None:
        logger.debug("No qualifying contour found for perspective detection.")
        return None

    # Translate box coords from ROI-space back to full-image space
    best_box[:, 0] += roi_x0
    best_box[:, 1] += roi_y0

    # Sort corners: top-left, top-right, bottom-right, bottom-left
    tl, tr, br, bl = _sort_corners(best_box)

    W = int(max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl)))
    H = int(max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr)))

    if W <= 0 or H <= 0:
        return None

    src = np.float32([tl, tr, br, bl])
    dst = np.float32([[0, 0], [W, 0], [W, H], [0, H]])
    M   = cv2.getPerspectiveTransform(src, dst)

    # Annotate debug image
    if debug_dir:
        annotated = image.copy()
        cv2.drawContours(annotated, [best_box], 0, (0, 255, 0), 3)
        _save_debug(annotated, debug_dir, "01_detected_box.png")

    return cv2.warpPerspective(image, M, (W, H)), best_box


def _sort_corners(box: np.ndarray) -> tuple:
    """Return (top-left, top-right, bottom-right, bottom-left) from 4 points."""
    pts = box[np.argsort(box[:, 1])]   # sort by y ascending
    top = pts[:2][np.argsort(pts[:2, 0])]  # top row: sort by x
    bot = pts[2:][np.argsort(pts[2:, 0])]  # bottom row: sort by x
    return top[0], top[1], bot[1], bot[0]  # TL, TR, BR, BL


def _crop_digit_region(warped: np.ndarray, cfg: dict) -> np.ndarray:
    """
    Crop the warped display image to the digit-only area.

    The default fractions work for the Landis+Gyr meter in energy_reading.png.
    Adjust digit_crop_x / digit_crop_y in config.yaml if your meter differs.
    """
    h, w = warped.shape[:2]

    y0_f, y1_f = cfg.get("digit_crop_y", [0.18, 0.92])
    x0_f, x1_f = cfg.get("digit_crop_x", [0.14, 0.85])

    y0, y1 = int(y0_f * h), int(y1_f * h)
    x0, x1 = int(x0_f * w), int(x1_f * w)

    return warped[y0:y1, x0:x1]


# ── Image enhancement ────────────────────────────────────────────────────────

def _to_grayscale(image: np.ndarray, debug_dir: Optional[str]) -> np.ndarray:
    """
    Convert a display crop to a grayscale image using the HSV value channel.

    The HSV value channel provides the best contrast for 7-segment LCD
    displays by reducing colour-temperature artefacts from artificial
    lighting.  Thresholding and scaling are left to the OCR multi-pass stage.
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)[:, :, 2]
    else:
        gray = image.copy()
    _save_debug(gray, debug_dir, "04_grayscale.png")
    return gray


# ── Utilities ────────────────────────────────────────────────────────────────

def _save_debug(image: np.ndarray, debug_dir: Optional[str], filename: str) -> None:
    if not debug_dir:
        return
    Path(debug_dir).mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(Path(debug_dir) / filename), image)
