"""
Camera capture module for energy meter reading.

Supports picamera2 (Raspberry Pi Camera Module) and file input for
local testing without hardware.
"""

import logging
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def capture_image(config: dict, test_image_path: str = None) -> tuple:
    """
    Capture an image from the Pi camera or load a file for testing.

    Args:
        config: Full config dict (uses 'camera' and 'pipeline' sections).
        test_image_path: If provided, load this file instead of using camera.

    Returns:
        (image as BGR numpy array, path where the image was saved/loaded from)
    """
    if test_image_path:
        return _load_image(test_image_path), test_image_path

    try:
        return _capture_picamera2(config)
    except ImportError:
        raise RuntimeError(
            "picamera2 is not installed. Run on a Raspberry Pi or pass "
            "test_image_path= for local testing."
        )


def _capture_picamera2(config: dict) -> tuple:
    """Capture a still image using picamera2."""
    from picamera2 import Picamera2  # only available on Pi

    image_dir = Path(config.get("pipeline", {}).get("image_dir", "/tmp/energy_monitor"))
    image_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    image_path = image_dir / f"meter_{timestamp}.jpg"

    camera_cfg = config.get("camera", {})
    resolution = tuple(camera_cfg.get("resolution", [1920, 1080]))

    cam = Picamera2()
    still_config = cam.create_still_configuration(main={"size": resolution})
    cam.configure(still_config)

    # Apply manual exposure/gain if configured
    controls = {}
    if camera_cfg.get("exposure_time"):
        controls["ExposureTime"] = int(camera_cfg["exposure_time"])
    if camera_cfg.get("analogue_gain"):
        controls["AnalogueGain"] = float(camera_cfg["analogue_gain"])
    if controls:
        cam.set_controls(controls)

    cam.start()
    time.sleep(2)  # Allow auto-exposure to settle

    # Trigger autofocus (Camera Module 3). Falls back gracefully on fixed-focus
    # cameras (Module 1/2) which don't support AfMode.
    if camera_cfg.get("autofocus", True):
        try:
            success = cam.autofocus_cycle()
            if not success:
                logger.warning("Autofocus did not converge; capturing anyway.")
        except Exception as exc:
            logger.debug("Autofocus not supported on this camera: %s", exc)

    cam.capture_file(str(image_path))
    cam.stop()
    cam.close()

    logger.info("Captured image: %s", image_path)
    image = cv2.imread(str(image_path))
    if image is None:
        raise RuntimeError(f"Failed to read captured image: {image_path}")

    # Rotate via OpenCV (picamera2 doesn't expose a Rotation control)
    rotation = camera_cfg.get("rotation", 0)
    if rotation == 90:
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    elif rotation == 180:
        image = cv2.rotate(image, cv2.ROTATE_180)
    elif rotation == 270:
        image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)

    return image, str(image_path)


def _load_image(path: str) -> np.ndarray:
    """Load an image from disk (used for testing)."""
    image = cv2.imread(path)
    if image is None:
        raise ValueError(f"Could not load image: {path}")
    logger.info("Loaded test image: %s", path)
    return image


def delete_image(image_path: str) -> None:
    """Remove an image file after it has been processed."""
    try:
        Path(image_path).unlink(missing_ok=True)
        logger.debug("Deleted image: %s", image_path)
    except OSError as exc:
        logger.warning("Could not delete image %s: %s", image_path, exc)
