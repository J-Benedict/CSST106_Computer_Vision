"""Shared OpenCV / Streamlit helper utilities for CSST106 laboratory modules."""

from __future__ import annotations

from pathlib import Path
from typing import Optional
import cv2
import numpy as np


def load_bgr_image(
    default_path: Path,
    uploaded_file=None,
    camera_file=None,
) -> tuple[np.ndarray, str]:
    """Load an OpenCV BGR image from camera input, uploaded file, or default sample path."""
    source_file = camera_file if camera_file is not None else uploaded_file
    if source_file is not None:
        raw_bytes = np.frombuffer(source_file.getvalue(), dtype=np.uint8)
        decoded = cv2.imdecode(raw_bytes, cv2.IMREAD_COLOR)
        if decoded is not None:
            source_name = getattr(source_file, "name", "camera_capture.jpg")
            return decoded, f"User Input ({source_name})"

    img = cv2.imread(str(default_path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Default sample image not found: {default_path}")
    return img, f"Bundled Sample ({default_path.name})"


def bgr_to_rgb(img: np.ndarray) -> np.ndarray:
    """Convert BGR (or single-channel grayscale) image safely to RGB for Streamlit."""
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def to_uint8(arr: np.ndarray) -> np.ndarray:
    """Normalize float64/int16 arrays cleanly to uint8 [0, 255] for display."""
    if arr.dtype == np.uint8:
        return arr
    return cv2.convertScaleAbs(arr)


def compute_image_entropy(gray: np.ndarray) -> float:
    """Compute Shannon entropy (bits/pixel) of an 8-bit grayscale image."""
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel()
    probs = hist / (hist.sum() + 1e-12)
    nonzero = probs[probs > 0]
    return float(-np.sum(nonzero * np.log2(nonzero)))
