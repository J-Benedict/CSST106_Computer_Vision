"""Lab Activity 2 (Manual Exercise 3): Edge Detection Technique (Sobel & Canny)."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from utils.cv_helpers import load_bgr_image


def _minmax_uint8(arr: np.ndarray) -> np.ndarray:
    """Normalize a non-negative floating-point magnitude map to uint8 [0, 255]."""
    max_val = float(np.max(arr))
    if max_val <= 1e-12:
        return np.zeros(arr.shape, dtype=np.uint8)
    scaled = np.clip((arr / max_val) * 255.0, 0, 255)
    return scaled.astype(np.uint8)


def run_lab2_pipeline(
    img_bgr: np.ndarray,
    blur_ksize: int = 3,
    blur_sigma: float = 0.0,
    sobel_ksize: int = 3,
    canny_low: int = 100,
    canny_high: int = 200,
) -> Dict[str, Any]:
    """Run Gaussian smoothing, Sobel X/Y/Magnitude, and Canny edge detection."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    k_blur = blur_ksize if blur_ksize % 2 == 1 else blur_ksize + 1
    img_blur = cv2.GaussianBlur(gray, (k_blur, k_blur), float(blur_sigma))

    # Sobel gradients in CV_64F to preserve negative slopes before magnitude calculation
    t0 = time.perf_counter()
    sobelx = cv2.Sobel(img_blur, cv2.CV_64F, 1, 0, ksize=sobel_ksize)
    sobely = cv2.Sobel(img_blur, cv2.CV_64F, 0, 1, ksize=sobel_ksize)
    sobel_mag_f64 = cv2.magnitude(sobelx, sobely)
    sobel_ms = (time.perf_counter() - t0) * 1000.0

    # Safe uint8 conversions for Streamlit rendering
    sobelx_u8 = cv2.convertScaleAbs(sobelx)
    sobely_u8 = cv2.convertScaleAbs(sobely)
    sobel_combined_u8 = _minmax_uint8(sobel_mag_f64)

    # Canny edge detection
    t1 = time.perf_counter()
    edges_canny = cv2.Canny(img_blur, int(canny_low), int(canny_high))
    canny_ms = (time.perf_counter() - t1) * 1000.0

    total_pixels = gray.size
    sobel_active = int(np.sum(sobel_combined_u8 >= 50))
    canny_edges_count = int(np.sum(edges_canny > 0))

    metrics_df = pd.DataFrame(
        [
            {
                "Detector": "Sobel Combined (Gradient Magnitude)",
                "Kernel / Thresholds": f"ksize={sobel_ksize}, CV_64F",
                "Edge / High-Grad Pixels": sobel_active,
                "Edge Coverage (%)": round(100.0 * sobel_active / total_pixels, 2),
                "Mean Intensity": round(float(np.mean(sobel_combined_u8)), 2),
                "Execution Time (ms)": round(sobel_ms, 3),
            },
            {
                "Detector": "Canny Edge Detector",
                "Kernel / Thresholds": f"low={canny_low}, high={canny_high}",
                "Edge / High-Grad Pixels": canny_edges_count,
                "Edge Coverage (%)": round(100.0 * canny_edges_count / total_pixels, 2),
                "Mean Intensity": round(float(np.mean(edges_canny)), 2),
                "Execution Time (ms)": round(canny_ms, 3),
            },
        ]
    )

    return {
        "gray": gray,
        "blurred": img_blur,
        "sobelx_u8": sobelx_u8,
        "sobely_u8": sobely_u8,
        "sobel_combined_u8": sobel_combined_u8,
        "sobel_mag_max": float(np.max(sobel_mag_f64)),
        "sobel_mag_mean": float(np.mean(sobel_mag_f64)),
        "edges_canny": edges_canny,
        "sobel_ms": sobel_ms,
        "canny_ms": canny_ms,
        "sobel_active": sobel_active,
        "canny_count": canny_edges_count,
        "total_pixels": total_pixels,
        "metrics_df": metrics_df,
    }


def make_lab2_figure(res: Dict[str, Any]):
    """Build Matplotlib side-by-side summary figure matching Lab Manual style."""
    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    panels = [
        (res["gray"], "Original Grayscale"),
        (res["blurred"], "Gaussian Blurred"),
        (res["sobelx_u8"], "Sobel X (Vertical Edges)"),
        (res["sobely_u8"], "Sobel Y (Horizontal Edges)"),
        (res["sobel_combined_u8"], "Sobel Combined Magnitude"),
        (res["edges_canny"], "Canny Edge Detection"),
    ]
    for ax, (img, title) in zip(axes.ravel(), panels):
        ax.imshow(img, cmap="gray")
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.axis("off")
    plt.tight_layout()
    return fig


def render(sample_paths: Dict[str, Path]) -> None:
    """Render Lab Activity 2 Streamlit UI."""
    st.header("Lab Activity 2 (Manual Exercise 3): Edge Detection Technique")
    st.caption("Course: CSST106 | Topic: Edge Detection Technique | Week No.: 3 – 4")

    with st.expander("Objective & Procedure Overview", expanded=False):
        st.markdown(
            """
            **Objective:** Understand and implement fundamental edge detection algorithms using Python and OpenCV —
            specifically **Sobel** (first-order directional gradient) and **Canny** (multi-stage optimal edge detector) —
            and compare their edge maps to prepare for segmentation, feature extraction, and object detection.

            **Procedure / Algorithms:**
            1. Load the image in grayscale (`cv2.IMREAD_GRAYSCALE` / `cv2.COLOR_BGR2GRAY`).
            2. Apply Gaussian smoothing (`cv2.GaussianBlur`) to suppress high-frequency pixel noise prior to differentiation.
            3. Compute horizontal (`dx=1, dy=0`) and vertical (`dx=0, dy=1`) Sobel derivatives in `cv2.CV_64F`, combine them
               with `cv2.magnitude(sobelx, sobely)`, and normalize gradients safely to `uint8`.
            4. Compute Canny edges (`cv2.Canny(img_blur, threshold1, threshold2)`) using Non-Maximum Suppression (NMS) and
               dual-threshold hysteresis edge tracking.
            """
        )

    uploaded = st.file_uploader(
        "Upload a custom image (optional — bundled sample pre-loaded)",
        type=["png", "jpg", "jpeg", "bmp"],
        key="lab2_upload",
    )
    img_bgr, src_label = load_bgr_image(sample_paths["lab2_edges"], uploaded_file=uploaded)
    st.info(f"**Active Input:** {src_label} | **Resolution:** `{img_bgr.shape[1]} × {img_bgr.shape[0]}`")

    st.subheader("Interactive Controls")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        blur_ksize = st.select_slider("Gaussian Blur Kernel", options=[1, 3, 5, 7, 9], value=3, key="lab2_blur_k")
    with c2:
        sobel_ksize = st.select_slider("Sobel Kernel Size", options=[1, 3, 5, 7], value=3, key="lab2_sobel_k")
    with c3:
        canny_low = st.slider("Canny Lower Threshold", 10, 250, 100, step=5, key="lab2_canny_low")
    with c4:
        canny_high = st.slider("Canny Upper Threshold", 20, 350, 200, step=5, key="lab2_canny_high")

    if canny_high <= canny_low:
        canny_high = min(350, canny_low + 20)
        st.warning(f"Adjusted Canny Upper Threshold to {canny_high} so it exceeds Lower Threshold ({canny_low}).")

    res = run_lab2_pipeline(
        img_bgr=img_bgr,
        blur_ksize=blur_ksize,
        blur_sigma=0.0,
        sobel_ksize=sobel_ksize,
        canny_low=canny_low,
        canny_high=canny_high,
    )

    st.subheader("Side-by-Side Visual Outputs")
    r1c1, r1c2, r1c3 = st.columns(3)
    with r1c1:
        st.image(res["gray"], caption="Original Grayscale", use_container_width=True)
    with r1c2:
        st.image(res["blurred"], caption=f"Gaussian Blur ({blur_ksize}×{blur_ksize})", use_container_width=True)
    with r1c3:
        st.image(res["sobelx_u8"], caption="Sobel X (Vertical Boundaries)", use_container_width=True)

    r2c1, r2c2, r2c3 = st.columns(3)
    with r2c1:
        st.image(res["sobely_u8"], caption="Sobel Y (Horizontal Boundaries)", use_container_width=True)
    with r2c2:
        st.image(res["sobel_combined_u8"], caption="Sobel Combined Magnitude (uint8)", use_container_width=True)
    with r2c3:
        st.image(res["edges_canny"], caption=f"Canny Edges ({canny_low}, {canny_high})", use_container_width=True)

    fig = make_lab2_figure(res)
    with st.expander("Combined Matplotlib Edge Map Comparison Plot", expanded=False):
        st.pyplot(fig)
    plt.close(fig)

    st.subheader("Quantitative Edge Detection Metrics")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Peak Sobel Magnitude", f"{res['sobel_mag_max']:.1f}")
    m2.metric("Mean Sobel Gradient", f"{res['sobel_mag_mean']:.2f}")
    m3.metric("Canny Edge Pixels", f"{res['canny_count']:,}")
    m4.metric("Canny Edge Density", f"{100.0 * res['canny_count'] / res['total_pixels']:.2f}%")
    st.dataframe(res["metrics_df"], use_container_width=True, hide_index=True)

    st.subheader("Post-Lab Question & Comparative Analysis")
    st.markdown(
        """
        **Question:** *Explain the differences between the edge maps produced by Sobel and Canny edge detection.*

        - **Gradient Magnitude vs. Binary Edge Map:** The **Sobel operator** computes continuous first-order spatial derivatives
          (`Gx`, `Gy`) and combines them as `G = sqrt(Gx² + Gy²)`. Its output is a grayscale **gradient-strength map** where
          pixel brightness encodes edge intensity, producing thick, graded edge ridges that remain sensitive to residual noise
          and require manual thresholding to binarize.
        - **Non-Maximum Suppression (NMS) & Thin Edges:** **Canny** goes beyond raw gradient computation by applying
          **Non-Maximum Suppression** along the quantized gradient direction, preserving only local maxima so edges are
          thinned cleanly to **1-pixel-wide contours**.
        - **Hysteresis Thresholding & Noise Robustness:** Canny applies two thresholds (`threshold1` low, `threshold2` high).
          Pixels above `threshold2` are definite strong edges, pixels below `threshold1` are discarded, and intermediate pixels
          are retained *only* if 8-connected to a strong edge. This eliminates isolated speckle noise while maintaining continuous,
          unbroken object boundaries — making Canny ideal for contour tracing and structural segmentation, whereas Sobel is
          preferable when directional gradient magnitude or edge orientation weights are needed.
        """
    )
