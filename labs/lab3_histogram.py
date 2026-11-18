"""Lab Activity 3: Histogram Equalization and Contrast Adjustment."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from utils.cv_helpers import bgr_to_rgb, compute_image_entropy, load_bgr_image


def _apply_linear_stretch(channel: np.ndarray, p_low: float, p_high: float, alpha: float, beta: float) -> np.ndarray:
    """Apply percentile min-max contrast stretching followed by optional linear gain/bias."""
    lo = float(np.percentile(channel, p_low))
    hi = float(np.percentile(channel, p_high))
    if hi - lo < 1e-5:
        stretched = channel.astype(np.float32)
    else:
        stretched = (channel.astype(np.float32) - lo) * (255.0 / (hi - lo))
        stretched = np.clip(stretched, 0.0, 255.0)
    adjusted = np.clip(alpha * stretched + beta, 0.0, 255.0)
    return adjusted.astype(np.uint8)


def _compute_hist_cdf(channel: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute 256-bin histogram and normalized cumulative distribution function (CDF)."""
    hist = cv2.calcHist([channel], [0], None, [256], [0, 256]).ravel()
    cdf = hist.cumsum()
    cdf_norm = cdf / (cdf[-1] + 1e-12)
    return hist, cdf_norm


def run_lab3_pipeline(
    img_bgr: np.ndarray,
    color_mode: str = "Grayscale",
    p_low: float = 2.0,
    p_high: float = 98.0,
    alpha: float = 1.0,
    beta: float = 0.0,
    clahe_clip: float = 2.0,
    clahe_tile: int = 8,
) -> Dict[str, Any]:
    """Execute Global Histogram Equalization, Linear Contrast Stretching, and CLAHE."""
    clahe = cv2.createCLAHE(clipLimit=float(clahe_clip), tileGridSize=(int(clahe_tile), int(clahe_tile)))

    if color_mode == "Grayscale":
        orig_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        eq_gray = cv2.equalizeHist(orig_gray)
        stretch_gray = _apply_linear_stretch(orig_gray, p_low, p_high, alpha, beta)
        clahe_gray = clahe.apply(orig_gray)

        displays = {
            "Original": orig_gray,
            "Global Hist Eq": eq_gray,
            "Linear Stretch": stretch_gray,
            "CLAHE": clahe_gray,
        }
        lum_channels = displays
    else:
        # Color enhancement via YUV Luminance channel (Y) to preserve chromaticity (U, V)
        yuv_orig = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YUV)
        y_orig, u_ch, v_ch = cv2.split(yuv_orig)

        y_eq = cv2.equalizeHist(y_orig)
        y_stretch = _apply_linear_stretch(y_orig, p_low, p_high, alpha, beta)
        y_clahe = clahe.apply(y_orig)

        eq_bgr = cv2.cvtColor(cv2.merge([y_eq, u_ch, v_ch]), cv2.COLOR_YUV2BGR)
        stretch_bgr = cv2.cvtColor(cv2.merge([y_stretch, u_ch, v_ch]), cv2.COLOR_YUV2BGR)
        clahe_bgr = cv2.cvtColor(cv2.merge([y_clahe, u_ch, v_ch]), cv2.COLOR_YUV2BGR)

        displays = {
            "Original": bgr_to_rgb(img_bgr),
            "Global Hist Eq": bgr_to_rgb(eq_bgr),
            "Linear Stretch": bgr_to_rgb(stretch_bgr),
            "CLAHE": bgr_to_rgb(clahe_bgr),
        }
        lum_channels = {
            "Original": y_orig,
            "Global Hist Eq": y_eq,
            "Linear Stretch": y_stretch,
            "CLAHE": y_clahe,
        }

    metrics_rows = []
    hist_data = {}
    for name, lum in lum_channels.items():
        hist, cdf_norm = _compute_hist_cdf(lum)
        hist_data[name] = (hist, cdf_norm)
        metrics_rows.append(
            {
                "Method": name,
                "Mean Luminance": round(float(np.mean(lum)), 2),
                "Contrast (Std Dev)": round(float(np.std(lum)), 2),
                "Min–Max Range": f"[{int(np.min(lum))}, {int(np.max(lum))}]",
                "Entropy (bits/px)": round(compute_image_entropy(lum), 3),
            }
        )

    return {
        "color_mode": color_mode,
        "displays": displays,
        "lum_channels": lum_channels,
        "hist_data": hist_data,
        "metrics_df": pd.DataFrame(metrics_rows),
    }


def make_lab3_hist_figure(res: Dict[str, Any]):
    """Build 4-panel Histogram & CDF curve Matplotlib figure."""
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.6))
    bins = np.arange(256)
    titles = ["Original", "Global Hist Eq", "Linear Stretch", "CLAHE"]
    colors = ["#4c78a8", "#f58518", "#54a24b", "#e45756"]

    for ax, title, col in zip(axes, titles, colors):
        hist, cdf_norm = res["hist_data"][title]
        ax.bar(bins, hist, width=1.0, color=col, alpha=0.55, label="Histogram")
        ax.set_title(f"{title}\nHist & CDF", fontsize=10, fontweight="bold")
        ax.set_xlim(0, 255)
        ax.set_xlabel("Intensity (0–255)", fontsize=8)
        ax.tick_params(axis="both", labelsize=8)

        ax2 = ax.twinx()
        ax2.plot(bins, cdf_norm, color="#111111", linewidth=1.6, label="CDF")
        ax2.set_ylim(0, 1.05)
        ax2.tick_params(axis="y", labelsize=8)
        if title == "CLAHE":
            ax2.set_ylabel("Normalized CDF", fontsize=8)
    axes[0].set_ylabel("Pixel Count", fontsize=8)
    plt.tight_layout()
    return fig


def render(sample_paths: Dict[str, Path]) -> None:
    """Render Lab Activity 3 Streamlit UI."""
    st.header("Lab Activity 3: Histogram Equalization & Contrast Adjustment")
    st.caption("Course: CSST106 | Topic: Histogram Equalization & Contrast Adjustment | Week No.: 4 – 5")

    with st.expander("Objective & Procedure Overview", expanded=False):
        st.markdown(
            """
            **Objective:** Understand and implement histogram equalization to enhance image contrast, compare
            global equalization vs. linear contrast stretching vs. Contrast Limited Adaptive Histogram Equalization (CLAHE),
            and analyze intensity histograms, Cumulative Distribution Functions (CDFs), and noise amplification.

            **Procedure / Algorithms:**
            1. Load a low-contrast grayscale or color image.
            2. Apply **Global Histogram Equalization** (`cv2.equalizeHist` directly on grayscale, or on the `Y` luminance
               channel in `cv2.COLOR_BGR2YUV` for color images so hue and saturation remain natural).
            3. Apply **Linear Contrast Stretching** via percentile range mapping `[p_low, p_high] → [0, 255]` and linear gain/bias `I_out = α·I_in + β`.
            4. Apply **CLAHE** (`cv2.createCLAHE(clipLimit, tileGridSize)`) for localized tile-based contrast enhancement.
            5. Compare side-by-side image clarity alongside intensity histograms and CDF curves.
            """
        )

    col_src1, col_src2 = st.columns([1, 2])
    with col_src1:
        color_mode = st.radio(
            "Processing Mode",
            ["Grayscale", "Color (YUV Luminance)"],
            horizontal=False,
            key="lab3_mode",
        )
    with col_src2:
        uploaded = st.file_uploader(
            "Upload a custom image (optional — bundled low-contrast samples pre-loaded)",
            type=["png", "jpg", "jpeg", "bmp"],
            key="lab3_upload",
        )

    default_key = "lab3_gray" if color_mode == "Grayscale" else "lab3_color"
    img_bgr, src_label = load_bgr_image(sample_paths[default_key], uploaded_file=uploaded)
    st.info(f"**Active Input:** {src_label} | **Mode:** `{color_mode}` | **Resolution:** `{img_bgr.shape[1]} × {img_bgr.shape[0]}`")

    st.subheader("Interactive Controls")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        p_low = st.slider("Stretch Low Percentile (%)", 0.0, 20.0, 2.0, step=0.5, key="lab3_plow")
        p_high = st.slider("Stretch High Percentile (%)", 80.0, 100.0, 98.0, step=0.5, key="lab3_phigh")
    with c2:
        alpha = st.slider("Linear Gain (α)", 0.5, 2.0, 1.0, step=0.05, key="lab3_alpha")
        beta = st.slider("Brightness Bias (β)", -50, 50, 0, step=2, key="lab3_beta")
    with c3:
        clahe_clip = st.slider("CLAHE Clip Limit", 0.5, 10.0, 2.5, step=0.5, key="lab3_clip")
    with c4:
        clahe_tile = st.select_slider("CLAHE Tile Grid Size", options=[2, 4, 8, 12, 16], value=8, key="lab3_tile")

    res = run_lab3_pipeline(
        img_bgr=img_bgr,
        color_mode=color_mode,
        p_low=p_low,
        p_high=p_high,
        alpha=alpha,
        beta=beta,
        clahe_clip=clahe_clip,
        clahe_tile=clahe_tile,
    )

    st.subheader("Visual Comparison (Original vs. Global Eq vs. Linear Stretch vs. CLAHE)")
    cols = st.columns(4)
    ordered_keys = ["Original", "Global Hist Eq", "Linear Stretch", "CLAHE"]
    for col_ui, name in zip(cols, ordered_keys):
        with col_ui:
            st.image(res["displays"][name], caption=name, use_container_width=True)

    st.subheader("Pixel Intensity Histograms & Cumulative Distribution Functions (CDFs)")
    fig = make_lab3_hist_figure(res)
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Quantitative Contrast & Entropy Metrics")
    st.dataframe(res["metrics_df"], use_container_width=True, hide_index=True)

    st.subheader("Lab Observations & Required Post-Lab Questions Answered")
    st.markdown(
        """
        **Observations on Detail Visibility & Noise Amplification:**
        - **Global Histogram Equalization** expands the overall dynamic range across `[0, 255]`, making faint structures
          immediately visible, but flattens global brightness uniformly and **amplifies background sensor noise** in smooth
          regions where the CDF slope is very steep.
        - **Linear Contrast Stretching** preserves relative gray-level ordering smoothly without artificial flattening, though
          regions with non-uniform illumination remain unevenly lit.
        - **CLAHE** achieves the sharpest local detail recovery in both shadowed and brightly lit regions while capping the
          CDF slope via `clipLimit`, dramatically curbing noise amplification.

        **Answered Lab Questions:**
        1. **What is histogram equalization?** It is a non-linear point-processing intensity transformation that redistributes
           pixel brightness values so the output histogram approximates a uniform distribution across the full dynamic range `[0, 255]`.
        2. **Why do we need histogram equalization?** Captured images often suffer from poor exposure, fog, backlighting, or
           sensor compression where pixel intensities cluster in a narrow band; equalization maximizes global contrast and
           improves visibility for both human inspection and downstream vision algorithms.
        3. **How does histogram equalization work mathematically?** For an 8-bit image with `N` total pixels and normalized
           histogram `p(r_k) = n_k / N`, the discrete **Cumulative Distribution Function (CDF)** is `CDF(r_k) = Σ_{j=0}^{k} p(r_j)`.
           Each pixel intensity `r_k` is mapped to `s_k = round((L - 1) × CDF(r_k))` where `L = 256`, turning steep histogram
           peaks into wider intensity intervals.
        4. **How does CLAHE work?** Contrast Limited Adaptive Histogram Equalization partitions the image into an `M × M` grid
           of local contextual **tiles** (e.g., `8 × 8`). Before computing each tile's CDF, histogram bins exceeding `clipLimit`
           are clipped and their excess counts are redistributed uniformly across all bins (capping contrast gain and preventing
           noise blow-up), followed by bilinear interpolation across tile boundaries to eliminate blocking artifacts.
        5. **Real-world applications:**
           - **Medical Imaging (X-rays, CT, MRI):** Enhancing subtle bone fractures, lung nodules, or micro-calcifications in mammograms.
           - **Satellite & Aerial Remote Sensing:** Revealing terrain, vegetation, and coastal structures under atmospheric haze.
           - **Low-Light Autonomous Driving & Surveillance:** Improving night-time pedestrian and lane visibility in dashcams and thermal feeds.
        """
    )
