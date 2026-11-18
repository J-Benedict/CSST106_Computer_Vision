"""Lab Activity 1: Introduction to Computer Vision & Image Manipulation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple
import cv2
import numpy as np
import pandas as pd
import streamlit as st

from utils.cv_helpers import bgr_to_rgb, load_bgr_image

INTERPOLATION_MAP = {
    "INTER_LINEAR (Bilinear)": cv2.INTER_LINEAR,
    "INTER_NEAREST (Nearest Neighbor)": cv2.INTER_NEAREST,
    "INTER_CUBIC (Bicubic)": cv2.INTER_CUBIC,
    "INTER_AREA (Area Resampling)": cv2.INTER_AREA,
}

ROTATION_MODES = (
    "90° Clockwise",
    "180°",
    "270° Counterclockwise (90° CCW)",
    "Custom Angle",
)


def run_lab1_pipeline(
    img_bgr: np.ndarray,
    pixel_x: int = 120,
    pixel_y: int = 120,
    patch_size: int = 24,
    new_rgb: Tuple[int, int, int] = (255, 220, 0),
    crop_box: Tuple[int, int, int, int] = (20, 60, 400, 440),
    scale_pct: int = 50,
    interp_name: str = "INTER_LINEAR (Bilinear)",
    rot_mode: str = "90° Clockwise",
    custom_angle: float = 45.0,
) -> Dict[str, Any]:
    """Execute Part 1 and Part 2 image representation and manipulation tasks."""
    h, w, c = img_bgr.shape
    px = int(np.clip(pixel_x, 0, w - 1))
    py = int(np.clip(pixel_y, 0, h - 1))

    # Part 1: Color Spaces & Channel Splitting
    rgb_display = bgr_to_rgb(img_bgr)
    bgr_misinterpreted = img_bgr.copy()  # Raw BGR array displayed directly as RGB
    b_ch, g_ch, r_ch = cv2.split(img_bgr)
    zeros = np.zeros_like(b_ch)

    # Tinted channel visualizations (in RGB order for Streamlit)
    blue_tint_rgb = cv2.merge([zeros, zeros, b_ch])
    green_tint_rgb = cv2.merge([zeros, g_ch, zeros])
    red_tint_rgb = cv2.merge([r_ch, zeros, zeros])

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Interactive Pixel Explorer & Patch Modification
    bgr_val = [int(v) for v in img_bgr[py, px]]
    rgb_val = [bgr_val[2], bgr_val[1], bgr_val[0]]
    gray_val = int(gray[py, px])

    modified_bgr = img_bgr.copy()
    half = max(1, patch_size // 2)
    y_min, y_max = max(0, py - half), min(h, py + half + 1)
    x_min, x_max = max(0, px - half), min(w, px + half + 1)
    new_bgr = (int(new_rgb[2]), int(new_rgb[1]), int(new_rgb[0]))
    modified_bgr[y_min:y_max, x_min:x_max] = new_bgr

    # Crosshair marker on inspected preview
    inspected_preview = modified_bgr.copy()
    cv2.drawMarker(
        inspected_preview,
        (px, py),
        (255, 255, 255),
        markerType=cv2.MARKER_CROSS,
        markerSize=18,
        thickness=2,
    )

    # Part 2: Cropping, Resizing, Rotation
    x1, y1, x2, y2 = crop_box
    x1, x2 = sorted((int(np.clip(x1, 0, w - 1)), int(np.clip(x2, 1, w))))
    y1, y2 = sorted((int(np.clip(y1, 0, h - 1)), int(np.clip(y2, 1, h))))
    if x2 <= x1:
        x2 = min(w, x1 + 1)
    if y2 <= y1:
        y2 = min(h, y1 + 1)
    cropped_bgr = img_bgr[y1:y2, x1:x2].copy()

    roi_overlay_bgr = img_bgr.copy()
    cv2.rectangle(roi_overlay_bgr, (x1, y1), (x2, y2), (0, 255, 255), 3)

    interp_flag = INTERPOLATION_MAP.get(interp_name, cv2.INTER_LINEAR)
    new_w = max(1, int(round(w * scale_pct / 100.0)))
    new_h = max(1, int(round(h * scale_pct / 100.0)))
    resized_bgr = cv2.resize(img_bgr, (new_w, new_h), interpolation=interp_flag)

    if rot_mode == "90° Clockwise":
        rotated_bgr = cv2.rotate(img_bgr, cv2.ROTATE_90_CLOCKWISE)
    elif rot_mode == "180°":
        rotated_bgr = cv2.rotate(img_bgr, cv2.ROTATE_180)
    elif rot_mode == "270° Counterclockwise (90° CCW)":
        rotated_bgr = cv2.rotate(img_bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)
    else:
        center = (w / 2.0, h / 2.0)
        rot_matrix = cv2.getRotationMatrix2D(center, float(custom_angle), 1.0)
        rotated_bgr = cv2.warpAffine(
            img_bgr, rot_matrix, (w, h), flags=cv2.INTER_LINEAR, borderValue=(24, 24, 24)
        )

    return {
        "shape": (h, w, c),
        "dtype": str(img_bgr.dtype),
        "rgb_display": rgb_display,
        "bgr_misinterpreted": bgr_misinterpreted,
        "blue_gray": b_ch,
        "green_gray": g_ch,
        "red_gray": r_ch,
        "blue_tint_rgb": blue_tint_rgb,
        "green_tint_rgb": green_tint_rgb,
        "red_tint_rgb": red_tint_rgb,
        "gray": gray,
        "pixel_coords": (px, py),
        "bgr_val": bgr_val,
        "rgb_val": rgb_val,
        "gray_val": gray_val,
        "modified_rgb": bgr_to_rgb(inspected_preview),
        "roi_overlay_rgb": bgr_to_rgb(roi_overlay_bgr),
        "cropped_rgb": bgr_to_rgb(cropped_bgr),
        "cropped_shape": cropped_bgr.shape,
        "resized_rgb": bgr_to_rgb(resized_bgr),
        "resized_shape": resized_bgr.shape,
        "rotated_rgb": bgr_to_rgb(rotated_bgr),
        "rotated_shape": rotated_bgr.shape,
        "crop_box": (x1, y1, x2, y2),
    }


def render(sample_paths: Dict[str, Path]) -> None:
    """Render Lab Activity 1 Streamlit UI."""
    st.header("Lab Activity 1: Introduction to Computer Vision & Image Manipulation")
    st.caption("Course: CSST106 | Topic: Image Manipulation | Week No.: 1 – 2")

    with st.expander("Objective & Procedure Overview", expanded=False):
        st.markdown(
            """
            **Objective:** Familiarize with computer vision, understand how digital images are represented as
            2D/3D NumPy arrays (`uint8` intensity matrices), inspect OpenCV's BGR vs. RGB channel ordering,
            convert color spaces to grayscale, inspect/modify pixel values, and perform ROI cropping, resizing,
            and geometric rotation.

            **Procedure / Algorithms:**
            1. Load a color image (`cv2.imread`) and inspect `(Height, Width, Channels)` via `img.shape`.
            2. Split channels (`cv2.split`) to examine Blue, Green, and Red planes and contrast BGR vs. RGB.
            3. Convert BGR to Grayscale (`cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)`).
            4. Inspect `(x, y)` coordinates (`img[y, x]`) and overwrite a pixel patch with a custom RGB color.
            5. Execute Region of Interest (ROI) array slicing (`img[y1:y2, x1:x2]`), `cv2.resize()`, and `cv2.rotate()` / `cv2.warpAffine()`.
            """
        )

    uploaded = st.file_uploader(
        "Upload a custom color image (optional — bundled sample pre-loaded)",
        type=["png", "jpg", "jpeg", "bmp"],
        key="lab1_upload",
    )
    img_bgr, src_label = load_bgr_image(sample_paths["lab1_color"], uploaded_file=uploaded)
    h, w, c = img_bgr.shape

    st.info(f"**Active Input:** {src_label} | **Shape:** `({h}, {w}, {c})` | **Total Pixels:** `{h * w:,}` | **Total Bytes:** `{img_bgr.nbytes:,}`")

    st.subheader("Interactive Controls")
    col_c1, col_c2, col_c3 = st.columns(3)
    with col_c1:
        st.markdown("**Part 1: Pixel Explorer & Patch Edit**")
        pixel_x = st.slider("Pixel X (Column)", 0, w - 1, min(480, w - 1), key="lab1_px")
        pixel_y = st.slider("Pixel Y (Row)", 0, h - 1, min(90, h - 1), key="lab1_py")
        patch_size = st.slider("Modification Patch Size (px)", 1, 60, 25, key="lab1_patch")
        hex_color = st.color_picker("New Patch RGB Color", "#FFD700", key="lab1_color_pick")
        new_rgb = tuple(int(hex_color.lstrip("#")[i : i + 2], 16) for i in (0, 2, 4))

    with col_c2:
        st.markdown("**Part 2: ROI Cropping**")
        x1 = st.slider("ROI x1 (Left)", 0, w - 2, min(20, w - 2), key="lab1_x1")
        x2 = st.slider("ROI x2 (Right)", x1 + 1, w, min(400, w), key="lab1_x2")
        y1 = st.slider("ROI y1 (Top)", 0, h - 2, min(60, h - 2), key="lab1_y1")
        y2 = st.slider("ROI y2 (Bottom)", y1 + 1, h, min(440, h), key="lab1_y2")

    with col_c3:
        st.markdown("**Part 2: Resize & Rotation**")
        scale_pct = st.slider("Resize Scale (%)", 10, 200, 50, step=5, key="lab1_scale")
        interp_name = st.selectbox("Interpolation Method", list(INTERPOLATION_MAP.keys()), key="lab1_interp")
        rot_mode = st.selectbox("Rotation Mode", ROTATION_MODES, key="lab1_rot_mode")
        custom_angle = 45.0
        if rot_mode == "Custom Angle":
            custom_angle = st.slider("Custom Angle (degrees CCW)", -180, 180, 45, key="lab1_angle")

    res = run_lab1_pipeline(
        img_bgr=img_bgr,
        pixel_x=pixel_x,
        pixel_y=pixel_y,
        patch_size=patch_size,
        new_rgb=new_rgb,
        crop_box=(x1, y1, x2, y2),
        scale_pct=scale_pct,
        interp_name=interp_name,
        rot_mode=rot_mode,
        custom_angle=custom_angle,
    )

    # Part 1 Display
    st.subheader("Part 1: Digital Image Representation, BGR vs. RGB & Grayscale")
    p1_c1, p1_c2, p1_c3 = st.columns(3)
    with p1_c1:
        st.image(res["rgb_display"], caption=f"Proper RGB Display {res['shape']}", use_container_width=True)
    with p1_c2:
        st.image(
            res["bgr_misinterpreted"],
            caption="Raw OpenCV BGR Array (R & B Swapped if un-converted)",
            use_container_width=True,
        )
    with p1_c3:
        st.image(res["gray"], caption=f"Grayscale ({h}, {w})", use_container_width=True)

    st.markdown("**Split BGR Color Channels (`b, g, r = cv2.split(img)`)**")
    ch1, ch2, ch3 = st.columns(3)
    with ch1:
        st.image(res["blue_tint_rgb"], caption="Blue Channel (B)", use_container_width=True)
    with ch2:
        st.image(res["green_tint_rgb"], caption="Green Channel (G)", use_container_width=True)
    with ch3:
        st.image(res["red_tint_rgb"], caption="Red Channel (R)", use_container_width=True)

    st.markdown("**Interactive Pixel Inspector & Live Patch Modification**")
    px_c1, px_c2 = st.columns([1, 2])
    with px_c1:
        px_df = pd.DataFrame(
            {
                "Property": [
                    "Coordinates (x, y)",
                    "NumPy Index",
                    "OpenCV [B, G, R]",
                    "Standard [R, G, B]",
                    "Grayscale Intensity",
                    "Applied Patch RGB",
                ],
                "Value": [
                    str(res["pixel_coords"]),
                    f"img[{res['pixel_coords'][1]}, {res['pixel_coords'][0]}]",
                    str(res["bgr_val"]),
                    str(res["rgb_val"]),
                    str(res["gray_val"]),
                    str(list(new_rgb)),
                ],
            }
        )
        st.dataframe(px_df, use_container_width=True, hide_index=True)
    with px_c2:
        st.image(
            res["modified_rgb"],
            caption=f"Live Modified Patch ({patch_size}×{patch_size} px) centered at {res['pixel_coords']}",
            use_container_width=True,
        )

    # Part 2 Display
    st.subheader("Part 2: Basic Image Manipulation (ROI Crop, Resize, Rotation)")
    p2_c1, p2_c2, p2_c3, p2_c4 = st.columns(4)
    with p2_c1:
        st.image(res["roi_overlay_rgb"], caption=f"ROI Bounds {res['crop_box']}", use_container_width=True)
    with p2_c2:
        st.image(res["cropped_rgb"], caption=f"Cropped ROI {res['cropped_shape']}", use_container_width=True)
    with p2_c3:
        st.image(
            res["resized_rgb"],
            caption=f"Resized {scale_pct}% → {res['resized_shape']}",
            use_container_width=True,
        )
    with p2_c4:
        st.image(
            res["rotated_rgb"],
            caption=f"Rotated ({rot_mode}) {res['rotated_shape']}",
            use_container_width=True,
        )

    # Quantitative Summary Metrics
    st.subheader("Quantitative Image Metrics")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Original Dimensions", f"{w} × {h} × {c}")
    m2.metric("Cropped ROI Size", f"{res['cropped_shape'][1]} × {res['cropped_shape'][0]}")
    m3.metric("Resized Dimensions", f"{res['resized_shape'][1]} × {res['resized_shape'][0]}")
    m4.metric("Rotated Dimensions", f"{res['rotated_shape'][1]} × {res['rotated_shape'][0]}")

    # Report Summary & Discussion Questions
    st.subheader("Lab Discussion & Report Questions Answered")
    st.markdown(
        """
        1. **2D vs. 3D Array Representation:** A grayscale digital image is a **2D matrix** of shape `(Height, Width)`
           where each element stores a single luminance intensity in `[0, 255]` (`uint8`). A color image is a **3D tensor**
           of shape `(Height, Width, 3)` stacking three 2D intensity planes. Note that NumPy indexes rows first (`y`, height)
           and columns second (`x`, width): `img[y, x]`.
        2. **Color Spaces & OpenCV BGR Ordering:** OpenCV stores color channels in **BGR** (Blue, Green, Red) order for
           historical compatibility with early camera hardware, whereas Matplotlib, Pillow, and web displays expect **RGB**.
           Displaying an un-converted BGR image directly swaps red and blue hues (turning blue skies orange and red swatches blue),
           making `cv2.cvtColor(img, cv2.COLOR_BGR2RGB)` mandatory for accurate visualization. Grayscale (`cv2.COLOR_BGR2GRAY`)
           computes perceptual luminance `Y = 0.299R + 0.587G + 0.114B`, reducing memory footprint by **3×**.
        3. **Real-Life Computer Vision Applications:** Medical imaging (DICOM X-ray/MRI slice inspection), autonomous vehicles
           (lane ROI cropping and camera calibration), industrial quality assurance (pixel-level defect inspection), and
           document scanning / biometric verification (perspective rotation and image normalization).
        """
    )
