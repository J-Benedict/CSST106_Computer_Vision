"""Lab Activity 7 (Manual Lab 6): Image Segmentation Techniques."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, Tuple
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
import torch
from torchvision.models.segmentation import (
    DeepLabV3_MobileNet_V3_Large_Weights,
    deeplabv3_mobilenet_v3_large,
)

from utils.cv_helpers import bgr_to_rgb, load_bgr_image


def _pascal_voc_palette(num_classes: int = 21) -> np.ndarray:
    """Create deterministic PASCAL VOC color palette (N, 3) uint8 in RGB."""
    palette = np.zeros((num_classes, 3), dtype=np.uint8)
    for j in range(num_classes):
        lab = j
        r = g = b = 0
        i = 0
        while lab:
            r |= ((lab >> 0) & 1) << (7 - i)
            g |= ((lab >> 1) & 1) << (7 - i)
            b |= ((lab >> 2) & 1) << (7 - i)
            i += 1
            lab >>= 3
        palette[j] = [r, g, b]
    return palette


def _compute_seg_metrics(pred_mask: np.ndarray, gt_mask: np.ndarray) -> Tuple[float, float, float]:
    """Compute IoU (Jaccard Index), Dice (F1), and Pixel Accuracy for binary masks."""
    pred = pred_mask > 0
    gt = gt_mask > 0
    intersection = int(np.logical_and(pred, gt).sum())
    union = int(np.logical_or(pred, gt).sum())
    iou = intersection / union if union > 0 else 0.0
    denom_dice = int(pred.sum() + gt.sum())
    dice = (2.0 * intersection) / denom_dice if denom_dice > 0 else 0.0
    acc = float(np.mean(pred == gt))
    return iou, dice, acc


def region_growing_from_seed(gray: np.ndarray, seed_x: int, seed_y: int, tau: int = 28) -> np.ndarray:
    """Fast 8-connected region growing from seed (x, y) with intensity similarity threshold tau."""
    h, w = gray.shape
    sx = int(np.clip(seed_x, 0, w - 1))
    sy = int(np.clip(seed_y, 0, h - 1))
    seed_val = int(gray[sy, sx])
    within_tol = (np.abs(gray.astype(np.int16) - seed_val) <= int(tau)).astype(np.uint8)
    _, labels = cv2.connectedComponents(within_tol, connectivity=8)
    target_lbl = int(labels[sy, sx])
    if target_lbl == 0:
        return np.zeros((h, w), dtype=np.uint8)
    return ((labels == target_lbl) * 255).astype(np.uint8)


def run_lab7_classical(
    img_bgr: np.ndarray,
    gt_mask: np.ndarray | None = None,
    adaptive_block: int = 105,
    adaptive_c: int = -12,
    seed_point: Tuple[int, int] = (105, 110),
    reg_tau: int = 35,
    ws_dist_ratio: float = 0.42,
    canny_low: int = 60,
    canny_high: int = 150,
) -> Dict[str, Any]:
    """Run Thresholding (Otsu & Adaptive), Region Growing, Watershed, and Edge-Based Segmentation."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # 1. Thresholding: Global Otsu + Adaptive Mean & Gaussian
    t0 = time.perf_counter()
    otsu_val, otsu_mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    otsu_ms = (time.perf_counter() - t0) * 1000.0

    blk = adaptive_block if adaptive_block % 2 == 1 else adaptive_block + 1
    blk = max(3, blk)
    t1 = time.perf_counter()
    adapt_mean = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, blk, int(adaptive_c)
    )
    adapt_gauss = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, blk, int(adaptive_c)
    )
    adapt_ms = (time.perf_counter() - t1) * 1000.0

    # 2A. Region Growing from interactive seed
    t2 = time.perf_counter()
    rg_mask = region_growing_from_seed(blur, seed_point[0], seed_point[1], tau=reg_tau)
    rg_ms = (time.perf_counter() - t2) * 1000.0

    rg_overlay_bgr = img_bgr.copy()
    rg_overlay_bgr[rg_mask > 0] = (
        0.55 * rg_overlay_bgr[rg_mask > 0] + 0.45 * np.array([40, 210, 90])
    ).astype(np.uint8)
    cv2.drawMarker(
        rg_overlay_bgr,
        (int(seed_point[0]), int(seed_point[1])),
        (0, 0, 255),
        cv2.MARKER_CROSS,
        16,
        2,
    )

    # 2B. Watershed Algorithm for separating touching objects
    t3 = time.perf_counter()
    kernel = np.ones((3, 3), dtype=np.uint8)
    opening = cv2.morphologyEx(otsu_mask, cv2.MORPH_OPEN, kernel, iterations=2)
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(
        dist_transform, float(ws_dist_ratio) * float(dist_transform.max() + 1e-6), 255, 0
    )
    sure_fg_u8 = sure_fg.astype(np.uint8)
    unknown = cv2.subtract(sure_bg, sure_fg_u8)

    num_markers, markers = cv2.connectedComponents(sure_fg_u8)
    markers = markers + 1
    markers[unknown == 255] = 0
    ws_markers = cv2.watershed(img_bgr.copy(), markers)
    ws_ms = (time.perf_counter() - t3) * 1000.0

    # Colorize distinct watershed instances & draw red boundaries (-1)
    ws_vis_bgr = img_bgr.copy()
    rng_col = np.random.default_rng(7)
    unique_lbls = [int(lbl) for lbl in np.unique(ws_markers) if lbl > 1]
    for lbl in unique_lbls:
        col = rng_col.integers(60, 245, size=3).astype(np.uint8)
        mask_lbl = ws_markers == lbl
        ws_vis_bgr[mask_lbl] = (0.5 * ws_vis_bgr[mask_lbl] + 0.5 * col).astype(np.uint8)
    ws_vis_bgr[ws_markers == -1] = [0, 0, 255]
    ws_fg_mask = ((ws_markers > 1) * 255).astype(np.uint8)

    # 3. Edge-Based Segmentation (Canny + Morph Close + Contour Fill)
    t4 = time.perf_counter()
    canny_edges = cv2.Canny(blur, int(canny_low), int(canny_high))
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    closed_edges = cv2.morphologyEx(canny_edges, cv2.MORPH_CLOSE, close_kernel)
    contours, _ = cv2.findContours(closed_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    edge_seg_mask = np.zeros_like(gray)
    valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 80.0]
    cv2.drawContours(edge_seg_mask, valid_contours, -1, 255, thickness=-1)
    edge_ms = (time.perf_counter() - t4) * 1000.0

    edge_overlay_bgr = img_bgr.copy()
    cv2.drawContours(edge_overlay_bgr, valid_contours, -1, (255, 180, 0), 2)

    ref_gt = gt_mask if gt_mask is not None else otsu_mask
    gt_source = "Ground-Truth Mask" if gt_mask is not None else "Otsu Reference Mask"

    methods = [
        ("Global Otsu Thresholding", otsu_mask, otsu_ms),
        ("Adaptive Gaussian Thresholding", adapt_gauss, adapt_ms),
        ("Adaptive Mean Thresholding", adapt_mean, adapt_ms),
        ("Region Growing (Single Seed)", rg_mask, rg_ms),
        ("Watershed Algorithm (Touching Objects)", ws_fg_mask, ws_ms),
        ("Edge-Based (Canny + Closed Contours)", edge_seg_mask, edge_ms),
    ]
    metrics_rows = []
    for name, m_mask, ms in methods:
        iou, dice, acc = _compute_seg_metrics(m_mask, ref_gt)
        metrics_rows.append(
            {
                "Method": name,
                f"IoU vs {gt_source}": round(iou, 4),
                "Dice (F1)": round(dice, 4),
                "Pixel Accuracy": round(acc, 4),
                "Foreground (%)": round(100.0 * float(np.mean(m_mask > 0)), 2),
                "Time (ms)": round(ms, 2),
            }
        )

    return {
        "gray": gray,
        "otsu_val": float(otsu_val),
        "otsu_mask": otsu_mask,
        "adapt_mean": adapt_mean,
        "adapt_gauss": adapt_gauss,
        "rg_mask": rg_mask,
        "rg_overlay_rgb": bgr_to_rgb(rg_overlay_bgr),
        "dist_transform_u8": cv2.normalize(dist_transform, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8),
        "ws_vis_rgb": bgr_to_rgb(ws_vis_bgr),
        "ws_fg_mask": ws_fg_mask,
        "ws_object_count": len(unique_lbls),
        "canny_edges": canny_edges,
        "edge_seg_mask": edge_seg_mask,
        "edge_overlay_rgb": bgr_to_rgb(edge_overlay_bgr),
        "edge_contour_count": len(valid_contours),
        "ref_gt": ref_gt,
        "metrics_df": pd.DataFrame(metrics_rows),
    }


@st.cache_resource(show_spinner="Loading Pre-trained DeepLabV3-MobileNetV3-Large...")
def load_deeplab_model():
    """Load and cache DeepLabV3-MobileNetV3-Large weights and metadata."""
    weights = DeepLabV3_MobileNet_V3_Large_Weights.DEFAULT
    model = deeplabv3_mobilenet_v3_large(weights=weights).eval()
    return model, weights


def run_lab7_semantic(scene_bgr: np.ndarray, overlay_alpha: float = 0.55) -> Dict[str, Any]:
    """Run DeepLabV3 semantic segmentation on a natural BGR image."""
    model, weights = load_deeplab_model()
    categories = weights.meta["categories"]
    palette = _pascal_voc_palette(len(categories))

    rgb_img = bgr_to_rgb(scene_bgr)
    pil_img = Image.fromarray(rgb_img)
    preprocess = weights.transforms()

    t0 = time.perf_counter()
    input_tensor = preprocess(pil_img).unsqueeze(0)
    with torch.no_grad():
        output_logits = model(input_tensor)["out"][0]
        pred_small = output_logits.argmax(0).byte().cpu().numpy()
    infer_ms = (time.perf_counter() - t0) * 1000.0

    # Resize label map cleanly back to original image resolution with NEAREST interpolation
    h, w = rgb_img.shape[:2]
    pred_labels = cv2.resize(pred_small, (w, h), interpolation=cv2.INTER_NEAREST)
    color_mask_rgb = palette[pred_labels]
    overlay_rgb = np.clip(
        (1.0 - float(overlay_alpha)) * rgb_img.astype(np.float32)
        + float(overlay_alpha) * color_mask_rgb.astype(np.float32),
        0,
        255,
    ).astype(np.uint8)

    unique_ids, counts = np.unique(pred_labels, return_counts=True)
    total_px = pred_labels.size
    legend_rows = []
    for cls_id, cnt in sorted(zip(unique_ids, counts), key=lambda x: x[1], reverse=True):
        c_rgb = [int(v) for v in palette[int(cls_id)]]
        legend_rows.append(
            {
                "Class ID": int(cls_id),
                "Class Label": categories[int(cls_id)],
                "RGB Color": f"rgb({c_rgb[0]}, {c_rgb[1]}, {c_rgb[2]})",
                "Pixels": int(cnt),
                "Coverage (%)": round(100.0 * int(cnt) / total_px, 2),
            }
        )

    return {
        "orig_rgb": rgb_img,
        "mask_rgb": color_mask_rgb,
        "overlay_rgb": overlay_rgb,
        "legend_df": pd.DataFrame(legend_rows),
        "infer_ms": infer_ms,
        "num_classes_detected": len(unique_ids),
    }


def render(sample_paths: Dict[str, Path]) -> None:
    """Render Lab Activity 7 Streamlit UI."""
    st.header("Lab Activity 7 (Manual Lab 6): Image Segmentation Techniques")
    st.caption("Course: CSST106 | Topic: Segmentation Technique | Week No.: 13")

    with st.expander("Objective & Procedure Overview", expanded=False):
        st.markdown(
            """
            **Objective:** Understand and apply four fundamental image segmentation paradigms — **Thresholding**
            (Global Otsu & Adaptive), **Region-Based** (Region Growing & Watershed), **Edge-Based** (Canny + Closed Contours),
            and **Deep Semantic Segmentation** (`DeepLabV3-MobileNetV3-Large`) — and compare them quantitatively (IoU) and qualitatively.

            **Procedure Summary:**
            1. **Thresholding:** Apply `cv2.THRESH_OTSU` and `cv2.adaptiveThreshold` on grayscale images.
            2. **Region-Based:** Grow homogeneous regions from seed `(x, y)` with tolerance `τ`, and apply `cv2.watershed`
               with Distance Transform markers to separate touching objects.
            3. **Edge-Based:** Extract edges (`cv2.Canny`), close gaps morphologically, and fill closed contours (`cv2.findContours`).
            4. **Semantic:** Infer pixel-wise class masks using pre-trained `DeepLabV3-MobileNetV3-Large` on PASCAL VOC classes.
            """
        )

    tab_classic, tab_semantic = st.tabs(
        [
            "Methods 1–3: Thresholding, Region-Based & Edge-Based (IoU Benchmark)",
            "Method 4: Deep Semantic Segmentation (DeepLabV3)",
        ]
    )

    # ---------------- CLASSICAL SEGMENTATION ----------------
    with tab_classic:
        up_c = st.file_uploader(
            "Upload a custom segmentation image (optional — touching coins sample + GT pre-loaded)",
            type=["png", "jpg", "jpeg", "bmp"],
            key="lab7_up_classic",
        )
        img_bgr, src_label = load_bgr_image(sample_paths["lab7_coins"], uploaded_file=up_c)
        gt_mask = None
        if up_c is None and sample_paths["lab7_gt"].exists():
            gt_mask = cv2.imread(str(sample_paths["lab7_gt"]), cv2.IMREAD_GRAYSCALE)

        h, w = img_bgr.shape[:2]
        st.info(f"**Active Input:** {src_label} | **Resolution:** `{w} × {h}`")

        st.subheader("Interactive Controls")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            adapt_blk = st.slider("Adaptive Block Size (odd)", 15, 165, 105, step=6, key="lab7_blk")
            adapt_c = st.slider("Adaptive Constant C", -30, 20, -12, step=2, key="lab7_c")
        with c2:
            seed_x = st.slider("Region Seed X", 0, w - 1, min(105, w - 1), key="lab7_sx")
            seed_y = st.slider("Region Seed Y", 0, h - 1, min(110, h - 1), key="lab7_sy")
        with c3:
            reg_tau = st.slider("Region Similarity τ", 5, 80, 35, step=5, key="lab7_tau")
            ws_ratio = st.slider("Watershed Peak Ratio", 0.15, 0.80, 0.42, step=0.05, key="lab7_ws")
        with c4:
            canny_low = st.slider("Canny Low Threshold", 10, 200, 60, step=10, key="lab7_clow")
            canny_high = st.slider("Canny High Threshold", 50, 300, 150, step=10, key="lab7_chigh")

        res = run_lab7_classical(
            img_bgr=img_bgr,
            gt_mask=gt_mask,
            adaptive_block=adapt_blk,
            adaptive_c=adapt_c,
            seed_point=(seed_x, seed_y),
            reg_tau=reg_tau,
            ws_dist_ratio=ws_ratio,
            canny_low=canny_low,
            canny_high=canny_high,
        )

        st.markdown("#### 1. Thresholding Segmentation (Global Otsu vs. Adaptive)")
        t_c1, t_c2, t_c3, t_c4 = st.columns(4)
        with t_c1:
            st.image(res["gray"], caption="Grayscale Input", use_container_width=True)
        with t_c2:
            st.image(res["otsu_mask"], caption=f"Global Otsu (T = {res['otsu_val']:.1f})", use_container_width=True)
        with t_c3:
            st.image(res["adapt_mean"], caption="Adaptive Mean Thresholding", use_container_width=True)
        with t_c4:
            st.image(res["adapt_gauss"], caption="Adaptive Gaussian Thresholding", use_container_width=True)

        st.markdown("#### 2. Region-Based Segmentation (Region Growing & Watershed)")
        r_c1, r_c2, r_c3, r_c4 = st.columns(4)
        with r_c1:
            st.image(
                res["rg_overlay_rgb"],
                caption=f"Region Growing Overlay (Seed=({seed_x},{seed_y}), τ={reg_tau})",
                use_container_width=True,
            )
        with r_c2:
            st.image(res["rg_mask"], caption="Region Growing Binary Mask", use_container_width=True)
        with r_c3:
            st.image(res["dist_transform_u8"], caption="Distance Transform Peaks", use_container_width=True)
        with r_c4:
            st.image(
                res["ws_vis_rgb"],
                caption=f"Watershed Result ({res['ws_object_count']} separated regions)",
                use_container_width=True,
            )

        st.markdown("#### 3. Edge-Based Segmentation vs. Watershed Boundaries")
        e_c1, e_c2, e_c3, e_c4 = st.columns(4)
        with e_c1:
            st.image(res["canny_edges"], caption="Canny Edge Map", use_container_width=True)
        with e_c2:
            st.image(
                res["edge_seg_mask"],
                caption=f"Closed & Filled Contours ({res['edge_contour_count']} objects)",
                use_container_width=True,
            )
        with e_c3:
            st.image(res["edge_overlay_rgb"], caption="Edge-Based Contour Overlay", use_container_width=True)
        with e_c4:
            st.image(res["ref_gt"], caption="Reference / Ground-Truth Mask", use_container_width=True)

        st.subheader("Quantitative Benchmark (IoU / Jaccard Index & Execution Time)")
        st.dataframe(res["metrics_df"], use_container_width=True, hide_index=True)

    # ---------------- SEMANTIC SEGMENTATION ----------------
    with tab_semantic:
        up_sem = st.file_uploader(
            "Upload a natural scene for DeepLabV3 semantic segmentation (optional — bundled scene pre-loaded)",
            type=["png", "jpg", "jpeg", "bmp"],
            key="lab7_up_sem",
        )
        sem_bgr, sem_src = load_bgr_image(sample_paths["lab7_semantic"], uploaded_file=up_sem)
        overlay_alpha = st.slider("Mask Overlay Opacity (α)", 0.10, 0.90, 0.55, step=0.05, key="lab7_alpha")

        sem_res = run_lab7_semantic(sem_bgr, overlay_alpha=overlay_alpha)
        st.info(
            f"**Active Input:** {sem_src} | **Model:** `DeepLabV3-MobileNetV3-Large (PASCAL VOC)` | "
            f"**Inference Time:** `{sem_res['infer_ms']:.1f} ms` | **Classes Detected:** `{sem_res['num_classes_detected']}`"
        )

        s1, s2, s3 = st.columns(3)
        with s1:
            st.image(sem_res["orig_rgb"], caption="Original Natural Scene", use_container_width=True)
        with s2:
            st.image(sem_res["mask_rgb"], caption="Pixel-Level Semantic Class Mask", use_container_width=True)
        with s3:
            st.image(sem_res["overlay_rgb"], caption=f"Semantic Overlay (α={overlay_alpha})", use_container_width=True)

        st.markdown("**Detected Semantic Class Color Legend & Coverage**")
        st.dataframe(sem_res["legend_df"], use_container_width=True, hide_index=True)

    st.subheader("Required Post-Lab Question & Technique Comparison")
    st.markdown(
        """
        **Question:** *Explain the differences between thresholding, region-based, edge-based, and semantic segmentation techniques. Provide one advantage and one limitation for each method.*

        | Technique | Core Mechanism | Key Advantage | Key Limitation | Ideal Use Case |
        | :--- | :--- | :--- | :--- | :--- |
        | **Thresholding (Otsu / Adaptive)** | Classifies pixels solely by comparing grayscale intensity against a global or local threshold `T`. | Extremely fast (`O(N)`), trivial to implement, requires zero training data. | Ignores spatial continuity; global thresholding fails under uneven illumination or overlapping foreground/background histograms. | High-contrast document OCR, scanned forms, backlit silhouette inspection. |
        | **Region-Based (Region Growing / Watershed)** | Groups spatially connected pixels sharing intensity homogeneity or partitions topographic distance basins between markers. | Naturally produces closed, connected regions; **Watershed** cleanly separates touching objects (e.g., clustered coins/cells). | Sensitive to seed placement and threshold `τ`, prone to **over-segmentation** on textured surfaces without careful marker filtering. | Biomedical cell/nuclei counting, organ/tumor volume delineation, touching parts separation. |
        | **Edge-Based (Canny + Contours)** | Detects abrupt spatial intensity discontinuities and links/closes boundary contours around enclosed regions. | Captures sharp geometric shapes and local intensity transitions even when background brightness varies smoothly. | High sensitivity to noise and broken edges; unclosed edge gaps leak during contour filling and fail to separate touching objects cleanly. | Precision mechanical part gauging, crack detection, lane boundary extraction. |
        | **Semantic Segmentation (DeepLabV3 / U-Net)** | End-to-end Deep CNN encoder-decoder with atrous spatial pyramid pooling classifying every pixel into semantic categories. | Understands high-level object context across clutter, shadows, textures, and complex natural scenes with multi-class labels. | High compute/memory footprint, slower CPU latency, requires large pixel-annotated training datasets (e.g., PASCAL VOC/COCO), and merges same-class touching instances. | Autonomous driving drivable-area/pedestrian parsing, robotics perception, medical multi-organ segmentation. |
        """
    )
