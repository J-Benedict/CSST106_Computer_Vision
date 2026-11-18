"""Hard-copy laboratory report generator (.md, .docx, and .html) for CSST106."""

from __future__ import annotations

import html
import os
from pathlib import Path
from typing import Any, Dict, List
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH

PROJECT_ROOT = Path(__file__).resolve().parent.parent
_YOLO_CFG_DIR = PROJECT_ROOT / "assets" / ".ultralytics_cfg"
_YOLO_CFG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("YOLO_CONFIG_DIR", str(_YOLO_CFG_DIR))

from utils.cv_helpers import bgr_to_rgb  # noqa: E402
from utils.sample_generator import ensure_all_samples  # noqa: E402
from labs import (  # noqa: E402
    lab1_images,
    lab2_edges,
    lab3_histogram,
    lab4_surf,
    lab5_pca,
    lab6_yolo,
    lab7_segmentation,
)

FIG_DIR = PROJECT_ROOT / "assets" / "report_figures"


def _save_grid_figure(
    panels: List[tuple[np.ndarray, str, str | None]],
    out_path: Path,
    nrows: int,
    ncols: int,
    figsize: tuple[float, float] = (13.0, 7.5),
    suptitle: str = "",
) -> Path:
    """Save a multi-panel image comparison figure cleanly to PNG."""
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes_flat = np.array(axes).ravel()
    for i, ax in enumerate(axes_flat):
        if i < len(panels):
            img, title, cmap = panels[i]
            if cmap:
                ax.imshow(img, cmap=cmap)
            else:
                ax.imshow(img)
            ax.set_title(title, fontsize=10, fontweight="bold")
        ax.axis("off")
    if suptitle:
        plt.suptitle(suptitle, fontsize=12, fontweight="bold")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _df_to_markdown(df: pd.DataFrame) -> str:
    """Convert a DataFrame to a standard GitHub/MS-Word-friendly Markdown table."""
    cols = [str(c) for c in df.columns]
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join([":---"] * len(cols)) + " |"
    rows = []
    for _, row in df.iterrows():
        vals = [str(row[c]).replace("|", "/") for c in df.columns]
        rows.append("| " + " | ".join(vals) + " |")
    return "\n".join([header, sep, *rows])


def _collect_all_lab_data() -> Dict[str, Any]:
    """Execute all 7 labs on bundled default samples and persist summary figures."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    samples = ensure_all_samples()

    # ---------------- LAB 1 ----------------
    img1 = cv2.imread(str(samples["lab1_color"]), cv2.IMREAD_COLOR)
    r1 = lab1_images.run_lab1_pipeline(
        img1,
        pixel_x=480,
        pixel_y=90,
        patch_size=25,
        new_rgb=(255, 215, 0),
        crop_box=(20, 60, 400, 440),
        scale_pct=50,
        interp_name="INTER_LINEAR (Bilinear)",
        rot_mode="90° Clockwise",
    )
    fig1_path = _save_grid_figure(
        [
            (r1["rgb_display"], f"Original RGB {r1['shape']}", None),
            (r1["bgr_misinterpreted"], "Raw BGR (R/B Swapped)", None),
            (r1["gray"], "Grayscale (2D)", "gray"),
            (r1["modified_rgb"], "Modified Patch at (480, 90)", None),
            (r1["cropped_rgb"], f"Cropped ROI {r1['cropped_shape']}", None),
            (r1["rotated_rgb"], f"Rotated 90° CW {r1['rotated_shape']}", None),
        ],
        FIG_DIR / "lab1_summary.png",
        nrows=2,
        ncols=3,
        figsize=(12.5, 8.0),
        suptitle="Lab 1: Digital Image Representation & Manipulation Outputs",
    )

    # ---------------- LAB 2 ----------------
    img2 = cv2.imread(str(samples["lab2_edges"]), cv2.IMREAD_COLOR)
    r2 = lab2_edges.run_lab2_pipeline(img2, blur_ksize=3, sobel_ksize=3, canny_low=100, canny_high=200)
    fig2 = lab2_edges.make_lab2_figure(r2)
    fig2_path = FIG_DIR / "lab2_summary.png"
    fig2.savefig(fig2_path, dpi=150, bbox_inches="tight")
    plt.close(fig2)

    # ---------------- LAB 3 ----------------
    img3_g = cv2.imread(str(samples["lab3_gray"]), cv2.IMREAD_COLOR)
    img3_c = cv2.imread(str(samples["lab3_color"]), cv2.IMREAD_COLOR)
    r3_g = lab3_histogram.run_lab3_pipeline(
        img3_g, color_mode="Grayscale", p_low=2.0, p_high=98.0, clahe_clip=2.5, clahe_tile=8
    )
    r3_c = lab3_histogram.run_lab3_pipeline(
        img3_c, color_mode="Color (YUV Luminance)", p_low=2.0, p_high=98.0, clahe_clip=2.5, clahe_tile=8
    )
    fig3_img_path = _save_grid_figure(
        [
            (r3_g["displays"]["Original"], "Original Grayscale", "gray"),
            (r3_g["displays"]["Global Hist Eq"], "Global Hist Eq", "gray"),
            (r3_g["displays"]["Linear Stretch"], "Linear Stretch", "gray"),
            (r3_g["displays"]["CLAHE"], "CLAHE (clip=2.5, 8×8)", "gray"),
        ],
        FIG_DIR / "lab3_gray_summary.png",
        nrows=1,
        ncols=4,
        figsize=(14.0, 3.8),
        suptitle="Lab 3: Contrast Enhancement Visual Comparison (Grayscale)",
    )
    fig3_hist = lab3_histogram.make_lab3_hist_figure(r3_g)
    fig3_hist_path = FIG_DIR / "lab3_hist_cdf.png"
    fig3_hist.savefig(fig3_hist_path, dpi=150, bbox_inches="tight")
    plt.close(fig3_hist)

    # ---------------- LAB 4 ----------------
    img4_ref = cv2.imread(str(samples["lab4_ref"]), cv2.IMREAD_COLOR)
    img4_scene = cv2.imread(str(samples["lab4_scene"]), cv2.IMREAD_COLOR)
    r4 = lab4_surf.run_lab4_pipeline(img4_ref, img4_scene, hessian_threshold=400, upright=False, ratio_thresh=0.75)
    fig4_path = _save_grid_figure(
        [
            (r4["surf_kp_img1_rgb"], f"SURF Keypoints Ref ({r4['surf_kp1_count']})", None),
            (r4["surf_kp_img2_rgb"], f"SURF Keypoints Scene ({r4['surf_kp2_count']})", None),
            (r4["surf_matches_rgb"], f"SURF Matches ({r4['surf_matches_count']})", None),
            (r4["orb_matches_rgb"], f"ORB Matches ({r4['orb_matches_count']})", None),
        ],
        FIG_DIR / "lab4_summary.png",
        nrows=2,
        ncols=2,
        figsize=(13.0, 8.2),
        suptitle="Lab 4: SURF Keypoints and Feature Matching Comparison",
    )

    # ---------------- LAB 5 ----------------
    r5 = lab5_pca.run_lab5_pipeline(str(samples["lab5_lfw"]), n_components=150, whiten=True, test_size=0.2)
    fig5_gal = lab5_pca.make_eigenfaces_gallery_figure(
        r5["eigenfaces"], r5["dataset"]["h"], r5["dataset"]["w"]
    )
    fig5_gal_path = FIG_DIR / "lab5_eigenfaces.png"
    fig5_gal.savefig(fig5_gal_path, dpi=150, bbox_inches="tight")
    plt.close(fig5_gal)

    fig5_fl = lab5_pca.make_first_vs_last_figure(r5["eigenfaces"])
    fig5_fl_path = FIG_DIR / "lab5_first_vs_last.png"
    fig5_fl.savefig(fig5_fl_path, dpi=150, bbox_inches="tight")
    plt.close(fig5_fl)

    fig5_var = lab5_pca.make_variance_figure(r5)
    fig5_var_path = FIG_DIR / "lab5_variance.png"
    fig5_var.savefig(fig5_var_path, dpi=150, bbox_inches="tight")
    plt.close(fig5_var)

    fig5_cm = lab5_pca.make_confusion_matrix_figure(r5["confusion_matrix"], r5["dataset"]["target_names"])
    fig5_cm_path = FIG_DIR / "lab5_cm.png"
    fig5_cm.savefig(fig5_cm_path, dpi=150, bbox_inches="tight")
    plt.close(fig5_cm)

    fig5_fail = lab5_pca.make_failures_figure(r5)
    fig5_fail_path = FIG_DIR / "lab5_failures.png"
    fig5_fail.savefig(fig5_fail_path, dpi=150, bbox_inches="tight")
    plt.close(fig5_fail)

    # ---------------- LAB 6 ----------------
    img6 = cv2.imread(str(samples["lab6_scene"]), cv2.IMREAD_COLOR)
    r6_inf = lab6_yolo.run_lab6_inference(img6, conf=0.25, iou=0.45, weights_path=str(PROJECT_ROOT / "yolo11n.pt"))
    dataset_dir = samples["lab6_yaml"].parent
    train_img0 = sorted((dataset_dir / "images" / "train").glob("*.jpg"))[0]
    train_lbl0 = dataset_dir / "labels" / "train" / f"{train_img0.stem}.txt"
    gt_vis_rgb, _ = lab6_yolo.draw_yolo_ground_truth(
        train_img0, train_lbl0, {0: "person", 1: "cat", 2: "cup"}
    )
    r6_tv = lab6_yolo.run_lab6_train_and_val(dataset_dir, epochs=3, imgsz=320)
    fig6_path = _save_grid_figure(
        [
            (r6_inf["annotated_rgb"], f"Pre-trained YOLOv11n ({r6_inf['num_detections']} detections)", None),
            (gt_vis_rgb, f"Custom Dataset GT Box Verification ({train_img0.name})", None),
        ],
        FIG_DIR / "lab6_summary.png",
        nrows=1,
        ncols=2,
        figsize=(13.0, 5.0),
        suptitle="Lab 6: YOLOv11n Object Detection & Custom Dataset Annotations",
    )

    # ---------------- LAB 7 ----------------
    img7_c = cv2.imread(str(samples["lab7_coins"]), cv2.IMREAD_COLOR)
    gt7 = cv2.imread(str(samples["lab7_gt"]), cv2.IMREAD_GRAYSCALE)
    r7_c = lab7_segmentation.run_lab7_classical(img7_c, gt_mask=gt7)
    img7_s = cv2.imread(str(samples["lab7_semantic"]), cv2.IMREAD_COLOR)
    r7_s = lab7_segmentation.run_lab7_semantic(img7_s)

    fig7_c_path = _save_grid_figure(
        [
            (r7_c["gray"], "Grayscale Input", "gray"),
            (r7_c["otsu_mask"], f"Otsu (T={r7_c['otsu_val']:.0f})", "gray"),
            (r7_c["adapt_gauss"], "Adaptive Gaussian", "gray"),
            (r7_c["rg_overlay_rgb"], "Region Growing (Seed)", None),
            (r7_c["ws_vis_rgb"], f"Watershed ({r7_c['ws_object_count']} regions)", None),
            (r7_c["edge_overlay_rgb"], "Edge-Based Contours", None),
        ],
        FIG_DIR / "lab7_classical.png",
        nrows=2,
        ncols=3,
        figsize=(13.0, 8.2),
        suptitle="Lab 7: Classical Image Segmentation Methods",
    )
    fig7_s_path = _save_grid_figure(
        [
            (r7_s["orig_rgb"], "Original Natural Scene", None),
            (r7_s["mask_rgb"], "DeepLabV3 Semantic Mask", None),
            (r7_s["overlay_rgb"], "Blended Semantic Overlay", None),
        ],
        FIG_DIR / "lab7_semantic.png",
        nrows=1,
        ncols=3,
        figsize=(13.5, 4.4),
        suptitle="Lab 7: DeepLabV3-MobileNetV3-Large Semantic Segmentation",
    )

    return {
        "r1": r1,
        "fig1": fig1_path,
        "r2": r2,
        "fig2": fig2_path,
        "r3_g": r3_g,
        "r3_c": r3_c,
        "fig3_img": fig3_img_path,
        "fig3_hist": fig3_hist_path,
        "r4": r4,
        "fig4": fig4_path,
        "r5": r5,
        "fig5_gal": fig5_gal_path,
        "fig5_fl": fig5_fl_path,
        "fig5_var": fig5_var_path,
        "fig5_cm": fig5_cm_path,
        "fig5_fail": fig5_fail_path,
        "r6_inf": r6_inf,
        "r6_tv": r6_tv,
        "fig6": fig6_path,
        "r7_c": r7_c,
        "r7_s": r7_s,
        "fig7_c": fig7_c_path,
        "fig7_s": fig7_s_path,
    }


def _build_lab_sections(d: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Construct structured report sections with exact quantitative metrics from runs."""
    r1 = d["r1"]
    r2 = d["r2"]
    r3_g = d["r3_g"]
    r3_c = d["r3_c"]
    r4 = d["r4"]
    r5 = d["r5"]
    ds5 = r5["dataset"]
    r6_inf = d["r6_inf"]
    r6_tv = d["r6_tv"]
    r7_c = d["r7_c"]
    r7_s = d["r7_s"]

    lab1_metrics_df = pd.DataFrame(
        [
            {"Operation / Property": "Original Image Shape", "Result": f"{r1['shape']} (H×W×C)"},
            {"Operation / Property": "Inspected Pixel (x=480, y=90)", "Result": f"BGR={r1['bgr_val']} | RGB={r1['rgb_val']} | Gray={r1['gray_val']}"},
            {"Operation / Property": "Modified Patch Size & Color", "Result": "25×25 px overwritten to RGB [255, 215, 0]"},
            {"Operation / Property": "Cropped ROI Bounds (x1, y1, x2, y2)", "Result": f"{r1['crop_box']} → Shape {r1['cropped_shape']}"},
            {"Operation / Property": "Resized (50%, INTER_LINEAR)", "Result": f"Shape {r1['resized_shape']}"},
            {"Operation / Property": "Rotated (90° Clockwise)", "Result": f"Shape {r1['rotated_shape']}"},
        ]
    )

    lab5_summary_df = pd.DataFrame(
        [
            {"Metric": "Dataset Total Samples", "Value": f"{ds5['n_samples']}"},
            {"Metric": "Image Dimensions & Features", "Value": f"{ds5['h']}×{ds5['w']} ({ds5['n_features']} features)"},
            {"Metric": "Number of Classes", "Value": f"{ds5['n_classes']}"},
            {"Metric": "Train / Test Split (80% / 20%)", "Value": f"{r5['X_train_shape'][0]} train / {r5['X_test_shape'][0]} test"},
            {"Metric": "PCA Components & Whitening", "Value": f"n_components={r5['eff_components']}, whiten={r5['whiten']}"},
            {"Metric": "Components for ≥ 90% Variance (k90)", "Value": f"{r5['k_90']} components"},
            {"Metric": "Variance Retained at 150 Components", "Value": f"{r5['total_var_retained'] * 100:.2f}%"},
            {"Metric": "RBF SVM Test Accuracy", "Value": f"{r5['accuracy'] * 100:.2f}%"},
            {"Metric": "Misclassified Test Samples", "Value": f"{len(r5['misclassified_idx'])} / {len(r5['y_test'])}"},
        ]
    )

    lab6_summary_df = pd.DataFrame(
        [
            {"Stage": "Part 1 Pre-trained Inference (YOLOv11n)", "Metric": "Detections & Latency", "Value": f"{r6_inf['num_detections']} objects in {r6_inf['infer_ms']:.1f} ms"},
            {"Stage": "Part 3 Custom Training (3 epochs, imgsz=320)", "Metric": "Precision (B)", "Value": f"{r6_tv['precision']:.4f}"},
            {"Stage": "Part 4 Validation (model.val)", "Metric": "Recall (B)", "Value": f"{r6_tv['recall']:.4f}"},
            {"Stage": "Part 4 Validation (model.val)", "Metric": "mAP@50 (B)", "Value": f"{r6_tv['map50']:.4f}"},
            {"Stage": "Part 4 Validation (model.val)", "Metric": "mAP@50-95 (B)", "Value": f"{r6_tv['map50_95']:.4f}"},
        ]
    )

    return [
        {
            "lab_title": "Laboratory Activity 1: Introduction to Computer Vision & Image Manipulation",
            "header_table": pd.DataFrame(
                [
                    {"Field": "Course", "Details": "CSST106: Computer Vision and Perception"},
                    {"Field": "Laboratory Activity", "Details": "Lab Exercise No. 1 – Image Manipulation"},
                    {"Field": "Week Number", "Details": "Week 1 – 2"},
                    {"Field": "Term / Academic Year", "Details": "1st Semester, AY 2026–2027"},
                ]
            ),
            "objective": (
                "To familiarize students with the concept of computer vision, understand how digital images are "
                "represented as 2D/3D arrays of pixel intensity values, differentiate OpenCV BGR vs. RGB color "
                "ordering, inspect and modify individual pixels, and perform basic image manipulation operations "
                "(ROI cropping, resizing, and rotation) using Python and OpenCV."
            ),
            "procedure": [
                "Load the bundled color test scene (`assets/samples/lab1_color_scene.png`) using `cv2.imread()` and inspect its array dimensions `(Height, Width, Channels)` and `uint8` data type.",
                "Demonstrate OpenCV's BGR channel ordering by splitting B, G, and R color planes via `cv2.split()` and contrasting proper BGR→RGB conversion (`cv2.cvtColor(img, cv2.COLOR_BGR2RGB)`) against raw un-converted BGR rendering.",
                "Convert the color image to 2D single-channel Grayscale using `cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)`.",
                "Use the Web UI Interactive Pixel Explorer sliders to inspect `[B, G, R]`, `[R, G, B]`, and grayscale intensity at `(x=480, y=90)`, then overwrite a `25×25` pixel patch with RGB `[255, 215, 0]`.",
                "Execute Part 2 manipulation operations: array slicing ROI crop `img[y1:y2, x1:x2]`, resizing at `50%` scale with `cv2.INTER_LINEAR`, and geometric rotation by `90°` clockwise via `cv2.rotate()`.",
            ],
            "screenshot": "> **[PASTE WEB UI SCREENSHOT HERE: Lab 1 – Navigate to 'Lab 1: Image Manipulation' tab in the Web UI and capture the Original RGB vs. Raw BGR vs. Grayscale panels, split B/G/R channels, Pixel Explorer table, and Part 2 Crop/Resize/Rotate outputs]**",
            "results_intro": (
                "On the default `640×480` color scene (`(480, 640, 3)`), splitting channels clearly revealed pure Red swatches "
                "illuminating only the Red channel while appearing zero in Blue/Green, whereas un-converted BGR rendering visibly "
                "swapped red and blue regions. Overwriting the `25×25` patch around `(480, 90)` updated pixel intensities live, and "
                "ROI slicing, `50%` bilinear downscaling, and `90°` clockwise rotation updated output dimensions cleanly."
            ),
            "tables": [("Lab 1 Quantitative Measurements", lab1_metrics_df)],
            "figures": [("Lab 1 Visual Outputs Summary", d["fig1"])],
            "qa": [
                (
                    "1. How are digital images represented as 2D and 3D arrays in Computer Vision?",
                    "A grayscale image is a 2D matrix of shape `(Height, Width)` where each cell stores a single `uint8` luminance "
                    "value in `[0, 255]` (`0` = black, `255` = white). A color image is a 3D tensor of shape `(Height, Width, 3)` "
                    "stacking three 2D intensity planes. NumPy indexes rows (`y`, height) first and columns (`x`, width) second (`img[y, x]`).",
                ),
                (
                    "2. Why does OpenCV use BGR channel ordering and how do color spaces differ?",
                    "OpenCV historically adopted BGR (Blue, Green, Red) ordering to match early Windows/camera hardware buffers, "
                    "whereas Matplotlib, Pillow, and Streamlit display RGB. Failing to convert BGR→RGB swaps red and blue hues. "
                    "Converting BGR to Grayscale (`Y = 0.299R + 0.587G + 0.114B`) collapses 3 channels into 1 luminance plane, "
                    "reducing memory and compute cost by 3× when hue is unnecessary.",
                ),
                (
                    "3. What are real-life applications of basic image representation and manipulation?",
                    "Medical imaging (ROI isolation and resizing of X-ray/MRI scans), autonomous driving (camera calibration, "
                    "lane ROI cropping, and resolution normalization), automated manufacturing inspection (pixel-level defect localization), "
                    "and document digitization (orientation correction and contrast prep).",
                ),
            ],
            "conclusion": (
                "Digital images function as structured multi-dimensional numerical matrices whose pixel values and color channel "
                "layouts can be inspected, sliced, rescaled, and rotated deterministically with OpenCV, establishing the essential "
                "data representation foundation for filtering, edge detection, and deep vision pipelines."
            ),
        },
        {
            "lab_title": "Laboratory Activity 2 (Manual Exercise 3): Edge Detection Technique",
            "header_table": pd.DataFrame(
                [
                    {"Field": "Course", "Details": "CSST106: Computer Vision and Perception"},
                    {"Field": "Laboratory Activity", "Details": "Lab Activity 2 (Manual Exercise 3) – Edge Detection Technique"},
                    {"Field": "Week Number", "Details": "Week 3 – 4"},
                    {"Field": "Term / Academic Year", "Details": "1st Semester, AY 2026–2027"},
                ]
            ),
            "objective": (
                "To understand and implement fundamental edge detection algorithms using Python and OpenCV — specifically "
                "the Sobel gradient operator and the Canny multi-stage edge detector — apply Gaussian noise-reduction preprocessing, "
                "and compare the resulting edge maps to prepare for object detection, segmentation, and feature extraction."
            ),
            "procedure": [
                "Load the test image (`assets/samples/lab2_architecture_edges.png`, `768×512`) and convert it to single-channel grayscale.",
                "Apply Gaussian smoothing (`cv2.GaussianBlur(gray, (3, 3), 0)`) to attenuate high-frequency pixel noise before numerical differentiation.",
                "Compute horizontal (`dx=1, dy=0`) and vertical (`dx=0, dy=1`) Sobel derivatives in 64-bit floating-point precision (`cv2.CV_64F`, `ksize=3`) to preserve both positive and negative intensity slopes.",
                "Combine directional gradients via Euclidean magnitude `cv2.magnitude(sobelx, sobely)` and normalize safely to `uint8` `[0, 255]` via `cv2.convertScaleAbs()` / min-max scaling for display.",
                "Compute binary thin edges using `cv2.Canny(img_blur, 100, 200)` with interactive threshold sliders and record edge pixel density and execution latency.",
            ],
            "screenshot": "> **[PASTE WEB UI SCREENSHOT HERE: Lab 2 – Navigate to 'Lab 2: Edge Detection (Sobel & Canny)' tab and capture Original, Gaussian Blur, Sobel X, Sobel Y, Sobel Combined Magnitude, Canny Edges, and the Quantitative Metrics Table]**",
            "results_intro": (
                f"On the default `768×512` test image (`{r2['total_pixels']:,}` pixels), Sobel Combined produced continuous thick "
                f"gradient-strength ridges with peak magnitude `{r2['sobel_mag_max']:.1f}` and mean gradient `{r2['sobel_mag_mean']:.2f}` "
                f"in `{r2['sobel_ms']:.2f} ms`, whereas Canny `(100, 200)` extracted crisp 1-pixel binary contours covering "
                f"`{r2['canny_count']:,}` edge pixels (`{100.0 * r2['canny_count'] / r2['total_pixels']:.2f}%` density) in `{r2['canny_ms']:.2f} ms`."
            ),
            "tables": [("Lab 2 Edge Detector Quantitative Metrics", r2["metrics_df"])],
            "figures": [("Lab 2 Sobel vs. Canny Edge Detection Comparison", d["fig2"])],
            "qa": [
                (
                    "1. Explain the differences between the edge maps produced by Sobel and Canny edge detection.",
                    "The **Sobel operator** evaluates local first-order partial derivatives (`Gx` and `Gy`) using 3×3 convolution kernels "
                    "and outputs a continuous grayscale gradient magnitude map (`√(Gx² + Gy²)`). Its edges are thicker, vary smoothly in "
                    "intensity, remain sensitive to textured noise, and require a manual threshold to turn into binary contours. In contrast, "
                    "**Canny edge detection** builds on smoothed gradients with two extra refinement stages: (1) **Non-Maximum Suppression (NMS)**, "
                    "which suppresses non-peak gradient pixels perpendicular to the edge direction to yield crisp **1-pixel-thin ridges**, and "
                    "(2) **Hysteresis Double Thresholding** (`low=100`, `high=200`), which preserves weak edge pixels only when connected to "
                    "strong edges (`> 200`). Consequently, Canny produces clean, continuous, binary object outlines with minimal isolated speckle noise.",
                ),
            ],
            "conclusion": (
                "Comparing Sobel and Canny demonstrates the trade-off between raw directional gradient strength (Sobel, useful when edge "
                "magnitude or orientation weights are needed) and clean topological contour extraction (Canny, optimal for downstream "
                "shape analysis, Hough transforms, and contour-based segmentation)."
            ),
        },
        {
            "lab_title": "Laboratory Activity 3: Histogram Equalization and Contrast Adjustment",
            "header_table": pd.DataFrame(
                [
                    {"Field": "Course", "Details": "CSST106: Computer Vision and Perception"},
                    {"Field": "Laboratory Activity", "Details": "Lab Exercise No. 3 – Histogram Equalization and Contrast Adjustment"},
                    {"Field": "Week Number", "Details": "Week 4 – 5"},
                    {"Field": "Term / Academic Year", "Details": "1st Semester, AY 2026–2027"},
                ]
            ),
            "objective": (
                "To understand and implement histogram equalization to enhance image contrast, compare global histogram "
                "equalization against linear contrast stretching and Contrast Limited Adaptive Histogram Equalization (CLAHE), "
                "and analyze intensity histograms, Cumulative Distribution Functions (CDFs), and noise amplification."
            ),
            "procedure": [
                "Load the low-contrast grayscale sample (`assets/samples/lab3_low_contrast.png`) and low-contrast color sample (`assets/samples/lab3_color_low_contrast.png`).",
                "Apply **Global Histogram Equalization** via `cv2.equalizeHist()` on grayscale, or convert color BGR images to YUV (`cv2.COLOR_BGR2YUV`), equalize only the luminance `Y` channel, and convert back to BGR so chromaticity (`U`, `V`) stays undistorted.",
                "Apply **Linear Contrast Stretching** by mapping the `[2nd, 98th]` intensity percentiles linearly across `[0, 255]` with configurable gain `α` and bias `β` (`I_out = α·I_in + β`).",
                "Apply **CLAHE** via `cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))` to equalize localized contextual tiles while limiting histogram peak clipping.",
                "Plot 256-bin histograms and normalized CDF curves beneath all four output versions and compute Mean Luminance, Contrast (Std Dev), Dynamic Range, and Shannon Entropy.",
            ],
            "screenshot": "> **[PASTE WEB UI SCREENSHOT HERE: Lab 3 – Navigate to 'Lab 3: Histogram Equalization & Contrast' tab and capture the 4 side-by-side enhanced images (Original, Global Hist Eq, Linear Stretch, CLAHE), the 4-panel Histogram + CDF plot, and the Metrics Table]**",
            "results_intro": (
                "On the low-contrast grayscale image (`[43, 175]` initial intensity range), Global Histogram Equalization spread intensities "
                "across `[0, 255]` and boosted standard deviation sharply, while Linear Stretching preserved smooth midtone gradients. "
                "CLAHE (`clipLimit=2.5`, `8×8` tiles) recovered local coin surface markings across both dimly lit and brighter corners "
                "without over-saturating flat background regions or blowing up sensor noise. In Color (YUV) mode, equalizing `Y` restored "
                "vibrant cup contrast without shifting color hue."
            ),
            "tables": [
                ("Lab 3 Quantitative Metrics (Grayscale Mode)", r3_g["metrics_df"]),
                ("Lab 3 Quantitative Metrics (Color YUV Mode)", r3_c["metrics_df"]),
            ],
            "figures": [
                ("Lab 3 Visual Comparison (Grayscale)", d["fig3_img"]),
                ("Lab 3 Intensity Histograms & CDF Curves", d["fig3_hist"]),
            ],
            "qa": [
                (
                    "1. What is histogram equalization?",
                    "Histogram equalization is a non-linear point-processing intensity transformation that redistributes pixel brightness values "
                    "so the resulting image histogram spans the entire available dynamic range `[0, 255]` as uniformly as possible.",
                ),
                (
                    "2. Why do we need histogram equalization?",
                    "Under-exposed, over-exposed, hazy, or poorly lit images often confine pixel intensities to a narrow gray band, hiding "
                    "structural details. Equalization expands contrast automatically, improving human readability and boosting the reliability "
                    "of downstream edge detectors, feature extractors, and classifiers.",
                ),
                (
                    "3. How does histogram equalization work mathematically?",
                    "Given an 8-bit image with `N` pixels and normalized intensity probabilities `p(r_k) = n_k / N`, the **Cumulative Distribution "
                    "Function (CDF)** is `CDF(r_k) = Σ_{j=0}^{k} p(r_j)`. Each input gray level `r_k` is mapped to `s_k = round(255 × CDF(r_k))`. "
                    "Regions with dense pixel populations produce steep CDF slopes and are stretched across wide output intervals, whereas sparse "
                    "intensity ranges are compressed.",
                ),
                (
                    "4. How does CLAHE work?",
                    "Contrast Limited Adaptive Histogram Equalization (CLAHE) divides the image into a grid of small tiles (e.g., `8×8`). "
                    "Within each tile, histogram bins exceeding `clipLimit` are clipped and their excess pixel counts are redistributed uniformly "
                    "across all 256 bins before computing the local CDF—capping maximum contrast amplification to suppress background noise. "
                    "Adjacent tiles are then blended via bilinear interpolation to remove artificial tile seams.",
                ),
                (
                    "5. What are real-world applications for histogram equalization?",
                    "Medical radiography (enhancing subtle fractures in chest X-rays, mammograms, and CT slices), satellite/aerial imagery "
                    "(penetrating atmospheric haze to map terrain and coastlines), and night-time/tunnel autonomous driving and security cameras "
                    "(recovering pedestrian and obstacle contrast in low-light frames).",
                ),
                (
                    "Observations on Detail Visibility & Noise Amplification:",
                    "Global histogram equalization boosts overall contrast drastically but amplifies random background grain in uniform regions "
                    "where histogram spikes create very steep CDF slopes. CLAHE resolves this trade-off by adapting locally to uneven illumination "
                    "while bounding noise amplification via `clipLimit`.",
                ),
            ],
            "conclusion": (
                "Global histogram equalization, linear contrast stretching, and CLAHE each manipulate intensity distributions differently; "
                "applying equalization selectively on luminance (`Y` in YUV) preserves natural color fidelity, and CLAHE provides the "
                "strongest local detail enhancement with controlled noise amplification on unevenly illuminated imagery."
            ),
        },
        {
            "lab_title": "Laboratory Activity 4: SURF, SIFT, and ORB Feature Detection & Matching",
            "header_table": pd.DataFrame(
                [
                    {"Field": "Course", "Details": "CSST106: Computer Vision and Perception"},
                    {"Field": "Laboratory Activity", "Details": "Lab Exercise No. 4 – SURF Feature Detection in Python using OpenCV"},
                    {"Field": "Week Number", "Details": "Week 6"},
                    {"Field": "Term / Academic Year", "Details": "1st Semester, AY 2026–2027"},
                ]
            ),
            "objective": (
                "To understand and implement SURF (Speeded-Up Robust Features) keypoint detection and descriptor extraction in OpenCV, "
                "visualize rich keypoints with scale and orientation, perform cross-image feature matching using Lowe's ratio test, "
                "explore upright U-SURF mode, and benchmark SURF against SIFT and ORB."
            ),
            "procedure": [
                "Load the reference object (`assets/samples/lab4_object_ref.png`) and rotated (`-22°`), scaled (`0.88×`) cluttered target scene (`assets/samples/lab4_scene_target.png`) in grayscale.",
                "Invoke the safe SURF detector factory (`detect_and_compute_surf`) which attempts `cv2.xfeatures2d.SURF_create(hessianThreshold=400, upright=...)` and automatically falls back to a calibrated Multi-Scale Determinant-of-Hessian (DoH) + 64-D Haar Wavelet descriptor when OpenCV PyPI builds exclude patented non-free binaries.",
                "Draw keypoints with `cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS` to inspect circle scale radii and orientation arrows.",
                "Match descriptors between reference and scene images using `cv2.BFMatcher` + Lowe's Ratio Test (`0.75`) and visualize matches via `cv2.drawMatches`.",
                "Benchmark Standard SURF, U-SURF (`upright=True`), SIFT (`cv2.SIFT_create()`), and ORB (`cv2.ORB_create()`) across keypoint count, detection/matching time (ms), and good match count.",
            ],
            "screenshot": "> **[PASTE WEB UI SCREENSHOT HERE: Lab 4 – Navigate to 'Lab 4: SURF, SIFT & ORB Features' tab and capture the Rich Keypoints on Ref/Scene, Feature Matches tabs (SURF/SIFT/ORB), and the Benchmark Comparison Table]**",
            "results_intro": (
                f"Active SURF backend (`{r4['surf_backend']}`) detected `{r4['surf_kp1_count']}` keypoints on the reference object and "
                f"`{r4['surf_kp2_count']}` keypoints on the cluttered target scene at `hessianThreshold=400`. Rotation-invariant SURF "
                f"yielded `{r4['surf_matches_count']}` good Lowe-ratio matches across the `-22°` rotation and `0.88×` scale shift, "
                "while U-SURF skipped orientation estimation for faster extraction at the cost of lower rotation invariance. "
                "SIFT achieved dense high-accuracy matching, and ORB executed fastest using 256-bit binary descriptors."
            ),
            "tables": [("Lab 4 SURF vs. U-SURF vs. SIFT vs. ORB Benchmark", r4["benchmark_df"])],
            "figures": [("Lab 4 Rich Keypoints and Cross-Image Feature Matching", d["fig4"])],
            "qa": [
                (
                    "1. What effect does changing the Hessian threshold have on the number of detected keypoints?",
                    "The `hessianThreshold` acts as a minimum Determinant-of-Hessian saliency cutoff. **Increasing the Hessian threshold "
                    "substantially reduces the number of detected keypoints**, filtering out subtle textures and weak blobs so only the "
                    "strongest, most stable landmarks survive. Decreasing the threshold increases keypoint density for low-texture regions "
                    "but introduces more noise and redundant matches.",
                ),
                (
                    "2. What are the descriptors in SURF used for, and why are they important?",
                    "Each SURF descriptor is a normalized **64-element vector** constructed from Haar wavelet intensity responses "
                    "`[Σ dx, Σ dy, Σ |dx|, Σ |dy|]` across a `4×4` oriented grid centered on the keypoint. Descriptors encode local "
                    "appearance in a scale-, rotation-, and illumination-invariant signature, making it possible to reliably match identical "
                    "physical points across different viewpoints for object recognition, panorama stitching, AR tracking, and 3D reconstruction.",
                ),
                (
                    "3. How does SURF compare to SIFT and ORB in feature detection and speed?",
                    "**SIFT** (Difference-of-Gaussians + 128-D float descriptor) achieves the highest accuracy under large scale and perspective "
                    "changes, at the highest computational cost. **SURF** (Determinant-of-Hessian box approximations on integral images + 64-D "
                    "Haar descriptor) is faster than SIFT with comparable repeatability, though patented for commercial deployment. **ORB** "
                    "(FAST corner detector + rotated BRIEF 256-bit binary descriptor) is patent-free and orders of magnitude faster via "
                    "Hamming XOR matching, making it ideal for real-time embedded/SLAM pipelines despite weaker scale invariance than SIFT/SURF.",
                ),
            ],
            "conclusion": (
                "Local invariant feature detectors and descriptors enable robust correspondence matching even under geometric rotation, "
                "scaling, and cluttered backgrounds, with SURF balancing speed and robustness, SIFT maximizing precision, and ORB providing "
                "real-time binary efficiency."
            ),
        },
        {
            "lab_title": "Laboratory Activity 5: Facial Recognition using Principal Component Analysis (Eigenfaces)",
            "header_table": pd.DataFrame(
                [
                    {"Field": "Course", "Details": "CSST106: Computer Vision and Perception"},
                    {"Field": "Laboratory Activity", "Details": "Lab Exercise No. 5 – PCA Dimensionality Reduction & Eigenfaces"},
                    {"Field": "Week Number", "Details": "Week 7 – 8"},
                    {"Field": "Term / Academic Year", "Details": "1st Semester, AY 2026–2027"},
                ]
            ),
            "objective": (
                "To load and flatten high-dimensional facial image data, apply randomized Principal Component Analysis (PCA) "
                "to extract Eigenfaces, quantify the trade-off between retained components and Cumulative Explained Variance (≥ 90%), "
                "and train an RBF Support Vector Machine (SVM) classifier in the reduced PCA projection space."
            ),
            "procedure": [
                "**Task 1 (Data Acquisition):** Load the Labeled Faces in the Wild (LFW) subset via `fetch_lfw_people(min_faces_per_person=70, resize=0.4)` (`1288` images, `50×37` pixels = `1850` features across `7` classes) and perform an 80/20 stratified-random split (`test_size=0.2, random_state=42`).",
                "**Task 2 (Computing PCA):** Initialize `PCA(n_components=150, svd_solver='randomized', whiten=True, random_state=42)`, fit strictly on `X_train` to prevent data leakage, and project both `X_train` and `X_test` into the 150-D Eigenface space.",
                "**Task 3 (Visualizing Eigenfaces):** Reshape `pca.components_` back to `(150, 50, 37)` and plot the first 12 Eigenfaces plus a direct side-by-side comparison of the First 3 vs. Last 3 Eigenfaces.",
                "**Task 4 (SVM Classification):** Fit `SVC(kernel='rbf', class_weight='balanced', random_state=42)` on `(X_train_pca, y_train)`, evaluate predictions on `X_test_pca` with `classification_report` and Confusion Matrix, plot Cumulative Explained Variance with the `90%` threshold marker, and programmatically extract misclassified test faces.",
            ],
            "screenshot": "> **[PASTE WEB UI SCREENSHOT HERE: Lab 5 – Navigate to 'Lab 5: PCA Eigenfaces & SVM' tab and capture the First 12 Eigenfaces Gallery, First 3 vs. Last 3 Eigenfaces, Confusion Matrix, Cumulative Explained Variance Plot (showing k90), and Programmatic Misclassified Faces Gallery]**",
            "results_intro": (
                f"Compressing `{ds5['n_features']}` pixel features down to `{r5['eff_components']}` whitened PCA components "
                f"(`{(1.0 - r5['eff_components'] / ds5['n_features']) * 100:.1f}%` dimensionality reduction) retained "
                f"`{r5['total_var_retained'] * 100:.2f}%` of facial variance in `{r5['pca_time']:.3f}s`. Programmatic variance "
                f"analysis determined that exact **`{r5['k_90']}` components** suffice to preserve `≥ 90%` variance. The RBF SVM "
                f"achieved **`{r5['accuracy'] * 100:.2f}%` test accuracy** (`{len(r5['y_test']) - len(r5['misclassified_idx'])}/{len(r5['y_test'])}` "
                f"correct) in `{r5['svm_time']:.3f}s`."
            ),
            "tables": [("Lab 5 Eigenfaces & SVM Quantitative Summary", lab5_summary_df)],
            "figures": [
                ("Lab 5 First 12 Eigenfaces Gallery", d["fig5_gal"]),
                ("Lab 5 First 3 vs. Last 3 Eigenfaces", d["fig5_fl"]),
                ("Lab 5 Cumulative Explained Variance Curve (90% Threshold)", d["fig5_var"]),
                ("Lab 5 RBF SVM Confusion Matrix", d["fig5_cm"]),
                ("Lab 5 Programmatic Failure Analysis (Misclassified Faces)", d["fig5_fail"]),
            ],
            "qa": [
                (
                    "Post-Lab Q1 — Dimensionality vs. Variance: How many components are required to retain 90% of the variance?",
                    f"Programmatic evaluation of `np.cumsum(pca.explained_variance_ratio_) ≥ 0.90` confirms that **`{r5['k_90']}` principal "
                    f"components** are required to retain at least **90%** of the total dataset variance (out of `{ds5['n_features']}` raw pixel "
                    f"dimensions, representing over a `{(1.0 - r5['k_90'] / ds5['n_features']) * 100:.1f}%` feature compression).",
                ),
                (
                    "Post-Lab Q2 — Visual Interpretation: What do the first 3 Eigenfaces capture vs. the last 3 Eigenfaces?",
                    "The **first 3 Eigenfaces (0, 1, 2)** exhibit smooth, recognizable low-frequency facial structures corresponding to dominant "
                    "global variations: overall ambient brightness, left-to-right directional illumination gradients, jawline/oval contour, and "
                    "eye-socket/nose-bridge symmetry. Conversely, the **last 3 Eigenfaces (147, 148, 149)** correspond to tiny trailing eigenvalues "
                    "and appear as speckled, grainy **high-frequency spatial patterns** capturing localized background noise, hair edges, and "
                    "minor pixel misalignments rather than global facial identity.",
                ),
                (
                    "Post-Lab Q3 — The Importance of Whitening: What does whiten=True do mathematically and why does it help an SVM?",
                    "Standard PCA yields uncorrelated features `z_i = u_i^T (x - μ)` whose variance equals eigenvalue `λ_i` (`Var(z_i) = λ_i`). "
                    "Because early lighting components have huge `λ_i` compared to subtle identity components (`i > 15`), raw PCA features are "
                    "severely anisotropic. Setting `whiten=True` divides every component by `√λ_i` (`z_white,i = z_i / √λ_i`), scaling every "
                    "projected dimension to **unit variance (`Cov = I`)**. Since an RBF SVM relies on isotropic Euclidean distances inside "
                    "`K(a, b) = exp(-γ ||a - b||²)`, whitening prevents high-variance illumination components from dominating the decision kernel "
                    "and allows finer discriminative Eigenfaces to contribute equally.",
                ),
                (
                    "Post-Lab Q4 — Programmatic Failure Analysis: Why did the Eigenface SVM misclassify specific test faces?",
                    f"Extracting `np.where(y_pred != y_test)` identified `{len(r5['misclassified_idx'])}` misclassified portraits. Visual "
                    "inspection indicates four primary computer-vision failure modes: (1) **strong directional shadows / specular glare** distorting "
                    "pixel intensities along lighting eigenmodes, (2) **out-of-plane head yaw/tilt** misaligning eyes and mouths from frontal templates, "
                    "(3) **partial occlusions** (eyeglasses, microphones, extreme expressions), and (4) **class imbalance** (George W. Bush has "
                    "530 samples vs. ~71–144 for smaller classes), biasing borderline test projections.",
                ),
                (
                    "Follow-Up Q1 — Why is dimensionality reduction important in computer vision?",
                    "Raw images live in massive correlated spaces (`1850` dimensions even at `50×37`). Reducing dimensionality mitigates the "
                    "curse of dimensionality, eliminates redundant pixel noise, accelerates training and inference, cuts storage overhead, and "
                    "reduces classifier overfitting on limited datasets.",
                ),
                (
                    "Follow-Up Q2 — What is the difference between the first principal component and the second principal component?",
                    "PC1 is the eigenvector of the data covariance matrix pointing along the single axis of **maximum global variance** in the centered "
                    "faces, whereas PC2 is constrained to be **orthogonal (90° perpendicular)** to PC1 while capturing the **maximum remaining variance**.",
                ),
                (
                    "Follow-Up Q3 — If PCA reduces an image from 1,000 features to 100 components, what information might be lost?",
                    "The discarded 900 trailing eigenvectors contain **high-frequency spatial details** — sharp edges, fine skin pores, subtle wrinkles, "
                    "eyelashes, micro-expressions, and sensor noise — resulting in slightly smoother reconstructions.",
                ),
                (
                    "Follow-Up Q4 — How can PCA be applied to face recognition?",
                    "By subtracting the average face `μ`, projecting training faces onto the top `k` Eigenfaces (`z = W^T (x - μ)`), and classifying "
                    "query faces in that compact `k`-dimensional coordinate space using distance matching or an SVM classifier.",
                ),
                (
                    "Follow-Up Q5 — Why should image features generally be centered before applying PCA?",
                    "PCA solves for orthogonal variance axes passing through the origin. Without zero-centering (`x - μ`), PC1 is pulled directly "
                    "toward the non-zero **mean brightness vector `μ`** instead of capturing genuine structural deviations among individuals.",
                ),
            ],
            "conclusion": (
                "Randomized PCA successfully compresses high-dimensional facial images by over 91% into orthogonal Eigenfaces while preserving "
                "~95% of variance; pairing whitened PCA projections with a balanced RBF SVM achieves strong facial recognition accuracy while "
                "exposing classical linear-template limitations under lighting, pose, and occlusion shifts."
            ),
        },
        {
            "lab_title": "Laboratory Activity 6 (Manual Lab 5/6): Object Detection with YOLOv11",
            "header_table": pd.DataFrame(
                [
                    {"Field": "Course", "Details": "CSST106: Computer Vision and Perception"},
                    {"Field": "Laboratory Activity", "Details": "Lab Exercise No. 6 (Manual Lab 5/6) – YOLOV11 Object Detection"},
                    {"Field": "Week Number", "Details": "Week 8 – 9"},
                    {"Field": "Term / Academic Year", "Details": "1st Semester, AY 2026–2027"},
                ]
            ),
            "objective": (
                "To understand and execute real-time object detection using pre-trained YOLOv11 (`yolo11n.pt`), prepare and "
                "verify a custom YOLO-format dataset (`images/`, `labels/`, and `data.yaml`), fine-tune YOLOv11n on custom classes, "
                "and interpret validation metrics (Precision, Recall, mAP@50, and mAP@50-95)."
            ),
            "procedure": [
                "**Part 1 (Pre-trained Inference):** Load `YOLO('yolo11n.pt')` (YOLOv11n) and run inference on the bundled multi-object scene (`assets/samples/lab6_street_scene.png`), uploaded images, videos, or live webcam snapshots with interactive Confidence (`0.25`) and IoU NMS (`0.45`) thresholds.",
                "**Part 2 (Custom Dataset & Annotations):** Generate and inspect `assets/custom_yolo_dataset/` organized into `images/train` (8 images), `images/val` (4 images), `labels/train`, `labels/val`, and `data.yaml` (`nc: 3`, `names: ['person', 'cat', 'cup']`), verifying normalized `[class_id x_center y_center width height]` bounding boxes.",
                "**Part 3 & 4 (Training, Validation & Evaluation):** Fine-tune `YOLO('yolo11n.pt').train(data=..., epochs=..., imgsz=...)` on the custom dataset and evaluate using `model.val()`, reporting Precision, Recall, mAP@50, and mAP@50-95.",
            ],
            "screenshot": "> **[PASTE WEB UI SCREENSHOT HERE: Lab 6 – Navigate to 'Lab 6: YOLOv11 Object Detection' tab and capture Part 1 Annotated Bounding Boxes + Detection Table, Part 2 Ground-Truth Annotation Explorer, and Part 3/4 Training/Validation Metrics]**",
            "results_intro": (
                f"Pre-trained `YOLOv11n` detected `{r6_inf['num_detections']}` objects (`cat`, `person`, `cup`, `dining table`) on the "
                f"bundled natural scene in `{r6_inf['infer_ms']:.1f} ms` at `conf=0.25`. Custom fine-tuning on the 3-class dataset "
                f"achieved Precision `{r6_tv['precision']:.4f}`, Recall `{r6_tv['recall']:.4f}`, mAP@50 `{r6_tv['map50']:.4f}`, and "
                f"mAP@50-95 `{r6_tv['map50_95']:.4f}` during `model.val()`."
            ),
            "tables": [
                ("Lab 6 Part 1 Detections on Default Scene", r6_inf["detections_df"]),
                ("Lab 6 End-to-End Inference, Training & Validation Summary", lab6_summary_df),
            ],
            "figures": [("Lab 6 YOLOv11n Inference & Custom Dataset Verification", d["fig6"])],
            "qa": [
                (
                    "1. What is object detection in computer vision, and how does it differ from image classification? Explain using a real-world example.",
                    "**Image Classification** assigns a single holistic category label to an entire image (e.g., predicting `'cat'` for a whole photo), "
                    "providing no coordinates and failing when multiple items appear together. **Object Detection** performs both **spatial localization** "
                    "(regressing tight bounding boxes `[x1, y1, x2, y2]`) and **multi-object classification** with per-instance confidence scores across "
                    "every object present in the frame. For example, in an **autonomous vehicle's forward camera**, image classification could only output "
                    "`'street'`, which cannot guide steering or braking; YOLOv11 object detection simultaneously locates a `pedestrian` at `(460, 185, 515, 360)`, "
                    "a `car` ahead at `(120, 210, 310, 340)`, and an overhead `traffic light` — enabling exact distance estimation and collision avoidance. "
                    "Similarly, in an **automated retail checkout system**, object detection locates and counts multiple individual items (`cup`, `bottle`, `apple`) "
                    "on a tray simultaneously.",
                ),
            ],
            "conclusion": (
                "YOLOv11n delivers fast, single-stage multi-object localization and classification out of the box and adapts seamlessly to "
                "domain-specific classes via standardized normalized bounding-box annotations, `data.yaml` configuration, transfer learning, and "
                "mAP validation."
            ),
        },
        {
            "lab_title": "Laboratory Activity 7 (Manual Lab 6): Image Segmentation Techniques",
            "header_table": pd.DataFrame(
                [
                    {"Field": "Course", "Details": "CSST106: Computer Vision and Perception"},
                    {"Field": "Laboratory Activity", "Details": "Lab Exercise No. 7 (Manual Lab 6) – Segmentation Technique"},
                    {"Field": "Week Number", "Details": "Week 13"},
                    {"Field": "Term / Academic Year", "Details": "1st Semester, AY 2026–2027"},
                ]
            ),
            "objective": (
                "To understand and apply four major image segmentation methods — Thresholding (Global Otsu & Adaptive), "
                "Region-Based (Region Growing & Watershed), Edge-Based (Canny + Closed Contours), and Deep Semantic Segmentation "
                "(DeepLabV3) — and evaluate their qualitative separation quality and quantitative IoU performance."
            ),
            "procedure": [
                "**Thresholding Segmentation:** Load the unevenly illuminated touching-coins test image (`assets/samples/lab7_coins_touching.png`) in grayscale and apply Global Otsu thresholding (`cv2.THRESH_OTSU`) alongside Adaptive Mean and Adaptive Gaussian thresholding (`blockSize=105, C=-12`).",
                "**Region-Based Segmentation:** Run interactive 8-connected Region Growing from seed `(105, 110)` with tolerance `τ=35`, then apply the Watershed algorithm (`cv2.watershed`) using morphological opening, Euclidean Distance Transform (`cv2.distanceTransform`), sure foreground/background markers, and boundary delineation (`markers == -1`).",
                "**Edge-Based Segmentation:** Extract edges with Canny (`60, 150`), bridge small gaps using elliptical morphological closing (`7×7`), and detect/fill closed external contours (`cv2.findContours` + `cv2.drawContours`).",
                "**Semantic Segmentation:** Execute pre-trained PyTorch `deeplabv3_mobilenet_v3_large` (`DeepLabV3_MobileNet_V3_Large_Weights.DEFAULT`) on `assets/samples/lab7_semantic_scene.png`, render the color-coded PASCAL VOC pixel mask overlay, and compute IoU, Dice, Pixel Accuracy, and latency across methods.",
            ],
            "screenshot": "> **[PASTE WEB UI SCREENSHOT HERE: Lab 7 – Navigate to 'Lab 7: Image Segmentation Techniques' tab and capture BOTH tabs: (1) Thresholding, Region Growing, Watershed, Edge-Based + IoU Table, and (2) DeepLabV3 Semantic Mask, Overlay, and Class Color Legend]**",
            "results_intro": (
                f"On the unevenly illuminated touching-coins scene, Global Otsu selected threshold `T = {r7_c['otsu_val']:.1f}` but merged "
                "adjacent touching coins into unified blobs. Adaptive Gaussian handled spatial shading better locally, single-seed Region "
                f"Growing isolated the target coin cleanly around `(105, 110)`, and **Watershed separated all `{r7_c['ws_object_count']}` "
                "distinct coin regions** with clean boundary ridges (`markers == -1`) while achieving strong foreground IoU against ground truth. "
                f"On the complex natural scene, **DeepLabV3-MobileNetV3-Large** classified every pixel into `{r7_s['num_classes_detected']}` "
                f"PASCAL VOC semantic labels in `{r7_s['infer_ms']:.1f} ms`."
            ),
            "tables": [
                ("Lab 7 Classical Segmentation Quantitative Benchmark (IoU vs. Ground Truth)", r7_c["metrics_df"]),
                ("Lab 7 DeepLabV3 Semantic Class Coverage Legend", r7_s["legend_df"]),
            ],
            "figures": [
                ("Lab 7 Classical Segmentation Methods Comparison", d["fig7_c"]),
                ("Lab 7 DeepLabV3 Semantic Segmentation Output", d["fig7_s"]),
            ],
            "qa": [
                (
                    "1. Explain the differences between thresholding, region-based, edge-based, and semantic segmentation techniques. Provide one advantage and one limitation for each method.",
                    "**Thresholding Segmentation** partitions pixels independently by intensity relative to a global (`Otsu`) or local neighborhood "
                    "(`Adaptive`) threshold. *Advantage:* Extremely fast (`O(N)`) and parameter-light. *Limitation:* Ignores spatial topology and "
                    "fails when foreground and background brightness overlap or lighting varies severely.\n\n"
                    "**Region-Based Segmentation (Region Growing & Watershed)** groups spatially connected pixels satisfying intensity homogeneity "
                    "or partitions topographic distance basins between foreground/background markers. *Advantage:* Produces guaranteed closed regions "
                    "and Watershed cleanly separates touching objects (e.g., clustered cells or coins). *Limitation:* Sensitive to seed selection and "
                    "prone to over-segmentation on noisy textures without careful marker filtering.\n\n"
                    "**Edge-Based Segmentation** detects sharp intensity discontinuities (`Canny`), links gaps morphologically, and fills enclosed "
                    "contours. *Advantage:* Works well when object borders are crisp even across smoothly varying backgrounds. *Limitation:* "
                    "Highly sensitive to noise and broken edges; unclosed gaps cause contour-filling leaks and fail to split touching objects.\n\n"
                    "**Semantic Segmentation (DeepLabV3 / U-Net)** uses deep convolutional encoder-decoders with multi-scale context aggregation "
                    "to assign a semantic class label to every individual pixel. *Advantage:* Understands high-level object semantics across clutter, "
                    "shadows, and varied textures. *Limitation:* Computationally heavy, requires large pixel-annotated training datasets, and merges "
                    "adjacent same-class instances unless extended to instance segmentation.",
                ),
            ],
            "conclusion": (
                "Classical thresholding, region-based, and edge-based techniques offer fast, interpretable segmentation when intensity or "
                "boundary assumptions hold (with Watershed excelling on touching objects), whereas deep semantic networks like DeepLabV3 "
                "unlock accurate pixel-level understanding in complex real-world scenes."
            ),
        },
    ]


def _render_markdown_report(sections: List[Dict[str, Any]]) -> str:
    """Render clean MS-Word-compatible LAB_REPORT.md without raw LaTeX dollar delimiters."""
    lines: List[str] = [
        "# CSST106: Computer Vision and Perception — Compiled Laboratory Report (Labs 1 – 7)",
        "",
        "**Course Code:** CSST106  ",
        "**Course Title:** Computer Vision and Perception  ",
        "**Institution:** College of Computer Studies  ",
        "**Term / Academic Year:** 1st Semester, AY 2026–2027  ",
        "**Unified Submission Platform:** Streamlit Web UI (`app.py`) + Hard-Copy Laboratory Report  ",
        "",
        "---",
        "",
    ]

    for idx, sec in enumerate(sections, start=1):
        lines.append(f"## {sec['lab_title']}")
        lines.append("")
        lines.append("### 1. Lab Header")
        lines.append(_df_to_markdown(sec["header_table"]))
        lines.append("")
        lines.append("### 2. Objective")
        lines.append(sec["objective"])
        lines.append("")
        lines.append("### 3. Algorithms / Techniques & Step-by-Step Procedure")
        for step_i, step_text in enumerate(sec["procedure"], start=1):
            lines.append(f"{step_i}. {step_text}")
        lines.append("")
        lines.append("### 4. Web UI Screenshot Placeholder")
        lines.append(sec["screenshot"])
        lines.append("")
        lines.append("### 5. Results & Observations")
        lines.append(sec["results_intro"])
        lines.append("")
        for tbl_title, df in sec["tables"]:
            lines.append(f"**{tbl_title}**")
            lines.append("")
            lines.append(_df_to_markdown(df))
            lines.append("")
        lines.append("### 6. Complete Lab Questions & Post-Lab Analysis Q&A")
        for q_title, q_ans in sec["qa"]:
            lines.append(f"**{q_title}**")
            lines.append("")
            lines.append(q_ans)
            lines.append("")
        lines.append("### 7. Conclusion")
        lines.append(sec["conclusion"])
        lines.append("")
        if idx < len(sections):
            lines.append("---")
            lines.append("")

    md_text = "\n".join(lines).strip() + "\n"
    # Guarantee clean MS Word compatibility with zero raw LaTeX $ delimiters
    md_text = md_text.replace("$", "")
    return md_text


def _add_docx_table(doc: Document, df: pd.DataFrame) -> None:
    """Insert a styled pandas DataFrame table into a python-docx Document."""
    table = doc.add_table(rows=1, cols=len(df.columns))
    table.style = "Table Grid"
    hdr_cells = table.rows[0].cells
    for j, col_name in enumerate(df.columns):
        hdr_cells[j].text = str(col_name)
        for p in hdr_cells[j].paragraphs:
            for r in p.runs:
                r.bold = True
                r.font.size = Pt(9.5)
    for _, row in df.iterrows():
        row_cells = table.add_row().cells
        for j, col_name in enumerate(df.columns):
            row_cells[j].text = str(row[col_name])
            for p in row_cells[j].paragraphs:
                for r in p.runs:
                    r.font.size = Pt(9.0)
    doc.add_paragraph("")


def _render_docx_report(sections: List[Dict[str, Any]], out_path: Path) -> None:
    """Generate formatted Microsoft Word hard-copy report (.docx) with embedded figures."""
    doc = Document()
    title = doc.add_heading("CSST106: Computer Vision and Perception", level=0)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    sub = doc.add_paragraph("Compiled Laboratory Activity Report (Labs 1 – 7) | 1st Semester, AY 2026–2027")
    sub.alignment = WD_ALIGN_PARAGRAPH.CENTER

    for idx, sec in enumerate(sections, start=1):
        if idx > 1:
            doc.add_page_break()
        doc.add_heading(sec["lab_title"], level=1)

        doc.add_heading("1. Lab Header", level=2)
        _add_docx_table(doc, sec["header_table"])

        doc.add_heading("2. Objective", level=2)
        doc.add_paragraph(sec["objective"])

        doc.add_heading("3. Algorithms / Techniques & Step-by-Step Procedure", level=2)
        for step_text in sec["procedure"]:
            doc.add_paragraph(step_text.replace("`", ""), style="List Number")

        doc.add_heading("4. Web UI Screenshot Placeholder", level=2)
        p_shot = doc.add_paragraph()
        run_shot = p_shot.add_run(sec["screenshot"].replace("> **[", "[").replace("]**", "]"))
        run_shot.bold = True
        run_shot.font.color.rgb = RGBColor(0x1F, 0x4E, 0x79)

        doc.add_heading("5. Results & Observations", level=2)
        doc.add_paragraph(sec["results_intro"].replace("`", "").replace("**", ""))

        for tbl_title, df in sec["tables"]:
            p_t = doc.add_paragraph()
            p_t.add_run(tbl_title).bold = True
            _add_docx_table(doc, df)

        for cap, fig_path in sec["figures"]:
            if fig_path.exists():
                doc.add_picture(str(fig_path), width=Inches(6.3))
                p_cap = doc.add_paragraph(cap)
                p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER

        doc.add_heading("6. Complete Lab Questions & Post-Lab Analysis Q&A", level=2)
        for q_title, q_ans in sec["qa"]:
            p_q = doc.add_paragraph()
            p_q.add_run(q_title.replace("`", "")).bold = True
            doc.add_paragraph(q_ans.replace("`", "").replace("**", ""))

        doc.add_heading("7. Conclusion", level=2)
        doc.add_paragraph(sec["conclusion"])

    doc.save(str(out_path))


def _render_html_report(sections: List[Dict[str, Any]], out_path: Path) -> None:
    """Generate clean printable HTML report linking generated figures."""
    parts = [
        "<!DOCTYPE html><html lang='en'><head><meta charset='utf-8'>",
        "<title>CSST106 Computer Vision Laboratory Report (Labs 1–7)</title>",
        "<style>body{font-family:Segoe UI,Arial,sans-serif;max-width:980px;margin:24px auto;line-height:1.55;color:#1b1f23;padding:0 16px;}",
        "h1,h2{color:#0f3d68;border-bottom:1px solid #d0d7de;padding-bottom:6px;}",
        "table{border-collapse:collapse;width:100%;margin:12px 0;font-size:13px;}",
        "th,td{border:1px solid #c7d0d9;padding:6px 10px;text-align:left;} th{background:#f0f4f8;}",
        ".callout{background:#eef6ff;border-left:4px solid #1f6feb;padding:10px 14px;margin:12px 0;font-weight:600;}",
        "figure{margin:16px 0;text-align:center;} figure img{max-width:100%;border:1px solid #d0d7de;border-radius:4px;}",
        "@media print{.lab-section{page-break-after:always;}}</style></head><body>",
        "<h1>CSST106: Computer Vision and Perception — Laboratory Report (Labs 1 – 7)</h1>",
        "<p><strong>Course:</strong> CSST106 | <strong>Term:</strong> 1st Semester, AY 2026–2027 | <strong>College of Computer Studies</strong></p>",
    ]

    for sec in sections:
        parts.append("<section class='lab-section'>")
        parts.append(f"<h2>{html.escape(sec['lab_title'])}</h2>")
        parts.append(sec["header_table"].to_html(index=False, border=0))
        parts.append(f"<h3>Objective</h3><p>{html.escape(sec['objective'])}</p>")
        parts.append("<h3>Algorithms &amp; Procedure</h3><ol>")
        for step in sec["procedure"]:
            parts.append(f"<li>{html.escape(step.replace('`', ''))}</li>")
        parts.append("</ol>")
        parts.append(f"<div class='callout'>{html.escape(sec['screenshot'])}</div>")
        parts.append(f"<h3>Results &amp; Observations</h3><p>{html.escape(sec['results_intro'])}</p>")
        for tbl_title, df in sec["tables"]:
            parts.append(f"<p><strong>{html.escape(tbl_title)}</strong></p>")
            parts.append(df.to_html(index=False, border=0))
        for cap, fig_path in sec["figures"]:
            rel = fig_path.relative_to(PROJECT_ROOT).as_posix()
            parts.append(f"<figure><img src='{html.escape(rel)}' alt='{html.escape(cap)}'><figcaption>{html.escape(cap)}</figcaption></figure>")
        parts.append("<h3>Post-Lab Questions &amp; Answers</h3>")
        for q_title, q_ans in sec["qa"]:
            parts.append(f"<p><strong>{html.escape(q_title)}</strong></p><p>{html.escape(q_ans)}</p>")
        parts.append(f"<h3>Conclusion</h3><p>{html.escape(sec['conclusion'])}</p>")
        parts.append("</section>")

    parts.append("</body></html>")
    out_path.write_text("\n".join(parts), encoding="utf-8")


def generate_all_reports() -> Dict[str, Path]:
    """Execute all 7 labs on bundled samples and write LAB_REPORT.md, .docx, and .html."""
    data = _collect_all_lab_data()
    sections = _build_lab_sections(data)

    md_path = PROJECT_ROOT / "LAB_REPORT.md"
    docx_path = PROJECT_ROOT / "CSST106_Lab_Report.docx"
    html_path = PROJECT_ROOT / "CSST106_Lab_Report.html"

    md_text = _render_markdown_report(sections)
    md_path.write_text(md_text, encoding="utf-8")
    _render_docx_report(sections, docx_path)
    _render_html_report(sections, html_path)

    return {"md": md_path, "docx": docx_path, "html": html_path}
