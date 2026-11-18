"""Automated end-to-end verification across Labs 1–7 and report generation."""

from __future__ import annotations

import os
from pathlib import Path
import cv2
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent
os.environ.setdefault("YOLO_CONFIG_DIR", str(PROJECT_ROOT / "assets" / ".ultralytics_cfg"))

from utils.sample_generator import ensure_all_samples  # noqa: E402
from utils.report_generator import generate_all_reports  # noqa: E402
from labs import (  # noqa: E402
    lab1_images,
    lab2_edges,
    lab3_histogram,
    lab4_surf,
    lab5_pca,
    lab6_yolo,
    lab7_segmentation,
)


def main() -> None:
    print("[1/8] Ensuring all bundled sample assets...")
    samples = ensure_all_samples()
    for name, path in samples.items():
        assert path.exists(), f"Missing sample asset {name}: {path}"

    print("[2/8] Testing Lab 1 (Digital Images & Manipulation)...")
    img1 = cv2.imread(str(samples["lab1_color"]), cv2.IMREAD_COLOR)
    r1 = lab1_images.run_lab1_pipeline(img1)
    assert r1["shape"] == (480, 640, 3)
    assert r1["gray"].ndim == 2
    assert r1["cropped_rgb"].shape[0] > 0

    print("[3/8] Testing Lab 2 (Sobel & Canny Edge Detection)...")
    img2 = cv2.imread(str(samples["lab2_edges"]), cv2.IMREAD_COLOR)
    r2 = lab2_edges.run_lab2_pipeline(img2)
    assert r2["sobel_combined_u8"].dtype == "uint8"
    assert r2["canny_count"] > 0
    fig2 = lab2_edges.make_lab2_figure(r2)
    plt.close(fig2)

    print("[4/8] Testing Lab 3 (Histogram Equalization & CLAHE)...")
    img3_g = cv2.imread(str(samples["lab3_gray"]), cv2.IMREAD_COLOR)
    img3_c = cv2.imread(str(samples["lab3_color"]), cv2.IMREAD_COLOR)
    r3_g = lab3_histogram.run_lab3_pipeline(img3_g, color_mode="Grayscale")
    r3_c = lab3_histogram.run_lab3_pipeline(img3_c, color_mode="Color (YUV Luminance)")
    assert len(r3_g["metrics_df"]) == 4 and len(r3_c["metrics_df"]) == 4
    fig3 = lab3_histogram.make_lab3_hist_figure(r3_g)
    plt.close(fig3)

    print("[5/8] Testing Lab 4 (SURF / U-SURF / SIFT / ORB)...")
    img4_ref = cv2.imread(str(samples["lab4_ref"]), cv2.IMREAD_COLOR)
    img4_scn = cv2.imread(str(samples["lab4_scene"]), cv2.IMREAD_COLOR)
    r4_std = lab4_surf.run_lab4_pipeline(img4_ref, img4_scn, hessian_threshold=400, upright=False)
    r4_up = lab4_surf.run_lab4_pipeline(img4_ref, img4_scn, hessian_threshold=400, upright=True)
    assert r4_std["surf_kp1_count"] > 0 and r4_up["surf_kp1_count"] > 0
    assert len(r4_std["benchmark_df"]) == 4

    print("[6/8] Testing Lab 5 (PCA Eigenfaces & RBF SVM)...")
    r5 = lab5_pca.run_lab5_pipeline(str(samples["lab5_lfw"]), n_components=150, whiten=True)
    assert r5["k_90"] is not None and r5["k_90"] > 0
    assert r5["accuracy"] > 0.50
    for f in (
        lab5_pca.make_eigenfaces_gallery_figure(r5["eigenfaces"], r5["dataset"]["h"], r5["dataset"]["w"]),
        lab5_pca.make_first_vs_last_figure(r5["eigenfaces"]),
        lab5_pca.make_variance_figure(r5),
        lab5_pca.make_confusion_matrix_figure(r5["confusion_matrix"], r5["dataset"]["target_names"]),
        lab5_pca.make_failures_figure(r5),
    ):
        plt.close(f)

    print("[7/8] Testing Lab 6 (YOLOv11n Inference & Custom Dataset)...")
    img6 = cv2.imread(str(samples["lab6_scene"]), cv2.IMREAD_COLOR)
    r6 = lab6_yolo.run_lab6_inference(img6, weights_path=str(PROJECT_ROOT / "yolo11n.pt"))
    assert r6["num_detections"] >= 1

    print("[8/8] Testing Lab 7 (Thresholding, Region, Watershed, Canny, DeepLabV3)...")
    img7_c = cv2.imread(str(samples["lab7_coins"]), cv2.IMREAD_COLOR)
    gt7 = cv2.imread(str(samples["lab7_gt"]), cv2.IMREAD_GRAYSCALE)
    r7_c = lab7_segmentation.run_lab7_classical(img7_c, gt_mask=gt7)
    assert r7_c["ws_object_count"] >= 5
    assert len(r7_c["metrics_df"]) == 6
    img7_s = cv2.imread(str(samples["lab7_semantic"]), cv2.IMREAD_COLOR)
    r7_s = lab7_segmentation.run_lab7_semantic(img7_s)
    assert r7_s["num_classes_detected"] >= 2

    print("[REPORT] Running generate_all_reports()...")
    rep = generate_all_reports()
    for k, p in rep.items():
        assert p.exists() and p.stat().st_size > 1000, f"Report artifact invalid: {k} -> {p}"
    md_text = rep["md"].read_text(encoding="utf-8")
    assert "$" not in md_text, "LAB_REPORT.md must not contain raw LaTeX $ delimiters"

    print("ALL SELF-TESTS AND REPORT CHECKS PASSED SUCCESSFULLY!")


if __name__ == "__main__":
    main()
