"""Unified Streamlit Web UI for CSST106: Computer Vision and Perception (Labs 1–7)."""

from __future__ import annotations

import os
from pathlib import Path
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
_YOLO_CFG_DIR = PROJECT_ROOT / "assets" / ".ultralytics_cfg"
_YOLO_CFG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("YOLO_CONFIG_DIR", str(_YOLO_CFG_DIR))

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

st.set_page_config(
    page_title="CSST106: Computer Vision & Perception Web UI",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource(show_spinner="Initializing bundled sample images & datasets...")
def get_samples() -> dict[str, Path]:
    """Ensure bundled default test assets exist and return their paths."""
    return ensure_all_samples()


MODULES = {
    "Course Overview": None,
    "Lab 1: Image Manipulation": lab1_images.render,
    "Lab 2: Edge Detection (Sobel & Canny)": lab2_edges.render,
    "Lab 3: Histogram Equalization & Contrast": lab3_histogram.render,
    "Lab 4: SURF, SIFT & ORB Features": lab4_surf.render,
    "Lab 5: PCA Eigenfaces & SVM": lab5_pca.render,
    "Lab 6: YOLOv11 Object Detection": lab6_yolo.render,
    "Lab 7: Image Segmentation Techniques": lab7_segmentation.render,
}


def render_overview() -> None:
    """Render unified course landing page & module directory."""
    st.title("CSST106: Computer Vision and Perception")
    st.subheader("Unified Laboratory Compilation Web User Interface (Labs 1 – 7)")
    st.markdown(
        """
        **College of Computer Studies** | **Academic Year:** 2026–2027 | **Term:** 1st Semester

        Welcome to the unified Web UI compiling all seven **CSST106 Computer Vision and Perception** laboratory
        exercises. Every module runs out-of-the-box on bundled sample images and datasets while also supporting
        interactive parameter tuning and custom image/video/webcam uploads.
        """
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Compiled Labs", "7 / 7")
    c2.metric("CV Backends", "OpenCV + scikit-learn + PyTorch")
    c3.metric("Deep Models", "YOLOv11n & DeepLabV3")
    c4.metric("Input Modes", "Bundled + Upload + Camera")

    st.markdown("### Laboratory Activity Directory")
    st.markdown(
        """
        | Module | Manual Mapping | Week No. | Core Algorithms & Interactive Features |
        | :--- | :--- | :---: | :--- |
        | **Lab 1** | Lab Exercise 1 | 1 – 2 | 2D/3D Pixel Arrays, BGR vs. RGB Channel Splitting, Grayscale, Pixel Inspector/Editor, ROI Crop, Resize, Rotate |
        | **Lab 2** | Lab Exercise 3 | 3 – 4 | Gaussian Smoothing, Sobel X/Y/Magnitude (`CV_64F` → `uint8`), Canny Edge Detection (`NMS` + Hysteresis) |
        | **Lab 3** | Lab Exercise 3 | 4 – 5 | Global Histogram Equalization (Grayscale & YUV `Y`), Percentile Contrast Stretching, CLAHE, Histograms & CDFs |
        | **Lab 4** | Lab Exercise 4 | 6 | SURF (with DoH + 64-D Haar fallback), U-SURF Toggle, Rich Keypoints, Lowe Ratio Matching, SIFT & ORB Benchmark |
        | **Lab 5** | Lab Exercise 5 | 7 – 8 | LFW Dataset, Randomized PCA Eigenfaces (`whiten=True`), RBF SVM, Explained Variance (`k90`), Failure Analysis |
        | **Lab 6** | Lab Exercise 5/6 | 8 – 9 | Pre-trained `YOLOv11n` (Image/Video/Webcam), Custom YOLO Dataset & `data.yaml` Explorer, Live Train & `model.val()` |
        | **Lab 7** | Lab Exercise 6 | 13 | Otsu & Adaptive Thresholding, Region Growing, Watershed, Edge Contours, DeepLabV3 Semantic Segmentation, IoU Benchmark |
        """
    )
    st.info("Select any Laboratory Activity from the **Sidebar Navigation** on the left to inspect and run it.")


def main() -> None:
    """Main Streamlit navigation loop."""
    sample_paths = get_samples()

    with st.sidebar:
        st.title("CSST106 Web UI")
        st.caption("Computer Vision & Perception")
        st.divider()
        selected = st.radio(
            "Select Laboratory Module",
            list(MODULES.keys()),
            key="nav_module",
        )
        st.divider()
        st.markdown(
            """
            **Submission Stack**
            - Unified Web UI (`app.py`)
            - 7 Modular Labs (`labs/`)
            - Bundled Samples (`assets/`)
            - Hard-Copy Report (`LAB_REPORT.md` & `.docx`)
            """
        )

    renderer = MODULES[selected]
    if renderer is None:
        render_overview()
    else:
        renderer(sample_paths)


if __name__ == "__main__":
    main()
