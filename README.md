# CSST106: Computer Vision and Perception — Unified Laboratory Web UI

A complete, modular **Streamlit Web User Interface** and hard-copy laboratory documentation suite compiling all seven **CSST106: Computer Vision and Perception** laboratory activities (`1st Semester, AY 2026–2027`, College of Computer Studies).

## Features

- **Single Unified Multi-Module Web UI (`app.py`)**: Clean sidebar navigation menu covering Course Overview + Lab Activities 1 through 7.
- **Out-of-the-Box Bundled Samples**: Every module auto-generates and pre-loads test images, cached LFW facial data, and a valid mini YOLO dataset in `assets/`, while supporting custom uploads (`st.file_uploader`) and camera input (`st.camera_input`).
- **Performance Caching**: Uses `@st.cache_resource` for PCA/SVM, YOLOv11n, and DeepLabV3 models, and `@st.cache_data` for dataset loading.
- **Hard-Copy Report Suite (`LAB_REPORT.md`, `CSST106_Lab_Report.docx`, `CSST106_Lab_Report.html`)**: Generated directly from the exact Web UI pipelines and quantitative metrics on the default sample images.

## Repository Structure

```text
CSST_106/
├── app.py                         # Main Streamlit Web UI entry point
├── generate_report.py             # Generates LAB_REPORT.md, .docx, and .html
├── self_test.py                   # End-to-end automated verification across Labs 1–7
├── requirements.txt               # Python package dependencies
├── LAB_REPORT.md                  # MS-Word-ready compiled laboratory report
├── CSST106_Lab_Report.docx        # Printable Microsoft Word report with embedded figures
├── CSST106_Lab_Report.html        # Printable HTML laboratory report
├── yolo11n.pt                     # Pre-trained Ultralytics YOLOv11n weights
├── labs/
│   ├── lab1_images.py             # Lab 1: Digital Images, BGR/RGB, Pixel Explorer, Crop/Resize/Rotate
│   ├── lab2_edges.py              # Lab 2 (Manual Ex 3): Sobel X/Y/Magnitude & Canny Edge Detection
│   ├── lab3_histogram.py          # Lab 3: Histogram Equalization (Gray & YUV), Linear Stretch, CLAHE
│   ├── lab4_surf.py               # Lab 4: SURF (with DoH+Haar fallback), U-SURF, SIFT & ORB Benchmark
│   ├── lab5_pca.py                # Lab 5: PCA Eigenfaces, RBF SVM, Variance k90 & Failure Analysis
│   ├── lab6_yolo.py               # Lab 6 (Manual Lab 5/6): YOLOv11n Inference, Custom Dataset & Train/Val
│   └── lab7_segmentation.py       # Lab 7 (Manual Lab 6): Otsu/Adaptive, Region Growing, Watershed, Canny, DeepLabV3
├── utils/
│   ├── cv_helpers.py              # Shared BGR/RGB, image loading, entropy & uint8 helpers
│   ├── sample_generator.py        # Deterministic sample image & custom YOLO dataset generator
│   └── report_generator.py        # Hard-copy report generator (.md, .docx, .html)
└── assets/
    ├── samples/                   # Bundled test images + cached LFW dataset
    ├── custom_yolo_dataset/       # Mini 3-class YOLO dataset (images/, labels/, data.yaml)
    └── report_figures/            # Figures embedded into .docx and .html reports
```

## Setup & Execution

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
2. **Launch the unified Streamlit Web UI:**
   ```bash
   streamlit run app.py
   ```
3. **Regenerate laboratory reports (`.md`, `.docx`, `.html`):**
   ```bash
   python generate_report.py
   ```
4. **Run automated self-tests across all 7 modules:**
   ```bash
   python self_test.py
   ```

## Laboratory Module Summary

| Module | Topic | Key Techniques Implemented |
| :--- | :--- | :--- |
| **Lab 1** | Image Manipulation | `cv2.split`, BGR vs. RGB, Grayscale, Pixel Inspector/Patch Edit, ROI Crop, Resize (`INTER_*`), Rotate |
| **Lab 2** | Edge Detection | `cv2.GaussianBlur`, `cv2.Sobel` (`CV_64F` → `uint8`), `cv2.magnitude`, `cv2.Canny` |
| **Lab 3** | Histogram & Contrast | `cv2.equalizeHist` (Gray & YUV `Y`), Percentile Contrast Stretch, `cv2.createCLAHE`, Histograms & CDFs |
| **Lab 4** | Feature Detection | Safe `SURF` + U-SURF (`DoH + 64-D Haar` fallback), `SIFT`, `ORB`, `BFMatcher` + Lowe's Ratio Test |
| **Lab 5** | PCA Eigenfaces | LFW Dataset, `PCA(n_components=150, whiten=True)`, `SVC(kernel='rbf')`, `k90` Variance, Failure Gallery |
| **Lab 6** | YOLOv11 Detection | Pre-trained `yolo11n.pt` (Image/Video/Webcam), YOLO `.txt` & `data.yaml` Verifier, `model.train()` & `model.val()` |
| **Lab 7** | Image Segmentation | Global Otsu, Adaptive Mean/Gaussian, Region Growing, `cv2.watershed`, Canny Contours, `DeepLabV3-MobileNetV3`, IoU |
