"""Lab Activity 6 (Manual Lab 5/6): Object Detection with YOLOv11."""

from __future__ import annotations

import os
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
import cv2
import numpy as np
import pandas as pd
import streamlit as st
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
_YOLO_CFG_DIR = PROJECT_ROOT / "assets" / ".ultralytics_cfg"
_YOLO_CFG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("YOLO_CONFIG_DIR", str(_YOLO_CFG_DIR))

from ultralytics import YOLO  # noqa: E402
from utils.cv_helpers import bgr_to_rgb, load_bgr_image  # noqa: E402

CUSTOM_CLASS_COLORS = {
    0: (255, 80, 80),   # person
    1: (60, 200, 90),   # cat
    2: (60, 140, 255),  # cup
}


@st.cache_resource(show_spinner="Loading YOLOv11n Model...")
def load_yolo_model(weights_path: str = "yolo11n.pt") -> YOLO:
    """Load and cache Ultralytics YOLOv11 model weights."""
    return YOLO(weights_path)


def _resolve_custom_yaml(dataset_dir: Path) -> Path:
    """Write a runtime YAML with absolute dataset path so Ultralytics resolves reliably."""
    resolved_yaml = _YOLO_CFG_DIR / "resolved_custom_data.yaml"
    cfg = {
        "path": str(dataset_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "nc": 3,
        "names": {0: "person", 1: "cat", 2: "cup"},
    }
    resolved_yaml.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return resolved_yaml


def run_lab6_inference(
    img_bgr: np.ndarray,
    conf: float = 0.25,
    iou: float = 0.45,
    weights_path: str = "yolo11n.pt",
) -> Dict[str, Any]:
    """Run YOLOv11n inference on a single BGR frame and extract detections table."""
    model = load_yolo_model(weights_path)
    t0 = time.perf_counter()
    results = model.predict(source=img_bgr, conf=float(conf), iou=float(iou), verbose=False)
    infer_ms = (time.perf_counter() - t0) * 1000.0

    res0 = results[0]
    annotated_bgr = res0.plot()
    annotated_rgb = bgr_to_rgb(annotated_bgr)

    rows: List[Dict[str, Any]] = []
    if res0.boxes is not None and len(res0.boxes) > 0:
        for idx, box in enumerate(res0.boxes, start=1):
            cls_id = int(box.cls[0])
            cls_name = str(model.names.get(cls_id, f"class_{cls_id}"))
            conf_val = float(box.conf[0])
            x1, y1, x2, y2 = [round(float(v), 1) for v in box.xyxy[0].tolist()]
            rows.append(
                {
                    "Detection #": idx,
                    "Class": cls_name,
                    "Confidence": round(conf_val, 3),
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "Box Size (W×H)": f"{int(round(x2 - x1))}×{int(round(y2 - y1))}",
                }
            )

    return {
        "annotated_rgb": annotated_rgb,
        "detections_df": pd.DataFrame(rows),
        "num_detections": len(rows),
        "infer_ms": infer_ms,
    }


def draw_yolo_ground_truth(image_path: Path, label_path: Path, class_names: Dict[int, str]) -> tuple[np.ndarray, pd.DataFrame]:
    """Draw normalized YOLO .txt annotations on an image for visual verification."""
    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read dataset image: {image_path}")
    h, w = img.shape[:2]
    ann_rows = []
    if label_path.exists():
        lines = [ln.strip() for ln in label_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
        for ln in lines:
            parts = ln.split()
            cls_id = int(parts[0])
            xc, yc, bw, bh = [float(v) for v in parts[1:5]]
            x1 = int(round((xc - bw / 2.0) * w))
            y1 = int(round((yc - bh / 2.0) * h))
            x2 = int(round((xc + bw / 2.0) * w))
            y2 = int(round((yc + bh / 2.0) * h))
            color = CUSTOM_CLASS_COLORS.get(cls_id, (0, 255, 255))
            name = class_names.get(cls_id, str(cls_id))
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
            cv2.putText(
                img,
                f"{cls_id}: {name}",
                (x1 + 4, max(22, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2,
                cv2.LINE_AA,
            )
            ann_rows.append(
                {
                    "class_id": cls_id,
                    "class_name": name,
                    "x_center": round(xc, 4),
                    "y_center": round(yc, 4),
                    "width": round(bw, 4),
                    "height": round(bh, 4),
                }
            )
    return bgr_to_rgb(img), pd.DataFrame(ann_rows)


def _build_dataset_tree(dataset_dir: Path) -> str:
    """Format directory tree string matching the Lab 6 manual specification."""
    lines = [f"{dataset_dir.name}/"]
    for sub in ("images", "labels"):
        lines.append(f"├── {sub}/")
        for split in ("train", "val"):
            files = sorted((dataset_dir / sub / split).glob("*"))
            branch = "└──" if split == "val" else "├──"
            lines.append(f"│   {branch} {split}/ ({len(files)} files)")
    lines.append("└── data.yaml")
    return "\n".join(lines)


def run_lab6_train_and_val(
    dataset_dir: Path,
    epochs: int = 3,
    imgsz: int = 640,
) -> Dict[str, Any]:
    """Train YOLOv11n on the custom mini dataset and run validation metrics."""
    resolved_yaml = _resolve_custom_yaml(dataset_dir)
    runs_project = str((PROJECT_ROOT / "runs" / "detect").resolve())
    model = YOLO(str(PROJECT_ROOT / "yolo11n.pt"))

    t0 = time.perf_counter()
    train_res = model.train(
        data=str(resolved_yaml),
        epochs=int(epochs),
        imgsz=int(imgsz),
        batch=4,
        project=runs_project,
        name="custom_train",
        exist_ok=True,
        verbose=False,
        plots=False,
    )
    train_sec = time.perf_counter() - t0

    save_dir = Path(getattr(train_res, "save_dir", getattr(model.trainer, "save_dir", runs_project)))
    best_pt = save_dir / "weights" / "best.pt"
    eval_weights = str(best_pt) if best_pt.exists() else str(PROJECT_ROOT / "yolo11n.pt")
    eval_model = YOLO(eval_weights)
    val_res = eval_model.val(
        data=str(resolved_yaml),
        imgsz=int(imgsz),
        project=runs_project,
        name="custom_val",
        exist_ok=True,
        verbose=False,
        plots=False,
    )
    rd = val_res.results_dict
    return {
        "best_pt": eval_weights,
        "train_sec": train_sec,
        "precision": float(rd.get("metrics/precision(B)", 0.0)),
        "recall": float(rd.get("metrics/recall(B)", 0.0)),
        "map50": float(rd.get("metrics/mAP50(B)", 0.0)),
        "map50_95": float(rd.get("metrics/mAP50-95(B)", 0.0)),
    }


def render(sample_paths: Dict[str, Path]) -> None:
    """Render Lab Activity 6 Streamlit UI."""
    st.header("Lab Activity 6 (Manual Lab 5/6): Object Detection with YOLOv11")
    st.caption("Course: CSST106 | Topic: YOLOv11 Object Detection | Week No.: 8 – 9")

    with st.expander("Objective & Procedure Overview", expanded=False):
        st.markdown(
            """
            **Learning Outcomes:** Run real-time inference with pre-trained **YOLOv11n** (`yolo11n.pt`),
            inspect YOLO-format normalized dataset annotations and `data.yaml`, fine-tune YOLOv11 on a custom
            dataset (`person`, `cat`, `cup`), and evaluate Precision, Recall, mAP@50, and mAP@50-95.

            **Workflow:**
            1. **Part 1:** Pre-trained `YOLOv11n` inference on scenes, uploaded images, uploaded videos, or camera snapshots.
            2. **Part 2:** Explore `assets/custom_yolo_dataset/` structure, `data.yaml`, and normalized bounding boxes.
            3. **Part 3 & 4:** Configure `epochs` & `imgsz`, execute training and validation (`model.val()`), and inspect metrics.
            """
        )

    part1_tab, part2_tab, part3_tab = st.tabs(
        [
            "Part 1: Pre-trained YOLOv11n Inference",
            "Part 2: Custom Dataset & Annotation Explorer",
            "Part 3 & 4: Training, Validation & Evaluation",
        ]
    )

    # ---------------- PART 1 ----------------
    with part1_tab:
        c_s1, c_s2 = st.columns([1.2, 1.0])
        with c_s1:
            input_mode = st.radio(
                "Input Source",
                ["Bundled Scene / Uploaded Image", "Webcam Snapshot", "Uploaded Video (MP4/AVI)"],
                horizontal=True,
                key="lab6_input_mode",
            )
        with c_s2:
            conf_col, iou_col = st.columns(2)
            with conf_col:
                conf = st.slider("Confidence Threshold", 0.05, 0.95, 0.25, step=0.05, key="lab6_conf")
            with iou_col:
                iou = st.slider("IoU NMS Threshold", 0.10, 0.95, 0.45, step=0.05, key="lab6_iou")

        if input_mode == "Bundled Scene / Uploaded Image":
            up_img = st.file_uploader(
                "Upload custom image (optional — bundled multi-object scene pre-loaded)",
                type=["png", "jpg", "jpeg", "bmp"],
                key="lab6_up_img",
            )
            img_bgr, src_label = load_bgr_image(sample_paths["lab6_scene"], uploaded_file=up_img)
            res = run_lab6_inference(img_bgr, conf=conf, iou=iou, weights_path=str(PROJECT_ROOT / "yolo11n.pt"))

            st.info(f"**Active Input:** {src_label} | **Model:** `YOLOv11n (yolo11n.pt)` | **Inference Latency:** `{res['infer_ms']:.1f} ms`")
            v1, v2 = st.columns(2)
            with v1:
                st.image(bgr_to_rgb(img_bgr), caption="Input Scene", use_container_width=True)
            with v2:
                st.image(
                    res["annotated_rgb"],
                    caption=f"YOLOv11n Detections ({res['num_detections']} objects)",
                    use_container_width=True,
                )

            if res["num_detections"] > 0:
                st.dataframe(res["detections_df"], use_container_width=True, hide_index=True)
            else:
                st.warning("No objects detected above the current Confidence Threshold. Try lowering `conf`.")

        elif input_mode == "Webcam Snapshot":
            cam = st.camera_input("Capture frame from webcam for YOLOv11n detection", key="lab6_cam")
            if cam is not None:
                img_bgr, src_label = load_bgr_image(sample_paths["lab6_scene"], camera_file=cam)
                res = run_lab6_inference(img_bgr, conf=conf, iou=iou, weights_path=str(PROJECT_ROOT / "yolo11n.pt"))
                st.image(
                    res["annotated_rgb"],
                    caption=f"Webcam YOLOv11n Output ({res['num_detections']} objects, {res['infer_ms']:.1f} ms)",
                    use_container_width=True,
                )
                if res["num_detections"] > 0:
                    st.dataframe(res["detections_df"], use_container_width=True, hide_index=True)
            else:
                st.info("Take a snapshot above to run YOLOv11n detection on your webcam frame.")

        else:
            up_vid = st.file_uploader("Upload a short video file", type=["mp4", "avi", "mov"], key="lab6_vid")
            max_frames = st.slider("Sampled Frames to Inspect", 1, 8, 4, key="lab6_vid_frames")
            if up_vid is not None:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
                    tmp.write(up_vid.getvalue())
                    tmp_path = tmp.name
                cap = cv2.VideoCapture(tmp_path)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
                step = max(1, total_frames // max_frames)
                cols = st.columns(min(2, max_frames))
                shown = 0
                frame_idx = 0
                all_rows = []
                while cap.isOpened() and shown < max_frames:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    if frame_idx % step == 0:
                        f_res = run_lab6_inference(frame, conf=conf, iou=iou, weights_path=str(PROJECT_ROOT / "yolo11n.pt"))
                        with cols[shown % len(cols)]:
                            st.image(
                                f_res["annotated_rgb"],
                                caption=f"Frame {frame_idx} ({f_res['num_detections']} dets, {f_res['infer_ms']:.1f} ms)",
                                use_container_width=True,
                            )
                        if not f_res["detections_df"].empty:
                            df_tmp = f_res["detections_df"].copy()
                            df_tmp.insert(0, "Frame", frame_idx)
                            all_rows.append(df_tmp)
                        shown += 1
                    frame_idx += 1
                cap.release()
                if all_rows:
                    st.dataframe(pd.concat(all_rows, ignore_index=True), use_container_width=True, hide_index=True)
            else:
                st.info("Upload an MP4/AVI/MOV clip to inspect sampled YOLOv11n video frame detections.")

    # ---------------- PART 2 ----------------
    with part2_tab:
        dataset_dir = sample_paths["lab6_yaml"].parent
        yaml_text = sample_paths["lab6_yaml"].read_text(encoding="utf-8")
        yaml_cfg = yaml.safe_load(yaml_text)
        class_names = {int(k): v for k, v in yaml_cfg["names"].items()}

        t_c1, t_c2 = st.columns(2)
        with t_c1:
            st.markdown("**Dataset Directory Structure**")
            st.code(_build_dataset_tree(dataset_dir), language="text")
        with t_c2:
            st.markdown("**`data.yaml` Configuration**")
            st.code(yaml_text, language="yaml")

        st.markdown("**Interactive Ground-Truth Bounding Box Verifier**")
        split_choice = st.selectbox("Select Dataset Split", ["train", "val"], key="lab6_split_view")
        imgs_list = sorted((dataset_dir / "images" / split_choice).glob("*.jpg"))
        chosen_img_path = st.selectbox(
            "Select Sample Image",
            imgs_list,
            format_func=lambda p: p.name,
            key="lab6_sample_select",
        )
        chosen_lbl_path = dataset_dir / "labels" / split_choice / f"{chosen_img_path.stem}.txt"
        gt_vis_rgb, gt_df = draw_yolo_ground_truth(chosen_img_path, chosen_lbl_path, class_names)

        v_c1, v_c2 = st.columns([1.15, 1.0])
        with v_c1:
            st.image(gt_vis_rgb, caption=f"Ground Truth Boxes: {chosen_img_path.name}", use_container_width=True)
        with v_c2:
            st.markdown(f"**Annotation File (`{chosen_lbl_path.name}`)**")
            st.code(chosen_lbl_path.read_text(encoding="utf-8"), language="text")
            st.dataframe(gt_df, use_container_width=True, hide_index=True)

    # ---------------- PART 3 & 4 ----------------
    with part3_tab:
        dataset_dir = sample_paths["lab6_yaml"].parent
        st.markdown("**Training & Validation Commands (Lab Manual Reference)**")
        code_c1, code_c2 = st.columns(2)
        with code_c1:
            st.code(
                "# CLI Training & Validation\n"
                "yolo task=detect mode=train model=yolo11n.pt data=data.yaml epochs=100 imgsz=640\n"
                "yolo task=detect mode=val model=runs/detect/train/weights/best.pt data=data.yaml",
                language="bash",
            )
        with code_c2:
            st.code(
                "# Python API\n"
                "from ultralytics import YOLO\n"
                "model = YOLO('yolo11n.pt')\n"
                "model.train(data='data.yaml', epochs=100, imgsz=640)\n"
                "metrics = model.val()",
                language="python",
            )

        st.markdown("**Interactive Custom Training & Validation Runner**")
        tr_c1, tr_c2, tr_c3 = st.columns([1, 1, 1.2])
        with tr_c1:
            epochs = st.slider("Epochs", 1, 100, 3, key="lab6_epochs")
        with tr_c2:
            imgsz = st.selectbox("Image Size (imgsz)", [640, 320], index=1, key="lab6_imgsz")
        with tr_c3:
            st.write("")
            run_train_btn = st.button("Train & Validate YOLOv11n on Custom Dataset", type="primary", key="lab6_train_btn")

        if run_train_btn:
            with st.spinner(f"Training YOLOv11n for {epochs} epochs (imgsz={imgsz}) and running model.val()..."):
                train_metrics = run_lab6_train_and_val(dataset_dir, epochs=epochs, imgsz=imgsz)
                st.session_state["lab6_train_metrics"] = train_metrics

        if "lab6_train_metrics" in st.session_state:
            tm = st.session_state["lab6_train_metrics"]
            st.success(f"Training & validation complete in `{tm['train_sec']:.2f}s`. Weights: `{Path(tm['best_pt']).name}`")
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Precision (B)", f"{tm['precision']:.4f}")
            k2.metric("Recall (B)", f"{tm['recall']:.4f}")
            k3.metric("mAP@50 (B)", f"{tm['map50']:.4f}")
            k4.metric("mAP@50-95 (B)", f"{tm['map50_95']:.4f}")

            # Run custom trained model inference on a validation image
            val_img_path = sorted((dataset_dir / "images" / "val").glob("*.jpg"))[0]
            val_bgr = cv2.imread(str(val_img_path), cv2.IMREAD_COLOR)
            custom_inf = run_lab6_inference(val_bgr, conf=0.15, iou=0.45, weights_path=tm["best_pt"])
            st.image(
                custom_inf["annotated_rgb"],
                caption=f"Trained Model Inference on {val_img_path.name} ({custom_inf['num_detections']} detections)",
                use_container_width=True,
            )
        else:
            st.info("Click **Train & Validate YOLOv11n on Custom Dataset** above to execute live fine-tuning and `model.val()` (default 3 epochs @ 320px runs in ~10s on CPU; raise up to 100 epochs / 640px for full convergence).")

    st.subheader("Required Post-Lab Question Answered")
    st.markdown(
        """
        **Question:** *What is object detection in computer vision, and how does it differ from image classification? Explain your answer using a real-world example.*

        - **Image Classification** assigns **one global category label** to an entire image (e.g., classifying a photo simply as `"cat"` or `"street"`), providing *zero spatial location* and failing when multiple distinct objects coexist.
        - **Object Detection** solves **both localization and multi-object classification simultaneously**: it predicts exact spatial **bounding boxes** `(x_min, y_min, x_max, y_max)` around every individual object instance in a scene along with each box's **class label** and **confidence score**.
        - **Real-World Example (Autonomous Vehicle / Smart Retail Checkout):**
          - In an **autonomous self-driving car** dashboard camera frame, image classification could only output `"road scene"`, which cannot guide steering or braking. Object detection (such as YOLOv11) simultaneously localizes `car` at `(120, 210, 310, 340)` (conf `0.94`), `pedestrian` stepping into a crosswalk at `(460, 185, 515, 360)` (conf `0.91`), and `traffic light` overhead — allowing the vehicle's control planner to measure exact distances and apply emergency braking.
          - Similarly, in an **automated retail checkout scanner**, object detection identifies and counts every item on the tray (`2× cup`, `1× bottle`, `1× apple`) by bounding box rather than predicting a single category for the whole tray.
        """
    )
