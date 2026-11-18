"""Sample data and mini YOLO dataset generator for CSST106 Computer Vision Labs."""

from __future__ import annotations

from pathlib import Path
from typing import Dict
import cv2
import numpy as np
import yaml
from skimage import data as skdata

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ASSETS_DIR = PROJECT_ROOT / "assets"
SAMPLES_DIR = ASSETS_DIR / "samples"
CUSTOM_YOLO_DIR = ASSETS_DIR / "custom_yolo_dataset"


def _to_bgr(rgb_img: np.ndarray) -> np.ndarray:
    """Convert an RGB numpy array to OpenCV BGR uint8."""
    return cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)


def _generate_lab1_sample(out_path: Path) -> None:
    """Create rich 640x480 color scene combining astronaut photo and BGR/RGB swatches."""
    canvas = np.full((480, 640, 3), 240, dtype=np.uint8)
    astro = _to_bgr(skdata.astronaut())
    astro_resized = cv2.resize(astro, (380, 380), interpolation=cv2.INTER_AREA)
    canvas[60:440, 20:400] = astro_resized

    # Header bar
    cv2.rectangle(canvas, (0, 0), (640, 48), (32, 42, 58), -1)
    cv2.putText(
        canvas,
        "CSST106 LAB 1: DIGITAL IMAGE & BGR CHANNELS",
        (22, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    # Pure BGR swatches on the right panel
    swatches = [
        ("PURE RED (RGB)", (0, 0, 255)),
        ("PURE GREEN (RGB)", (0, 255, 0)),
        ("PURE BLUE (RGB)", (255, 0, 0)),
        ("YELLOW (R+G)", (0, 255, 255)),
        ("CYAN (G+B)", (255, 255, 0)),
        ("MAGENTA (R+B)", (255, 0, 255)),
    ]
    y_start = 60
    for idx, (label, bgr) in enumerate(swatches):
        y1 = y_start + idx * 64
        y2 = y1 + 52
        cv2.rectangle(canvas, (420, y1), (620, y2), bgr, -1)
        cv2.rectangle(canvas, (420, y1), (620, y2), (40, 40, 40), 1)
        text_color = (20, 20, 20) if sum(bgr) > 400 else (255, 255, 255)
        cv2.putText(
            canvas,
            label,
            (430, y1 + 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            text_color,
            1,
            cv2.LINE_AA,
        )

    cv2.imwrite(str(out_path), canvas)


def _generate_lab2_sample(out_path: Path) -> None:
    """Create 768x512 high-detail test image for Sobel and Canny edge detection."""
    cam = skdata.camera()  # 512x512 grayscale
    panel = np.full((512, 256), 230, dtype=np.uint8)

    # Checkerboard pattern
    tile = 32
    for r in range(0, 192, tile):
        for c in range(0, 192, tile):
            if ((r // tile) + (c // tile)) % 2 == 0:
                panel[24 + r : 24 + r + tile, 32 + c : 32 + c + tile] = 35

    # Crisp geometric shapes
    cv2.circle(panel, (92, 310), 54, 45, 3, cv2.LINE_AA)
    cv2.circle(panel, (92, 310), 28, 110, -1, cv2.LINE_AA)
    pts = np.array([[160, 255], [232, 360], [140, 365]], dtype=np.int32)
    cv2.polylines(panel, [pts], True, 25, 3, cv2.LINE_AA)
    cv2.rectangle(panel, (32, 395), (224, 480), 40, 3)
    cv2.line(panel, (32, 438), (224, 438), 75, 2)
    cv2.line(panel, (128, 395), (128, 480), 75, 2)

    combined = np.hstack([cam, panel])
    bgr = cv2.cvtColor(combined, cv2.COLOR_GRAY2BGR)
    cv2.imwrite(str(out_path), bgr)


def _generate_lab3_samples(gray_path: Path, color_path: Path) -> None:
    """Create low-contrast unevenly illuminated grayscale and color images."""
    # Grayscale: coins with compressed dynamic range and smooth illumination gradient
    coins = skdata.coins().astype(np.float32)
    coins_resized = cv2.resize(coins, (512, 400), interpolation=cv2.INTER_AREA)
    norm_coins = (coins_resized - coins_resized.min()) / (coins_resized.max() - coins_resized.min() + 1e-6)
    h, w = norm_coins.shape
    grad_x = np.linspace(0.75, 1.25, w, dtype=np.float32)[None, :]
    grad_y = np.linspace(0.85, 1.15, h, dtype=np.float32)[:, None]
    low_gray = np.clip(62.0 + norm_coins * 72.0 * grad_x * grad_y, 35, 175).astype(np.uint8)
    cv2.imwrite(str(gray_path), low_gray)

    # Color: coffee cup compressed toward foggy low-contrast range
    coffee = _to_bgr(skdata.coffee()).astype(np.float32)
    coffee_resized = cv2.resize(coffee, (600, 400), interpolation=cv2.INTER_AREA)
    low_color = np.clip(68.0 + coffee_resized * 0.36, 0, 255).astype(np.uint8)
    cv2.imwrite(str(color_path), low_color)


def _generate_lab4_samples(ref_path: Path, scene_path: Path) -> None:
    """Create reference object image and rotated/scaled clutter target scene."""
    coffee = _to_bgr(skdata.coffee())
    # Crisp 280x220 reference card from textured coffee image + geometric fiducials
    ref = cv2.resize(coffee, (280, 220), interpolation=cv2.INTER_AREA)
    cv2.rectangle(ref, (6, 6), (274, 214), (245, 245, 245), 2)
    cv2.putText(
        ref,
        "CSST106 REF",
        (16, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 240, 180),
        2,
        cv2.LINE_AA,
    )
    cv2.imwrite(str(ref_path), ref)

    # Textured brick background 640x480
    brick = skdata.brick()
    brick_bgr = cv2.cvtColor(cv2.resize(brick, (640, 480)), cv2.COLOR_GRAY2BGR)
    brick_bgr = cv2.convertScaleAbs(brick_bgr, alpha=0.65, beta=25)

    # Rotate (-22 deg) and scale (0.88x) the reference object cleanly into the scene
    h_ref, w_ref = ref.shape[:2]
    scene = brick_bgr.copy()
    mask_ref = np.full((h_ref, w_ref), 255, dtype=np.uint8)

    # Place centered at (340, 245) in 640x480 scene
    center_src = (w_ref / 2.0, h_ref / 2.0)
    M = cv2.getRotationMatrix2D(center_src, -22.0, 0.88)
    M[0, 2] += 340.0 - center_src[0]
    M[1, 2] += 245.0 - center_src[1]

    warped_ref = cv2.warpAffine(ref, M, (640, 480), flags=cv2.INTER_LINEAR)
    warped_mask = cv2.warpAffine(mask_ref, M, (640, 480), flags=cv2.INTER_NEAREST)
    # Slight illumination change
    warped_ref = cv2.convertScaleAbs(warped_ref, alpha=1.05, beta=8)
    scene[warped_mask > 0] = warped_ref[warped_mask > 0]
    cv2.imwrite(str(scene_path), scene)


def _generate_lfw_cache(cache_path: Path) -> None:
    """Fetch LFW dataset via scikit-learn and cache locally, with deterministic fallback."""
    try:
        from sklearn.datasets import fetch_lfw_people

        lfw = fetch_lfw_people(min_faces_per_person=70, resize=0.4)
        np.savez_compressed(
            cache_path,
            images=lfw.images.astype(np.float32),
            data=lfw.data.astype(np.float32),
            target=lfw.target.astype(np.int64),
            target_names=lfw.target_names,
            is_synthetic=np.array([False]),
        )
        return
    except Exception:
        pass

    # Deterministic synthetic face-like fallback (7 classes, ~1288 samples, 50x37)
    rng = np.random.default_rng(42)
    h, w = 50, 37
    class_names = np.array(
        [
            "Ariel Sharon",
            "Colin Powell",
            "Donald Rumsfeld",
            "George W Bush",
            "Gerhard Schroeder",
            "Hugo Chavez",
            "Tony Blair",
        ]
    )
    counts = [77, 236, 121, 530, 109, 71, 144]
    yy, xx = np.mgrid[0:h, 0:w]
    yy_n = (yy - h / 2.0) / (h / 2.0)
    xx_n = (xx - w / 2.0) / (w / 2.0)
    oval = (((xx_n / 0.78) ** 2 + (yy_n / 0.88) ** 2) <= 1.0).astype(np.float32)

    images_list = []
    targets_list = []
    for cls_idx, count in enumerate(counts):
        # Distinct prototype face geometry per subject
        eye_y = -0.18 + (cls_idx % 3) * 0.04
        eye_x = 0.28 + (cls_idx % 2) * 0.05
        brow_w = 0.65 + (cls_idx * 0.04)
        proto = np.full((h, w), 0.25, dtype=np.float32)
        proto += oval * (0.45 + 0.03 * cls_idx)
        # Eyes
        left_eye = np.exp(-(((xx_n + eye_x) ** 2 + (yy_n - eye_y) ** 2) / 0.014))
        right_eye = np.exp(-(((xx_n - eye_x) ** 2 + (yy_n - eye_y) ** 2) / 0.014))
        nose = np.exp(-((xx_n ** 2) / 0.008 + ((yy_n - 0.12) ** 2) / 0.05))
        mouth = np.exp(-((xx_n ** 2) / (0.06 * brow_w) + ((yy_n - 0.45) ** 2) / 0.009))
        proto = proto - 0.32 * (left_eye + right_eye) + 0.15 * nose - 0.22 * mouth

        for _ in range(count):
            illum = 1.0 + rng.normal(0.0, 0.18) * xx_n + rng.normal(0.0, 0.12) * yy_n
            shift_x = int(rng.integers(-2, 3))
            shift_y = int(rng.integers(-2, 3))
            face = np.roll(proto, shift=(shift_y, shift_x), axis=(0, 1))
            face = face * illum + rng.normal(0.0, 0.04, size=(h, w))
            face = np.clip(face, 0.0, 1.0).astype(np.float32)
            images_list.append(face)
            targets_list.append(cls_idx)

    images = np.stack(images_list, axis=0)
    data = images.reshape(len(images), h * w)
    target = np.array(targets_list, dtype=np.int64)
    perm = rng.permutation(len(images))
    np.savez_compressed(
        cache_path,
        images=images[perm],
        data=data[perm],
        target=target[perm],
        target_names=class_names,
        is_synthetic=np.array([True]),
    )


def _generate_lab6_scene_and_dataset(scene_path: Path, dataset_dir: Path) -> None:
    """Create YOLOv11 default detection scene and valid custom YOLO dataset."""
    astro = cv2.resize(_to_bgr(skdata.astronaut()), (300, 300), interpolation=cv2.INTER_AREA)
    cat = cv2.resize(_to_bgr(skdata.chelsea()), (300, 210), interpolation=cv2.INTER_AREA)
    coffee = cv2.resize(_to_bgr(skdata.coffee()), (300, 210), interpolation=cv2.INTER_AREA)

    # Composite 960x420 natural scene for Part 1 detection
    scene = np.full((420, 960, 3), 228, dtype=np.uint8)
    scene[60:360, 20:320] = astro
    scene[105:315, 340:640] = cat
    scene[105:315, 650:950] = coffee
    cv2.imwrite(str(scene_path), scene)

    # Custom YOLO dataset: 8 train, 4 val images (640x640)
    for split in ("train", "val"):
        (dataset_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (dataset_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    crops = [
        (0, astro),   # person
        (1, cat),     # cat
        (2, coffee),  # cup
    ]
    rng = np.random.default_rng(106)

    def _make_split(split_name: str, num_imgs: int) -> None:
        img_size = 640
        for idx in range(1, num_imgs + 1):
            bg_val = int(rng.integers(170, 240))
            canvas = np.full((img_size, img_size, 3), bg_val, dtype=np.uint8)
            # Subtle gradient background
            grad = np.linspace(-20, 20, img_size, dtype=np.int16)[None, :, None]
            canvas = np.clip(canvas.astype(np.int16) + grad, 0, 255).astype(np.uint8)

            # Place 2 or 3 objects into distinct quadrants to prevent occlusion
            quadrants = [(20, 20), (330, 20), (20, 330), (330, 330)]
            rng.shuffle(quadrants)
            chosen_classes = rng.choice([0, 1, 2], size=int(rng.integers(2, 4)), replace=False)

            label_lines = []
            for q_idx, cls_id in enumerate(chosen_classes):
                _, patch = crops[int(cls_id)]
                scale = float(rng.uniform(0.68, 0.92))
                new_w = int(patch.shape[1] * scale)
                new_h = int(patch.shape[0] * scale)
                resized = cv2.resize(patch, (new_w, new_h), interpolation=cv2.INTER_AREA)
                if rng.random() > 0.5:
                    resized = cv2.flip(resized, 1)

                qx, qy = quadrants[q_idx]
                max_dx = max(1, 290 - new_w)
                max_dy = max(1, 290 - new_h)
                x1 = min(img_size - new_w - 4, qx + int(rng.integers(0, max_dx)))
                y1 = min(img_size - new_h - 4, qy + int(rng.integers(0, max_dy)))
                x2 = x1 + new_w
                y2 = y1 + new_h
                canvas[y1:y2, x1:x2] = resized

                xc = ((x1 + x2) / 2.0) / img_size
                yc = ((y1 + y2) / 2.0) / img_size
                bw = new_w / img_size
                bh = new_h / img_size
                label_lines.append(f"{cls_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")

            stem = f"{split_name}_{idx:02d}"
            cv2.imwrite(str(dataset_dir / "images" / split_name / f"{stem}.jpg"), canvas)
            (dataset_dir / "labels" / split_name / f"{stem}.txt").write_text(
                "\n".join(label_lines) + "\n", encoding="utf-8"
            )

    _make_split("train", 8)
    _make_split("val", 4)

    data_yaml = {
        "path": "assets/custom_yolo_dataset",
        "train": "images/train",
        "val": "images/val",
        "nc": 3,
        "names": {0: "person", 1: "cat", 2: "cup"},
    }
    (dataset_dir / "data.yaml").write_text(
        yaml.safe_dump(data_yaml, sort_keys=False), encoding="utf-8"
    )


def _generate_lab7_samples(coins_path: Path, gt_path: Path, semantic_path: Path) -> None:
    """Create touching coins segmentation image + GT mask and DeepLabV3 natural scene."""
    h, w = 440, 520
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    # Non-uniform dark-to-medium background illumination (challenges global thresholding)
    bg = 48.0 + 52.0 * (xx / w) + 22.0 * (yy / h)
    rng = np.random.default_rng(77)
    bg += rng.normal(0.0, 4.0, size=(h, w))
    img = np.clip(bg, 0, 255).astype(np.uint8)
    gt = np.zeros((h, w), dtype=np.uint8)

    # Pairs of touching/slightly overlapping coins and isolated coins: (cx, cy, radius, intensity)
    coins_spec = [
        (105, 110, 44, 205),
        (182, 118, 42, 212),  # touches coin 1
        (330, 105, 45, 222),
        (410, 125, 40, 228),  # touches coin 3
        (120, 260, 46, 208),
        (255, 225, 42, 218),
        (330, 235, 43, 225),  # touches coin 6
        (205, 345, 44, 210),
        (415, 330, 48, 232),
    ]
    for cx, cy, rad, val in coins_spec:
        dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
        inside = dist <= rad
        # Smooth radial texture like real metallic coins
        shading = val - 26.0 * (dist / rad) ** 2 + rng.normal(0.0, 5.0, size=(h, w))
        img[inside] = np.clip(shading[inside], 140, 252).astype(np.uint8)
        gt[inside] = 255

    img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.imwrite(str(coins_path), img_bgr)
    cv2.imwrite(str(gt_path), gt)

    # Complex natural scene for DeepLabV3 semantic segmentation (person + cat)
    astro = cv2.resize(_to_bgr(skdata.astronaut()), (320, 360), interpolation=cv2.INTER_AREA)
    cat = cv2.resize(_to_bgr(skdata.chelsea()), (320, 240), interpolation=cv2.INTER_AREA)
    sem_scene = np.full((400, 680, 3), 215, dtype=np.uint8)
    sem_scene[20:380, 15:335] = astro
    sem_scene[80:320, 345:665] = cat
    cv2.imwrite(str(semantic_path), sem_scene)


def ensure_all_samples(force: bool = False) -> Dict[str, Path]:
    """Ensure all bundled sample images, cached LFW data, and mini YOLO dataset exist."""
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    CUSTOM_YOLO_DIR.mkdir(parents=True, exist_ok=True)

    paths = {
        "lab1_color": SAMPLES_DIR / "lab1_color_scene.png",
        "lab2_edges": SAMPLES_DIR / "lab2_architecture_edges.png",
        "lab3_gray": SAMPLES_DIR / "lab3_low_contrast.png",
        "lab3_color": SAMPLES_DIR / "lab3_color_low_contrast.png",
        "lab4_ref": SAMPLES_DIR / "lab4_object_ref.png",
        "lab4_scene": SAMPLES_DIR / "lab4_scene_target.png",
        "lab5_lfw": SAMPLES_DIR / "lfw_cached.npz",
        "lab6_scene": SAMPLES_DIR / "lab6_street_scene.png",
        "lab6_yaml": CUSTOM_YOLO_DIR / "data.yaml",
        "lab7_coins": SAMPLES_DIR / "lab7_coins_touching.png",
        "lab7_gt": SAMPLES_DIR / "lab7_coins_gt.png",
        "lab7_semantic": SAMPLES_DIR / "lab7_semantic_scene.png",
    }

    if force or not paths["lab1_color"].exists():
        _generate_lab1_sample(paths["lab1_color"])
    if force or not paths["lab2_edges"].exists():
        _generate_lab2_sample(paths["lab2_edges"])
    if force or not (paths["lab3_gray"].exists() and paths["lab3_color"].exists()):
        _generate_lab3_samples(paths["lab3_gray"], paths["lab3_color"])
    if force or not (paths["lab4_ref"].exists() and paths["lab4_scene"].exists()):
        _generate_lab4_samples(paths["lab4_ref"], paths["lab4_scene"])
    if force or not paths["lab5_lfw"].exists():
        _generate_lfw_cache(paths["lab5_lfw"])
    if force or not (paths["lab6_scene"].exists() and paths["lab6_yaml"].exists()):
        _generate_lab6_scene_and_dataset(paths["lab6_scene"], CUSTOM_YOLO_DIR)
    if force or not (
        paths["lab7_coins"].exists() and paths["lab7_gt"].exists() and paths["lab7_semantic"].exists()
    ):
        _generate_lab7_samples(paths["lab7_coins"], paths["lab7_gt"], paths["lab7_semantic"])

    return paths
