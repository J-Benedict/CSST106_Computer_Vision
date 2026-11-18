"""Lab Activity 4: SURF, SIFT, and ORB Feature Detection & Matching."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Tuple
import cv2
import numpy as np
import pandas as pd
import streamlit as st

from utils.cv_helpers import bgr_to_rgb, load_bgr_image


def _nms_3d_doh(response_pyr: List[np.ndarray], scales: List[float], threshold: float, max_kps: int = 1000):
    """3D non-maximum suppression across scale-space Determinant-of-Hessian maps."""
    keypoints_data = []
    n_scales = len(response_pyr)
    h, w = response_pyr[0].shape

    for s_idx in range(n_scales):
        cur = response_pyr[s_idx]
        # 2D 3x3 spatial local maximum mask
        dilated = cv2.dilate(cur, np.ones((3, 3), dtype=np.uint8))
        mask = (cur == dilated) & (cur >= float(threshold))
        if s_idx > 0:
            mask &= cur >= response_pyr[s_idx - 1]
        if s_idx < n_scales - 1:
            mask &= cur >= response_pyr[s_idx + 1]

        # Exclude 10px border
        mask[:10, :] = False
        mask[-10:, :] = False
        mask[:, :10] = False
        mask[:, -10:] = False

        ys, xs = np.where(mask)
        if len(xs) == 0:
            continue
        resps = cur[ys, xs]
        sigma = scales[s_idx]
        for x, y, r in zip(xs, ys, resps):
            keypoints_data.append((float(r), int(x), int(y), float(sigma), s_idx))

    keypoints_data.sort(key=lambda item: item[0], reverse=True)
    return keypoints_data[:max_kps]


def _compute_surf_approximation(
    gray: np.ndarray,
    hessian_threshold: float = 400.0,
    upright: bool = False,
) -> Tuple[List[cv2.KeyPoint], np.ndarray]:
    """
    Faithful Multi-Scale Determinant-of-Hessian (DoH) + 64-D Haar Wavelet SURF approximation
    used automatically when OpenCV's patented cv2.xfeatures2d.SURF_create is unavailable.
    """
    gray_f = gray.astype(np.float32) / 255.0
    scales = [1.4, 2.0, 2.8, 4.0, 5.6, 8.0]
    response_pyr: List[np.ndarray] = []

    for sigma in scales:
        ksize = int(2 * round(2.5 * sigma) + 1)
        blurred = cv2.GaussianBlur(gray_f, (ksize, ksize), sigmaX=sigma, sigmaY=sigma)
        # Second-order spatial derivatives scaled by sigma^2 (Lindeberg scale normalization)
        lxx = cv2.Sobel(blurred, cv2.CV_32F, 2, 0, ksize=3) * (sigma ** 2)
        lyy = cv2.Sobel(blurred, cv2.CV_32F, 0, 2, ksize=3) * (sigma ** 2)
        lxy = cv2.Sobel(blurred, cv2.CV_32F, 1, 1, ksize=3) * (sigma ** 2)
        # Bay et al. (2006) approximated Determinant of Hessian: det(H) = Dxx*Dyy - (0.9 * Dxy)^2
        det_h = (lxx * lyy) - ((0.9 * lxy) ** 2)
        # Scale factor calibrated so hessianThreshold=400 matches typical OpenCV SURF density
        response_pyr.append(np.maximum(det_h, 0.0) * 12000.0)

    candidates = _nms_3d_doh(response_pyr, scales, threshold=hessian_threshold, max_kps=1000)
    if not candidates:
        return [], np.empty((0, 64), dtype=np.float32)

    # First-order Haar-like gradients for orientation and 4x4x4 descriptor extraction
    dx_map = cv2.Sobel(gray_f, cv2.CV_32F, 1, 0, ksize=3)
    dy_map = cv2.Sobel(gray_f, cv2.CV_32F, 0, 1, ksize=3)
    h, w = gray.shape

    keypoints: List[cv2.KeyPoint] = []
    angles_rad = np.zeros(len(candidates), dtype=np.float32)

    for idx, (resp, cx, cy, sigma, oct_idx) in enumerate(candidates):
        if upright:
            angle_deg = 0.0
            angle_rad = 0.0
        else:
            rad_win = max(4, int(round(3.0 * sigma)))
            x0, x1 = max(0, cx - rad_win), min(w, cx + rad_win + 1)
            y0, y1 = max(0, cy - rad_win), min(h, cy + rad_win + 1)
            sum_dx = float(np.sum(dx_map[y0:y1, x0:x1]))
            sum_dy = float(np.sum(dy_map[y0:y1, x0:x1]))
            angle_rad = float(np.arctan2(sum_dy, sum_dx))
            angle_deg = float(np.degrees(angle_rad) % 360.0)

        angles_rad[idx] = angle_rad
        keypoints.append(
            cv2.KeyPoint(
                x=float(cx),
                y=float(cy),
                size=float(sigma * 9.0),
                angle=angle_deg,
                response=float(resp),
                octave=int(oct_idx),
            )
        )

    # Vectorized 64-D SURF descriptor: 4x4 subregions, 3x3 samples per subregion
    n_kps = len(candidates)
    xs_kp = np.array([c[1] for c in candidates], dtype=np.float32)
    ys_kp = np.array([c[2] for c in candidates], dtype=np.float32)
    sigmas = np.array([c[3] for c in candidates], dtype=np.float32)

    # Local grid offsets in units of sigma covering [-6, +6] across 4x4 subregions (3x3 samples each)
    sub_centers = np.linspace(-4.5, 4.5, 4, dtype=np.float32)
    sample_offsets = np.linspace(-1.0, 1.0, 3, dtype=np.float32)
    u_grid, v_grid = np.meshgrid(
        (sub_centers[:, None] + sample_offsets[None, :]).ravel(),
        (sub_centers[:, None] + sample_offsets[None, :]).ravel(),
    )  # shape (12, 12) -> reshape to (4, 3, 4, 3) -> (16, 9)
    u_sub = u_grid.reshape(4, 3, 4, 3).transpose(0, 2, 1, 3).reshape(16, 9)
    v_sub = v_grid.reshape(4, 3, 4, 3).transpose(0, 2, 1, 3).reshape(16, 9)

    cos_a = np.cos(angles_rad)[:, None, None]
    sin_a = np.sin(angles_rad)[:, None, None]
    scale_b = sigmas[:, None, None]

    # Sample coordinates (N, 16, 9)
    sx = xs_kp[:, None, None] + scale_b * (u_sub[None, :, :] * cos_a - v_sub[None, :, :] * sin_a)
    sy = ys_kp[:, None, None] + scale_b * (u_sub[None, :, :] * sin_a + v_sub[None, :, :] * cos_a)

    ix = np.clip(np.rint(sx).astype(np.int32), 0, w - 1)
    iy = np.clip(np.rint(sy).astype(np.int32), 0, h - 1)

    gx = dx_map[iy, ix]
    gy = dy_map[iy, ix]

    # Rotate Haar responses into keypoint local frame
    rx = gx * cos_a + gy * sin_a
    ry = -gx * sin_a + gy * cos_a

    # Accumulate [sum(dx), sum(dy), sum(|dx|), sum(|dy|)] per 16 subregions -> (N, 64)
    feat = np.stack(
        [
            np.sum(rx, axis=2),
            np.sum(ry, axis=2),
            np.sum(np.abs(rx), axis=2),
            np.sum(np.abs(ry), axis=2),
        ],
        axis=2,
    ).reshape(n_kps, 64)

    norms = np.linalg.norm(feat, axis=1, keepdims=True) + 1e-7
    descriptors = (feat / norms).astype(np.float32)
    return keypoints, descriptors


def detect_and_compute_surf(
    gray: np.ndarray,
    hessian_threshold: float = 400.0,
    upright: bool = False,
) -> Tuple[List[cv2.KeyPoint], np.ndarray, str]:
    """
    Safe SURF factory that attempts native cv2.xfeatures2d.SURF_create first
    and seamlessly falls back to the DoH + 64-D Haar Wavelet approximation if non-free is disabled.
    """
    try:
        surf = cv2.xfeatures2d.SURF_create(
            hessianThreshold=float(hessian_threshold),
            upright=bool(upright),
        )
        kps, desc = surf.detectAndCompute(gray, None)
        mode = "Native OpenCV xfeatures2d.SURF"
        if desc is None:
            desc = np.empty((0, 64), dtype=np.float32)
        return list(kps), desc, mode
    except (AttributeError, cv2.error):
        kps, desc = _compute_surf_approximation(
            gray,
            hessian_threshold=float(hessian_threshold),
            upright=bool(upright),
        )
        mode = "DoH + 64-D Haar Wavelet SURF Fallback (Non-free module excluded in PyPI wheel)"
        return kps, desc, mode


def _match_descriptors(
    desc1: np.ndarray,
    desc2: np.ndarray,
    norm_type: int,
    ratio_thresh: float = 0.75,
) -> List[cv2.DMatch]:
    """Run kNN matching with Lowe's ratio test safely."""
    if desc1 is None or desc2 is None or len(desc1) < 2 or len(desc2) < 2:
        return []
    bf = cv2.BFMatcher(norm_type, crossCheck=False)
    knn = bf.knnMatch(desc1, desc2, k=2)
    good = []
    for pair in knn:
        if len(pair) == 2:
            m, n = pair
            if m.distance < ratio_thresh * n.distance:
                good.append(m)
    good.sort(key=lambda m: m.distance)
    return good


def run_lab4_pipeline(
    img1_bgr: np.ndarray,
    img2_bgr: np.ndarray,
    hessian_threshold: float = 400.0,
    upright: bool = False,
    ratio_thresh: float = 0.75,
    max_draw_matches: int = 40,
) -> Dict[str, Any]:
    """Run SURF/U-SURF, SIFT, and ORB detection, matching, and benchmark comparison."""
    gray1 = cv2.cvtColor(img1_bgr, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2_bgr, cv2.COLOR_BGR2GRAY)

    # 1. Active SURF (or U-SURF)
    t0 = time.perf_counter()
    surf_kp1, surf_desc1, surf_backend = detect_and_compute_surf(gray1, hessian_threshold, upright)
    surf_kp2, surf_desc2, _ = detect_and_compute_surf(gray2, hessian_threshold, upright)
    surf_det_ms = (time.perf_counter() - t0) * 1000.0

    t_m0 = time.perf_counter()
    surf_matches = _match_descriptors(surf_desc1, surf_desc2, cv2.NORM_L2, ratio_thresh)
    surf_match_ms = (time.perf_counter() - t_m0) * 1000.0

    surf_kp_img1 = cv2.drawKeypoints(
        img1_bgr, surf_kp1, None, (255, 80, 0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )
    surf_kp_img2 = cv2.drawKeypoints(
        img2_bgr, surf_kp2, None, (255, 80, 0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )
    surf_matches_img = cv2.drawMatches(
        img1_bgr,
        surf_kp1,
        img2_bgr,
        surf_kp2,
        surf_matches[:max_draw_matches],
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )

    # 2. U-SURF vs Standard SURF comparison (run opposite mode cleanly for benchmark table)
    t_alt = time.perf_counter()
    alt_kp1, alt_desc1, _ = detect_and_compute_surf(gray1, hessian_threshold, not upright)
    alt_kp2, alt_desc2, _ = detect_and_compute_surf(gray2, hessian_threshold, not upright)
    alt_det_ms = (time.perf_counter() - t_alt) * 1000.0
    alt_matches = _match_descriptors(alt_desc1, alt_desc2, cv2.NORM_L2, ratio_thresh)

    # 3. SIFT
    sift = cv2.SIFT_create()
    t1 = time.perf_counter()
    sift_kp1, sift_desc1 = sift.detectAndCompute(gray1, None)
    sift_kp2, sift_desc2 = sift.detectAndCompute(gray2, None)
    sift_det_ms = (time.perf_counter() - t1) * 1000.0
    t_m1 = time.perf_counter()
    sift_matches = _match_descriptors(sift_desc1, sift_desc2, cv2.NORM_L2, ratio_thresh)
    sift_match_ms = (time.perf_counter() - t_m1) * 1000.0
    sift_matches_img = cv2.drawMatches(
        img1_bgr,
        sift_kp1,
        img2_bgr,
        sift_kp2,
        sift_matches[:max_draw_matches],
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )

    # 4. ORB
    orb = cv2.ORB_create(nfeatures=1000)
    t2 = time.perf_counter()
    orb_kp1, orb_desc1 = orb.detectAndCompute(gray1, None)
    orb_kp2, orb_desc2 = orb.detectAndCompute(gray2, None)
    orb_det_ms = (time.perf_counter() - t2) * 1000.0
    t_m2 = time.perf_counter()
    orb_matches = _match_descriptors(orb_desc1, orb_desc2, cv2.NORM_HAMMING, ratio_thresh)
    orb_match_ms = (time.perf_counter() - t_m2) * 1000.0
    orb_matches_img = cv2.drawMatches(
        img1_bgr,
        orb_kp1,
        img2_bgr,
        orb_kp2,
        orb_matches[:max_draw_matches],
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )

    surf_label = "U-SURF (Upright)" if upright else "SURF (Rotation-Invariant)"
    alt_label = "SURF (Rotation-Invariant)" if upright else "U-SURF (Upright)"

    benchmark_df = pd.DataFrame(
        [
            {
                "Algorithm": surf_label,
                "Keypoints (Ref)": len(surf_kp1),
                "Keypoints (Scene)": len(surf_kp2),
                "Descriptor Type": "64-D Float32 (Haar)",
                "Detect+Compute (ms)": round(surf_det_ms, 2),
                "Match Time (ms)": round(surf_match_ms, 2),
                "Good Matches": len(surf_matches),
            },
            {
                "Algorithm": alt_label,
                "Keypoints (Ref)": len(alt_kp1),
                "Keypoints (Scene)": len(alt_kp2),
                "Descriptor Type": "64-D Float32 (Haar)",
                "Detect+Compute (ms)": round(alt_det_ms, 2),
                "Match Time (ms)": round(surf_match_ms, 2),
                "Good Matches": len(alt_matches),
            },
            {
                "Algorithm": "SIFT",
                "Keypoints (Ref)": len(sift_kp1),
                "Keypoints (Scene)": len(sift_kp2),
                "Descriptor Type": "128-D Float32 (DoG)",
                "Detect+Compute (ms)": round(sift_det_ms, 2),
                "Match Time (ms)": round(sift_match_ms, 2),
                "Good Matches": len(sift_matches),
            },
            {
                "Algorithm": "ORB",
                "Keypoints (Ref)": len(orb_kp1),
                "Keypoints (Scene)": len(orb_kp2),
                "Descriptor Type": "256-bit Binary (BRIEF)",
                "Detect+Compute (ms)": round(orb_det_ms, 2),
                "Match Time (ms)": round(orb_match_ms, 2),
                "Good Matches": len(orb_matches),
            },
        ]
    )

    return {
        "surf_backend": surf_backend,
        "surf_kp1_count": len(surf_kp1),
        "surf_kp2_count": len(surf_kp2),
        "surf_matches_count": len(surf_matches),
        "sift_matches_count": len(sift_matches),
        "orb_matches_count": len(orb_matches),
        "surf_kp_img1_rgb": bgr_to_rgb(surf_kp_img1),
        "surf_kp_img2_rgb": bgr_to_rgb(surf_kp_img2),
        "surf_matches_rgb": bgr_to_rgb(surf_matches_img),
        "sift_matches_rgb": bgr_to_rgb(sift_matches_img),
        "orb_matches_rgb": bgr_to_rgb(orb_matches_img),
        "benchmark_df": benchmark_df,
    }


def render(sample_paths: Dict[str, Path]) -> None:
    """Render Lab Activity 4 Streamlit UI."""
    st.header("Lab Activity 4: SURF, SIFT, and ORB Feature Detection & Matching")
    st.caption("Course: CSST106 | Topic: SURF Feature Detection in Python using OpenCV | Week No.: 6")

    with st.expander("Objective & Procedure Overview", expanded=False):
        st.markdown(
            """
            **Objective:** Understand and implement **SURF** (Speeded-Up Robust Features) keypoint detection and
            descriptor extraction in OpenCV, visualize scale/orientation rich keypoints, perform cross-image
            feature matching, explore **U-SURF** (Upright SURF), and benchmark against **SIFT** and **ORB**.

            **Procedure / Algorithms:**
            1. Load reference object and rotated/scaled target scene in grayscale.
            2. Instantiate SURF safely (`cv2.xfeatures2d.SURF_create` with automatic DoH + 64-D Haar Wavelet fallback
               for non-free PyPI builds) with configurable **Hessian Threshold** and **U-SURF** toggle.
            3. Draw scale/orientation keypoints via `cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS`.
            4. Match descriptors via `BFMatcher` + Lowe's Ratio Test (`cv2.drawMatches`) and benchmark against `cv2.SIFT_create()` and `cv2.ORB_create()`.
            """
        )

    col_u1, col_u2 = st.columns(2)
    with col_u1:
        up1 = st.file_uploader("Upload Reference Object Image (optional)", type=["png", "jpg", "jpeg"], key="lab4_up1")
    with col_u2:
        up2 = st.file_uploader("Upload Target Scene Image (optional)", type=["png", "jpg", "jpeg"], key="lab4_up2")

    img1_bgr, src1 = load_bgr_image(sample_paths["lab4_ref"], uploaded_file=up1)
    img2_bgr, src2 = load_bgr_image(sample_paths["lab4_scene"], uploaded_file=up2)

    st.subheader("Interactive Controls")
    c1, c2, c3 = st.columns(3)
    with c1:
        hessian_thresh = st.slider("SURF Hessian Threshold", 50, 2000, 400, step=50, key="lab4_hessian")
    with c2:
        upright_mode = st.checkbox("Enable U-SURF Mode (Upright, No Rotation Invariance)", value=False, key="lab4_upright")
    with c3:
        ratio_thresh = st.slider("Lowe's Ratio Test Threshold", 0.50, 0.95, 0.75, step=0.05, key="lab4_ratio")

    res = run_lab4_pipeline(
        img1_bgr=img1_bgr,
        img2_bgr=img2_bgr,
        hessian_threshold=hessian_thresh,
        upright=upright_mode,
        ratio_thresh=ratio_thresh,
    )

    st.info(
        f"**Active SURF Engine:** `{res['surf_backend']}` | "
        f"**Inputs:** `{src1}` vs. `{src2}`"
    )

    st.subheader("Rich Keypoint Visualization (`DRAW_RICH_KEYPOINTS`)")
    k_c1, k_c2 = st.columns(2)
    with k_c1:
        st.image(
            res["surf_kp_img1_rgb"],
            caption=f"Reference Object Keypoints ({res['surf_kp1_count']} detected)",
            use_container_width=True,
        )
    with k_c2:
        st.image(
            res["surf_kp_img2_rgb"],
            caption=f"Target Scene Keypoints ({res['surf_kp2_count']} detected)",
            use_container_width=True,
        )

    st.subheader("Feature Matching & Benchmark Comparison (SURF vs. SIFT vs. ORB)")
    tab_surf, tab_sift, tab_orb = st.tabs(["SURF Matches", "SIFT Matches", "ORB Matches"])
    with tab_surf:
        st.image(
            res["surf_matches_rgb"],
            caption=f"SURF Matching ({res['surf_matches_count']} Lowe ratio matches, top 40 shown)",
            use_container_width=True,
        )
    with tab_sift:
        st.image(
            res["sift_matches_rgb"],
            caption=f"SIFT Matching ({res['sift_matches_count']} Lowe ratio matches, top 40 shown)",
            use_container_width=True,
        )
    with tab_orb:
        st.image(
            res["orb_matches_rgb"],
            caption=f"ORB Matching ({res['orb_matches_count']} Lowe ratio matches, top 40 shown)",
            use_container_width=True,
        )

    st.dataframe(res["benchmark_df"], use_container_width=True, hide_index=True)

    st.subheader("Required Post-Lab Questions Answered")
    st.markdown(
        """
        1. **What effect does changing the Hessian threshold have on the number of detected keypoints?**
           Raising `hessianThreshold` acts as a strict minimum-response filter on the Determinant of the Hessian matrix:
           **higher thresholds sharply reduce the number of detected keypoints**, retaining only the strongest, most salient
           blob corners while rejecting weak texture variations. Conversely, lowering the threshold produces many more keypoints,
           increasing sensitivity to fine details but also admitting background clutter and noise.
        2. **What are the descriptors in SURF used for, and why are they important?**
           A SURF descriptor is a compact **64-element floating-point vector** (computed from Haar wavelet responses `dx`, `dy`,
           `|dx|`, `|dy|` summed over a `4 × 4` oriented grid around each keypoint) that encodes the local intensity distribution.
           Descriptors make keypoints **uniquely identifiable and invariant to scale, rotation (unless using U-SURF), and linear
           illumination changes**, enabling reliable cross-image matching, homography estimation, panorama stitching, and object tracking.
        3. **How does SURF compare to SIFT and ORB in feature detection and speed?**
           - **SIFT (128-D float):** Highest sub-pixel accuracy and unmatched scale/rotation invariance using Difference-of-Gaussians (DoG),
             but computationally slowest due to Gaussian pyramid convolutions and 128-D floating-point descriptors.
           - **SURF (64-D float):** Designed as a faster alternative to SIFT by approximating Laplacian/Hessian convolutions with box
             filters on integral images and smaller 64-D Haar descriptors; slightly less accurate under extreme viewpoint shifts than SIFT
             but significantly faster (patented for commercial use).
           - **ORB (256-bit binary):** Built from FAST corner detection and rotated BRIEF binary descriptors; **orders of magnitude faster**
             using Hamming distance (`XOR + popcount`), patent-free, and ideal for real-time mobile/embedded SLAM, though less invariant
             across large scale changes than SIFT or SURF.
        """
    )
