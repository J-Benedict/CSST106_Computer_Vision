"""Lab Activity 5: Facial Recognition using Principal Component Analysis (Eigenfaces) & SVM."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import streamlit as st


@st.cache_data(show_spinner="Loading LFW Facial Dataset...")
def load_lfw_dataset(cache_path_str: str) -> Dict[str, Any]:
    """Load cached LFW dataset (min_faces_per_person=70, resize=0.4) or fetch directly."""
    cache_path = Path(cache_path_str)
    if cache_path.exists():
        npz = np.load(cache_path, allow_pickle=False)
        images = npz["images"]
        data = npz["data"]
        target = npz["target"]
        target_names = npz["target_names"]
        is_synthetic = bool(npz["is_synthetic"][0]) if "is_synthetic" in npz else False
    else:
        from sklearn.datasets import fetch_lfw_people

        lfw = fetch_lfw_people(min_faces_per_person=70, resize=0.4)
        images = lfw.images.astype(np.float32)
        data = lfw.data.astype(np.float32)
        target = lfw.target.astype(np.int64)
        target_names = lfw.target_names
        is_synthetic = False

    n_samples, h, w = images.shape
    return {
        "images": images,
        "data": data,
        "target": target,
        "target_names": target_names,
        "n_samples": int(n_samples),
        "h": int(h),
        "w": int(w),
        "n_features": int(data.shape[1]),
        "n_classes": int(len(target_names)),
        "is_synthetic": is_synthetic,
    }


@st.cache_resource(show_spinner="Fitting PCA Eigenfaces & RBF SVM Classifier...")
def run_lab5_pipeline(
    cache_path_str: str,
    n_components: int = 150,
    whiten: bool = True,
    test_size: float = 0.2,
) -> Dict[str, Any]:
    """Run Tasks 1–4 and Post-Lab Q1–Q4 programmatic analytics for Eigenfaces + SVM."""
    ds = load_lfw_dataset(cache_path_str)
    X = ds["data"]
    y = ds["target"]
    target_names = ds["target_names"]
    h, w = ds["h"], ds["w"]

    # Task 1: 80/20 Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=float(test_size), random_state=42
    )

    eff_components = min(int(n_components), X_train.shape[0], X_train.shape[1])

    # Task 2: PCA strictly fitted on X_train
    t0 = time.perf_counter()
    pca = PCA(
        n_components=eff_components,
        svd_solver="randomized",
        whiten=bool(whiten),
        random_state=42,
    )
    pca.fit(X_train)
    X_train_pca = pca.transform(X_train)
    X_test_pca = pca.transform(X_test)
    pca_time = time.perf_counter() - t0

    # Task 3: Reshape Eigenfaces
    eigenfaces = pca.components_.reshape((eff_components, h, w))

    # Task 4: RBF SVM Classifier
    t1 = time.perf_counter()
    clf = SVC(kernel="rbf", class_weight="balanced", random_state=42)
    clf.fit(X_train_pca, y_train)
    svm_time = time.perf_counter() - t1

    y_pred = clf.predict(X_test_pca)
    acc = float(accuracy_score(y_test, y_pred))
    report_dict = classification_report(
        y_test, y_pred, target_names=target_names, output_dict=True, zero_division=0
    )
    report_text = classification_report(
        y_test, y_pred, target_names=target_names, zero_division=0
    )
    cm = confusion_matrix(y_test, y_pred)

    # Q1: Cumulative Explained Variance & 90% threshold
    cum_var = np.cumsum(pca.explained_variance_ratio_)
    reached_90 = bool(np.any(cum_var >= 0.90))
    k_90 = int(np.argmax(cum_var >= 0.90) + 1) if reached_90 else None
    total_var_retained = float(cum_var[-1])

    # Q4: Programmatic Failure Analysis
    misclassified_idx = np.where(y_pred != y_test)[0]

    return {
        "dataset": ds,
        "X_train_shape": X_train.shape,
        "X_test_shape": X_test.shape,
        "X_test": X_test,
        "y_test": y_test,
        "y_pred": y_pred,
        "eff_components": eff_components,
        "whiten": whiten,
        "pca_time": pca_time,
        "svm_time": svm_time,
        "eigenfaces": eigenfaces,
        "explained_variance_ratio": pca.explained_variance_ratio_,
        "cum_var": cum_var,
        "reached_90": reached_90,
        "k_90": k_90,
        "total_var_retained": total_var_retained,
        "accuracy": acc,
        "report_dict": report_dict,
        "report_text": report_text,
        "confusion_matrix": cm,
        "misclassified_idx": misclassified_idx,
    }


def make_eigenfaces_gallery_figure(eigenfaces: np.ndarray, h: int, w: int, n_row: int = 3, n_col: int = 4):
    """Plot gallery of the first 12 Eigenfaces."""
    count = min(n_row * n_col, eigenfaces.shape[0])
    fig, axes = plt.subplots(n_row, n_col, figsize=(2.1 * n_col, 2.5 * n_row))
    axes_flat = np.array(axes).ravel()
    for i, ax in enumerate(axes_flat):
        if i < count:
            ax.imshow(eigenfaces[i], cmap="gray")
            ax.set_title(f"Eigenface {i}", fontsize=10, fontweight="bold")
        ax.axis("off")
    plt.suptitle("First 12 Principal Components (Eigenfaces)", fontsize=12, fontweight="bold")
    plt.tight_layout()
    return fig


def make_first_vs_last_figure(eigenfaces: np.ndarray):
    """Plot First 3 vs. Last 3 Eigenfaces side-by-side."""
    n = eigenfaces.shape[0]
    indices = [0, 1, 2, max(0, n - 3), max(0, n - 2), max(0, n - 1)]
    fig, axes = plt.subplots(2, 3, figsize=(8.2, 5.6))
    for plot_i, comp_idx in enumerate(indices):
        ax = axes[plot_i // 3, plot_i % 3]
        ax.imshow(eigenfaces[comp_idx], cmap="gray")
        group = "Global Structure" if plot_i < 3 else "High-Freq Noise"
        ax.set_title(f"Eigenface {comp_idx} ({group})", fontsize=9.5, fontweight="bold")
        ax.axis("off")
    plt.suptitle("Visual Interpretation: First 3 vs. Last 3 Eigenfaces", fontsize=11.5, fontweight="bold")
    plt.tight_layout()
    return fig


def make_variance_figure(res: Dict[str, Any]):
    """Plot Cumulative Explained Variance vs. Component Count with 90% threshold line."""
    cum_var = res["cum_var"]
    xs = np.arange(1, len(cum_var) + 1)
    fig, ax = plt.subplots(figsize=(8.0, 4.2))
    ax.plot(xs, cum_var, color="#1f77b4", linewidth=2.2, label="Cumulative Explained Variance")
    ax.axhline(0.90, color="#d62728", linestyle="--", linewidth=1.6, label="90% Variance Threshold")
    if res["reached_90"] and res["k_90"] is not None:
        k90 = res["k_90"]
        ax.axvline(k90, color="#2ca02c", linestyle=":", linewidth=1.6, label=f"90% Reached at k = {k90}")
        ax.scatter([k90], [cum_var[k90 - 1]], color="#2ca02c", zorder=5, s=55)
    ax.set_title("Cumulative Explained Variance vs. Number of PCA Components", fontsize=11.5, fontweight="bold")
    ax.set_xlabel("Number of Principal Components")
    ax.set_ylabel("Cumulative Explained Variance Ratio")
    ax.set_ylim(0.0, 1.03)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right")
    plt.tight_layout()
    return fig


def make_confusion_matrix_figure(cm: np.ndarray, target_names: np.ndarray):
    """Plot confusion matrix heatmap for SVM test predictions."""
    fig, ax = plt.subplots(figsize=(6.8, 5.6))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    tick_marks = np.arange(len(target_names))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(target_names, rotation=35, ha="right", fontsize=8.5)
    ax.set_yticklabels(target_names, fontsize=8.5)
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=8.5,
            )
    ax.set_ylabel("True Identity")
    ax.set_xlabel("Predicted Identity")
    ax.set_title("RBF SVM Confusion Matrix on PCA Eigenfaces", fontsize=11, fontweight="bold")
    plt.tight_layout()
    return fig


def make_failures_figure(res: Dict[str, Any], max_show: int = 8):
    """Plot gallery of programmatically extracted misclassified test faces."""
    idx_list = res["misclassified_idx"]
    ds = res["dataset"]
    h, w = ds["h"], ds["w"]
    target_names = ds["target_names"]
    X_test = res["X_test"]
    y_test = res["y_test"]
    y_pred = res["y_pred"]

    count = min(max_show, len(idx_list))
    if count == 0:
        fig, ax = plt.subplots(figsize=(5, 2))
        ax.text(0.5, 0.5, "No Misclassified Samples on Test Split (100% Accuracy)", ha="center", va="center")
        ax.axis("off")
        return fig

    cols = min(4, count)
    rows = int(np.ceil(count / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(2.7 * cols, 3.0 * rows))
    axes_flat = np.array(axes).ravel()
    for i, ax in enumerate(axes_flat):
        if i < count:
            idx = int(idx_list[i])
            face_img = X_test[idx].reshape((h, w))
            true_name = target_names[y_test[idx]].split()[-1]
            pred_name = target_names[y_pred[idx]].split()[-1]
            ax.imshow(face_img, cmap="gray")
            ax.set_title(f"True: {true_name}\nPred: {pred_name}", fontsize=9.5, color="#b22222", fontweight="bold")
        ax.axis("off")
    plt.suptitle("Programmatic Failure Analysis: Misclassified Test Portraits", fontsize=11.5, fontweight="bold")
    plt.tight_layout()
    return fig


def render(sample_paths: Dict[str, Path]) -> None:
    """Render Lab Activity 5 Streamlit UI."""
    st.header("Lab Activity 5: Facial Recognition using PCA (Eigenfaces) & SVM")
    st.caption("Course: CSST106 | Topic: Dimensionality Reduction & Feature Extraction (PCA) | Week No.: 7 – 8")

    with st.expander("Objective & Laboratory Tasks Overview", expanded=False):
        st.markdown(
            """
            **Learning Objectives:** Load and flatten high-dimensional facial image data (`h × w`), extract principal
            variance components (**Eigenfaces**) via randomized PCA strictly fitted on training data, quantify the
            trade-off between dimensionality and cumulative explained variance (`≥ 90%`), and classify identities using an RBF SVM.

            **Tasks Implemented:**
            - **Task 1:** Load LFW (`min_faces_per_person=70, resize=0.4`) and split 80% Train / 20% Test (`random_state=42`).
            - **Task 2:** Fit `PCA(n_components=150, svd_solver='randomized', whiten=True)` on `X_train` and project train/test sets.
            - **Task 3:** Reshape `pca.components_` to `(n_components, h, w)` and render First 12 + First 3 vs. Last 3 Eigenfaces.
            - **Task 4:** Train `SVC(kernel='rbf', class_weight='balanced')` and evaluate Classification Report & Confusion Matrix.
            """
        )

    st.subheader("Interactive Controls")
    c1, c2, c3 = st.columns(3)
    with c1:
        n_components = st.slider("Number of PCA Components", 20, 250, 150, step=10, key="lab5_k")
    with c2:
        whiten = st.checkbox("Enable PCA Whitening (whiten=True)", value=True, key="lab5_whiten")
    with c3:
        test_size = st.slider("Test Split Fraction", 0.10, 0.40, 0.20, step=0.05, key="lab5_split")

    res = run_lab5_pipeline(
        cache_path_str=str(sample_paths["lab5_lfw"]),
        n_components=n_components,
        whiten=whiten,
        test_size=test_size,
    )
    ds = res["dataset"]
    if ds["is_synthetic"]:
        st.warning("Using deterministic offline synthetic facial dataset fallback (same shape & class structure as LFW).")

    # Task 1 Metrics
    st.subheader("Task 1: Dataset Dimensions & Split Summary")
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Total Samples", f"{ds['n_samples']:,}")
    m2.metric("Image Shape", f"{ds['h']} × {ds['w']}")
    m3.metric("Raw Features", f"{ds['n_features']:,}")
    m4.metric("Classes (People)", f"{ds['n_classes']}")
    m5.metric("PCA Components", f"{res['eff_components']}")

    st.write(
        f"**Train Shape:** `{res['X_train_shape']}` | **Test Shape:** `{res['X_test_shape']}` | "
        f"**PCA Fit/Transform Time:** `{res['pca_time']:.3f}s` | **SVM Fit Time:** `{res['svm_time']:.3f}s`"
    )

    # Task 3 Visualizations
    st.subheader("Task 3: Eigenfaces Gallery & First 3 vs. Last 3 Comparison")
    g_c1, g_c2 = st.columns([1.25, 1.0])
    with g_c1:
        fig_gal = make_eigenfaces_gallery_figure(res["eigenfaces"], ds["h"], ds["w"])
        st.pyplot(fig_gal)
        plt.close(fig_gal)
    with g_c2:
        fig_fl = make_first_vs_last_figure(res["eigenfaces"])
        st.pyplot(fig_fl)
        plt.close(fig_fl)

    # Task 4 Classification Report & Confusion Matrix
    st.subheader("Task 4: RBF SVM Classification Performance")
    s1, s2, s3 = st.columns(3)
    s1.metric("Test Accuracy", f"{res['accuracy'] * 100:.2f}%")
    s2.metric("Total Variance Retained", f"{res['total_var_retained'] * 100:.2f}%")
    s3.metric("Misclassified Test Faces", f"{len(res['misclassified_idx'])} / {len(res['y_test'])}")

    rep_c1, rep_c2 = st.columns([1.15, 1.0])
    with rep_c1:
        rep_df = pd.DataFrame(res["report_dict"]).T.round(3)
        st.dataframe(rep_df, use_container_width=True)
    with rep_c2:
        fig_cm = make_confusion_matrix_figure(res["confusion_matrix"], ds["target_names"])
        st.pyplot(fig_cm)
        plt.close(fig_cm)

    # Post-Lab Deliverables Q1–Q4
    st.subheader("Post-Lab Analysis Deliverables (Q1 – Q4)")
    q1_c1, q1_c2 = st.columns([1.2, 1.0])
    with q1_c1:
        fig_var = make_variance_figure(res)
        st.pyplot(fig_var)
        plt.close(fig_var)
    with q1_c2:
        k90_msg = (
            f"**{res['k_90']} components** are required to retain **≥ 90%** of total dataset variance."
            if res["reached_90"]
            else f"Current `n_components={res['eff_components']}` retains **{res['total_var_retained']*100:.2f}%** variance (< 90%); increase slider to find exact `k_90`."
        )
        st.markdown(
            f"""
            #### Q1. Dimensionality vs. Variance
            {k90_msg}
            Reducing raw pixels from **{ds['n_features']}** dimensions down to **{res['eff_components']}** principal
            components compresses the representation by **{(1.0 - res['eff_components'] / ds['n_features']) * 100:.1f}%**
            while retaining **{res['total_var_retained'] * 100:.2f}%** of facial variance.

            #### Q2. Visual Interpretation (First 3 vs. Last 3 Eigenfaces)
            - **First 3 Eigenfaces (0, 1, 2):** Capture the highest-variance **global facial structure** — left-to-right
              directional lighting gradients, overall facial oval/jawline silhouette, hairline shading, and primary eye-socket/nose-bridge symmetry.
            - **Last 3 Eigenfaces ({res['eff_components']-3}, {res['eff_components']-2}, {res['eff_components']-1}):** Capture
              tiny residual eigenvalues consisting primarily of **high-frequency spatial noise**, background texture clutter,
              and pixel-level alignment artifacts rather than identity-defining anatomy.
            """
        )

    st.markdown(
        """
        #### Q3. The Importance of Whitening (`whiten=True`)
        Mathematically, standard PCA produces uncorrelated projections `z_i = u_i^T (x - μ)` whose variances equal their
        corresponding eigenvalues `λ_i` (`Var(z_i) = λ_i`). Setting `whiten=True` divides each projection by the square root
        of its eigenvalue: `z_white,i = z_i / √λ_i`, transforming the feature covariance matrix into the **identity matrix**
        (`Cov(z_white) = I`, unit variance across every component). Because an **RBF SVM** computes isotropic Euclidean distances
        `K(a, b) = exp(-γ ||a - b||²)`, unwhitened projections allow the top 1–3 components (often dominated by lighting variations)
        with massive `λ_i` values to overwhelm subtle identity-carrying components (`z_10..z_80`). Whitening equalizes feature
        scales so all discriminative Eigenfaces contribute proportionally to the decision boundary.
        """
    )

    st.markdown("#### Q4. Programmatic Failure Analysis")
    fail_c1, fail_c2 = st.columns([1.3, 1.0])
    with fail_c1:
        fig_fail = make_failures_figure(res)
        st.pyplot(fig_fail)
        plt.close(fig_fail)
    with fail_c2:
        st.markdown(
            """
            **Computer-Vision Hypotheses for Eigenface Misclassifications:**
            1. **Illumination & Shadow Casts:** PCA is linear and pixel-aligned; harsh directional sunlight or shadows shift raw
               pixel intensities along the top illumination eigenmodes, pulling a test face closer to another person under similar lighting.
            2. **Out-of-Plane Pose & Head Rotation:** Eigenfaces assume aligned frontal landmarks (eyes/nose/mouth at fixed `(x, y)` pixels).
               Yaw/pitch head turns misalign facial landmarks relative to the training templates.
            3. **Occlusions, Eyeglasses & Expressions:** Sunglasses, microphones, strong smiles, or frowns alter local pixel neighborhoods,
               distorting linear projection weights.
            4. **Class Imbalance:** In LFW (`min_faces_per_person=70`), dominant classes like George W. Bush (~530 images) exert
               stronger density near ambiguous boundary regions even with `class_weight='balanced'`.
            """
        )

    with st.expander("Follow-up Questions (1 – 5) Complete Answers", expanded=True):
        st.markdown(
            """
            1. **Why is dimensionality reduction important in computer vision?**
               Raw images contain thousands to millions of strongly correlated pixels (`1850` here, or `307,200` for 640×480 grayscale).
               Dimensionality reduction mitigates the **curse of dimensionality**, removes redundant pixel correlations and sensor noise,
               dramatically speeds up training/inference, reduces memory storage, and lowers overfitting risk when labeled samples are limited.
            2. **What is the difference between the first principal component and the second principal component?**
               The **first principal component (PC1)** is the unit-length eigenvector of the covariance matrix pointing along the
               direction of **maximum total variance** in the centered dataset (typically overall lighting/skin-tone brightness).
               The **second principal component (PC2)** is constrained to be **strictly orthogonal** (`u_1^T u_2 = 0`) to PC1 while
               capturing the largest *remaining* variance (commonly left-vs-right directional shading or face width/contour).
            3. **If PCA reduces an image from 1,000 features to 100 components, what information might be lost?**
               The discarded 900 trailing components contain **fine-grained high-frequency spatial details** — sharp edges, subtle skin
               textures, eyelashes, faint wrinkles, small expression nuances, and sensor/background noise — causing reconstructed faces
               to appear slightly smoothed.
            4. **How can PCA be applied to face recognition?**
               Training portraits are flattened, mean-centered, and projected via PCA into a compact `k`-dimensional **"face space"** spanned
               by the top `k` Eigenfaces (`z = W^T (x - μ)`). New query faces are projected onto the exact same Eigenface basis, and
               identities are recognized by matching or classifying the `k`-element weight vector using nearest-neighbor distance or an SVM.
            5. **Why should image features generally be centered before applying PCA?**
               PCA seeks orthogonal axes of maximum variance **passing through the origin**. If data is not zero-centered by subtracting
               the mean face (`x - μ`), the first principal component gets pulled toward the **sample mean vector** (`μ`) rather than
               capturing genuine inter-person structural variance around the average face.
            """
        )
