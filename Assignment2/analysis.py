#!/usr/bin/env python3

"""
Advanced Facial Recognition Evaluation Pipeline
Comprehensive FMR/FNMR analysis with varying thresholds and detailed metrics
"""

import os
import json
import csv
import numpy as np
import cv2
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from sklearn.metrics import roc_curve, auc, confusion_matrix, precision_recall_curve
import seaborn as sns
from datetime import datetime
import torch
from facenet_pytorch import InceptionResnetV1


print("=" * 80)
print("ADVANCED FACIAL RECOGNITION EVALUATION PIPELINE")
print("FMR/FNMR Analysis with Varying Thresholds")
print("=" * 80)


BASE_DIR = Path(__file__).resolve().parent
dataset_path = str(BASE_DIR / "age_gap_dataset")
metadata_path = os.path.join(dataset_path, "metadata.csv")
VARIANTS = ["AGE_GAP"]
os.environ.setdefault("TORCH_HOME", str(BASE_DIR / "torch_cache"))


# ============================================================
# LOAD METADATA
# ============================================================

with open(metadata_path, "r", newline="", encoding="utf-8-sig") as f:
    metadata = list(csv.DictReader(f))

# Normalize the CSV representation to the pair format used by the evaluator.
# Paths in the CSV are relative to age_gap_dataset and use Windows separators.
for record in metadata:
    record["subject_id"] = str(record["subject_id"])
    record["image1_age_gap"] = record["image_1"]
    record["image2_age_gap"] = record["image_2"]

print(f"\n✓ Loaded metadata for {len(metadata)} subjects")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FACE_MODEL = InceptionResnetV1(pretrained="vggface2").eval().to(DEVICE)


# ============================================================
# EVALUATOR CLASS
# ============================================================

class AdvancedFaceRecognitionEvaluator:
    """Advanced evaluation with comprehensive metrics and analysis"""

    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.embeddings = {}
        self.results = {}

    # --------------------------------------------------------
    # Load image
    # --------------------------------------------------------

    def load_image(self, image_path):
        """Load image from file"""

        full_path = os.path.join(self.dataset_path, image_path)

        img = cv2.imread(full_path)

        if img is None:
            raise FileNotFoundError(
                f"Cannot load image: {full_path}"
            )

        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # --------------------------------------------------------
    # Preprocess face
    # --------------------------------------------------------

    def preprocess_face(self, face_img):
        """Preprocessing: normalize, resize, histogram equalization"""

        face = cv2.resize(face_img, (256, 256))

        face = face.astype(np.float32) / 255.0

        if len(face.shape) == 3:

            for i in range(3):

                face[:, :, i] = (
                    cv2.equalizeHist(
                        (face[:, :, i] * 255).astype(np.uint8)
                    ) / 255.0
                )

        return face

    # --------------------------------------------------------
    # Extract features
    # --------------------------------------------------------

    def extract_features(self, face_img):
        """Extract a 512-D FaceNet identity embedding."""
        image = cv2.resize(face_img, (160, 160), interpolation=cv2.INTER_AREA)
        tensor = torch.from_numpy(image).permute(2, 0, 1).float()
        tensor = (tensor - 0.5) / 0.5
        tensor = tensor.unsqueeze(0).to(DEVICE)
        with torch.inference_mode():
            embedding = FACE_MODEL(tensor).cpu().numpy().ravel()
        return embedding / (np.linalg.norm(embedding) + 1e-8)

        hog = cv2.HOGDescriptor(
            (128, 128), (16, 16), (8, 8), (8, 8), 9
        ).compute(gray).ravel()
        hog /= np.linalg.norm(hog) + 1e-8

        # Low-frequency structure is more age-stable than individual pixels.
        structure = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA)
        structure = (structure.astype(np.float32) - structure.mean()) / (structure.std() + 1e-6)
        structure = structure.ravel()
        structure /= np.linalg.norm(structure) + 1e-8

        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        appearance = []
        for channel, upper in ((hsv[:, :, 0], 180), (hsv[:, :, 1], 256), (gray, 256)):
            hist = cv2.calcHist([channel], [0], None, [32], [0, upper]).ravel()
            hist /= hist.sum() + 1e-8
            appearance.extend(hist)
        appearance = np.asarray(appearance, dtype=np.float32)

        # Weight the parts explicitly so no single feature family dominates.
        features = np.concatenate([
            0.60 * hog,
            0.25 * structure,
            0.15 * appearance
        ])
        return features / (np.linalg.norm(features) + 1e-8)

    # --------------------------------------------------------
    # Similarity
    # --------------------------------------------------------

    def calculate_similarity(self, embedding1, embedding2):
        """Calculate cosine similarity between FaceNet embeddings."""
        return float(np.dot(embedding1, embedding2))

    # --------------------------------------------------------
    # Process all images
    # --------------------------------------------------------

    def process_all_images(self):
        """Process all images for all variants"""

        print("\n[1] PROCESSING ALL IMAGES")
        print("-" * 80)

        for i, record in enumerate(metadata, 1):

            subject_id = record['subject_id']

            for variant in VARIANTS:

                img1 = self.load_image(
                    record[f'image1_{variant.lower()}']
                )

                img2 = self.load_image(
                    record[f'image2_{variant.lower()}']
                )

                face1_prep = self.preprocess_face(img1)
                face2_prep = self.preprocess_face(img2)

                emb1 = self.extract_features(face1_prep)
                emb2 = self.extract_features(face2_prep)

                key = f"{subject_id}_{variant}"

                self.embeddings[key] = (
                    emb1,
                    emb2
                )

            if i % 50 == 0:
                print(
                    f"  Processed {i}/{len(metadata)} subjects..."
                )

        print("✓ Processed all subjects and variants")

    # --------------------------------------------------------
    # Evaluate
    # --------------------------------------------------------

    def evaluate_with_thresholds(
        self,
        variant_name,
        num_impostor_pairs=5000
    ):
        """Comprehensive evaluation with multiple thresholds"""

        print(f"\n[2] EVALUATION - {variant_name}")
        print("-" * 80)

        genuine_scores = []
        impostor_scores = []

        # ====================================================
        # Genuine pairs
        # ====================================================

        for record in metadata:

            subject_id = record['subject_id']

            key = f"{subject_id}_{variant_name}"

            emb1, emb2 = self.embeddings[key]

            score = self.calculate_similarity(
                emb1,
                emb2
            )

            genuine_scores.append(score)

        # ====================================================
        # Impostor pairs
        # ====================================================

        for _ in range(num_impostor_pairs):

            idx1, idx2 = np.random.choice(
                len(metadata),
                2,
                replace=False
            )

            subject1 = metadata[idx1]['subject_id']
            subject2 = metadata[idx2]['subject_id']

            key1 = f"{subject1}_{variant_name}"
            key2 = f"{subject2}_{variant_name}"

            emb1, _ = self.embeddings[key1]
            _, emb2 = self.embeddings[key2]

            score = self.calculate_similarity(
                emb1,
                emb2
            )

            impostor_scores.append(score)

        genuine_scores = np.array(genuine_scores)
        impostor_scores = np.array(impostor_scores)

        # ====================================================
        # Statistics
        # ====================================================

        print("\nGenuine Pairs Statistics:")

        print(f"  Count:   {len(genuine_scores)}")
        print(f"  Mean:    {np.mean(genuine_scores):.6f}")
        print(f"  Std:     {np.std(genuine_scores):.6f}")
        print(f"  Min:     {np.min(genuine_scores):.6f}")
        print(f"  Max:     {np.max(genuine_scores):.6f}")
        print(f"  Median:  {np.median(genuine_scores):.6f}")

        print("\nImpostor Pairs Statistics:")

        print(f"  Count:   {len(impostor_scores)}")
        print(f"  Mean:    {np.mean(impostor_scores):.6f}")
        print(f"  Std:     {np.std(impostor_scores):.6f}")
        print(f"  Min:     {np.min(impostor_scores):.6f}")
        print(f"  Max:     {np.max(impostor_scores):.6f}")
        print(f"  Median:  {np.median(impostor_scores):.6f}")

        # ====================================================
        # Thresholds
        # ====================================================

        # FIX:
        # Use both genuine and impostor scores to determine
        # the complete threshold range.

        all_scores = np.concatenate([
            genuine_scores,
            impostor_scores
        ])

        thresholds = np.arange(
            np.min(all_scores) - 0.01,
            np.max(all_scores) + 0.01,
            0.005
        )

        fmr_values = []
        fnmr_values = []

        # ====================================================
        # FMR / FNMR
        # ====================================================

        for threshold in thresholds:

            # False Match Rate:
            # impostor accepted
            fmr = (
                np.sum(impostor_scores >= threshold)
                / len(impostor_scores)
            )

            # False Non-Match Rate:
            # genuine rejected
            fnmr = (
                np.sum(genuine_scores < threshold)
                / len(genuine_scores)
            )

            fmr_values.append(fmr)
            fnmr_values.append(fnmr)

        fmr_values = np.array(fmr_values)
        fnmr_values = np.array(fnmr_values)

        # ====================================================
        # EER
        # ====================================================

        eer_idx = np.argmin(
            np.abs(
                fmr_values - fnmr_values
            )
        )

        eer_threshold = thresholds[eer_idx]

        # Better EER estimate:
        eer = (
            fmr_values[eer_idx]
            + fnmr_values[eer_idx]
        ) / 2

        # ====================================================
        # ROC
        # ====================================================

        y_true = np.concatenate([
            np.ones(len(genuine_scores)),
            np.zeros(len(impostor_scores))
        ])

        y_scores = np.concatenate([
            genuine_scores,
            impostor_scores
        ])

        fpr, tpr, roc_thresholds = roc_curve(
            y_true,
            y_scores
        )

        roc_auc = auc(
            fpr,
            tpr
        )

        # ====================================================
        # Precision-Recall
        # ====================================================

        precision, recall, pr_thresholds = (
            precision_recall_curve(
                y_true,
                y_scores
            )
        )

        # ====================================================
        # Specific threshold values
        # ====================================================

        # FaceNet cosine scores are not restricted to the old 0.55--0.90
        # range; choose operating points from the observed score range.
        key_thresholds = np.linspace(
            np.percentile(all_scores, 10),
            np.percentile(all_scores, 90),
            8
        ).tolist()

        threshold_metrics = []

        # FIX:
        # Calculate directly instead of depending on
        # generated threshold array.

        for t in key_thresholds:

            fmr_t = (
                np.sum(impostor_scores >= t)
                / len(impostor_scores)
            )

            fnmr_t = (
                np.sum(genuine_scores < t)
                / len(genuine_scores)
            )

            threshold_metrics.append({

                'threshold': t,

                'FMR': fmr_t,

                'FNMR': fnmr_t,

                'MATED': 1 - fnmr_t,

                'NONMATED': fmr_t

            })

        # ====================================================
        # Results
        # ====================================================

        results = {

            'variant': variant_name,

            'genuine_scores': genuine_scores,

            'impostor_scores': impostor_scores,

            'thresholds': thresholds,

            'fmr_values': fmr_values,

            'fnmr_values': fnmr_values,

            'eer': float(eer),

            'eer_threshold': float(eer_threshold),

            'auc': float(roc_auc),

            'fpr': fpr,

            'tpr': tpr,

            'precision': precision,

            'recall': recall,

            'threshold_metrics': threshold_metrics,

            'genuine_stats': {

                'mean': float(
                    np.mean(genuine_scores)
                ),

                'std': float(
                    np.std(genuine_scores)
                ),

                'min': float(
                    np.min(genuine_scores)
                ),

                'max': float(
                    np.max(genuine_scores)
                )

            },

            'impostor_stats': {

                'mean': float(
                    np.mean(impostor_scores)
                ),

                'std': float(
                    np.std(impostor_scores)
                ),

                'min': float(
                    np.min(impostor_scores)
                ),

                'max': float(
                    np.max(impostor_scores)
                )

            }

        }

        # ====================================================
        # Print metrics
        # ====================================================

        print("\nPerformance Metrics:")

        print(
            f"  EER:  {eer:.4f} "
            f"({eer * 100:.2f}%) "
            f"at threshold {eer_threshold:.4f}"
        )

        print(
            f"  AUC:  {roc_auc:.4f}"
        )

        print("\nThreshold Analysis:")

        print(
            f"  {'Threshold':<12}"
            f"{'FMR':<12}"
            f"{'FNMR':<12}"
            f"{'TAR':<12}"
        )

        print(
            f"  {'-' * 48}"
        )

        for tm in threshold_metrics:

            print(
                f"  {tm['threshold']:<12.4f}"
                f"{tm['FMR']:<12.6f}"
                f"{tm['FNMR']:<12.6f}"
                f"{tm['MATED']:<12.6f}"
            )

        return results


# ============================================================
# RUN EVALUATION
# ============================================================

print("\nInitializing evaluator...")

evaluator = AdvancedFaceRecognitionEvaluator(
    dataset_path
)

evaluator.process_all_images()


# ============================================================
# Evaluate all variants
# ============================================================

all_results = {}

for variant in VARIANTS:

    all_results[variant] = (
        evaluator.evaluate_with_thresholds(
            variant
        )
    )


# ============================================================
# COMPREHENSIVE VISUALIZATIONS
# ============================================================

print("\n[3] GENERATING COMPREHENSIVE VISUALIZATIONS")
print("-" * 80)


# ============================================================
# Figure 1: FMR/FNMR vs Threshold
# ============================================================

fig1, axes = plt.subplots(
    1,
    len(VARIANTS),
    figsize=(18, 5)
)
axes = np.atleast_1d(axes)

fig1.suptitle(
    'FMR/FNMR vs Threshold - All Variants',
    fontsize=14,
    fontweight='bold'
)

for col, variant in enumerate(VARIANTS):

    ax = axes[col]

    data = all_results[variant]

    ax.plot(
        data['thresholds'],
        data['fmr_values'],
        'r-',
        linewidth=2,
        label='FMR (False Match Rate)'
    )

    ax.plot(
        data['thresholds'],
        data['fnmr_values'],
        'b-',
        linewidth=2,
        label='FNMR (False Non-Match Rate)'
    )

    ax.axvline(
        data['eer_threshold'],
        color='g',
        linestyle='--',
        linewidth=2,
        label=f'EER={data["eer"]:.4f}'
    )

    ax.fill_between(
        data['thresholds'],
        data['fmr_values'],
        data['fnmr_values'],
        alpha=0.1
    )

    ax.set_xlabel(
        'Threshold',
        fontsize=10
    )

    ax.set_ylabel(
        'Error Rate',
        fontsize=10
    )

    ax.set_title(
        f'{variant}\n'
        f'(EER: {data["eer"]:.4f}, '
        f'AUC: {data["auc"]:.4f})'
    )

    ax.legend(fontsize=9)

    ax.grid(
        True,
        alpha=0.3
    )

    ax.set_xlim(
        data['thresholds'].min(),
        data['thresholds'].max()
    )

    ax.set_ylim(
        0,
        1
    )

plt.tight_layout()

plt.savefig(
    f"{dataset_path}/fmr_fnmr_vs_threshold.png",
    dpi=150,
    bbox_inches='tight'
)

print("✓ Saved: fmr_fnmr_vs_threshold.png")

plt.close()


# ============================================================
# Figure 2: ROC Curves
# ============================================================

fig2, axes = plt.subplots(
    1,
    len(VARIANTS),
    figsize=(18, 5)
)
axes = np.atleast_1d(axes)

fig2.suptitle(
    'ROC Curves - All Variants',
    fontsize=14,
    fontweight='bold'
)

for col, variant in enumerate(VARIANTS):

    ax = axes[col]

    data = all_results[variant]

    ax.plot(
        data['fpr'],
        data['tpr'],
        'b-',
        linewidth=2.5,
        label=f'ROC (AUC = {data["auc"]:.4f})'
    )

    ax.plot(
        [0, 1],
        [0, 1],
        'k--',
        linewidth=1.5,
        label='Random Classifier'
    )

    ax.fill_between(
        data['fpr'],
        data['tpr'],
        alpha=0.1
    )

    ax.set_xlabel(
        'False Positive Rate (FMR)',
        fontsize=10
    )

    ax.set_ylabel(
        'True Positive Rate (1-FNMR)',
        fontsize=10
    )

    ax.set_title(
        f'{variant} ROC Curve'
    )

    ax.legend(
        fontsize=10,
        loc='lower right'
    )

    ax.grid(
        True,
        alpha=0.3
    )

    ax.set_xlim(
        0,
        1
    )

    ax.set_ylim(
        0,
        1
    )

plt.tight_layout()

plt.savefig(
    f"{dataset_path}/roc_curves.png",
    dpi=150,
    bbox_inches='tight'
)

print("✓ Saved: roc_curves.png")

plt.close()


# ============================================================
# Figure 3: Score Distributions
# ============================================================

fig3, axes = plt.subplots(
    len(VARIANTS),
    3,
    figsize=(16, 12)
)
axes = np.atleast_2d(axes)

fig3.suptitle(
    'Score Distributions - Genuine vs Impostor',
    fontsize=14,
    fontweight='bold'
)

for row, variant in enumerate(VARIANTS):

    data = all_results[variant]

    # --------------------------------------------------------
    # Score distribution
    # --------------------------------------------------------

    ax1 = axes[row, 0]

    ax1.hist(
        data['genuine_scores'],
        bins=30,
        alpha=0.6,
        label='Genuine',
        color='green',
        density=True
    )

    ax1.hist(
        data['impostor_scores'],
        bins=30,
        alpha=0.6,
        label='Impostor',
        color='red',
        density=True
    )

    ax1.axvline(
        data['eer_threshold'],
        color='black',
        linestyle='--',
        linewidth=2,
        label='EER Threshold'
    )

    ax1.set_xlabel(
        'Similarity Score'
    )

    ax1.set_ylabel(
        'Density'
    )

    ax1.set_title(
        f'{variant} - Score Distribution'
    )

    ax1.legend(
        fontsize=9
    )

    ax1.grid(
        True,
        alpha=0.3
    )

    # --------------------------------------------------------
    # Precision-Recall
    # --------------------------------------------------------

    ax2 = axes[row, 1]

    ax2.plot(
        data['recall'],
        data['precision'],
        'b-',
        linewidth=2.5
    )

    ax2.fill_between(
        data['recall'],
        data['precision'],
        alpha=0.1
    )

    ax2.set_xlabel(
        'Recall (Sensitivity)',
        fontsize=10
    )

    ax2.set_ylabel(
        'Precision',
        fontsize=10
    )

    ax2.set_title(
        f'{variant} - Precision-Recall'
    )

    ax2.grid(
        True,
        alpha=0.3
    )

    ax2.set_xlim(
        0,
        1
    )

    ax2.set_ylim(
        0,
        1
    )

    # --------------------------------------------------------
    # Threshold metrics table
    # --------------------------------------------------------

    ax3 = axes[row, 2]

    ax3.axis('off')

    table_data = []

    table_data.append([
        'Threshold',
        'FMR',
        'FNMR',
        'TAR'
    ])

    for tm in data['threshold_metrics'][::2]:

        table_data.append([
            f"{tm['threshold']:.2f}",
            f"{tm['FMR']:.4f}",
            f"{tm['FNMR']:.4f}",
            f"{tm['MATED']:.4f}"
        ])

    table = ax3.table(
        cellText=table_data,
        cellLoc='center',
        loc='center',
        colWidths=[
            0.25,
            0.25,
            0.25,
            0.25
        ]
    )

    table.auto_set_font_size(False)

    table.set_fontsize(9)

    table.scale(
        1,
        1.8
    )

    for i in range(4):

        table[(0, i)].set_facecolor(
            '#40466e'
        )

        table[(0, i)].set_text_props(
            weight='bold',
            color='white'
        )

    ax3.set_title(
        f'{variant} - Threshold Metrics',
        fontweight='bold'
    )

plt.tight_layout()

plt.savefig(
    f"{dataset_path}/score_distributions_detailed.png",
    dpi=150,
    bbox_inches='tight'
)

print("✓ Saved: score_distributions_detailed.png")

plt.close()


# ============================================================
# Figure 4: FMR/FNMR Heatmap
# ============================================================

fig4, axes = plt.subplots(
    1,
    len(VARIANTS),
    figsize=(16, 4)
)
axes = np.atleast_1d(axes)

fig4.suptitle(
    'FMR/FNMR Heatmap - Key Threshold Points',
    fontsize=14,
    fontweight='bold'
)

for col, variant in enumerate(VARIANTS):

    ax = axes[col]

    data = all_results[variant]

    metrics_matrix = []
    thresholds_list = []

    for tm in data['threshold_metrics']:

        thresholds_list.append(
            f"{tm['threshold']:.2f}"
        )

        metrics_matrix.append([
            tm['FMR'],
            tm['FNMR']
        ])

    # ========================================================
    # FIX:
    # This is now guaranteed to be 2 x 8
    # ========================================================

    metrics_array = np.array(
        metrics_matrix,
        dtype=float
    ).T

    # Safety check
    if metrics_array.size == 0:

        print(
            f"WARNING: No threshold metrics "
            f"available for {variant}"
        )

        continue

    im = ax.imshow(
        metrics_array,
        cmap='RdYlGn_r',
        aspect='auto',
        vmin=0,
        vmax=0.5
    )

    ax.set_xticks(
        range(len(thresholds_list))
    )

    ax.set_xticklabels(
        thresholds_list,
        rotation=45
    )

    ax.set_yticks([
        0,
        1
    ])

    ax.set_yticklabels([
        'FMR',
        'FNMR'
    ])

    ax.set_title(
        f'{variant}'
    )

    # --------------------------------------------------------
    # Text annotations
    # --------------------------------------------------------

    for i in range(
        metrics_array.shape[0]
    ):

        for j in range(
            metrics_array.shape[1]
        ):

            text = ax.text(
                j,
                i,
                f'{metrics_array[i, j]:.3f}',
                ha="center",
                va="center",
                color="black",
                fontsize=8
            )

    plt.colorbar(
        im,
        ax=ax,
        label='Error Rate'
    )

plt.tight_layout()

plt.savefig(
    f"{dataset_path}/threshold_heatmap.png",
    dpi=150,
    bbox_inches='tight'
)

print("✓ Saved: threshold_heatmap.png")

plt.close()


# ============================================================
# Figure 5: Performance Comparison
# ============================================================

fig5, axes = plt.subplots(
    2,
    2,
    figsize=(14, 10)
)

fig5.suptitle(
    'Performance Comparison Across Variants',
    fontsize=14,
    fontweight='bold'
)

variants = VARIANTS


# ------------------------------------------------------------
# EER Comparison
# ------------------------------------------------------------

ax1 = axes[0, 0]

eer_values = [
    all_results[v]['eer']
    for v in variants
]

colors = [
    '#2ecc71',
    '#f39c12',
    '#e74c3c'
]

bars1 = ax1.bar(
    variants,
    eer_values,
    color=colors,
    alpha=0.7,
    edgecolor='black'
)

ax1.set_ylabel(
    'EER',
    fontsize=11,
    fontweight='bold'
)

ax1.set_title(
    'Equal Error Rate (Lower is Better)'
)

ax1.set_ylim(
    0,
    max(eer_values) * 1.2
)

for i, (bar, val) in enumerate(
    zip(bars1, eer_values)
):

    ax1.text(
        bar.get_x()
        + bar.get_width() / 2,
        val + 0.01,
        f'{val:.4f}',
        ha='center',
        va='bottom',
        fontsize=10,
        fontweight='bold'
    )

ax1.grid(
    True,
    alpha=0.3,
    axis='y'
)


# ------------------------------------------------------------
# AUC Comparison
# ------------------------------------------------------------

ax2 = axes[0, 1]

auc_values = [
    all_results[v]['auc']
    for v in variants
]

bars2 = ax2.bar(
    variants,
    auc_values,
    color=colors,
    alpha=0.7,
    edgecolor='black'
)

ax2.set_ylabel(
    'AUC',
    fontsize=11,
    fontweight='bold'
)

ax2.set_title(
    'Area Under ROC Curve (Higher is Better)'
)

ax2.set_ylim(
    0,
    1
)

for i, (bar, val) in enumerate(
    zip(bars2, auc_values)
):

    ax2.text(
        bar.get_x()
        + bar.get_width() / 2,
        val + 0.02,
        f'{val:.4f}',
        ha='center',
        va='bottom',
        fontsize=10,
        fontweight='bold'
    )

ax2.grid(
    True,
    alpha=0.3,
    axis='y'
)


# ------------------------------------------------------------
# Genuine vs Impostor
# ------------------------------------------------------------

ax3 = axes[1, 0]

x = np.arange(
    len(variants)
)

width = 0.35

genuine_means = [
    all_results[v]['genuine_stats']['mean']
    for v in variants
]

impostor_means = [
    all_results[v]['impostor_stats']['mean']
    for v in variants
]

bars3a = ax3.bar(
    x - width / 2,
    genuine_means,
    width,
    label='Genuine',
    color='green',
    alpha=0.7
)

bars3b = ax3.bar(
    x + width / 2,
    impostor_means,
    width,
    label='Impostor',
    color='red',
    alpha=0.7
)

ax3.set_ylabel(
    'Mean Similarity Score',
    fontsize=11,
    fontweight='bold'
)

ax3.set_title(
    'Mean Scores Comparison'
)

ax3.set_xticks(x)

ax3.set_xticklabels(
    variants
)

ax3.legend()

ax3.grid(
    True,
    alpha=0.3,
    axis='y'
)


# ------------------------------------------------------------
# Score Gap
# ------------------------------------------------------------

ax4 = axes[1, 1]

gaps = [
    all_results[v]['genuine_stats']['mean']
    -
    all_results[v]['impostor_stats']['mean']
    for v in variants
]

bars4 = ax4.bar(
    variants,
    gaps,
    color=colors,
    alpha=0.7,
    edgecolor='black'
)

ax4.set_ylabel(
    'Score Gap',
    fontsize=11,
    fontweight='bold'
)

ax4.set_title(
    'Genuine-Impostor Separability (Higher is Better)'
)

for i, (bar, val) in enumerate(
    zip(bars4, gaps)
):

    ax4.text(
        bar.get_x()
        + bar.get_width() / 2,
        val + 0.01,
        f'{val:.4f}',
        ha='center',
        va='bottom',
        fontsize=10,
        fontweight='bold'
    )

ax4.grid(
    True,
    alpha=0.3,
    axis='y'
)

plt.tight_layout()

plt.savefig(
    f"{dataset_path}/performance_comparison.png",
    dpi=150,
    bbox_inches='tight'
)

print("✓ Saved: performance_comparison.png")

plt.close()


# ============================================================
# Figure 6: Detailed Threshold Operating Points
# ============================================================

fig6 = plt.figure(
    figsize=(16, 10)
)

gs = GridSpec(
    3,
    2,
    figure=fig6,
    hspace=0.35,
    wspace=0.3
)

fig6.suptitle(
    'Threshold Operating Points Analysis',
    fontsize=14,
    fontweight='bold'
)


# Helper function
def get_threshold_metric(data, threshold):

    for tm in data['threshold_metrics']:

        if abs(
            tm['threshold'] - threshold
        ) < 1e-9:

            return tm

    return {
        'FMR': 0.0,
        'FNMR': 0.0,
        'MATED': 1.0,
        'NONMATED': 0.0
    }


for idx, variant in enumerate(VARIANTS):

    data = all_results[variant]

    # --------------------------------------------------------
    # FMR/FNMR comparison
    # --------------------------------------------------------

    ax1 = fig6.add_subplot(
        gs[idx, 0]
    )

    thres_vals = [
        tm['threshold']
        for tm in data['threshold_metrics']
    ]

    fmr_vals = [
        tm['FMR']
        for tm in data['threshold_metrics']
    ]

    fnmr_vals = [
        tm['FNMR']
        for tm in data['threshold_metrics']
    ]

    tar_vals = [
        tm['MATED']
        for tm in data['threshold_metrics']
    ]

    ax1.plot(
        thres_vals,
        fmr_vals,
        'ro-',
        linewidth=2.5,
        markersize=8,
        label='FMR'
    )

    ax1.plot(
        thres_vals,
        fnmr_vals,
        'bs-',
        linewidth=2.5,
        markersize=8,
        label='FNMR'
    )

    ax1.plot(
        thres_vals,
        tar_vals,
        'g^-',
        linewidth=2.5,
        markersize=8,
        label='TAR (1-FNMR)'
    )

    ax1.set_xlabel(
        'Threshold',
        fontsize=10,
        fontweight='bold'
    )

    ax1.set_ylabel(
        'Error/Accept Rate',
        fontsize=10,
        fontweight='bold'
    )

    ax1.set_title(
        f'{variant} - Operating Points'
    )

    ax1.legend(
        fontsize=9,
        loc='best'
    )

    ax1.grid(
        True,
        alpha=0.3
    )

    ax1.set_ylim(
        -0.05,
        1.05
    )

    # --------------------------------------------------------
    # Statistics box
    # --------------------------------------------------------

    ax2 = fig6.add_subplot(
        gs[idx, 1]
    )

    ax2.axis('off')

    tm65 = get_threshold_metric(
        data,
        0.65
    )

    tm75 = get_threshold_metric(
        data,
        0.75
    )

    tm85 = get_threshold_metric(
        data,
        0.85
    )

    stats_text = f"""
{variant} - Summary Statistics

Genuine Pair Statistics:
  Count: {len(data['genuine_scores'])}
  Mean:  {data['genuine_stats']['mean']:.6f}
  Std:   {data['genuine_stats']['std']:.6f}
  Range: [{data['genuine_stats']['min']:.6f}, {data['genuine_stats']['max']:.6f}]

Impostor Pair Statistics:
  Count: {len(data['impostor_scores'])}
  Mean:  {data['impostor_stats']['mean']:.6f}
  Std:   {data['impostor_stats']['std']:.6f}
  Range: [{data['impostor_stats']['min']:.6f}, {data['impostor_stats']['max']:.6f}]

Performance Metrics:
  EER:  {data['eer']:.6f} @ T={data['eer_threshold']:.4f}
  AUC:  {data['auc']:.6f}
  Gap:  {data['genuine_stats']['mean'] - data['impostor_stats']['mean']:.6f}

Key Operating Points:
  @ T=0.65: FMR={tm65['FMR']:.4f}, FNMR={tm65['FNMR']:.4f}
  @ T=0.75: FMR={tm75['FMR']:.4f}, FNMR={tm75['FNMR']:.4f}
  @ T=0.85: FMR={tm85['FMR']:.4f}, FNMR={tm85['FNMR']:.4f}
"""

    ax2.text(
        0.05,
        0.95,
        stats_text,
        transform=ax2.transAxes,
        fontsize=9,
        verticalalignment='top',
        family='monospace',
        bbox=dict(
            boxstyle='round',
            facecolor='wheat',
            alpha=0.5
        )
    )


plt.savefig(
    f"{dataset_path}/threshold_operating_points.png",
    dpi=150,
    bbox_inches='tight'
)

print(
    "✓ Saved: threshold_operating_points.png"
)

plt.close()


# ============================================================
# SAVE COMPREHENSIVE RESULTS
# ============================================================

print("\n[4] SAVING COMPREHENSIVE RESULTS")
print("-" * 80)


comprehensive_results = {

    'timestamp':
        datetime.now().isoformat(),

    'dataset_info': {

        'total_subjects':
            len(metadata),

        'genuine_pairs':
            len(metadata),

        'impostor_pairs_tested':
            len(
            all_results[VARIANTS[0]]['impostor_scores']
            )

    },

    'variants': {}

}


for variant in VARIANTS:

    data = all_results[variant]

    comprehensive_results[
        'variants'
    ][variant] = {

        'statistics': {

            'genuine':
                data['genuine_stats'],

            'impostor':
                data['impostor_stats']

        },

        'performance_metrics': {

            'eer':
                data['eer'],

            'eer_threshold':
                data['eer_threshold'],

            'auc':
                data['auc'],

            'separability_gap':
                data['genuine_stats']['mean']
                -
                data['impostor_stats']['mean']

        },

        'threshold_operating_points':
            data['threshold_metrics']

    }


with open(
    f"{dataset_path}/comprehensive_evaluation_results.json",
    'w'
) as f:

    json.dump(
        comprehensive_results,
        f,
        indent=2
    )

print(
    "✓ Saved: comprehensive_evaluation_results.json"
)


# ============================================================
# CREATE DETAILED METRICS TABLE
# ============================================================

metrics_table_path = (
    f"{dataset_path}/detailed_threshold_metrics.txt"
)

with open(
    metrics_table_path,
    'w'
) as f:

    f.write(
        "=" * 100 + "\n"
    )

    f.write(
        "DETAILED THRESHOLD METRICS FOR ALL VARIANTS\n"
    )

    f.write(
        "=" * 100 + "\n\n"
    )

    for variant in VARIANTS:

        data = all_results[variant]

        f.write(
            f"\n{'=' * 100}\n"
        )

        f.write(
            f"{variant} - COMPLETE ANALYSIS\n"
        )

        f.write(
            f"{'=' * 100}\n\n"
        )

        f.write(
            "Genuine Pairs Statistics:\n"
        )

        f.write(
            f"  Count:   "
            f"{len(data['genuine_scores'])}\n"
        )

        f.write(
            f"  Mean:    "
            f"{data['genuine_stats']['mean']:.6f}\n"
        )

        f.write(
            f"  Std Dev: "
            f"{data['genuine_stats']['std']:.6f}\n"
        )

        f.write(
            f"  Min:     "
            f"{data['genuine_stats']['min']:.6f}\n"
        )

        f.write(
            f"  Max:     "
            f"{data['genuine_stats']['max']:.6f}\n\n"
        )

        f.write(
            "Impostor Pairs Statistics:\n"
        )

        f.write(
            f"  Count:   "
            f"{len(data['impostor_scores'])}\n"
        )

        f.write(
            f"  Mean:    "
            f"{data['impostor_stats']['mean']:.6f}\n"
        )

        f.write(
            f"  Std Dev: "
            f"{data['impostor_stats']['std']:.6f}\n"
        )

        f.write(
            f"  Min:     "
            f"{data['impostor_stats']['min']:.6f}\n"
        )

        f.write(
            f"  Max:     "
            f"{data['impostor_stats']['max']:.6f}\n\n"
        )

        f.write(
            "Performance Metrics:\n"
        )

        f.write(
            f"  EER:       "
            f"{data['eer']:.6f} "
            f"({data['eer'] * 100:.3f}%)\n"
        )

        f.write(
            f"  EER Threshold: "
            f"{data['eer_threshold']:.6f}\n"
        )

        f.write(
            f"  AUC:       "
            f"{data['auc']:.6f}\n"
        )

        f.write(
            f"  Gap:       "
            f"{data['genuine_stats']['mean'] - data['impostor_stats']['mean']:.6f}\n\n"
        )

        f.write(
            "Threshold Operating Points:\n"
        )

        f.write(
            f"  {'Threshold':<12}"
            f"{'FMR':<15}"
            f"{'FNMR':<15}"
            f"{'TAR':<15}"
            f"{'FAR':<15}\n"
        )

        f.write(
            f"  {'-' * 72}\n"
        )

        for tm in data['threshold_metrics']:

            f.write(
                f"  "
                f"{tm['threshold']:<12.4f}"
                f"{tm['FMR']:<15.6f}"
                f"{tm['FNMR']:<15.6f}"
                f"{tm['MATED']:<15.6f}"
                f"{tm['NONMATED']:<15.6f}\n"
            )

        f.write("\n")


print(
    "✓ Saved: detailed_threshold_metrics.txt"
)


# ============================================================
# SUMMARY STATISTICS
# ============================================================

print("\n" + "=" * 80)
print("EVALUATION SUMMARY")
print("=" * 80)


summary_data = []


for variant in VARIANTS:

    data = all_results[variant]

    summary_data.append({

        'Variant':
            variant,

        'EER':
            f"{data['eer']:.6f}",

        'EER Threshold':
            f"{data['eer_threshold']:.4f}",

        'AUC':
            f"{data['auc']:.6f}",

        'Genuine Mean':
            f"{data['genuine_stats']['mean']:.6f}",

        'Impostor Mean':
            f"{data['impostor_stats']['mean']:.6f}",

        'Gap':
            f"{data['genuine_stats']['mean'] - data['impostor_stats']['mean']:.6f}"

    })


print(
    f"\n{'Variant':<15}"
    f"{'EER':<12}"
    f"{'AUC':<12}"
    f"{'Genuine μ':<15}"
    f"{'Impostor μ':<15}"
    f"{'Gap':<12}"
)

print(
    "-" * 80
)


for sd in summary_data:

    print(
        f"{sd['Variant']:<15}"
        f"{sd['EER']:<12}"
        f"{sd['AUC']:<12}"
        f"{sd['Genuine Mean']:<15}"
        f"{sd['Impostor Mean']:<15}"
        f"{sd['Gap']:<12}"
    )


# ============================================================
# DONE
# ============================================================

print("\n" + "=" * 80)
print("✓ COMPREHENSIVE EVALUATION COMPLETE!")
print("=" * 80)


print("\nGenerated Files:")

print(
    "  1. fmr_fnmr_vs_threshold.png "
    "- FMR/FNMR curves for all variants"
)

print(
    "  2. roc_curves.png "
    "- ROC curves with AUC values"
)

print(
    "  3. score_distributions_detailed.png "
    "- Detailed score analysis"
)

print(
    "  4. threshold_heatmap.png "
    "- Error rate heatmap"
)

print(
    "  5. performance_comparison.png "
    "- Variant comparison"
)

print(
    "  6. threshold_operating_points.png "
    "- Detailed operating points"
)

print(
    "  7. comprehensive_evaluation_results.json "
    "- All metrics in JSON"
)

print(
    "  8. detailed_threshold_metrics.txt "
    "- Detailed text report"
)

print("\n✓ All analyses complete and saved!")
