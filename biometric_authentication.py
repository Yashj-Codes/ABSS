"""
User Authentication using Biometric Features
Advanced Biometric Systems and Security (AI461) - Assignment 1
SVNIT Surat

Author: Yash
--------------------------------------------------------------------
Pipeline:
1. Load 144-dim feature vectors for 1000 samples (100 users x 10 images each,
   stored column-major: 144 rows (features) x 1000 cols (samples), samples
   grouped consecutively in blocks of 10 per user).
2. Split each user's 10 samples -> first 5 = enrollment (train), last 5 = test.
3. Build ONE template per user by averaging the 5 enrollment feature vectors.
4. Score every test sample against:
      - its own template   -> GENUINE score
      - every OTHER user's template -> IMPOSTOR score
   using two distance/similarity measures:
      - Euclidean distance   (lower  = more similar)
      - Cosine similarity    (higher = more similar)
5. From the genuine / impostor score pools compute:
      A. Genuine score distribution
      B. Impostor score distribution
      C. FAR vs threshold
      D. FRR vs threshold
      E. ROC curve (TAR vs FAR)
      F. Equal Error Rate (EER) and Decidability Index (d')
6. Compare Euclidean vs Cosine and conclude which is better.

Total comparisons (matches the assignment's design diagram):
   genuine  : 100 users x 5 test samples                = 500
   impostor : 500 test samples x 99 other-user templates = 49,500
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

# --------------------------------------------------------------------------
# CONFIG
# --------------------------------------------------------------------------
DATA_FILE = "data/biomet_data.csv"
OUT_DIR = Path("plots")
OUT_DIR.mkdir(exist_ok=True)

N_USERS = 100
N_SAMPLES_PER_USER = 10
N_TRAIN = 5
N_TEST = 5
N_FEATURES = 144

np.random.seed(42)

# --------------------------------------------------------------------------
# 1. LOAD DATA
# --------------------------------------------------------------------------
def load_data(path):
    """
    Raw file: 144 rows (features) x 1000 tab-separated columns (samples),
    with a trailing empty column from the trailing tab on every line.
    Returns X of shape (1000, 144): rows = samples, cols = features.
    """
    df = pd.read_csv(path, delimiter="\t", header=None)
    df = df.dropna(axis=1, how="all")          # drop the trailing empty column
    raw = df.values.astype(float)               # (144, 1000)
    assert raw.shape == (N_FEATURES, N_USERS * N_SAMPLES_PER_USER), raw.shape
    X = raw.T                                    # (1000, 144)
    return X


def reshape_by_user(X):
    """(1000, 144) -> (100 users, 10 samples, 144 features)"""
    return X.reshape(N_USERS, N_SAMPLES_PER_USER, N_FEATURES)


# --------------------------------------------------------------------------
# 2 & 3. TRAIN/TEST SPLIT + TEMPLATE CREATION
# --------------------------------------------------------------------------
def build_templates(X_user):
    """
    X_user: (100, 10, 144)
    returns:
      templates : (100, 144)  -- mean of first 5 (enrollment) samples per user
      test_set  : (100, 5, 144) -- remaining 5 samples per user
    """
    train = X_user[:, :N_TRAIN, :]
    test = X_user[:, N_TRAIN:, :]
    templates = train.mean(axis=1)
    return templates, test


# --------------------------------------------------------------------------
# 4. MATCHING SCORES (Euclidean distance & Cosine similarity)
# --------------------------------------------------------------------------
def euclidean_distance(a, b):
    return np.linalg.norm(a - b)


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)


def compute_scores(templates, test_set):
    """
    Returns dict with genuine/impostor score arrays for both metrics.
    genuine  : 100 users * 5 test samples          = 500 scores
    impostor : 500 test samples * 99 other templates = 49500 scores
    """
    genuine_euc, genuine_cos = [], []
    impostor_euc, impostor_cos = [], []

    for u in range(N_USERS):
        for t in range(N_TEST):
            probe = test_set[u, t]
            # genuine: compare against own template
            genuine_euc.append(euclidean_distance(probe, templates[u]))
            genuine_cos.append(cosine_similarity(probe, templates[u]))
            # impostor: compare against every other user's template
            for v in range(N_USERS):
                if v == u:
                    continue
                impostor_euc.append(euclidean_distance(probe, templates[v]))
                impostor_cos.append(cosine_similarity(probe, templates[v]))

    return {
        "euclidean": {
            "genuine": np.array(genuine_euc),
            "impostor": np.array(impostor_euc),
            "higher_is_similar": False,
        },
        "cosine": {
            "genuine": np.array(genuine_cos),
            "impostor": np.array(impostor_cos),
            "higher_is_similar": True,
        },
    }


# --------------------------------------------------------------------------
# 5C/5D. FAR / FRR vs THRESHOLD
# --------------------------------------------------------------------------
def far_frr_curves(genuine, impostor, higher_is_similar, n_points=1000):
    lo = min(genuine.min(), impostor.min())
    hi = max(genuine.max(), impostor.max())
    thresholds = np.linspace(lo, hi, n_points)

    if higher_is_similar:
        # accept if score >= threshold
        FAR = np.array([(impostor >= th).mean() for th in thresholds])
        FRR = np.array([(genuine < th).mean() for th in thresholds])
    else:
        # accept if score <= threshold (distance)
        FAR = np.array([(impostor <= th).mean() for th in thresholds])
        FRR = np.array([(genuine > th).mean() for th in thresholds])

    return thresholds, FAR, FRR


# --------------------------------------------------------------------------
# 5F. EER + DECIDABILITY INDEX
# --------------------------------------------------------------------------
def compute_eer(thresholds, FAR, FRR):
    diff = FAR - FRR
    sign_change = np.where(np.diff(np.sign(diff)) != 0)[0]
    if len(sign_change) == 0:
        idx = np.argmin(np.abs(diff))
        return FAR[idx], thresholds[idx]
    i = sign_change[0]
    # linear interpolation between i and i+1 for the crossing point
    x0, x1 = thresholds[i], thresholds[i + 1]
    f0, f1 = diff[i], diff[i + 1]
    t_cross = x0 + (x1 - x0) * (-f0) / (f1 - f0)
    far0, far1 = FAR[i], FAR[i + 1]
    eer = far0 + (far1 - far0) * (t_cross - x0) / (x1 - x0)
    return eer, t_cross


def decidability_index(genuine, impostor):
    mu_g, mu_i = genuine.mean(), impostor.mean()
    var_g, var_i = genuine.var(), impostor.var()
    return abs(mu_g - mu_i) / np.sqrt((var_g + var_i) / 2)


# --------------------------------------------------------------------------
# PLOTTING
# --------------------------------------------------------------------------
def plot_genuine_impostor(genuine, impostor, metric_name, fname):
    plt.figure(figsize=(7, 5))
    plt.hist(genuine, bins=50, alpha=0.6, density=True, label="Genuine", color="seagreen")
    plt.hist(impostor, bins=50, alpha=0.6, density=True, label="Impostor", color="crimson")
    plt.title(f"Genuine vs Impostor Score Distribution — {metric_name}")
    xlabel = "Euclidean Distance" if metric_name == "Euclidean" else "Cosine Similarity"
    plt.xlabel(xlabel)
    plt.ylabel("Probability Density")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / fname, dpi=150)
    plt.close()


def plot_far_frr(thresholds, FAR, FRR, eer, eer_th, metric_name, fname):
    plt.figure(figsize=(7, 5))
    plt.plot(thresholds, FAR, label="FAR", color="crimson")
    plt.plot(thresholds, FRR, label="FRR", color="steelblue")
    plt.axvline(eer_th, color="gray", linestyle="--", linewidth=1)
    plt.scatter([eer_th], [eer], color="black", zorder=5,
                label=f"EER = {eer*100:.2f}% @ th={eer_th:.3f}")
    plt.title(f"FAR / FRR vs Threshold — {metric_name}")
    plt.xlabel("Threshold")
    plt.ylabel("Error Rate")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / fname, dpi=150)
    plt.close()


def plot_roc(results, fname):
    plt.figure(figsize=(7, 6))
    colors = {"euclidean": "crimson", "cosine": "steelblue"}
    for key, res in results.items():
        FAR = res["FAR"]
        TAR = 1 - res["FRR"]
        # sort by FAR for a clean log-scale plot
        order = np.argsort(FAR)
        FAR_sorted, TAR_sorted = FAR[order], TAR[order]
        plt.plot(FAR_sorted, TAR_sorted, label=f"{key.capitalize()} (EER={res['eer']*100:.2f}%)",
                  color=colors[key])
    plt.xscale("log")
    plt.xlim(1e-4, 1)
    plt.xlabel("False Acceptance Rate (FAR) [log scale]")
    plt.ylabel("True Acceptance Rate (TAR = 1-FRR)")
    plt.title("ROC Curve: Euclidean vs Cosine")
    plt.plot([1e-4, 1], [1e-4, 1], linestyle=":", color="gray", label="Chance line")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_DIR / fname, dpi=150)
    plt.close()


# --------------------------------------------------------------------------
# MAIN
# --------------------------------------------------------------------------
def main():
    X = load_data(DATA_FILE)
    X_user = reshape_by_user(X)
    templates, test_set = build_templates(X_user)
    scores = compute_scores(templates, test_set)

    summary_lines = []
    summary_lines.append("=" * 70)
    summary_lines.append("BIOMETRIC AUTHENTICATION — RESULTS SUMMARY")
    summary_lines.append("=" * 70)
    summary_lines.append(f"Users: {N_USERS} | Enrollment/user: {N_TRAIN} | Test/user: {N_TEST}")
    summary_lines.append(f"Genuine comparisons : {N_USERS * N_TEST}")
    summary_lines.append(f"Impostor comparisons: {N_USERS * N_TEST * (N_USERS - 1)}")
    summary_lines.append("")

    roc_results = {}

    for metric_key, metric_label in [("euclidean", "Euclidean"), ("cosine", "Cosine")]:
        g = scores[metric_key]["genuine"]
        i = scores[metric_key]["impostor"]
        higher_is_similar = scores[metric_key]["higher_is_similar"]

        # A & B: distributions
        plot_genuine_impostor(g, i, metric_label, f"AB_genuine_impostor_{metric_key}.png")

        # C & D: FAR / FRR curves
        thresholds, FAR, FRR = far_frr_curves(g, i, higher_is_similar)
        eer, eer_th = compute_eer(thresholds, FAR, FRR)
        plot_far_frr(thresholds, FAR, FRR, eer, eer_th, metric_label,
                     f"CD_FAR_FRR_{metric_key}.png")

        # F: decidability index
        d_prime = decidability_index(g, i)

        roc_results[metric_key] = {"FAR": FAR, "FRR": FRR, "eer": eer}

        summary_lines.append(f"--- {metric_label} Distance ---")
        summary_lines.append(f"  Genuine  mean={g.mean():.4f}  std={g.std():.4f}")
        summary_lines.append(f"  Impostor mean={i.mean():.4f}  std={i.std():.4f}")
        summary_lines.append(f"  EER  = {eer*100:.3f}%  (threshold = {eer_th:.4f})")
        summary_lines.append(f"  Decidability Index d' = {d_prime:.4f}")
        summary_lines.append("")

    # E: combined ROC curve
    plot_roc(roc_results, "E_ROC_comparison.png")

    eer_euc = roc_results["euclidean"]["eer"]
    eer_cos = roc_results["cosine"]["eer"]
    better = "Cosine Similarity" if eer_cos < eer_euc else "Euclidean Distance"
    summary_lines.append("=" * 70)
    summary_lines.append(f"CONCLUSION: {better} gives the lower EER "
                          f"({min(eer_cos, eer_euc)*100:.3f}% vs "
                          f"{max(eer_cos, eer_euc)*100:.3f}%) and is the better "
                          f"matching criterion for this dataset.")
    summary_lines.append("=" * 70)

    summary_text = "\n".join(summary_lines)
    print(summary_text)
    with open("results_summary.txt", "w") as f:
        f.write(summary_text)


if __name__ == "__main__":
    main()
