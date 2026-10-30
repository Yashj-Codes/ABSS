import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


DATA_PATH = Path('biomet_data.csv')
OUTPUT_DIR = Path('results')
OUTPUT_DIR.mkdir(exist_ok=True)


def load_data(path: Path) -> np.ndarray:
    data = np.loadtxt(path, delimiter=None)
    if data.ndim != 2:
        raise ValueError(f'Expected 2D data, got shape {data.shape}')
    return data


def build_user_templates(data: np.ndarray, users: int = 100, samples_per_user: int = 10) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # The CSV stores one feature vector per column. We assume the columns are arranged as:
    # user 1: samples 1-10, user 2: samples 1-10, ..., user 100: samples 1-10.
    samples = data.T  # shape: (1000, 144)
    reshaped = samples.reshape(users, samples_per_user, data.shape[0])
    train = reshaped[:, :5, :]
    test = reshaped[:, 5:, :]

    templates = train.mean(axis=1)  # shape: (100, 144)
    train_vectors = train  # shape: (100, 5, 144)
    test_vectors = test    # shape: (100, 5, 144)
    return templates, train_vectors, test_vectors


def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    return np.linalg.norm(a - b)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)
    if a_norm == 0 or b_norm == 0:
        return 0.0
    return float(np.dot(a, b) / (a_norm * b_norm))


def compute_scores(templates: np.ndarray, test_vectors: np.ndarray, metric: str) -> tuple[np.ndarray, np.ndarray]:
    genuine_scores = []
    imposter_scores = []

    for user_idx in range(templates.shape[0]):
        template = templates[user_idx]
        for test_vector in test_vectors[user_idx]:
            if metric == 'euclidean':
                genuine_scores.append(euclidean_distance(test_vector, template))
                imposter_scores.append(
                    np.mean([
                        euclidean_distance(test_vector, templates[other_idx])
                        for other_idx in range(templates.shape[0])
                        if other_idx != user_idx
                    ])
                )
            else:
                genuine_scores.append(cosine_similarity(test_vector, template))
                imposter_scores.append(
                    np.mean([
                        cosine_similarity(test_vector, templates[other_idx])
                        for other_idx in range(templates.shape[0])
                        if other_idx != user_idx
                    ])
                )

    return np.array(genuine_scores), np.array(imposter_scores)


def compute_metrics(genuine: np.ndarray, imposter: np.ndarray, metric: str) -> dict:
    if metric == 'euclidean':
        thresholds = np.linspace(min(np.min(genuine), np.min(imposter)) - 1e-6,
                                 max(np.max(genuine), np.max(imposter)) + 1e-6,
                                 400)
        far = []
        frr = []
        for threshold in thresholds:
            far.append(np.mean(imposter <= threshold))
            frr.append(np.mean(genuine > threshold))
    else:
        thresholds = np.linspace(min(np.min(genuine), np.min(imposter)) - 1e-6,
                                 max(np.max(genuine), np.max(imposter)) + 1e-6,
                                 400)
        far = []
        frr = []
        for threshold in thresholds:
            far.append(np.mean(imposter >= threshold))
            frr.append(np.mean(genuine < threshold))

    far = np.array(far)
    frr = np.array(frr)
    gar = 1.0 - frr

    # EER via interpolation between the two points that bracket the equal error.
    diff = np.abs(far - frr)
    idx = np.argmin(diff)
    if idx == 0:
        eer = float(far[0])
    elif idx == len(thresholds) - 1:
        eer = float(far[-1])
    else:
        x1, x2 = thresholds[idx - 1], thresholds[idx]
        y1, y2 = far[idx - 1] - frr[idx - 1], far[idx] - frr[idx]
        if y2 == y1:
            eer = float((far[idx] + far[idx - 1]) / 2.0)
        else:
            eer = float(np.interp(0.0, [y1, y2], [x1, x2]))

    decidability = abs(np.mean(genuine) - np.mean(imposter)) / np.sqrt(0.5 * (np.var(genuine) + np.var(imposter)))

    return {
        'thresholds': thresholds,
        'far': far,
        'frr': frr,
        'gar': gar,
        'eer': float(eer),
        'decidability': float(decidability),
        'genuine_mean': float(np.mean(genuine)),
        'genuine_std': float(np.std(genuine)),
        'imposter_mean': float(np.mean(imposter)),
        'imposter_std': float(np.std(imposter)),
    }


def save_summary(summary: dict, metric: str) -> None:
    with open(OUTPUT_DIR / f'{metric}_summary.txt', 'w', encoding='utf-8') as f:
        f.write(f'{metric.upper()} RESULTS\n')
        f.write('===================\n')
        f.write(f"Genuine mean: {summary['genuine_mean']:.4f}\n")
        f.write(f"Genuine std : {summary['genuine_std']:.4f}\n")
        f.write(f"Imposter mean: {summary['imposter_mean']:.4f}\n")
        f.write(f"Imposter std : {summary['imposter_std']:.4f}\n")
        f.write(f"EER: {summary['eer']:.4f}\n")
        f.write(f"Decidability index: {summary['decidability']:.4f}\n")


def plot_histograms(genuine: np.ndarray, imposter: np.ndarray, metric: str) -> None:
    plt.figure(figsize=(8, 4))
    plt.hist(genuine, bins=30, density=True, alpha=0.7, label='Genuine', color='green')
    plt.hist(imposter, bins=30, density=True, alpha=0.7, label='Imposter', color='red')
    plt.xlabel(metric.replace('_', ' ').title())
    plt.ylabel('Density')
    plt.title(f'{metric.replace("_", " ").title()} score distributions')
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f'{metric}_distribution.png', dpi=200)
    plt.close()


def plot_far_frr(thresholds: np.ndarray, far: np.ndarray, frr: np.ndarray, metric: str) -> None:
    plt.figure(figsize=(8, 4))
    plt.plot(thresholds, far, label='FAR', color='red')
    plt.plot(thresholds, frr, label='FRR', color='blue')
    plt.xlabel(metric.replace('_', ' ').title() + ' threshold')
    plt.ylabel('Error rate')
    plt.title(f'FAR/FRR curve ({metric})')
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f'{metric}_far_frr.png', dpi=200)
    plt.close()


def plot_roc(far: np.ndarray, gar: np.ndarray, metric: str) -> None:
    plt.figure(figsize=(8, 4))
    plt.plot(far, gar, color='purple')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel('FAR')
    plt.ylabel('GAR')
    plt.title(f'ROC curve ({metric})')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f'{metric}_roc.png', dpi=200)
    plt.close()


def main() -> None:
    data = load_data(DATA_PATH)
    templates, _, test_vectors = build_user_templates(data)

    for metric in ['euclidean', 'cosine']:
        genuine, imposter = compute_scores(templates, test_vectors, metric)
        metrics = compute_metrics(genuine, imposter, metric)
        save_summary(metrics, metric)
        plot_histograms(genuine, imposter, metric)
        plot_far_frr(metrics['thresholds'], metrics['far'], metrics['frr'], metric)
        plot_roc(metrics['far'], metrics['gar'], metric)

        print(f'[{metric.upper()}]')
        print(f"  Genuine mean/std: {metrics['genuine_mean']:.4f} / {metrics['genuine_std']:.4f}")
        print(f"  Imposter mean/std: {metrics['imposter_mean']:.4f} / {metrics['imposter_std']:.4f}")
        print(f"  EER: {metrics['eer']:.4f}")
        print(f"  Decidability index: {metrics['decidability']:.4f}")

    # Compare which metric performs better.
    euclidean_metrics = compute_metrics(*compute_scores(templates, test_vectors, 'euclidean'), 'euclidean')
    cosine_metrics = compute_metrics(*compute_scores(templates, test_vectors, 'cosine'), 'cosine')

    best_metric = 'cosine' if cosine_metrics['eer'] < euclidean_metrics['eer'] else 'euclidean'
    print('\nBest matching criterion based on EER:')
    print(f'  {best_metric.upper()}')


if __name__ == '__main__':
    main()
