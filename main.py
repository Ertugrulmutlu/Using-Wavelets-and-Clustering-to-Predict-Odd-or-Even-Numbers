#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path

import numpy as np
import pywt
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


def number_to_bit_signal(num: int) -> np.ndarray:
    """Convert integer to binary array and pad to next power-of-two length."""
    bit_str = bin(num)[2:]
    bit_arr = np.array([int(b) for b in bit_str], dtype=np.float64)
    next_pow2 = 1 << int(np.ceil(np.log2(len(bit_arr)))) if len(bit_arr) > 1 else 1
    pad_len = max(0, next_pow2 - len(bit_arr))
    return np.pad(bit_arr, (0, pad_len), mode="constant")


def extract_features(num: int, wavelet: str = "haar", max_level: int = 3):
    """Compute [energy, l2_norm, mean_abs] per DWT level (approx + details)."""
    sig = number_to_bit_signal(num)
    max_possible = pywt.dwt_max_level(len(sig), pywt.Wavelet(wavelet).dec_len)
    level = min(max_level, max_possible) if max_possible > 0 else 1
    coeffs = pywt.wavedec(sig, wavelet, level=level)

    feats = []
    for c in coeffs:
        energy = float(np.sum(c ** 2))
        l2 = float(np.linalg.norm(c))
        mav = float(np.mean(np.abs(c)))
        feats.append([energy, l2, mav])
    return feats  # list[level][feature_idx]


def run_experiment(
    start: int = 1,
    end: int = 1000,
    wavelet: str = "haar",
    max_level: int = 3,
    random_state: int = 42,
):
    """Core pipeline: feature extraction → per-(level,feature) k-means → probabilities → scores."""
    numbers = np.arange(start, end, dtype=int)
    labels = numbers % 2  # 1=odd, 0=even

    all_features = [extract_features(n, wavelet=wavelet, max_level=max_level) for n in numbers]
    max_level_eff = max(len(f) for f in all_features)

    # stack features per level; shape per level: (N, 3)
    features_by_level = []
    for lvl in range(max_level_eff):
        lvl_rows = []
        for f in all_features:
            lvl_rows.append(f[lvl] if lvl < len(f) else [0.0, 0.0, 0.0])
        features_by_level.append(np.asarray(lvl_rows, dtype=np.float64))

    # probabilities[n, level, feature] = P(odd | cluster of (n, level, feature))
    probabilities = np.zeros((len(numbers), max_level_eff, 3), dtype=np.float64)

    for lvl in range(max_level_eff):
        for feat_idx in range(3):
            X = features_by_level[lvl][:, feat_idx].reshape(-1, 1)
            km = KMeans(n_clusters=2, n_init="auto", random_state=random_state)
            clusters = km.fit_predict(X)

            # Probability of odd within each cluster
            for cl in (0, 1):
                members = (clusters == cl)
                if np.any(members):
                    p_odd = float(np.mean(labels[members]))
                    probabilities[members, lvl, feat_idx] = p_odd
                else:
                    probabilities[members, lvl, feat_idx] = 0.5  # degenerate safeguard

    # weights
    level_weights = np.linspace(0.5, 1.5, max_level_eff)  # shape (L,)
    feature_weights = np.array([1.0, 1.0, 1.0])           # energy, l2, mav

    weighted = (
        probabilities
        * level_weights.reshape(1, max_level_eff, 1)
        * feature_weights.reshape(1, 1, 3)
    )
    final_scores = np.sum(weighted, axis=(1, 2)) / (np.sum(level_weights) * np.sum(feature_weights))
    preds = (final_scores > 0.5).astype(int)
    acc = float(np.mean(preds == labels))

    results = {
        "start": int(start),
        "end": int(end),
        "wavelet": wavelet,
        "max_level": int(max_level),
        "random_state": int(random_state),
        "accuracy": acc,
        "n_samples": int(len(numbers)),
    }
    return numbers, labels, final_scores, preds, results


def plot_scores(numbers, labels, final_scores, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(12, 5))
    colors = ["red" if l == 1 else "blue" for l in labels]
    plt.scatter(numbers, final_scores, c=colors, s=10, alpha=0.8,
                label="Odd (red) / Even (blue)")
    plt.axhline(0.5, linestyle="--", label="Decision threshold (0.5)")
    plt.title("Wavelet Features + K-Means: Predicted Oddness Score")
    plt.xlabel("Integer n")
    plt.ylabel("Oddness score")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Wavelet + KMeans Parity Detection")
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--end", type=int, default=1000, help="exclusive")
    parser.add_argument("--wavelet", type=str, default="haar")
    parser.add_argument("--max_level", type=int, default=3)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--fig", type=str, default="figures/cluster_scores.png")
    parser.add_argument("--results_dir", type=str, default="results")
    args = parser.parse_args()

    numbers, labels, scores, preds, meta = run_experiment(
        start=args.start,
        end=args.end,
        wavelet=args.wavelet,
        max_level=args.max_level,
        random_state=args.random_state,
    )

    print(f"Final Accuracy: {meta['accuracy']*100:.2f}% ({meta['n_samples']} samples)")

    # save plot & results
    plot_scores(numbers, labels, scores, Path(args.fig))

    res_dir = Path(args.results_dir)
    res_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(res_dir / "scores.npz",
                        numbers=numbers, labels=labels, scores=scores, preds=preds)
    with open(res_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


if __name__ == "__main__":
    os.environ.setdefault("MPLCONFIGDIR", str(Path.home() / ".matplotlib"))
    main()
