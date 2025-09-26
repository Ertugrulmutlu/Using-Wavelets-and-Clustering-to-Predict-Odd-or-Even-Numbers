# Wavelet Parity Detection

This repository implements the code for the paper **"Wavelet-Based Feature Extraction and Clustering for Parity Detection: A Feature Engineering Perspective"**.

The goal is to solve a trivial mathematical problem — determining whether a number is odd or even — without using modular arithmetic. Instead, we explore a feature engineering approach based on wavelet decomposition and unsupervised clustering.

## 🧠 Overview

* Convert integers into binary signals.
* Apply multi-level discrete wavelet transform (DWT).
* Extract statistical features (energy, L2 norm, mean absolute value).
* Perform per-feature k-means clustering (unsupervised).
* Aggregate cluster probabilities to compute an "oddness score."
* Classify numbers as odd/even.

📊 **Result:** ~69.67% accuracy without using any arithmetic rule.

## ⚙️ Usage

```bash
pip install -r requirements.txt
python main.py --start 1 --end 1000 --wavelet haar --max_level 3
```

Outputs:

* 📈 `figures/cluster_scores.png` – visualization of oddness scores
* 📁 `results/metrics.json` – accuracy and experiment metadata
* 📁 `results/scores.npz` – all scores and predictions

## 📁 Project Structure

```
├─ main.py                # main script
├─ requirements.txt      # dependencies
├─ figures/              # generated plots
├─ results/              # saved results
└─ README.md            # this file
```

## 📄 Citation

If you use this code, please cite the corresponding student paper:

```
E. Mutlu, "Wavelet-Based Feature Extraction and Clustering for Parity Detection: A Feature Engineering Perspective," RWTH Aachen University, 2025.
```
