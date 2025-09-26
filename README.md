# Wavelet-Based Parity Detection 🧠

This project explores an unconventional approach to the classic **parity detection problem** — determining whether a number is odd or even — using **wavelet-based feature extraction** and **unsupervised clustering**.

Instead of relying on simple modular arithmetic, we transform integers into **binary signals**, apply **Discrete Wavelet Transform (DWT)** to extract **multi-scale features**, and then use **k-means clustering** to test whether these structural patterns reveal parity — even though the problem is purely symbolic.

---

## 🚀 Features
- Converts integers into binary signal representations
- Performs multi-level **wavelet decomposition** (`pywt`)
- Extracts signal-based statistical features: **Energy**, **L2 Norm**, and **Mean Absolute Value**
- Applies **unsupervised k-means clustering** for parity classification
- Calculates and prints classification accuracy

---

## 📦 Installation

Clone the repository and install the dependencies:

```bash
git clone https://github.com/Ertugrulmutlu/Using-Wavelets-and-Clustering-to-Predict-Odd-or-Even-Numbers.git
cd Using-Wavelets-and-Clustering-to-Predict-Odd-or-Even-Numbers
pip install -r requirements.txt
```

---

## ▶️ Run the Script

Simply run the main script:

```bash
python main.py
```

This will:
- Compute wavelet-based features for integers
- Cluster them without any labels
- Estimate the parity classification accuracy (~69.67%)
- Print the final results to the console

---

## 📚 Citation

If you use this code in your research, please cite:

```
Mutlu, Ertuğrul. "Wavelet-Based Feature Extraction and Clustering for Parity Detection: A Feature Engineering Perspective." 2025.
```

---

## 🧪 Requirements

- Python 3.8+
- numpy
- pywt
- scikit-learn
- matplotlib

Install them manually or use:

```bash
pip install -r requirements.txt
```

---

## 📁 Project Structure

```
├── main.py              # Core implementation
├── requirements.txt     # Dependencies
├── README.md            # Project documentation
└── run.sh               # Quick run script (optional)
```

---

## 🧠 Author

**Ertuğrul Mutlu**  
RWTH Aachen University - Department of Computer Engineering  
📧 Contact: ertugrulmutlu004@gmail.com
