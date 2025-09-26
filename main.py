import numpy as np
import pywt
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def number_to_bit_signal(num):
    bit_str = bin(num)[2:]
    bit_arr = np.array([int(b) for b in bit_str])
    pad_len = 2**int(np.ceil(np.log2(len(bit_arr)))) - len(bit_arr)
    return np.pad(bit_arr, (0, pad_len), 'constant')

def extract_features(num, wavelet='haar', max_level=3):
    sig = number_to_bit_signal(num)
    max_possible_level = pywt.dwt_max_level(len(sig), pywt.Wavelet(wavelet).dec_len)
    level = min(max_level, max_possible_level) if max_possible_level > 0 else 1
    coeffs = pywt.wavedec(sig, wavelet, level=level)
    features = []
    for c in coeffs:
        energy = np.sum(c**2)
        norm = np.linalg.norm(c)
        mean_abs = np.mean(np.abs(c))
        features.append([energy, norm, mean_abs])
    return features

numbers = np.arange(1, 1000)
labels = numbers % 2

all_features = [extract_features(n) for n in numbers]
max_level = max(len(f) for f in all_features)

features_by_level = []
for lvl in range(max_level):
    lvl_feats = []
    for f in all_features:
        if lvl < len(f):
            lvl_feats.append(f[lvl])
        else:
            lvl_feats.append([0,0,0])
    features_by_level.append(np.array(lvl_feats))

probabilities = np.zeros((len(numbers), max_level, 3))

for lvl in range(max_level):
    for feat_idx in range(3):
        X = features_by_level[lvl][:, feat_idx].reshape(-1, 1)
        kmeans = KMeans(n_clusters=2, random_state=42).fit(X)
        clusters = kmeans.labels_

        cluster0_mean = np.mean(labels[clusters==0])
        cluster1_mean = np.mean(labels[clusters==1])

        tek_cluster = 0 if cluster0_mean > cluster1_mean else 1

        for i, cl in enumerate(clusters):
            members = (clusters == cl)
            prob_tek = np.mean(labels[members])
            probabilities[i, lvl, feat_idx] = prob_tek

level_weights = np.linspace(0.5, 1.5, max_level)
feature_weights = np.array([1, 1, 1])

weighted_probs = probabilities * level_weights.reshape(1, max_level, 1) * feature_weights.reshape(1, 1, 3)
final_scores = np.sum(weighted_probs, axis=(1,2)) / np.sum(level_weights) / np.sum(feature_weights)

predicted_labels = (final_scores > 0.5).astype(int)
accuracy = np.mean(predicted_labels == labels)

print(f"Final Accuracy: {accuracy*100:.2f}%")

plt.figure(figsize=(14,6))
plt.scatter(numbers, final_scores, c=['red' if l==1 else 'blue' for l in labels], label='True Odd (Red) / Even (Blue)')
plt.axhline(0.5, color='green', linestyle='--', label='Decision Threshold (0.5)')
plt.title("Wavelet Features + KMeans: Predicted Probability of Being Odd")
plt.xlabel("Number")
plt.ylabel("Score")
plt.legend()
plt.show()
