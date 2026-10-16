# Unmix the Signal (Independent Component Analysis from Scratch)

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://unmix-ica-stream-app.streamlit.app/)

> **Interactive Web App:** [unmix-ica-stream-app.streamlit.app](https://unmix-ica-stream-app.streamlit.app/)

An implementation of Independent Component Analysis (ICA) built from scratch using NumPy and SciPy to solve the Cocktail Party Problem and recover clean, independent audio sources from linear mixtures.

## Pipeline Architecture

1. **Source Generation & Mixing:** $N$ clean, mutually independent non-Gaussian signals are mixed via an arbitrary invertible mixing matrix $A$: 
   $$X = AS$$
2. **Centering:** Zero-centering data by subtracting empirical channel means.
3. **PCA Whitening:** Decorrelating signals and normalizing variances to unity using eigendecomposition: 
   $$\Sigma = \frac{1}{N-1}XX^T = E D E^T \implies X_{white} = D^{-1/2}E^T X$$
4. **FastICA Fixed-Point Optimization:** Maximizing non-Gaussianity via hyperbolic tangent contrast functions $g(u) = \tanh(u)$ with Gram-Schmidt deflationary orthogonalization.
5. **Reconstruction & Normalization:** Projecting whitened signals through the unmixing matrix $W$ to recover source estimates $\hat{S}$.

## Results & Visualizations

### 1. Frequency Recovery (Spectrograms)
![Spectrograms](assets/spectrograms.png)

### 2. Time-Domain Waveform Separation
![Waveforms](assets/comparison.png)

### 3. PCA vs. ICA (Why Second-Order Statistics Fail)
![PCA vs ICA](assets/pca_vs_ica.png)

---

## Audio Demonstration

| Audio Source | Mixed Recording | Recovered (ICA) |
| :--- | :--- | :--- |
| **Source 1**<br><audio controls src="data/sources/source_1.wav"></audio> | **Mixed Channel 1**<br><audio controls src="data/mixed/mixed_1.wav"></audio> | **Recovered 1**<br><audio controls src="data/separated/recovered_1.wav"></audio> |
| **Source 2**<br><audio controls src="data/sources/source_2.wav"></audio> | **Mixed Channel 2**<br><audio controls src="data/mixed/mixed_2.wav"></audio> | **Recovered 2**<br><audio controls src="data/separated/recovered_2.wav"></audio> |
| **Source 3**<br><audio controls src="data/sources/source_3.wav"></audio> | **Mixed Channel 3**<br><audio controls src="data/mixed/mixed_3.wav"></audio> | **Recovered 3**<br><audio controls src="data/separated/recovered_3.wav"></audio> |

---

## Separation Metrics
* **Absolute Pearson Correlation:** $\ge 0.98$ for all aligned source pairs.
* **Signal-to-Distortion Ratio (SDR):** Evaluated against original sources accounting for permutation and sign inversion.
