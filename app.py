import os
import io
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

# --- 1. Dynamic Directory Resolution ---
# BASE_DIR points directly to the folder containing app.py
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(BASE_DIR, "data")
SOURCES_DIR = os.path.join(DATA_DIR, "sources")
MIXED_DIR = os.path.join(DATA_DIR, "mixed")
SEPARATED_DIR = os.path.join(DATA_DIR, "separated")

# Ensure required folders exist dynamically at runtime
for folder in [SOURCES_DIR, MIXED_DIR, SEPARATED_DIR]:
    os.makedirs(folder, exist_ok=True)

# --- 2. Streamlit Page Setup ---
st.set_page_config(page_title="Blind Audio Separation (ICA)", layout="wide")
st.title("Cocktail Party Problem: FastICA Audio Unmixer")

st.markdown("""
Separate mixed audio sources into independent acoustic components using **FastICA from scratch** (Centering, PCA Whitening, and fixed-point iteration).
""")

# Sidebar settings
st.sidebar.header("ICA Configuration")
n_sources = st.sidebar.slider("Number of Synthesized Sources", 2, 4, 3)
duration = st.sidebar.slider("Duration (seconds)", 2, 6, 4)
sr = 22050
t = np.linspace(0, duration, int(sr * duration), endpoint=False)

# Signal synthesis helper
def generate_signals(n, length):
    signals = []
    # Source 1: Harmonic pitch
    signals.append(np.sin(2 * np.pi * 330 * t) + 0.5 * np.sin(2 * np.pi * 660 * t))
    # Source 2: Tremolo / AM chime
    signals.append(np.sin(2 * np.pi * 550 * t) * np.sin(2 * np.pi * 4 * t))
    # Source 3: Square wave buzz
    signals.append(np.sign(np.sin(2 * np.pi * 120 * t)) * 0.4)
    # Source 4 (Optional): Frequency Chirp
    if n > 3:
        signals.append(np.sin(2 * np.pi * (100 + 50 * t) * t))
    
    S = np.array(signals[:n])
    for i in range(n):
        S[i] = S[i] / np.max(np.abs(S[i]))
        # Save to resolved absolute path
        out_path = os.path.join(SOURCES_DIR, f"source_{i+1}.wav")
        wavfile.write(out_path, sr, S[i].astype(np.float32))
    return S

S = generate_signals(n_sources, len(t))

# Mixing Matrix
np.random.seed(42)
A = np.random.uniform(0.2, 0.8, (n_sources, n_sources))
while abs(np.linalg.det(A)) < 1e-2:
    A = np.random.uniform(0.2, 0.8, (n_sources, n_sources))
X = np.dot(A, S)

# Save mixed signals
for i in range(n_sources):
    norm_mix = (X[i] / np.max(np.abs(X[i]))).astype(np.float32)
    out_path = os.path.join(MIXED_DIR, f"mixed_{i+1}.wav")
    wavfile.write(out_path, sr, norm_mix)

col1, col2 = st.columns(2)

with col1:
    st.subheader("1. Observed Mixtures")
    for i in range(n_sources):
        mix_file = os.path.join(MIXED_DIR, f"mixed_{i+1}.wav")
        st.write(f"Mixed Track {i+1}")
        st.audio(mix_file, format="audio/wav")

# ICA Pipeline Function
def run_pipeline(X_in, n_comp):
    # Centering
    X_cent = X_in - np.mean(X_in, axis=1, keepdims=True)
    
    # PCA Whitening
    Sigma = np.cov(X_cent)
    evals, E = np.linalg.eigh(Sigma)
    D_inv = np.diag(1.0 / np.sqrt(np.maximum(evals, 1e-12)))
    V = np.dot(D_inv, E.T)
    X_w = np.dot(V, X_cent)
    
    # FastICA loop
    W = np.zeros((n_comp, n_comp))
    for p in range(n_comp):
        w = np.random.randn(n_comp)
        w /= np.linalg.norm(w)
        for _ in range(300):
            u = np.dot(w, X_w)
            w_new = np.mean(X_w * np.tanh(u), axis=1) - np.mean(1.0 - np.tanh(u)**2) * w
            for j in range(p):
                w_new -= np.dot(w_new, W[j]) * W[j]
            w_new /= np.linalg.norm(w_new)
            if np.abs(np.abs(np.dot(w_new, w)) - 1.0) < 1e-5:
                w = w_new
                break
            w = w_new
        W[p] = w
    return np.dot(W, X_w)

if st.button("Run FastICA Separation", type="primary"):
    S_rec = run_pipeline(X, n_sources)
    
    with col2:
        st.subheader("2. Separated Sources")
        for i in range(n_sources):
            norm_rec = (S_rec[i] / np.max(np.abs(S_rec[i]))).astype(np.float32)
            out_path = os.path.join(SEPARATED_DIR, f"recovered_{i+1}.wav")
            wavfile.write(out_path, sr, norm_rec)
            st.write(f"Recovered Source {i+1}")
            st.audio(out_path, format="audio/wav")
            
    # Waveform Display
    st.subheader("Waveform Analysis (First 800 Samples)")
    fig, axes = plt.subplots(3, n_sources, figsize=(14, 6))
    for i in range(n_sources):
        axes[0, i].plot(t[:800], S[i][:800], color='navy')
        axes[0, i].set_title(f"Original {i+1}")
        axes[1, i].plot(t[:800], X[i][:800], color='firebrick')
        axes[1, i].set_title(f"Mixed {i+1}")
        axes[2, i].plot(t[:800], S_rec[i][:800], color='forestgreen')
        axes[2, i].set_title(f"Separated {i+1}")
    plt.tight_layout()
    st.pyplot(fig)
