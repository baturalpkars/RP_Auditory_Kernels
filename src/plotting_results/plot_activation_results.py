"""
This script visualizes the difference in normalized kernel activation patterns
between speech+noise reconstructions and pure noise reconstructions.

It computes histograms of activated auditory kernels, normalizes them,
compares speech vs noise per noise type, and plots bar charts highlighting
which kernels are more speech- or noise-selective.

Usage:
- Ensure your directories (`RECON_SPEECH_DIR`, `RECON_NOISE_DIR`) contain
  valid encoded waveforms from prior matching pursuit steps.
- Set `TARGET_SNR` to the desired SNR level for speech degradation analysis (e.g., '10', '0', etc.)

Generated output: PDF plots saved to OUTPUT_DIR.
"""

import os
import pickle
import numpy as np
from collections import Counter
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

# === CONFIGURATION ===
RECON_SPEECH_DIR = "../../reconstructed_speeches"  # Folder for reconstructed noisy speech
RECON_NOISE_DIR = "../../reconstructed_noises"  # Folder for reconstructed noise-only signals
OUTPUT_DIR = "../../results/plots/avg_kernel_diff_plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)

NOISE_TYPES = ["babble", "train_coming", "white_noise", "airportAnnouncement"]
TARGET_SNR = "10"  # <<< You can change this to "-5", "0", "5", or "10" to focus on different degradation levels


# === Histogram helpers ===
def normalize_histogram(hist):
    """Convert raw kernel counts into normalized proportions (sum = 1)."""
    total = sum(hist.values())
    return {k: v / total for k, v in hist.items()} if total > 0 else hist


def get_top_k_kernels(hist, k=50):
    """(Unused) Limit histogram to top-K frequent kernels."""
    return dict(sorted(hist.items(), key=lambda x: x[1], reverse=True)[:k])


def load_histogram(encoded_path):
    """Load kernel indices from an encoded waveform and return histogram (Counter)."""
    if not os.path.exists(encoded_path):
        return None
    with open(encoded_path, "rb") as f:
        encoded = pickle.load(f)
    return Counter([entry[0] for entry in encoded])  # entry[0] = kernel index


# === Normalize noise histograms from pre-saved encodings ===
average_noise_hist = {}
for folder_name in os.listdir(RECON_NOISE_DIR):
    folder_path = os.path.join(RECON_NOISE_DIR, folder_name)
    encoded_path = os.path.join(folder_path, "encoded_waveform.pkl")
    if not os.path.exists(encoded_path):
        continue
    hist = load_histogram(encoded_path)
    if hist:
        normalized_hist = normalize_histogram(hist)
        average_noise_hist[folder_name.lower()] = normalized_hist
        print(f"✅ Loaded & normalized noise histogram from: {folder_name} (total kernels: {len(hist)})")

# === Matching helper (to deal with inconsistent naming)
NOISE_FOLDER_MAP = {
    "babble": "babble",
    "white_noise": "white",
    "airportAnnouncement": "airport",
    "train_coming": "traincoming",
}


def match_noise_folder(noise_type):
    """Match a noise type to folder name in noise reconstruction directory."""
    target = NOISE_FOLDER_MAP.get(noise_type, noise_type).lower()
    for folder in average_noise_hist:
        if target in folder:
            return folder
    return None


# === Main loop per noise type ===
for noise_type in NOISE_TYPES:
    histograms = []

    # Gather normalized histograms from all reconstructed speech files with this noise
    for speaker_id in os.listdir(RECON_SPEECH_DIR):
        speaker_path = os.path.join(RECON_SPEECH_DIR, speaker_id)
        if not os.path.isdir(speaker_path):
            continue

        for folder in os.listdir(speaker_path):
            if not folder.endswith(f"{noise_type}_{TARGET_SNR}dB"):
                continue

            encoded_path = os.path.join(speaker_path, folder, "encoded_waveform.pkl")
            hist = load_histogram(encoded_path)
            if hist:
                histograms.append(normalize_histogram(hist))

    if not histograms:
        print(f"⚠️ No speech histograms for noise '{noise_type}' at {TARGET_SNR}dB")
        continue

    print(f"📊 Processing {len(histograms)} speech files for {noise_type}")

    # === Compute average speech histogram
    all_kernels = set()
    for h in histograms:
        all_kernels.update(h.keys())

    avg_speech_hist = {
        k: np.mean([h.get(k, 0) for h in histograms])
        for k in all_kernels
    }

    # === Load matching noise histogram
    noise_folder_key = match_noise_folder(noise_type)
    if not noise_folder_key:
        print(f"❌ No matching noise folder for: {noise_type}")
        continue
    noise_hist = average_noise_hist[noise_folder_key]

    # === Compute kernel-wise difference
    all_kernels.update(noise_hist.keys())
    diff_hist = {
        k: avg_speech_hist.get(k, 0) - noise_hist.get(k, 0)
        for k in all_kernels
    }

    # === Filter to significant differences (abs diff > 0.001)
    significant_diffs = {k: v for k, v in diff_hist.items() if abs(v) > 0.001}
    if not significant_diffs:
        print(f"⚠️ No significant differences found for {noise_type}")
        continue

    # === Plot bar chart of kernel difference (Speech - Noise)
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    kernels = list(significant_diffs.keys())
    values = list(significant_diffs.values())
    colors = ['green' if v > 0 else 'red' for v in values]

    bars = ax.bar(kernels, values, color=colors, alpha=0.7)
    ax.axhline(0, color='black', linewidth=0.8)

    # Set y-axis limits with margin for legend
    ymin = min(values)
    ymax = max(values)
    yrange = ymax - ymin
    ax.set_ylim(ymin - 0.05 * yrange, ymax + 0.25 * yrange)

    ax.set_title(
        f"Normalized Kernel Usage Difference (Speech - Noise)\n{noise_type} @ {TARGET_SNR}dB",
        fontsize=13
    )
    ax.set_xlabel("Kernel Index", fontsize=16)
    ax.set_ylabel("Proportion Difference", fontsize=16)
    ax.tick_params(axis='both', labelsize=12)
    ax.grid(True, alpha=0.4)

    # Add color legend
    speech_patch = mpatches.Patch(color='green', label='Speech-pref.')
    noise_patch = mpatches.Patch(color='red', label='Noise-pref.')
    ax.legend(handles=[speech_patch, noise_patch], fontsize=12, frameon=True, loc='upper right')

    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, f"{noise_type}_normalized_diff_{TARGET_SNR}dB.pdf"), format='pdf')
    plt.close()

    # === Print diagnostic stats
    speech_diversity = len([k for k, v in avg_speech_hist.items() if v > 0])
    noise_diversity = len([k for k, v in noise_hist.items() if v > 0])
    print(f"   Speech uses {speech_diversity} unique kernels")
    print(f"   Noise uses {noise_diversity} unique kernels")
    print(f"   Found {len(significant_diffs)} kernels with meaningful differences")
    print(f"   Max speech preference: {max(diff_hist.values()):.4f}")
    print(f"   Max noise preference: {min(diff_hist.values()):.4f}")
    print()

print(f"\n✅ All normalized kernel difference plots saved to: {OUTPUT_DIR}")
