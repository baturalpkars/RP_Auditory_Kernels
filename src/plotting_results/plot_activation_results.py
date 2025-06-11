import os
import pickle
import numpy as np
from collections import Counter, defaultdict
import matplotlib.pyplot as plt

# === CONFIGURATION ===
RECON_SPEECH_DIR = "../../reconstructed_speeches"
RECON_NOISE_DIR = "../../reconstructed_noises"
OUTPUT_DIR = "../../results/plots/avg_kernel_diff_plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)

NOISE_TYPES = ["babble", "train_coming", "white_noise", "airportAnnouncement"]
TARGET_SNR = "-5"


# === NORMALIZATION FUNCTIONS ===
def normalize_histogram(hist):
    """Convert raw counts to proportions (0-1)"""
    total = sum(hist.values())
    return {k: v / total for k, v in hist.items()} if total > 0 else hist


def get_top_k_kernels(hist, k=50):
    """Get only the top K most used kernels to reduce noise"""
    return dict(sorted(hist.items(), key=lambda x: x[1], reverse=True)[:k])


# === Utility to load kernel histogram from encoded_waveform.pkl
def load_histogram(encoded_path):
    if not os.path.exists(encoded_path):
        return None
    with open(encoded_path, "rb") as f:
        encoded = pickle.load(f)
    return Counter([entry[0] for entry in encoded])


# === Load and normalize average noise histograms
average_noise_hist = {}
for folder_name in os.listdir(RECON_NOISE_DIR):
    folder_path = os.path.join(RECON_NOISE_DIR, folder_name)
    encoded_path = os.path.join(folder_path, "encoded_waveform.pkl")
    if not os.path.exists(encoded_path):
        continue
    hist = load_histogram(encoded_path)
    if hist:
        # Normalize noise histogram
        normalized_hist = normalize_histogram(hist)
        average_noise_hist[folder_name.lower()] = normalized_hist
        print(f"✅ Loaded & normalized noise histogram from: {folder_name} (total kernels: {len(hist)})")

# === Match noise type to corresponding folder
NOISE_FOLDER_MAP = {
    "babble": "babble",
    "white_noise": "white",
    "airportAnnouncement": "airport",
    "train_coming": "traincoming",  # handle special case
}


def match_noise_folder(noise_type):
    target = NOISE_FOLDER_MAP.get(noise_type, noise_type).lower()
    for folder in average_noise_hist:
        if target in folder:
            return folder
    return None


# === For each noise type, process with normalization
for noise_type in NOISE_TYPES:
    histograms = []

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
                # Normalize each speech histogram
                normalized_hist = normalize_histogram(hist)
                histograms.append(normalized_hist)

    if not histograms:
        print(f"⚠️ No speech histograms for noise '{noise_type}' at {TARGET_SNR}dB")
        continue

    print(f"📊 Processing {len(histograms)} speech files for {noise_type}")

    # === Average normalized histograms
    all_kernels = set()
    for h in histograms:
        all_kernels.update(h.keys())

    avg_speech_hist = {
        k: np.mean([h.get(k, 0) for h in histograms])
        for k in all_kernels
    }

    # === Load matching noise histogram (already normalized)
    noise_folder_key = match_noise_folder(noise_type)
    if not noise_folder_key:
        print(f"❌ No matching noise folder for: {noise_type}")
        continue

    noise_hist = average_noise_hist[noise_folder_key]

    # === Calculate relative difference (now both are proportions)
    all_kernels.update(noise_hist.keys())
    diff_hist = {
        k: avg_speech_hist.get(k, 0) - noise_hist.get(k, 0)
        for k in all_kernels
    }

    # === Optional: Focus on top differences for cleaner plots
    # Get kernels with largest absolute differences
    significant_diffs = {k: v for k, v in diff_hist.items() if abs(v) > 0.001}  # 0.1% threshold

    if not significant_diffs:
        print(f"⚠️ No significant differences found for {noise_type}")
        continue

    # === Create clean single plot
    plt.figure(figsize=(12, 6))

    kernels = list(significant_diffs.keys())
    values = list(significant_diffs.values())
    colors = ['green' if v > 0 else 'red' for v in values]

    plt.bar(kernels, values, color=colors, alpha=0.7)
    plt.axhline(0, color='black', linewidth=0.8)
    plt.title(
        f"Normalized Kernel Usage Difference (Speech - Noise)\n{noise_type} @ {TARGET_SNR}dB | Green: Speech-preferred, Red: Noise-preferred")
    plt.xlabel("Kernel Index")
    plt.ylabel("Proportion Difference")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{noise_type}_normalized_diff_{TARGET_SNR}dB.png"), dpi=150)
    plt.close()

    # === Print summary statistics
    speech_diversity = len([k for k, v in avg_speech_hist.items() if v > 0])
    noise_diversity = len([k for k, v in noise_hist.items() if v > 0])

    print(f"   Speech uses {speech_diversity} unique kernels")
    print(f"   Noise uses {noise_diversity} unique kernels")
    print(f"   Found {len(significant_diffs)} kernels with meaningful differences")
    print(f"   Max speech preference: {max(diff_hist.values()):.4f}")
    print(f"   Max noise preference: {min(diff_hist.values()):.4f}")
    print()

print(f"\n✅ All normalized kernel difference plots saved to: {OUTPUT_DIR}")