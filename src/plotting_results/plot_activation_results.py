import os
import pickle
import numpy as np
from collections import Counter, defaultdict
import matplotlib.pyplot as plt

# === CONFIGURATION ===
RECON_SPEECH_DIR = "../../reconstructed_speeches"
RECON_NOISE_DIR = "../../reconstructed_noises"
OUTPUT_DIR = "../../plots/avg_kernel_diff_plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)

NOISE_TYPES = ["babble", "train_coming", "white_noise", "airportAnnouncement"]
TARGET_SNR = "-5"


# === Utility to load kernel histogram from encoded_waveform.pkl
def load_histogram(encoded_path):
    if not os.path.exists(encoded_path):
        return None
    with open(encoded_path, "rb") as f:
        encoded = pickle.load(f)
    return Counter([entry[0] for entry in encoded])


# === Load average noise histograms
average_noise_hist = {}
for folder_name in os.listdir(RECON_NOISE_DIR):
    folder_path = os.path.join(RECON_NOISE_DIR, folder_name)
    encoded_path = os.path.join(folder_path, "encoded_waveform.pkl")
    if not os.path.exists(encoded_path):
        continue
    hist = load_histogram(encoded_path)
    if hist:
        average_noise_hist[folder_name.lower()] = hist
        print(f"✅ Loaded noise histogram from: {folder_name}")

# === Match noise type to corresponding folder
# Map noise_type to actual noise folder naming pattern
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


# === For each noise type, average all TARGET_SNR speech histograms and subtract noise
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
                histograms.append(hist)

    if not histograms:
        print(f"⚠️ No speech histograms for noise '{noise_type}' at {TARGET_SNR}dB")
        continue

    # === Average histograms
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

    # === Subtract noise from average speech
    all_kernels.update(noise_hist.keys())
    diff_hist = {
        k: noise_hist.get(k, 0) - avg_speech_hist.get(k, 0)
        for k in all_kernels
    }

    # === Plot
    plt.figure(figsize=(10, 4))
    plt.bar(diff_hist.keys(), diff_hist.values(), color='teal')
    plt.axhline(0, color='black', linewidth=0.8)
    plt.title(f"Avg Kernel Usage (Speech - Noise) | {noise_type} @ {TARGET_SNR}dB")
    plt.xlabel("Kernel Index")
    plt.ylabel("Average Count Difference")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{noise_type}_avg_diff_{TARGET_SNR}dB.png"))
    plt.close()

print(f"\n✅ All average kernel difference plots saved to: {OUTPUT_DIR}")
