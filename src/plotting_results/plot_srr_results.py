"""
This script generates SRR (Signal-to-Residual Ratio) vs. Kernel Rate plots
for each noise type and SNR level using Matching Pursuit results.

It processes reconstructed speech files, extracts the norm list and encoded waveform,
then plots SRR curves averaged across speakers, with shaded standard deviations.

✅ Outputs are saved in: ../../results/plots/srr_plots/

🔧 You can change:
- Noise types via `NOISE_TYPES`
- SNR levels via `SNR_LEVELS`
- Sampling rate or kernel truncation limits
"""

import os
import numpy as np
import librosa
import pickle
import matplotlib.pyplot as plt

# === CONFIGURATION ===
RECON_DIR = "../../reconstructed_speeches"          # Folder with reconstructed speech subfolders
ORIG_WAV_DIR = "../../degraded_speeches"            # Folder with degraded .wav files
OUTPUT_PLOT_DIR = "../../results/plots/srr_plots"   # Where to save the final plots
os.makedirs(OUTPUT_PLOT_DIR, exist_ok=True)

SNR_LEVELS = ["-5", "0", "5", "10"]                  # Customize SNRs shown
NOISE_TYPES = ["babble", "train_coming", "white_noise", "airportAnnouncement"]
KERNEL_LIMIT = 1500                                  # Max number of kernels to consider (truncate long traces)
FS = 16000                                           # Sampling rate in Hz


# === Helper functions ===

def extract_noise_snr(folder_name):
    """
    Extract noise type and SNR value from folder name
    e.g., 'p234_003_babble_10dB' → ('babble', '10')
    """
    parts = folder_name.split("_")
    if len(parts) < 4:
        print(f"❌ Unexpected folder name format: {folder_name}")
        return None, None
    snr = parts[-1].replace("dB", "")
    noise = "_".join(parts[2:-1])
    return noise, snr


def compute_srr(norm_list, signal):
    """
    Compute the SRR in dB from norm list (residual norms)
    and original signal norm.
    """
    norm_y = np.linalg.norm(signal)
    norm_list = np.array(norm_list[:KERNEL_LIMIT])
    return 20 * np.log10(norm_y / norm_list)


def compute_kernels_per_second(encoded_waveform, signal_len):
    """
    Return x-axis: number of selected kernels per second.
    """
    num_kernels = min(len(encoded_waveform), KERNEL_LIMIT)
    return np.linspace(1, num_kernels / signal_len * FS, num_kernels)


# === Data containers ===
data = {noise: {snr: [] for snr in SNR_LEVELS} for noise in NOISE_TYPES}
kernel_axes = {noise: {snr: [] for snr in SNR_LEVELS} for noise in NOISE_TYPES}


# === Traverse each speaker folder and process ===
for speaker_id in os.listdir(RECON_DIR):
    speaker_folder = os.path.join(RECON_DIR, speaker_id)
    if not os.path.isdir(speaker_folder):
        continue

    for sample_folder in os.listdir(speaker_folder):
        full_path = os.path.join(speaker_folder, sample_folder)
        norm_path = os.path.join(full_path, "norm_list.npy")
        encoded_path = os.path.join(full_path, "encoded_waveform.pkl")

        noise, snr = extract_noise_snr(sample_folder)
        if noise not in NOISE_TYPES or snr not in SNR_LEVELS:
            continue

        print(f"📂 Folder: {sample_folder} → noise: '{noise}', snr: '{snr}'")

        degraded_wav = os.path.join(ORIG_WAV_DIR, speaker_id, f"{sample_folder}.wav")
        if not (os.path.exists(norm_path) and os.path.exists(degraded_wav) and os.path.exists(encoded_path)):
            continue

        try:
            # Load norm list and waveform
            norm_list = np.load(norm_path)
            with open(encoded_path, "rb") as f:
                encoded_waveform = pickle.load(f)
            y, _ = librosa.load(degraded_wav, sr=FS)

            # Compute SRR and kernel rate
            srr = compute_srr(norm_list, y)
            x_vals = compute_kernels_per_second(encoded_waveform, len(y))

            data[noise][snr].append(srr)
            kernel_axes[noise][snr].append(x_vals)
        except Exception as e:
            print(f"❌ Error processing {sample_folder}: {e}")


# === Generate SRR vs Kernel Rate plots ===
for noise in NOISE_TYPES:
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    for snr in SNR_LEVELS:
        srr_curves = data[noise][snr]
        x_curves = kernel_axes[noise][snr]
        if not srr_curves or not x_curves:
            continue

        # Truncate all curves to same length
        min_len = min(len(c) for c in srr_curves)
        srr_trunc = [c[:min_len] for c in srr_curves]
        x_trunc = [x[:min_len] for x in x_curves]

        avg_srr = np.mean(srr_trunc, axis=0)
        std_srr = np.std(srr_trunc, axis=0)
        avg_x = np.mean(x_trunc, axis=0)

        # === Plot
        ax.plot(avg_x, avg_srr, label=f"{snr} dB", linewidth=2)
        ax.fill_between(avg_x, avg_srr - std_srr, avg_srr + std_srr,
                        alpha=0.2, linewidth=0, label=None)

    ax.set_title(f"SRR vs. Kernels/sec — Noise: {noise}", fontsize=20)
    ax.set_xlabel("Kernels per Second", fontsize=18)
    ax.set_ylabel("SRR (dB)", fontsize=18)
    ax.tick_params(axis='both', labelsize=15)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(loc='upper left', fontsize=16, frameon=True, title="SNR Levels", title_fontsize=13)

    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_PLOT_DIR, f"{noise}_srr.pdf"), format='pdf')
    plt.close()

print(f"✅ All SRR plots saved to: {OUTPUT_PLOT_DIR}/")
