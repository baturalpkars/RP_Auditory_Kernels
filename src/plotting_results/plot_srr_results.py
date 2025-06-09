import os
import numpy as np
import librosa
import pickle
import matplotlib.pyplot as plt

# === CONFIGURATION ===
RECON_DIR = "../../reconstructed_speeches"
ORIG_WAV_DIR = "../../degraded_speeches"
OUTPUT_PLOT_DIR = "../../results/plots/srr_plots"
os.makedirs(OUTPUT_PLOT_DIR, exist_ok=True)

SNR_LEVELS = ["-5", "0", "5", "10"]
NOISE_TYPES = ["babble", "train_coming", "white_noise", "airportAnnouncement"]
KERNEL_LIMIT = 1500
FS = 16000  # Sampling rate


# === Helper functions ===
def extract_noise_snr(folder_name):
    parts = folder_name.split("_")
    if len(parts) < 4:
        print(f"❌ Unexpected folder name format: {folder_name}")
        return None, None
    snr = parts[-1].replace("dB", "")
    noise = "_".join(parts[2:-1])
    return noise, snr


def compute_srr(norm_list, signal):
    norm_y = np.linalg.norm(signal)
    norm_list = np.array(norm_list[:KERNEL_LIMIT])
    return 20 * np.log10(norm_y / norm_list)


def compute_kernels_per_second(encoded_waveform, signal_len):
    num_kernels = min(len(encoded_waveform), KERNEL_LIMIT)
    return np.linspace(1, num_kernels / signal_len * FS, num_kernels)


# === Data structure ===
data = {
    noise: {snr: [] for snr in SNR_LEVELS}
    for noise in NOISE_TYPES
}
kernel_axes = {
    noise: {snr: [] for snr in SNR_LEVELS}
    for noise in NOISE_TYPES
}

# === Traverse reconstructed files ===
for speaker_id in os.listdir(RECON_DIR):
    speaker_folder = os.path.join(RECON_DIR, speaker_id)
    if not os.path.isdir(speaker_folder):
        continue

    for sample_folder in os.listdir(speaker_folder):
        full_path = os.path.join(speaker_folder, sample_folder)
        norm_path = os.path.join(full_path, "norm_list.npy")
        encoded_path = os.path.join(full_path, "encoded_waveform.pkl")

        noise, snr = extract_noise_snr(sample_folder)
        # if noise not in NOISE_TYPES or snr not in SNR_LEVELS:
        #     continue
        print(f"📂 Folder: {sample_folder} → noise: '{noise}', snr: '{snr}'")

        degraded_wav = os.path.join(ORIG_WAV_DIR, speaker_id, f"{sample_folder}.wav")
        if not (os.path.exists(norm_path) and os.path.exists(degraded_wav) and os.path.exists(encoded_path)):
            continue

        try:
            norm_list = np.load(norm_path)
            with open(encoded_path, "rb") as f:
                encoded_waveform = pickle.load(f)
            y, _ = librosa.load(degraded_wav, sr=FS)

            srr = compute_srr(norm_list, y)
            x_vals = compute_kernels_per_second(encoded_waveform, len(y))

            data[noise][snr].append(srr)
            kernel_axes[noise][snr].append(x_vals)
        except Exception as e:
            print(f"❌ Error processing {sample_folder}: {e}")

# === Plotting ===
# === Plotting ===
for noise in NOISE_TYPES:
    plt.figure(figsize=(10, 6))
    for snr in SNR_LEVELS:
        srr_curves = data[noise][snr]
        x_curves = kernel_axes[noise][snr]
        if not srr_curves or not x_curves:
            continue

        # Find min length to align curves
        min_len = min(len(c) for c in srr_curves)

        # Truncate all curves to min_len
        srr_trunc = [c[:min_len] for c in srr_curves]
        x_trunc = [x[:min_len] for x in x_curves]

        avg_srr = np.mean(srr_trunc, axis=0)
        avg_x = np.mean(x_trunc, axis=0)

        plt.plot(avg_x, avg_srr, label=f"{snr} dB")

    plt.title(f"SRR vs Kernels/sec — Noise: {noise}")
    plt.xlabel("Kernels/second")
    plt.ylabel("SRR [dB]")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_PLOT_DIR, f"{noise}_srr.png"))
    plt.close()

print(f"✅ All SRR plots saved to: {OUTPUT_PLOT_DIR}/")
