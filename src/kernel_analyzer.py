import os
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt


def analyze_encoded_waveform(encoded_waveform, speech, signal_length, norm_list,
                             sr=16000, output_folder='../output_analysis', clean_id=""):
    """
    Analyzes kernel usage and reconstruction quality for a single reconstructed signal.

    Args:
        encoded_waveform: List of tuples (kernel_index, amplitude, position)
        speech: Original degraded input signal
        signal_length: Length of original speech signal (in samples)
        norm_list: Residual norm after each kernel addition (used for SRR curve)
        sr: Sampling rate (default 16 kHz)
        output_folder: Where to save plots
        clean_id: Base identifier for saving filenames
    """

    os.makedirs(output_folder, exist_ok=True)

    # === Kernel Usage Histogram ===
    kernel_indices = [entry[0] for entry in encoded_waveform]
    kernel_counts = Counter(kernel_indices)

    plt.figure(figsize=(10, 4))
    plt.bar(kernel_counts.keys(), kernel_counts.values())
    plt.title("Kernel Usage Frequency")
    plt.xlabel("Kernel Index")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, f'{clean_id}_bar_plot.png'))
    plt.close()

    # === Amplitude Distribution of Used Kernels ===
    amps = [abs(entry[1]) for entry in encoded_waveform]
    plt.figure(figsize=(8, 3))
    plt.hist(amps, bins=50)
    plt.title("Distribution of Amplitudes of Selected Kernels")
    plt.xlabel("Amplitude")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, f'{clean_id}_amp_dist.png'))
    plt.close()

    # === SRR Curve: Structural Fidelity over Kernels/sec ===
    # SRR = 20 * log10(||signal|| / ||residual||)
    SRR_ld = 20 * np.log10(np.linalg.norm(speech) / norm_list)

    # x-axis: kernel rate (kernels/sec) across signal duration
    kernel_rates = np.linspace(1, len(norm_list) / signal_length * sr, len(norm_list))

    plt.figure(figsize=(8, 3))
    plt.plot(kernel_rates, SRR_ld)
    plt.title("Kernels/sec vs. Signal-to-Residual Ratio (SRR)")
    plt.xlabel("Kernels/sec")
    plt.ylabel("SRR [dB]")
    plt.grid(True)
    plt.legend(["Learned"])
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, f'{clean_id}_srr_kernels.png'))
    plt.close()


def save_plots(original_signal, reconstructed_signal, output_folder):
    """
    Saves line plots of original, reconstructed, and overlayed waveforms.

    Args:
        original_signal: The degraded input signal
        reconstructed_signal: Output of kernel-based reconstruction
        output_folder: Where to save the plots
    """

    # === Original signal ===
    plt.figure(figsize=(12, 3))
    plt.plot(original_signal, label='Original Signal')
    plt.title('Original Signal')
    plt.xlabel('Sample')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'original_signal.png'))
    plt.close()

    # === Reconstructed signal ===
    plt.figure(figsize=(12, 3))
    plt.plot(reconstructed_signal, label='Reconstructed Signal', color='orange')
    plt.title('Reconstructed Signal')
    plt.xlabel('Sample')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'reconstructed_signal.png'))
    plt.close()

    # === Overlay plot ===
    plt.figure(figsize=(12, 3))
    plt.plot(original_signal, label='Original', alpha=0.7)
    plt.plot(reconstructed_signal, label='Reconstructed', alpha=0.7)
    plt.title('Overlay: Original vs Reconstructed')
    plt.xlabel('Sample')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'overlay.png'))
    plt.close()
