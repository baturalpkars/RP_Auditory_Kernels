import os
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt


def analyze_encoded_waveform(encoded_waveform, speech, signal_length, norm_list, sr=16000,
                             output_folder='../output_analysis', clean_id=""):
    os.makedirs(output_folder, exist_ok=True)  # Ensure directory exists

    # Bar plot
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

    # Amplitude distribution
    amps = [abs(entry[1]) for entry in encoded_waveform]
    plt.figure(figsize=(8, 3))
    plt.hist(amps, bins=50)
    plt.title("Distribution of Amplitudes of Selected Kernels")
    plt.xlabel("Amplitude")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, f'{clean_id}_amp_dist.png'))
    plt.close()

    # SRR vs Kernels/Second
    SRR_ld = 20 * np.log10(np.linalg.norm(speech) / norm_list)
    plt.figure(figsize=(8, 3))
    plt.plot(np.linspace(1, len(norm_list) / signal_length * sr, len(norm_list)), SRR_ld)
    plt.title("Number of kernels/second vs the SRR")
    plt.legend(["Learned"])
    plt.xlabel("kernels/second")
    plt.ylabel("SRR [dB]")
    plt.grid()
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, f'{clean_id}_srr_kernels.png'))
    plt.close()


def save_plots(original_signal, reconstructed_signal, output_folder):
    """Save separate and overlay plots for signals."""
    # Plot original
    plt.figure(figsize=(12, 3))
    plt.plot(original_signal, label='Original Signal')
    plt.title('Original Signal')
    plt.xlabel('Sample')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'original_signal.png'))
    plt.close()

    # Plot reconstructed
    plt.figure(figsize=(12, 3))
    plt.plot(reconstructed_signal, label='Reconstructed Signal', color='orange')
    plt.title('Reconstructed Signal')
    plt.xlabel('Sample')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'reconstructed_signal.png'))
    plt.close()

    # Overlay plot
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