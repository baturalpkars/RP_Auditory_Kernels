"""
This script reads metric evaluation results (like PESQ, STOI) from CSV files
and generates comparison bar plots across:
    - Noise types (e.g., babble, train, etc.)
    - SNR levels (e.g., -5 dB to 10 dB)
    - Reconstruction types (clean vs degraded vs hybrid)

It supports:
- Average plots per SNR
- 2-panel subplots for key SNRs (0 dB vs 10 dB)
- Custom y-axis range for PESQ (set to [0, 3.5] manually)

To run:
1. Ensure you have CSV metric outputs in `../../results/`.
2. Uncomment or add entries to `metric_files` as needed.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# === CONFIGURATION ===

metric_files = {
    # You can toggle PESQ/STOI/ViSQOL here
    # "PESQ": "../../results/results_pesq.csv",
    "STOI": "../../results/results_stoi.csv",
}

# Metrics will be averaged over these comparison types
comparison_columns = [
    "clean_vs_recon_clean",            # Reference reconstruction quality
    "degraded_vs_recon_degraded",      # Improvement from degradation
    "recon_clean_vs_recon_degraded",   # Drop due to degradation
]

# Color mapping for bar plots
comparison_colors = {
    "clean_vs_recon_clean": "blue",
    "degraded_vs_recon_degraded": "green",
    "recon_clean_vs_recon_degraded": "orange",
}

# All noise environments you evaluate
noise_types = ["babble", "train_coming", "white_noise", "airportAnnouncement"]

# You can add/remove more SNR levels here
snr_levels = ["-5", "0", "5", "10"]

# Output folder for the plots
output_dir = "../../results/plots/metric_avg_graphs"
os.makedirs(output_dir, exist_ok=True)


# === FILE NAME PARSER ===

def extract_info(filename):
    """
    Given a file name like 'p234_003_babble_10dB',
    extract (speaker_id, noise_type, snr).
    """
    parts = filename.split("_")
    speaker = parts[0] + "_" + parts[1]
    snr = parts[-1].replace("dB", "")
    noise = "_".join(parts[2:-1])
    return speaker, noise, snr


# === AVERAGE METRIC PLOTTER ===

def plot_avg_metric(df, metric_name):
    """
    For each SNR level, generate one bar plot showing
    average scores across noise types and reconstruction comparisons.
    """
    df["speaker"], df["noise"], df["snr"] = zip(*df["file"].map(extract_info))

    for snr in snr_levels:
        df_snr = df[df["snr"] == snr]
        avg_data = {comp: [] for comp in comparison_columns}

        for noise in noise_types:
            df_noise = df_snr[df_snr["noise"] == noise]
            for comp in comparison_columns:
                avg_score = df_noise[comp].mean()
                avg_data[comp].append(avg_score)

        x = np.arange(len(noise_types))
        bar_width = 0.2

        fig, ax = plt.subplots(figsize=(8, 5))  # Adjust figure size here
        all_scores = []
        for i, comp in enumerate(comparison_columns):
            scores = avg_data[comp]
            all_scores.extend(scores)
            ax.bar(x + i * bar_width, scores, width=bar_width,
                   label=comp.replace("_", " "), color=comparison_colors[comp])

        # Adjust y-limit dynamically (can hardcode for specific metrics if needed)
        ymax = max(all_scores) * 1.5
        ax.set_ylim(0, ymax)

        ax.set_xticks(x + bar_width)
        ax.set_xticklabels(noise_types, rotation=15, fontsize=13)
        ax.set_title(f"{metric_name} Average Scores @ {snr}dB", fontsize=15)
        ax.set_ylabel("Average Score", fontsize=14)
        ax.set_xlabel("Noise Type", fontsize=14)
        ax.tick_params(axis='y', labelsize=12)
        ax.tick_params(axis='x', labelsize=12)
        ax.legend(loc='upper right', fontsize=12, frameon=True)
        ax.grid(True, axis='y', linestyle='--', alpha=0.5)
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, f"{metric_name.lower()}_avg_snr{snr}.pdf"), format='pdf')
        plt.close()


# === SUBPLOT COMPARISON PLOTTER ===

def plot_snr_subplots(df, metric_name):
    """
    Generates a 2-row subplot for two SNR levels (default: 10dB and 0dB).
    Helps compare behavior under good and bad noise conditions.
    """
    df["speaker"], df["noise"], df["snr"] = zip(*df["file"].map(extract_info))

    snr_targets = ["10", "0"]  # <<< You can change this to ["10", "-5"] etc.
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(9, 8), sharex=True)
    all_scores = []

    for idx, snr in enumerate(snr_targets):
        ax = axes[idx]
        df_snr = df[df["snr"] == snr]
        avg_data = {comp: [] for comp in comparison_columns}

        for noise in noise_types:
            df_noise = df_snr[df_snr["noise"] == noise]
            for comp in comparison_columns:
                avg_score = df_noise[comp].mean()
                avg_data[comp].append(avg_score)
                all_scores.append(avg_score)

        x = np.arange(len(noise_types))
        bar_width = 0.2

        for i, comp in enumerate(comparison_columns):
            ax.bar(x + i * bar_width, avg_data[comp], width=bar_width,
                   label=comp.replace("_", " "), color=comparison_colors[comp])

        # === Y-LIMIT SETTINGS ===
        if metric_name.upper() == "PESQ":
            ax.set_ylim(0, 3.5)  # PESQ is bounded (typically 1–4.5, wideband ~3.5 upper limit)
        else:
            ax.set_ylim(0, 1)    # STOI and ViSQOL typically fall in [0,1] range

        ax.set_title(f"{metric_name} Scores @ {snr} dB", fontsize=20)
        ax.set_ylabel("Avg Score", fontsize=18)
        ax.tick_params(axis='y', labelsize=16)
        ax.set_xticks(x + bar_width)
        ax.set_xticklabels(noise_types, rotation=10, fontsize=14)
        ax.grid(True, axis='y', linestyle='--', alpha=0.4)

        # Uncomment this if you want to show the legend once at top
        # if idx == 0:
        #     ax.legend(fontsize=16, loc='upper right', frameon=True)

    fig.suptitle(f"{metric_name} Scores Comparison (0dB vs. 10dB)", fontsize=20)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(os.path.join(output_dir, f"{metric_name.lower()}_subplots_0_10.pdf"), format='pdf')
    plt.close()


# === MAIN EXECUTION ===

# Loop through all defined metrics and generate plots
for metric_name, csv_file in metric_files.items():
    print(f"Processing {metric_name}...")
    df = pd.read_csv(csv_file)
    df.columns = ["file"] + comparison_columns
    plot_snr_subplots(df, metric_name)
    # plot_avg_metric(df, metric_name)  # Optional: Uncomment to generate per-SNR plots

print(f"✅ Plots saved to: {output_dir}/")
