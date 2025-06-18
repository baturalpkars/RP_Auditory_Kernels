import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# === Config ===
metric_files = {
    "PESQ": "../../results/results_pesq.csv",
    "STOI": "../../results/results_stoi.csv",
}
comparison_columns = [
    "clean_vs_recon_clean",
    "degraded_vs_recon_degraded",
    "recon_clean_vs_recon_degraded",
]
comparison_colors = {
    "clean_vs_recon_clean": "blue",
    "degraded_vs_recon_degraded": "green",
    "recon_clean_vs_recon_degraded": "orange",
}
noise_types = ["babble", "train_coming", "white_noise", "airportAnnouncement"]
snr_levels = ["-5", "0", "5", "10"]
output_dir = "../../results/plots/metric_avg_graphs"
os.makedirs(output_dir, exist_ok=True)


# === Helper to Extract File Info ===
def extract_info(filename):
    parts = filename.split("_")
    speaker = parts[0] + "_" + parts[1]
    snr = parts[-1].replace("dB", "")
    noise = "_".join(parts[2:-1])
    return speaker, noise, snr


# === Plot Function for Averaged Metrics ===
# === Plot Function for Averaged Metrics ===
def plot_avg_metric(df, metric_name):
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

        fig, ax = plt.subplots(figsize=(8, 5))  # Wider & taller
        all_scores = []
        for i, comp in enumerate(comparison_columns):
            scores = avg_data[comp]
            all_scores.extend(scores)
            ax.bar(x + i * bar_width, scores, width=bar_width,
                   label=comp.replace("_", " "), color=comparison_colors[comp])

        # Adjust y-limit for legend placement
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


def plot_snr_subplots(df, metric_name):
    df["speaker"], df["noise"], df["snr"] = zip(*df["file"].map(extract_info))

    snr_targets = ["10", "0"]
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

        # Adjust y-limit for legend placement
        ymax = max(all_scores) * 1.5
        ax.set_ylim(0, ymax)

        ax.set_title(f"{metric_name} Scores @ {snr} dB", fontsize=14)
        ax.set_ylabel("Avg Score", fontsize=13)
        ax.tick_params(axis='y', labelsize=12)
        ax.set_xticks(x + bar_width)
        ax.set_xticklabels(noise_types, rotation=10, fontsize=12)
        ax.grid(True, axis='y', linestyle='--', alpha=0.4)
        ax.legend(fontsize=11, loc='upper right', frameon=True)

    fig.suptitle(f"{metric_name} Scores Comparison (0dB vs. 10dB)", fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(os.path.join(output_dir, f"{metric_name.lower()}_subplots_0_10.pdf"), format='pdf')
    plt.close()


# === Run for STOI and PESQ ===
for metric_name, csv_file in metric_files.items():
    print(f"Processing {metric_name}...")
    df = pd.read_csv(csv_file)
    df.columns = ["file"] + comparison_columns
    plot_snr_subplots(df, metric_name)

print(f"✅ Plots saved to: {output_dir}/")
