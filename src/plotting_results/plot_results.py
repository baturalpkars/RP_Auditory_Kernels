import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# === Config ===
metric_files = {
    "PESQ": "results/results_pesq.csv",
    "STOI": "results/results_stoi.csv",
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
output_dir = "../../plots/metric_avg_graphs"
os.makedirs(output_dir, exist_ok=True)


# === Helper to Extract File Info ===
def extract_info(filename):
    parts = filename.split("_")
    speaker = parts[0] + "_" + parts[1]
    snr = parts[-1].replace("dB", "")
    noise = "_".join(parts[2:-1])
    return speaker, noise, snr


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
        bar_width = 0.15

        plt.figure(figsize=(10, 6))
        for i, comp in enumerate(comparison_columns):
            scores = avg_data[comp]
            plt.bar(x + i * bar_width, scores, width=bar_width,
                    label=comp.replace("_", " "), color=comparison_colors[comp])

        plt.xticks(x + bar_width, noise_types, rotation=15)
        plt.title(f"{metric_name} Average Scores @ {snr}dB")
        plt.ylabel("Average Score")
        plt.xlabel("Noise Type")
        plt.legend()
        plt.grid(True, axis='y', linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{metric_name.lower()}_avg_snr{snr}.png"))
        plt.close()


# === Run for STOI and PESQ ===
for metric_name, csv_file in metric_files.items():
    print(f"Processing {metric_name}...")
    df = pd.read_csv(csv_file)
    df.columns = ["file"] + comparison_columns
    plot_avg_metric(df, metric_name)

print(f"✅ Plots saved to: {output_dir}/")
