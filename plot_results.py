import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

# === Config ===
metric_files = {
    "PESQ": "results/results_pesq.csv",
    "SNR": "results/results_snr.csv",
    "STOI": "results/results_stoi.csv",
    "ViSQOL": "results/results_visqol.csv",
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
desired_snr = "0"  # Change this if you want to use a different SNR level

output_dir = "metric_graphs"
os.makedirs(output_dir, exist_ok=True)


# === Helper to Extract File Info ===
def extract_info(filename):
    parts = filename.split("_")
    speaker = parts[0] + "_" + parts[1]
    snr = parts[-1].replace("dB", "")
    noise = "_".join(parts[2:-1])
    return f"{speaker}_{noise}", snr


# === Plot Function ===
def plot_metric(df, metric_name):
    df["file_cleaned"], df["snr"] = zip(*df["file"].map(extract_info))
    df = df[df["snr"] == desired_snr]

    melted = df.melt(id_vars=["file_cleaned"],
                     value_vars=df.columns[1:-1],
                     var_name="comparison", value_name="score")

    plt.figure(figsize=(16, 6))
    ax = plt.gca()

    bar_width = 0.25
    comparisons = comparison_columns
    x_labels = sorted(melted["file_cleaned"].unique())
    x = np.arange(len(x_labels))

    for i, comparison in enumerate(comparisons):
        subset = melted[melted["comparison"] == comparison]
        merged = pd.DataFrame({"file_cleaned": x_labels})
        merged = merged.merge(subset, on="file_cleaned", how="left")
        scores = merged["score"].values
        ax.bar(x + (i - 1) * bar_width, scores, width=bar_width,
               label=comparison.replace("_", " "),
               color=comparison_colors[comparison], alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=90, fontsize=8)
    ax.set_title(f"{metric_name} Scores @ {desired_snr}dB")
    ax.set_ylabel("Score")
    ax.set_xlabel("File (Speaker + Noise)")
    ax.legend(loc='lower right', bbox_to_anchor=(1.0, 0.0))
    plt.tight_layout()
    plt.grid(True, axis='y', linestyle='--', alpha=0.3)
    plt.savefig(os.path.join(output_dir, f"{metric_name.lower()}_snr{desired_snr}_new.png"))
    plt.close()


# === Run for All Metrics ===
for metric_name, csv_file in metric_files.items():
    print(f"Processing {metric_name}...")
    df = pd.read_csv(csv_file)
    df.columns = ["file"] + comparison_columns
    plot_metric(df, metric_name)

print(f"✅ Plots saved to: {output_dir}/")
