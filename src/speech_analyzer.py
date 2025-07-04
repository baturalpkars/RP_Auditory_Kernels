import os
import csv
from typing import List
from src.metrics import MetricStrategy

class AudioQualityAnalyzer:
    """
    A class to compute multiple audio quality metrics across clean, degraded, and reconstructed signals.
    """

    def __init__(self, clean_dir, degraded_dir, reconstructed_clean_dir, reconstructed_degraded_dir):
        """
        Initializes the analyzer with required folder paths.

        Args:
        - clean_dir: Path to clean original signals.
        - degraded_dir: Path to degraded versions of the clean signals.
        - reconstructed_clean_dir: Reconstructed outputs from clean input.
        - reconstructed_degraded_dir: Reconstructed outputs from degraded input.
        """
        self.clean_dir = clean_dir
        self.degraded_dir = degraded_dir
        self.reconstructed_clean_dir = reconstructed_clean_dir
        self.reconstructed_degraded_dir = reconstructed_degraded_dir
        self.metrics: List[MetricStrategy] = []
        self.results = []

    def add_metric(self, metric: MetricStrategy):
        """Add a metric (e.g., PESQ, STOI, SNR) to be computed."""
        self.metrics.append(metric)

    def analyze(self):
        """
        Iterates through all speech samples and computes metrics:
        - clean vs reconstructed clean
        - degraded vs reconstructed degraded
        - reconstructed clean vs reconstructed degraded
        """
        for clean_id in os.listdir(self.reconstructed_degraded_dir):
            if clean_id.startswith('.'):  # Skip hidden files
                continue

            recon_speaker_dir = os.path.join(self.reconstructed_degraded_dir, clean_id)
            if not os.path.isdir(recon_speaker_dir):
                continue

            for subfolder in os.listdir(recon_speaker_dir):
                # Define full paths for all necessary files
                recon_deg_path = os.path.join(recon_speaker_dir, subfolder, 'reconstructed.wav')
                degraded_path = os.path.join(self.degraded_dir, clean_id, f"{subfolder}.wav")
                clean_path = os.path.join(self.clean_dir, f"{clean_id}.wav")
                recon_clean_path = os.path.join(self.reconstructed_clean_dir, clean_id, 'reconstructed.wav')

                # File existence checks
                if not os.path.isfile(recon_deg_path):
                    print(f"Missing reconstructed degraded file: {recon_deg_path}")
                if not os.path.isfile(degraded_path):
                    print(f"Missing degraded file: {degraded_path}")
                if not os.path.isfile(clean_path):
                    print(f"Missing clean file: {clean_path}")
                if not os.path.isfile(recon_clean_path):
                    print(f"Missing reconstructed clean file: {recon_clean_path}")

                if not (os.path.isfile(recon_deg_path) and os.path.isfile(degraded_path)
                        and os.path.isfile(clean_path) and os.path.isfile(recon_clean_path)):
                    print(f"Skipping {subfolder} - one or more files missing")
                    continue

                # Score results container
                entry = {'file': subfolder}

                # Compute each metric
                for metric in self.metrics:
                    try:
                        entry[f"{metric.name}_clean_vs_recon_clean"] = round(
                            metric.calculate_score(clean_path, recon_clean_path), 3)
                        entry[f"{metric.name}_degraded_vs_recon_degraded"] = round(
                            metric.calculate_score(degraded_path, recon_deg_path), 3)
                        entry[f"{metric.name}_recon_clean_vs_recon_degraded"] = round(
                            metric.calculate_score(recon_clean_path, recon_deg_path), 3)
                    except Exception as e:
                        print(f"[{metric.name}] Error for {subfolder}: {e}")
                        # If an error occurs, fill with None
                        for key in ['clean_vs_recon_clean', 'degraded_vs_recon_degraded',
                                    'recon_clean_vs_recon_degraded']:
                            entry[f"{metric.name}_{key}"] = None

                self.results.append(entry)

    def export_csv(self, path='quality_results.csv'):
        """
        Exports the computed results into a CSV file.

        Args:
        - path: Output CSV file path
        """
        if not self.results:
            print("No results to export.")
            return

        with open(path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.results[0].keys())
            writer.writeheader()
            writer.writerows(self.results)

        print(f"✅ Results exported to {path}")