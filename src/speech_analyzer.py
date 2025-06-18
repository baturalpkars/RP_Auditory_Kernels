import os
import csv
from src.metrics import MetricStrategy
from typing import List


class AudioQualityAnalyzer:
    def __init__(self, clean_dir, degraded_dir, reconstructed_clean_dir, reconstructed_degraded_dir):
        self.clean_dir = clean_dir
        self.degraded_dir = degraded_dir
        self.reconstructed_clean_dir = reconstructed_clean_dir
        self.reconstructed_degraded_dir = reconstructed_degraded_dir
        self.metrics: List[MetricStrategy] = []
        self.results = []

    def add_metric(self, metric: MetricStrategy):
        self.metrics.append(metric)

    # This method analyzes the audio files in the specified directories with the given metrics.
    def analyze(self):
        for clean_id in os.listdir(self.reconstructed_degraded_dir):
            if clean_id.startswith('.'):  # Skip hidden files like .DS_Store
                continue

            recon_speaker_dir = os.path.join(self.reconstructed_degraded_dir, clean_id)
            if not os.path.isdir(recon_speaker_dir):
                continue

            for subfolder in os.listdir(recon_speaker_dir):
                recon_deg_path = os.path.join(recon_speaker_dir, subfolder, 'reconstructed.wav')
                degraded_path = os.path.join(self.degraded_dir, clean_id, f"{subfolder}.wav")
                clean_path = os.path.join(self.clean_dir, f"{clean_id}.wav")
                recon_clean_path = os.path.join(self.reconstructed_clean_dir, clean_id, 'reconstructed.wav')

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

                entry = {'file': subfolder}

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
                        for key in ['clean_vs_recon_clean', 'degraded_vs_recon_degraded',
                                    'recon_clean_vs_recon_degraded']:
                            entry[f"{metric.name}_{key}"] = None

                self.results.append(entry)

    def export_csv(self, path='quality_results.csv'):
        if not self.results:
            print("No results to export.")
            return

        with open(path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.results[0].keys())
            writer.writeheader()
            writer.writerows(self.results)

        print(f"✅ Results exported to {path}")
