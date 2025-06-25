# Run ViSQOL evaluation on reconstructed speech signals
# ---------------------------------------------------------------------
# ⚠️ ViSQOL is run using Docker. Before running this script:
#   1. Make sure Docker is installed and running on your system.
#   2. The following image must be available (it will pull if not):
#        mubtasimahasan/visqol:v3
#   3. This script will resample audio to 48kHz and mount the current
#      working directory to the Docker container under /data.
#   4. Output will be printed and parsed from the Docker container logs.
# ---------------------------------------------------------------------
#   You can follow the instructions here also; https://github.com/google/visqol

from metrics import ViSQOLMetric
from speech_analyzer import AudioQualityAnalyzer

# Instantiate the analyzer with paths to required directories
analyzer = AudioQualityAnalyzer(
    clean_dir='../clean_speeches',
    degraded_dir='../degraded_speeches',
    reconstructed_clean_dir='../reconstructed_clean_speeches',
    reconstructed_degraded_dir='../reconstructed_speeches'
)

# Add ViSQOL metric (perceptual similarity based on Google ViSQOL)
analyzer.add_metric(ViSQOLMetric())

# Run the evaluation
analyzer.analyze()

# Save results to CSV
analyzer.export_csv('results_visqol.csv')