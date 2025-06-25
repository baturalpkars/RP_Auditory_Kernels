# Run STOI evaluation on reconstructed speech signals

from metrics import STIOMetric
from speech_analyzer import AudioQualityAnalyzer

# Instantiate the analyzer with paths to required directories
analyzer = AudioQualityAnalyzer(
    clean_dir='../clean_speeches',
    degraded_dir='../degraded_speeches',
    reconstructed_clean_dir='../reconstructed_clean_speeches',
    reconstructed_degraded_dir='../reconstructed_speeches'
)

# Add STOI metric (short-time objective intelligibility)
analyzer.add_metric(STIOMetric())

# Run the evaluation
analyzer.analyze()

# Save results to CSV
analyzer.export_csv('results_stoi.csv')
