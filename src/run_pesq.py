# Run PESQ evaluation on reconstructed speech signals

from metrics import PESQMetric
from speech_analyzer import AudioQualityAnalyzer

# Instantiate the analyzer with paths to required directories
analyzer = AudioQualityAnalyzer(
    clean_dir='../clean_speeches',
    degraded_dir='../degraded_speeches',
    reconstructed_clean_dir='../reconstructed_clean_speeches',
    reconstructed_degraded_dir='../reconstructed_speeches'
)

# Add PESQ metric (perceptual evaluation of speech quality)
analyzer.add_metric(PESQMetric())

# Run the evaluation
analyzer.analyze()

# Save results to CSV
analyzer.export_csv('results_pesq.csv')
