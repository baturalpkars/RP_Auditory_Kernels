from metrics import PESQMetric
from speech_analyzer import AudioQualityAnalyzer

analyzer = AudioQualityAnalyzer(
    clean_dir='clean_speeches',
    degraded_dir='degraded_speeches',
    reconstructed_clean_dir='reconstructed_clean_speeches',
    reconstructed_degraded_dir='reconstructed_speeches'
)


analyzer.add_metric(PESQMetric())

analyzer.analyze()
analyzer.export_csv('results_pesq.csv')
