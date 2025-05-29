from metrics import SNRMetric, PESQMetric, ViSQOLMetric, STIOMetric
from speech_analyzer import AudioQualityAnalyzer

analyzer = AudioQualityAnalyzer(
    clean_dir='clean_speeches',
    degraded_dir='degraded_speeches',
    reconstructed_clean_dir='reconstructed_clean_speeches',
    reconstructed_degraded_dir='reconstructed_speeches'
)

# analyzer.add_metric(SNRMetric())
# analyzer.add_metric(PESQMetric())
# analyzer.add_metric(ViSQOLMetric())
analyzer.add_metric(STIOMetric())

analyzer.analyze()
analyzer.export_csv('results_stoi.csv')
