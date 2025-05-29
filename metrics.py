import numpy as np
import subprocess
import os
import librosa
import soundfile as sf
from pesq import pesq
from scipy.io import wavfile
from pystoi.stoi import stoi


# 1. Metric Base Class
class MetricStrategy:
    @property
    def name(self):
        raise NotImplementedError

    def calculate_score(self, ref_path, deg_path):
        raise NotImplementedError


# 2. SNR Metric
class SNRMetric(MetricStrategy):
    @property
    def name(self):
        return "SNR"

    def calculate_score(self, ref_path, deg_path):
        sr_ref, ref = wavfile.read(ref_path)
        sr_deg, deg = wavfile.read(deg_path)

        if sr_ref != sr_deg:
            raise ValueError("Sampling rates do not match.")

        min_len = min(len(ref), len(deg))
        ref = ref[:min_len]
        deg = deg[:min_len]

        noise = ref - deg
        signal_power = np.mean(ref ** 2)
        noise_power = np.mean(noise ** 2)
        snr = 10 * np.log10(signal_power / (noise_power + 1e-10))
        return snr


# 3. PESQ Metric
class PESQMetric(MetricStrategy):
    @property
    def name(self):
        return "PESQ"

    def calculate_score(self, ref_path, deg_path):
        sr_ref, ref = wavfile.read(ref_path)
        sr_deg, deg = wavfile.read(deg_path)

        if sr_ref != sr_deg:
            raise ValueError("Sampling rates do not match.")

        if sr_ref not in [8000, 16000]:
            raise ValueError("PESQ only supports 8000 or 16000 Hz.")

        min_len = min(len(ref), len(deg))
        ref = ref[:min_len]
        deg = deg[:min_len]

        mode = 'wb' if sr_ref == 16000 else 'nb'
        return pesq(sr_ref, ref, deg, mode)


class ViSQOLMetric(MetricStrategy):
    @property
    def name(self):
        return "ViSQOL"

    def resample_and_save(self, input_path, output_path, sr_target=48000):
        signal, sr = librosa.load(input_path, sr=None)
        if sr != sr_target:
            signal = librosa.resample(signal, orig_sr=sr, target_sr=sr_target)
        sf.write(output_path, signal, sr_target)

    def calculate_score(self, ref_path, deg_path):
        # Prepare resampled files
        ref_resampled = ref_path.replace('.wav', '_resampled.wav')
        deg_resampled = deg_path.replace('.wav', '_resampled.wav')
        self.resample_and_save(ref_path, ref_resampled)
        self.resample_and_save(deg_path, deg_resampled)

        # Docker command
        command = [
            "docker", "run", "--rm",
            "-v", f"{os.getcwd()}:/data",
            "mubtasimahasan/visqol:v3",
            "--reference_file", f"/data/{os.path.relpath(ref_resampled)}",
            "--degraded_file", f"/data/{os.path.relpath(deg_resampled)}"
        ]
        print(f"Running ViSQOL with command:\n{' '.join(command)}")
        try:
            result = subprocess.run(command, capture_output=True, text=True)
            print(f"[ViSQOL STDOUT for {os.path.basename(ref_path)}]:\n{result.stdout}")
            print(f"[ViSQOL STDERR for {os.path.basename(ref_path)}]:\n{result.stderr}")

            if result.returncode != 0:
                print(f"ViSQOL Docker returned error code: {result.returncode}")
                return None

            for line in result.stdout.splitlines():
                if "MOS-LQO" in line:
                    return float(line.split(":")[-1].strip())

            print(f"⚠️ ViSQOL MOS-LQO not found in output for: {ref_path}")
            return None

        except Exception as e:
            print(f"❌ ViSQOL failed for {ref_path} vs {deg_path}: {e}")
            return None


class STIOMetric(MetricStrategy):
    @property
    def name(self):
        return "STOI"

    def calculate_score(self, ref_path, deg_path):
        fs, ref = wavfile.read(ref_path)
        _, deg = wavfile.read(deg_path)

        # Truncate to equal length
        min_len = min(len(ref), len(deg))
        ref = ref[:min_len]
        deg = deg[:min_len]

        return stoi(ref, deg, fs, extended=False)
