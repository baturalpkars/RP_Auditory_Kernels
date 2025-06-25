import os
import numpy as np
import subprocess
import librosa
import soundfile as sf
from pesq import pesq
from scipy.io import wavfile
from pystoi.stoi import stoi


# ===============================
# Abstract Base Class for Metrics
# ===============================

class MetricStrategy:
    @property
    def name(self):
        """Name of the metric (to be overridden)."""
        raise NotImplementedError

    def calculate_score(self, ref_path, deg_path):
        """Compute metric score given reference and degraded paths."""
        raise NotImplementedError


# ===============
# SNR Metric
# ===============

class SNRMetric(MetricStrategy):
    @property
    def name(self):
        return "SNR"

    def calculate_score(self, ref_path, deg_path):
        """
        Signal-to-Noise Ratio (SNR) in dB.

        Higher values indicate better fidelity.
        """
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


# ===============
# PESQ Metric
# ===============

class PESQMetric(MetricStrategy):
    @property
    def name(self):
        return "PESQ"

    def calculate_score(self, ref_path, deg_path):
        """
        PESQ (Perceptual Evaluation of Speech Quality) score.
        Range: [1.0 - 4.5] (higher is better).
        """
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


# ===============
# ViSQOL Metric
# ===============

class ViSQOLMetric(MetricStrategy):
    @property
    def name(self):
        return "ViSQOL"

    def resample_and_save(self, input_path, output_path, sr_target=48000):
        """Resamples audio to 48kHz and saves as WAV (ViSQOL requires 48kHz)."""
        signal, sr = librosa.load(input_path, sr=None)
        if sr != sr_target:
            signal = librosa.resample(signal, orig_sr=sr, target_sr=sr_target)
        sf.write(output_path, signal, sr_target)

    def calculate_score(self, ref_path, deg_path):
        """
        ViSQOL (Virtual Speech Quality Objective Listener).
        Outputs a MOS-LQO score (1.0–5.0).
        Requires Docker + resampled 48kHz inputs.
        """
        ref_resampled = ref_path.replace('.wav', '_resampled.wav')
        deg_resampled = deg_path.replace('.wav', '_resampled.wav')
        self.resample_and_save(ref_path, ref_resampled)
        self.resample_and_save(deg_path, deg_resampled)

        # Compose Docker command
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

            # Parse output for MOS-LQO
            for line in result.stdout.splitlines():
                if "MOS-LQO" in line:
                    return float(line.split(":")[-1].strip())

            print(f"⚠️ ViSQOL MOS-LQO not found in output for: {ref_path}")
            return None

        except Exception as e:
            print(f"❌ ViSQOL failed for {ref_path} vs {deg_path}: {e}")
            return None


# ===============
# STOI Metric
# ===============

class STIOMetric(MetricStrategy):
    @property
    def name(self):
        return "STOI"

    def calculate_score(self, ref_path, deg_path):
        """
        Short-Time Objective Intelligibility (STOI) score.
        Range: [0.0 - 1.0] (higher = more intelligible).
        """
        fs, ref = wavfile.read(ref_path)
        _, deg = wavfile.read(deg_path)

        # Truncate to same length
        min_len = min(len(ref), len(deg))
        ref = ref[:min_len]
        deg = deg[:min_len]

        return stoi(ref, deg, fs, extended=False)
