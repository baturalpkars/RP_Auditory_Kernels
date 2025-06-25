import os
import numpy as np
import librosa
import soundfile as sf

def add_additive_noise_and_save(clean, noises, sr, clean_filename, base_folder, snr_db=10):
    """
    Adds various noise types to a clean signal at a specified SNR,
    and saves the degraded versions to disk.

    Args:
        clean (np.ndarray): Clean speech waveform.
        noises (dict): Dictionary mapping noise type names to noise waveforms.
        sr (int): Sampling rate.
        clean_filename (str): Identifier for the clean speech (e.g., 'p234_003').
        base_folder (str): Base path to store degraded outputs.
        snr_db (float): Desired Signal-to-Noise Ratio (in dB).
    """
    output_folder = os.path.join(base_folder, clean_filename)
    os.makedirs(output_folder, exist_ok=True)

    for environment_type, additive_noise in noises.items():
        # Repeat noise if it's shorter than the clean speech
        if len(additive_noise) < len(clean):
            repeat_factor = int(np.ceil(len(clean) / len(additive_noise)))
            additive_noise = np.tile(additive_noise, repeat_factor)

        # Trim to match clean length
        additive_noise = additive_noise[:len(clean)]

        # Add noise at desired SNR
        degraded_speech = add_noise_to_signal(clean, additive_noise, snr_db)

        # Save degraded version
        output_filename = f"{clean_filename}_{environment_type}_{snr_db}dB.wav"
        output_path = os.path.join(output_folder, output_filename)
        sf.write(output_path, degraded_speech, sr)
        print(f"Saved: {output_path}")


def add_noise_to_signal(clean, additive_noise, snr_db):
    """
    Mix clean speech with additive noise at the given SNR.

    Args:
        clean (np.ndarray): Clean speech signal.
        additive_noise (np.ndarray): Noise signal.
        snr_db (float): Desired Signal-to-Noise Ratio in dB.

    Returns:
        np.ndarray: Noisy signal.
    """
    # Calculate current power
    clean_power = np.mean(clean ** 2)
    noise_power = np.mean(additive_noise ** 2)

    # Calculate required noise power for desired SNR
    # SNR = 10 * log10(speech^2 / desired_noise_power^2)
    # Thus -> desired_noise_power = speech^2 / (10^(snr_db/10))
    desired_noise_power = clean_power / (10 ** (snr_db / 10))
    scaled_noise = additive_noise * np.sqrt(desired_noise_power / noise_power)

    return clean + scaled_noise


def generate_white_noise(duration=10.0, sample_rate=16000):
    """
    Generates white noise and saves it to disk.

    Args:
        duration (float): Duration in seconds.
        sample_rate (int): Sampling rate in Hz.
    """
    white_noise = np.random.normal(0, 1, int(sample_rate * duration))
    sf.write("../dataset/noises/white_noise_1.wav", white_noise, sample_rate)


def normalize_to_target_rms(signal, target_rms=0.1):
    """
    Normalize signal to a target Root Mean Square (RMS) energy.

    Args:
        signal (np.ndarray): Input signal.
        target_rms (float): Desired RMS energy.

    Returns:
        np.ndarray: Normalized signal.
    """
    current_rms = np.sqrt(np.mean(signal ** 2))
    if current_rms == 0:
        return signal
    return signal * (target_rms / current_rms)


# === Configuration ===

# Noise sources to use for degradation
noise_files = {
    'airportAnnouncement': './dataset/noises/AirportAnnouncements_7.wav',
    'babble': './dataset/noises/Babble_2.wav',
    'train_coming': './dataset/noises/TrainComing_1.wav',
    'white_noise': './dataset/noises/white_noise_1.wav',
}

# 100 speech files (50 speakers x 2 utterances each) used for degradation
all_speech_ids = [
    "p237_073", "p237_198", "p241_017", "p241_185", "p245_132", "p245_200", "p246_020", "p246_032",
    "p247_004", "p247_010", "p251_015", "p251_038", "p260_005", "p260_042", "p263_013", "p263_014",
    "p271_009", "p271_143", "p272_011", "p272_330", "p275_034", "p275_036", "p281_079", "p281_085",
    "p284_016", "p284_184", "p285_060", "p285_080", "p292_057", "p292_058", "p298_027", "p298_028",
    "p302_135", "p302_244", "p304_081", "p304_082", "p316_122", "p316_128", "p326_037", "p326_039",
    "p334_054", "p334_210", "p345_249", "p345_385", "p347_102", "p347_115", "p360_160", "p360_188",
    "p363_268", "p363_422",
    "p234_003", "p234_321", "p248_142", "p248_306", "p249_035", "p249_241", "p255_078", "p255_125",
    "p264_012", "p264_018", "p265_031", "p265_029", "p266_001", "p266_002", "p283_025", "p283_033",
    "p293_007", "p293_123", "p295_030", "p295_040", "p299_074", "p299_086", "p301_118", "p301_130",
    "p303_050", "p303_055", "p305_175", "p305_176", "p306_109", "p306_110", "p307_076", "p307_077",
    "p308_154", "p308_155", "p310_068", "p310_212", "p312_134", "p312_205", "p313_124", "p313_131",
    "p314_164", "p314_166", "p323_083", "p323_084", "p333_072", "p333_075", "p335_121", "p335_256",
    "p336_223", "p336_344"
]

# Full paths to clean speech files
speech_files = {f"./dataset/clean_train/{sid}.wav" for sid in all_speech_ids}

# Output directory for degraded samples
base_output_folder = './degraded_speeches'

# === Processing Loop ===

for speech_path in speech_files:
    speech, sr = librosa.load(speech_path, sr=None)
    speech = normalize_to_target_rms(speech, target_rms=0.1)

    clean_filename = os.path.splitext(os.path.basename(speech_path))[0]

    # Save normalized clean version separately
    clean_save_folder = './clean_speeches'
    os.makedirs(clean_save_folder, exist_ok=True)
    clean_output_path = os.path.join(clean_save_folder, f'{clean_filename}.wav')
    sf.write(clean_output_path, speech, sr)

    # Load all noise samples for this speech file
    loaded_noises = {}
    for noise_type, filepath in noise_files.items():
        noise, _ = librosa.load(filepath, sr=sr)
        loaded_noises[noise_type] = noise

    # Generate and save degraded versions at multiple SNR levels
    for snr in [-5, 0, 5, 10]:
        add_additive_noise_and_save(
            clean=speech,
            noises=loaded_noises,
            sr=sr,
            clean_filename=clean_filename,
            base_folder=base_output_folder,
            snr_db=snr
        )

# Generate white noise sample if it doesn't exist
# if not os.path.exists('./dataset/noises/white_noise_1.wav'):
#     generate_white_noise(duration=10.0, sample_rate=16000)
#     print("Generated white noise sample.")
