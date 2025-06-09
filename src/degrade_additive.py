import os
import numpy as np
import librosa
import soundfile as sf


def add_additive_noise_and_save(clean, noises, sr, clean_filename, base_folder, snr_db=10):
    """
    Adds different types of noise to the clean speech and saves the degraded speech files.

    Args:
    - clean: The clean speech signal (numpy array)
    - noises: Dictionary of noise signals
    - sr: Sampling rate
    - clean_filename: Name of the clean file (e.g., "p234_003")
    - output_folder: Where to save degraded speech files
    - snr_db: Desired SNR in dB (default: 10)
    """
    output_folder = os.path.join(base_folder, clean_filename)
    os.makedirs(output_folder, exist_ok=True)

    for environment_type, additive_noise in noises.items():
        # Match lengths
        if len(additive_noise) < len(clean):
            repeat_factor = int(np.ceil(len(clean) / len(additive_noise)))
            additive_noise = np.tile(additive_noise, repeat_factor)

        # Cut noise to exact speech length
        additive_noise = additive_noise[:len(clean)]

        # Add noise at desired SNR
        degraded_speech = add_noise_to_signal(clean, additive_noise, snr_db)

        # Create filename
        output_filename = f"{clean_filename}_{environment_type}_{snr_db}dB.wav"
        output_path = os.path.join(output_folder, output_filename)

        # Save the degraded speech
        sf.write(output_path, degraded_speech, sr)
        print(f"Saved: {output_path}")


def add_noise_to_signal(clean, additive_noise, snr_db):
    """

    Args:
        clean: The clean speech signal (numpy array)
        additive_noise: Additive noise that we want to add
        snr_db: Desired SNR

    Returns:
        noisy_signal: Degraded speech
    """
    # Calculate current power
    clean_power = np.mean(clean ** 2)
    noise_power = np.mean(additive_noise ** 2)

    # Calculate required noise power for desired SNR
    # SNR = 10 * log10(speech^2 / desired_noise_power^2)
    # Thus -> desired_noise_power = speech^2 / (10^(snr_db/10))
    desired_noise_power = clean_power / (10 ** (snr_db / 10))

    # Scale noise to desired power
    additive_noise = additive_noise * np.sqrt(desired_noise_power / noise_power)

    # Add noise to clean signal
    noisy_signal = clean + additive_noise
    return noisy_signal


def generate_white_noise(duration=10.0, sample_rate=16000):
    # Parameters
    duration = duration  # seconds
    sample_rate = sample_rate  # Hz

    # Generate white noise
    white_noise = np.random.normal(0, 1, int(sample_rate * duration))

    # Save as WAV if needed
    sf.write("../dataset/noises/white_noise_1.wav", white_noise, sample_rate)


def normalize_to_target_rms(signal, target_rms=0.1):
    current_rms = np.sqrt(np.mean(signal ** 2))
    if current_rms == 0:
        return signal
    scaling_factor = target_rms / current_rms
    return signal * scaling_factor


noise_files = {
    # 'airConditioner': './MS-SNSD/noise_train/AirConditioner_1.wav',
    # 'copyMachine': './MS-SNSD/noise_train/CopyMachine_5.wav',
    # 'munching': './MS-SNSD/noise_train/Munching_8.wav',
    'airportAnnouncement': './dataset/noises/AirportAnnouncements_7.wav',
    'babble': './dataset/noises/Babble_2.wav',
    'train_coming': './dataset/noises/TrainComing_1.wav',
    'white_noise': './dataset/noises/white_noise_1.wav',
    # Add more types here...
}

# speech_files = {'./dataset/clean_train/p234_003.wav', './dataset/clean_train/p234_321.wav',
#                 './dataset/clean_train/p237_073.wav', './dataset/clean_train/p237_198.wav',
#                 './dataset/clean_train/p241_017.wav', './dataset/clean_train/p241_185.wav',
#                 './dataset/clean_train/p245_132.wav', './dataset/clean_train/p245_200.wav'}

# List of all file IDs (both men and women, each with 2 utterances)
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

# Set of full file paths assuming they are in ./dataset/clean_train/
speech_files = {f"./dataset/clean_train/{sid}.wav" for sid in all_speech_ids}

# Base output folder
base_output_folder = './degraded_speeches'

# Process each clean speech
for speech_path in speech_files:
    speech, sr = librosa.load(speech_path, sr=None)
    speech = normalize_to_target_rms(speech, target_rms=0.1)  # Normalize the clean speech

    # Extract clean file name (e.g., p234_003)
    clean_filename = os.path.splitext(os.path.basename(speech_path))[0]

    clean_save_folder = './clean_speeches'
    os.makedirs(clean_save_folder, exist_ok=True)
    clean_output_path = os.path.join(clean_save_folder, f'{clean_filename}.wav')
    sf.write(clean_output_path, speech, sr)

    # Load and store all noise samples once per speech
    loaded_noises = {}
    for noise_type, filepath in noise_files.items():
        noise, _ = librosa.load(filepath, sr=sr)
        loaded_noises[noise_type] = noise

    # Apply for multiple SNR levels
    for snr in [-5, 0, 5, 10]:
        add_additive_noise_and_save(speech, loaded_noises, sr, clean_filename, base_output_folder, snr_db=snr)
