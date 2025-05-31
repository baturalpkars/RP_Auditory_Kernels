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
    sf.write("dataset/noises/white_noise_1.wav", white_noise, sample_rate)


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

speech_files = {'./dataset/clean_train/p234_003.wav', './dataset/clean_train/p234_321.wav',
                './dataset/clean_train/p237_073.wav', './dataset/clean_train/p237_198.wav',
                './dataset/clean_train/p241_017.wav', './dataset/clean_train/p241_185.wav',
                './dataset/clean_train/p245_132.wav', './dataset/clean_train/p245_200.wav'}

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
