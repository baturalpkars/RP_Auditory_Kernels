import os
import numpy as np
import pickle
import librosa
import soundfile as sf
import kernel_analyzer
from ExampleEncodingDecoding import mp_utils as mp


def use_kernels(KERNEL_PATH='../ExampleEncodingDecoding/kernels_15040.jld2',
                NOISES='../dataset/noises',
                OUTPUT_FOLDER='../reconstructed_noises',
                STOP_TYPE='amplitude',
                STOP_CONDITION=0.1):
    """
    Reconstructs noise-only samples using learned auditory kernels and saves both
    reconstructed audio and kernel analysis outputs.

    Args:
        KERNEL_PATH: Path to the learned auditory kernel dictionary (.jld2 format)
        NOISES: Folder containing noise-only .wav files
        OUTPUT_FOLDER: Where to save results (reconstructed .wav, encodings, plots)
        STOP_TYPE: Type of stopping criterion used in matching pursuit
        STOP_CONDITION: Threshold value for stopping
    """

    # Load learned dictionary of auditory kernels
    dictionary = mp.create_dictionary_from_JLD2(KERNEL_PATH)

    # Create output folder if it doesn't exist
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    # Process each noise .wav file
    for noise_file in os.listdir(NOISES):
        if not noise_file.endswith('.wav'):
            continue

        noise_path = os.path.join(NOISES, noise_file)

        # Load full noise signal
        noise, sr = librosa.load(noise_path, sr=None)

        # Extract first 6 seconds for analysis
        duration_sec = 6.0
        end_sample = int(sr * duration_sec)
        if len(noise) < end_sample:
            print(f"Skipping {noise_file} — shorter than 6 seconds.")
            continue

        sliced_noise = noise[:end_sample]

        print(f"Running matching pursuit on {noise_file}...")

        # Matching Pursuit encoding
        encoded_waveform, residual = mp.matching_pursuit(
            dictionary, sliced_noise,
            stop_type=STOP_TYPE,
            stop_condition=STOP_CONDITION
        )

        # Reconstruct signal and get SRR norm list
        reconstructed_speech, norm_list = mp.reconstruct_and_get_norm(
            dictionary, encoded_waveform, sliced_noise)

        print(f"Finished matching pursuit on {noise_file}.")

        # === Save results ===
        clean_id = os.path.splitext(noise_file)[0]
        output_subfolder = os.path.join(OUTPUT_FOLDER, clean_id)
        os.makedirs(output_subfolder, exist_ok=True)

        # Save reconstructed waveform and original trimmed noise
        sf.write(os.path.join(output_subfolder, 'reconstructed.wav'), reconstructed_speech, sr)
        sf.write(os.path.join(output_subfolder, 'trimmed_original.wav'), sliced_noise, sr)
        print(f"Saved trimmed and reconstructed audio for {noise_file}")

        # Save encoded waveform and SRR norm list
        with open(os.path.join(output_subfolder, 'encoded_waveform.pkl'), 'wb') as f:
            pickle.dump(encoded_waveform, f)
        np.save(os.path.join(output_subfolder, 'norm_list.npy'), norm_list)

        # Save waveform plots and SRR/kernel analysis
        kernel_analyzer.save_plots(sliced_noise, reconstructed_speech, output_subfolder)
        kernel_analyzer.analyze_encoded_waveform(
            encoded_waveform, noise, len(sliced_noise), norm_list,
            sr, output_subfolder, clean_id)


# Entry point
use_kernels()
