import os
import numpy as np
import pickle
import librosa
import soundfile as sf
import kernel_analyzer
from ExampleEncodingDecoding import mp_utils as mp


def use_kernels(KERNEL_PATH='./ExampleEncodingDecoding/kernels_15040.jld2',
                NOISES='./dataset/noises',
                OUTPUT_FOLDER='./reconstructed_noises',
                STOP_TYPE='amplitude', STOP_CONDITION=0.1):

    # Load the learned kernels (dictionary)
    dictionary = mp.create_dictionary_from_JLD2(KERNEL_PATH)

    # Make sure output folder exists
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    # Traverse each .wav file in NOISES directory
    for noise_file in os.listdir(NOISES):
        if not noise_file.endswith('.wav'):
            continue

        noise_path = os.path.join(NOISES, noise_file)

        # Load full noise
        noise, sr = librosa.load(noise_path, sr=None)

        # Trim to most energetic 2 seconds
        window_duration = 2.0
        window_length = int(sr * window_duration)
        hop_length = int(sr * 1.0)

        max_rms = -np.inf
        best_segment = None

        for start in range(0, len(noise) - window_length + 1, hop_length):
            segment = noise[start:start + window_length]
            rms = np.sqrt(np.mean(segment ** 2))
            if rms > max_rms:
                max_rms = rms
                best_segment = segment

        if best_segment is None:
            print(f"Skipping {noise_file} — too short or empty.")
            continue

        sliced_noise = best_segment

        print(f"Running matching pursuit on {noise_file}...")

        encoded_waveform, residual = mp.matching_pursuit(
            dictionary, sliced_noise,
            stop_type=STOP_TYPE,
            stop_condition=STOP_CONDITION
        )

        reconstructed_speech, norm_list = mp.reconstruct_and_get_norm(dictionary, encoded_waveform, sliced_noise)
        print(f"Finished matching pursuit on {noise_file}.")

        # Create output folder for this file
        clean_id = os.path.splitext(noise_file)[0]
        output_subfolder = os.path.join(OUTPUT_FOLDER, clean_id)
        os.makedirs(output_subfolder, exist_ok=True)

        # Save reconstructed result
        sf.write(os.path.join(output_subfolder, 'reconstructed.wav'), reconstructed_speech, sr)
        sf.write(os.path.join(output_subfolder, 'trimmed_original.wav'), sliced_noise, sr)
        print(f"Saved trimmed and reconstructed audio for {noise_file}")

        # Save encoded waveform and norm list
        with open(os.path.join(output_subfolder, 'encoded_waveform.pkl'), 'wb') as f:
            pickle.dump(encoded_waveform, f)
        np.save(os.path.join(output_subfolder, 'norm_list.npy'), norm_list)

        # Save plots
        kernel_analyzer.save_plots(sliced_noise, reconstructed_speech, output_subfolder)
        kernel_analyzer.analyze_encoded_waveform(encoded_waveform, noise, len(sliced_noise), norm_list,
                                                 sr, output_subfolder, clean_id)


use_kernels()