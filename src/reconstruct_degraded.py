import os
import numpy as np
import pickle
import librosa
import soundfile as sf

from src import kernel_analyzer
from ExampleEncodingDecoding import mp_utils as mp


def use_kernels(KERNEL_PATH='../ExampleEncodingDecoding/kernels_15040.jld2',
                DEGRADED_FOLDER='../degraded_speeches',
                OUTPUT_FOLDER='../reconstructed_speeches',
                STOP_TYPE='amplitude',
                STOP_CONDITION=0.1):
    """
    Reconstructs all degraded speech files using auditory kernel matching pursuit.

    Args:
        KERNEL_PATH (str): Path to learned dictionary of kernels.
        DEGRADED_FOLDER (str): Path to degraded input speech folders.
        OUTPUT_FOLDER (str): Where to save the reconstructed outputs.
        STOP_TYPE (str): Stopping criteria type for matching pursuit (e.g., 'amplitude').
        STOP_CONDITION (float): Threshold for stopping (e.g., residual amplitude < 0.1).
    """

    # Load pre-trained kernel dictionary
    dictionary = mp.create_dictionary_from_JLD2(KERNEL_PATH)

    # Optional: skip reprocessing these speaker IDs
    processed_ids = {
        # Add speaker IDs here if you want to skip them
    }

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    # Loop over each speaker directory
    for speaker_id in os.listdir(DEGRADED_FOLDER):
        if speaker_id in processed_ids:
            print(f"Skipping {speaker_id}, already processed.")
            continue

        speaker_path = os.path.join(DEGRADED_FOLDER, speaker_id)
        if not os.path.isdir(speaker_path):
            continue  # Skip files, only process directories

        # Loop over each degraded .wav file in the speaker directory
        for degraded_file in os.listdir(speaker_path):
            if not degraded_file.endswith('.wav'):
                continue

            degraded_path = os.path.join(speaker_path, degraded_file)
            speech, sr = librosa.load(degraded_path, sr=None)

            # To debug
            # print(f"Running matching pursuit on {degraded_file}...")

            # Encode using Matching Pursuit (MP)
            encoded_waveform, residual = mp.matching_pursuit(
                dictionary, speech,
                stop_type=STOP_TYPE,
                stop_condition=STOP_CONDITION
            )

            # Reconstruct signal from encoded representation
            reconstructed_speech, norm_list = mp.reconstruct_and_get_norm(
                dictionary, encoded_waveform, speech
            )

            print(f"Finished matching pursuit on {degraded_file}.")

            # Prepare output folder
            clean_id = os.path.splitext(degraded_file)[0]
            output_subfolder = os.path.join(OUTPUT_FOLDER, speaker_id, clean_id)
            os.makedirs(output_subfolder, exist_ok=True)

            # === Save outputs ===

            # Save reconstructed waveform
            reconstructed_path = os.path.join(output_subfolder, 'reconstructed.wav')
            sf.write(reconstructed_path, reconstructed_speech, sr)

            # Save encoded waveform (for reuse or analysis)
            with open(os.path.join(output_subfolder, 'encoded_waveform.pkl'), 'wb') as f:
                pickle.dump(encoded_waveform, f)

            # Save norm list (used for diagnostic SRR plots)
            np.save(os.path.join(output_subfolder, 'norm_list.npy'), norm_list)

            # Generate and save analysis plots
            kernel_analyzer.save_plots(speech, reconstructed_speech, output_subfolder)

            # Save kernel activation histogram and SRR analysis
            kernel_analyzer.analyze_encoded_waveform(
                encoded_waveform,
                speech,
                len(speech),
                norm_list,
                sr,
                output_subfolder,
                clean_id
            )


# === ENTRY POINT ===
# This will run when the script is executed directly
use_kernels()
