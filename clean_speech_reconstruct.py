import os
from collections import Counter
import kernel_analyzer
import pickle
import librosa
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from ExampleEncodingDecoding import mp_utils as mp


def reconstruct_clean_speeches(speech_files,
                               KERNEL_PATH='./ExampleEncodingDecoding/kernels_15040.jld2',
                               OUTPUT_FOLDER='./reconstructed_clean_speeches',
                               STOP_TYPE='amplitude',
                               STOP_CONDITION=0.1):
    # Load dictionary
    dictionary = mp.create_dictionary_from_JLD2(KERNEL_PATH)
    print(f"Loaded {len(dictionary)} kernels.")
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    for speech_path in speech_files:
        # Load the normalized speech
        speech, sr = librosa.load(speech_path, sr=None)

        # Get clean ID (e.g., p234_003)
        clean_id = os.path.splitext(os.path.basename(speech_path))[0]

        # Output folder: ./reconstructed_speeches/p234_003/original/
        output_subfolder = os.path.join(OUTPUT_FOLDER, clean_id)
        os.makedirs(output_subfolder, exist_ok=True)

        # Matching pursuit
        print(f"Running matching pursuit on clean speech: {clean_id}")
        encoded_waveform, residual = mp.matching_pursuit(dictionary, speech, stop_type=STOP_TYPE,
                                                         stop_condition=STOP_CONDITION)
        reconstructed, norm_list = mp.reconstruct_and_get_norm(dictionary, encoded_waveform, speech)

        # Save WAV
        reconstructed_path = os.path.join(output_subfolder, 'reconstructed.wav')
        sf.write(reconstructed_path, reconstructed, sr)

        # # # Save encoded waveforms to use later!
        # Save encoded waveform
        with open(os.path.join(output_subfolder, 'encoded_waveform.pkl'), 'wb') as f:
            pickle.dump(encoded_waveform, f)

        # Save norm list
        np.save(os.path.join(output_subfolder, 'norm_list.npy'), norm_list)

        # To use saved files.
        # with open(os.path.join(output_subfolder, 'encoded_waveform.pkl'), 'rb') as f:
        #     encoded_waveform = pickle.load(f)
        #
        # # Load norm list
        # norm_list = np.load(os.path.join(output_subfolder, 'norm_list.npy'))

        kernel_analyzer.analyze_encoded_waveform(encoded_waveform, speech, len(speech), norm_list, sr, output_subfolder,
                                                 clean_id)

        # Save MSE
        mse = np.mean((speech - reconstructed) ** 2)
        with open(os.path.join(output_subfolder, 'mse.txt'), 'w') as f:
            f.write(f"Mean Squared Error (MSE): {mse:.6f}\n")

        # Save plots
        kernel_analyzer.save_plots(speech, reconstructed, output_subfolder)
        print(f"Done: {clean_id} — MSE: {mse:.6f}")


# --- Your speech files
speech_files = {
    './clean_speeches/p234_003.wav',
    './clean_speeches/p234_321.wav',
    './clean_speeches/p237_073.wav',
    './clean_speeches/p237_198.wav',
    './clean_speeches/p241_017.wav',
    './clean_speeches/p241_185.wav',
    './clean_speeches/p245_132.wav',
    './clean_speeches/p245_200.wav',
}

# --- Run it
reconstruct_clean_speeches(speech_files)
