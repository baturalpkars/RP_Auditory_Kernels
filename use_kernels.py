import os
from collections import Counter
import numpy as np
import pickle
import librosa
import soundfile as sf
import matplotlib.pyplot as plt
import kernel_analyzer
from ExampleEncodingDecoding import mp_utils as mp


# def analyze_encoded_waveform(encoded_waveform, speech, signal_length, norm_list, sr=16000,
#                              output_folder='./output_analysis', clean_id=""):
#     os.makedirs(output_folder, exist_ok=True)  # Ensure directory exists
#
#     # Bar plot
#     kernel_indices = [entry[0] for entry in encoded_waveform]
#     kernel_counts = Counter(kernel_indices)
#
#     plt.figure(figsize=(10, 4))
#     plt.bar(kernel_counts.keys(), kernel_counts.values())
#     plt.title("Kernel Usage Frequency")
#     plt.xlabel("Kernel Index")
#     plt.ylabel("Count")
#     plt.tight_layout()
#     plt.savefig(os.path.join(output_folder, f'{clean_id}_bar_plot.png'))
#     plt.close()
#
#     # Amplitude distribution
#     amps = [abs(entry[1]) for entry in encoded_waveform]
#     plt.figure(figsize=(8, 3))
#     plt.hist(amps, bins=50)
#     plt.title("Distribution of Amplitudes of Selected Kernels")
#     plt.xlabel("Amplitude")
#     plt.ylabel("Frequency")
#     plt.tight_layout()
#     plt.savefig(os.path.join(output_folder, f'{clean_id}_amp_dist.png'))
#     plt.close()
#
#     # SRR vs Kernels/Second
#     SRR_ld = 20 * np.log10(np.linalg.norm(speech) / norm_list)
#     plt.figure(figsize=(8, 3))
#     plt.plot(np.linspace(1, len(norm_list) / signal_length * sr, len(norm_list)), SRR_ld)
#     plt.title("Number of kernels/second vs the SRR")
#     plt.legend(["Learned"])
#     plt.xlabel("kernels/second")
#     plt.ylabel("SRR [dB]")
#     plt.grid()
#     plt.tight_layout()
#     plt.savefig(os.path.join(output_folder, f'{clean_id}_srr_kernels.png'))
#     plt.close()


def use_kernels_once(KERNEL_PATH='./ExampleEncodingDecoding/kernels_15040.jld2',
                     SPEECH_PATH='./degraded_speeches/p245_132/p245_132_airConditioner_0dB.wav',
                     OUTPUT_RECONSTRUCTED_PATH='./output_reconstructed.wav', STOP_TYPE='amplitude',
                     STOP_CONDITION=0.1):
    # Load the learned kernels (dictionary)
    dictionary = mp.create_dictionary_from_JLD2(KERNEL_PATH)
    print(f"Loaded {len(dictionary)} kernels.")

    # Load the speech file
    speech, sr = librosa.load(SPEECH_PATH, sr=None)

    # Apply Matching Pursuit Encoding
    print("Running matching pursuit. This might take a while...")
    encoded_waveform, residual = mp.matching_pursuit(dictionary, speech,
                                                     stop_type=STOP_TYPE, stop_condition=STOP_CONDITION)
    print("Running matching pursuit finished.")

    # Reconstruct Speech
    reconstructed_speech, norm_list = mp.reconstruct_and_get_norm(dictionary, encoded_waveform, speech)

    # Save the reconstructed speech
    sf.write(OUTPUT_RECONSTRUCTED_PATH, reconstructed_speech, sr)
    print(f"Saved reconstructed speech to {OUTPUT_RECONSTRUCTED_PATH}")

    kernel_analyzer.analyze_encoded_waveform(encoded_waveform, speech, len(speech), norm_list, sr, './output_analysis')

    # Calculate Reconstruction Quality
    mse = np.mean((speech - reconstructed_speech) ** 2)
    print(f"Mean Squared Error (MSE) between original and reconstructed: {mse:.6f}")

    return reconstructed_speech


def use_kernels(KERNEL_PATH='./ExampleEncodingDecoding/kernels_15040.jld2',
                DEGRADED_FOLDER='./degraded_speeches',
                OUTPUT_FOLDER='./reconstructed_speeches', STOP_TYPE='amplitude', STOP_CONDITION=0.1):
    # Load the learned kernels (dictionary)
    dictionary = mp.create_dictionary_from_JLD2(KERNEL_PATH)

    # Make sure output folder exists
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    # Traverse each subfolder in degraded_speeches/
    for speaker_id in os.listdir(DEGRADED_FOLDER):
        speaker_path = os.path.join(DEGRADED_FOLDER, speaker_id)
        if not os.path.isdir(speaker_path):
            continue  # skip non-directories

        # Process each degraded wav inside the speaker folder
        for degraded_file in os.listdir(speaker_path):
            if not degraded_file.endswith('.wav'):
                continue

            degraded_path = os.path.join(speaker_path, degraded_file)

            # Load degraded speech
            speech, sr = librosa.load(degraded_path, sr=None)

            print(f"Running matching pursuit on {degraded_file}...")
            # Reconstruct using matching pursuit
            encoded_waveform, residual = mp.matching_pursuit(dictionary, speech,
                                                             stop_type=STOP_TYPE,
                                                             stop_condition=STOP_CONDITION)
            reconstructed_speech, norm_list = mp.reconstruct_and_get_norm(dictionary, encoded_waveform, speech)
            print(f"Finished matching pursuit on {degraded_file}.")

            # Create output folder for this file
            clean_id = os.path.splitext(degraded_file)[0]
            output_subfolder = os.path.join(OUTPUT_FOLDER, speaker_id, clean_id)
            os.makedirs(output_subfolder, exist_ok=True)

            # Save reconstructed speech
            reconstructed_path = os.path.join(output_subfolder, 'reconstructed.wav')
            sf.write(reconstructed_path, reconstructed_speech, sr)
            print(f"Saved reconstructed speech to {reconstructed_path}")

            # Save MSE
            mse = np.mean((speech - reconstructed_speech) ** 2)
            with open(os.path.join(output_subfolder, 'mse.txt'), 'w') as f:
                f.write(f"Mean Squared Error (MSE): {mse:.6f}\n")

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

            # Save plots
            kernel_analyzer.save_plots(speech, reconstructed_speech, output_subfolder)
            kernel_analyzer.analyze_encoded_waveform(encoded_waveform, speech, len(speech), norm_list,
                                                     sr, output_subfolder, clean_id)


# This script provides functions to use learned kernels for speech reconstruction.
use_kernels()
# use_kernels_once()
