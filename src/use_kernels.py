import os
import numpy as np
import pickle
import librosa
import soundfile as sf
from src import kernel_analyzer
from ExampleEncodingDecoding import mp_utils as mp


def use_kernels(KERNEL_PATH='../ExampleEncodingDecoding/kernels_15040.jld2',
                DEGRADED_FOLDER='../degraded_speeches',
                OUTPUT_FOLDER='../reconstructed_speeches', STOP_TYPE='amplitude', STOP_CONDITION=0.1):
    # Load the learned kernels (dictionary)
    dictionary = mp.create_dictionary_from_JLD2(KERNEL_PATH)

    processed_ids = {
        "p234_003", "p249_241", "p272_330", "p293_123", "p304_081", "p312_134", "p333_072",
        "p237_198", "p255_078", "p275_034", "p295_040", "p305_175", "p312_205", "p335_121",
        "p241_185", "p255_125", "p275_036", "p298_027", "p307_076", "p313_124", "p335_256",
        "p245_200", "p260_005", "p281_079", "p299_074", "p307_077", "p314_164", "p336_223",
        "p246_020", "p264_012", "p284_016", "p301_118", "p308_154", "p316_128", "p336_344",
        "p246_032", "p266_001", "p285_060", "p301_130", "p308_155", "p323_083", "p347_115",
        "p248_142", "p271_009", "p285_080", "p302_244", "p310_068", "p323_084", "p360_160",
        "p249_035", "p272_011", "p292_057", "p303_055", "p310_212", "p326_037", "p360_188",
        "p363_268"
    }

    # Make sure output folder exists
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    # Traverse each subfolder in degraded_speeches/
    for speaker_id in os.listdir(DEGRADED_FOLDER):
        if speaker_id in processed_ids:
            print(f"Skipping {speaker_id}, already processed.")
            continue

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
            # print(f"Saved reconstructed speech to {reconstructed_path}")

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
