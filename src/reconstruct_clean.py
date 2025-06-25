import os
import kernel_analyzer
import pickle
import librosa
import numpy as np
import soundfile as sf
from ExampleEncodingDecoding import mp_utils as mp


def reconstruct_clean_speeches(speech_files,
                               KERNEL_PATH='../ExampleEncodingDecoding/kernels_15040.jld2',
                               OUTPUT_FOLDER='../reconstructed_clean_speeches',
                               STOP_TYPE='amplitude',
                               STOP_CONDITION=0.1):
    """
    Reconstruct clean speech files using learned auditory kernels.
    Saves reconstructed WAVs, encoded waveform data, norm lists, and analysis plots.

    Args:
        speech_files: Set of clean speech file paths
        KERNEL_PATH: Path to kernel dictionary file (.jld2)
        OUTPUT_FOLDER: Folder to save reconstructed results
        STOP_TYPE: Stopping criterion for matching pursuit ('amplitude' recommended)
        STOP_CONDITION: Threshold for stopping (e.g., 0.1)
    """

    # Load the dictionary of learned kernels
    dictionary = mp.create_dictionary_from_JLD2(KERNEL_PATH)
    print(f"Loaded {len(dictionary)} kernels.")
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    for speech_path in speech_files:
        # Load normalized clean speech
        speech, sr = librosa.load(speech_path, sr=None)

        # Extract speaker utterance ID (e.g., p234_003)
        clean_id = os.path.splitext(os.path.basename(speech_path))[0]

        # Create subfolder for this reconstruction
        output_subfolder = os.path.join(OUTPUT_FOLDER, clean_id)
        os.makedirs(output_subfolder, exist_ok=True)

        # Matching pursuit encoding
        print(f"Running matching pursuit on clean speech: {clean_id}")
        encoded_waveform, residual = mp.matching_pursuit(dictionary, speech,
                                                         stop_type=STOP_TYPE,
                                                         stop_condition=STOP_CONDITION)
        # Reconstruct signal
        reconstructed, norm_list = mp.reconstruct_and_get_norm(dictionary, encoded_waveform, speech)

        # Save reconstructed audio
        sf.write(os.path.join(output_subfolder, 'reconstructed.wav'), reconstructed, sr)

        # Save encoded waveform as pickle
        with open(os.path.join(output_subfolder, 'encoded_waveform.pkl'), 'wb') as f:
            pickle.dump(encoded_waveform, f)

        # Save norm list (for SRR curve)
        np.save(os.path.join(output_subfolder, 'norm_list.npy'), norm_list)

        # Save SRR plot, histogram, and kernel usage visualization
        kernel_analyzer.analyze_encoded_waveform(encoded_waveform, speech, len(speech), norm_list,
                                                 sr, output_subfolder, clean_id)

        # Save original/reconstructed/overlay plots
        kernel_analyzer.save_plots(speech, reconstructed, output_subfolder)

        # Optional: Uncomment to also compute and save MSE
        # mse = np.mean((speech - reconstructed) ** 2)
        # with open(os.path.join(output_subfolder, 'mse.txt'), 'w') as f:
        #     f.write(f"Mean Squared Error (MSE): {mse:.6f}\n")


# ======= Set of clean files to process =======

# List of all speaker-utterance IDs used (100 total)
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

# Create full path strings for clean dataset
speech_files = {f"./dataset/clean_train/{sid}.wav" for sid in all_speech_ids}

# === Run the reconstruction ===
reconstruct_clean_speeches(speech_files)
