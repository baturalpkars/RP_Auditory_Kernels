# Author: Dimme
#
# These are functions used to explain/implement matching pursuit. The most important ones are
#
#   create_dictionary(*kernels)
#   matching_pursuit(dictionary, x, stop_condition, stop_type)
#   reconstruct_matching_pursuit(dictionary, encoded_waveform)
#
# Todo: it would be better if some stuff can run on gpu/multithreaded as well. Mostly during training this is handy. 


import numpy as np
import os
import h5py
from concurrent.futures import ThreadPoolExecutor
import scipy.signal as sig
import matplotlib.pyplot as plt


# Define a class to hold kernel data
class Kernel:
    def __init__(self, kernel, gradient, abs_amp):
        self.kernel = np.asarray(kernel, dtype=np.float64)         # The actual kernel (1D array)
        self.gradient = np.asarray(gradient, dtype=np.float64)     # The gradient of the kernel (1D array)
        self.abs_amp = float(abs_amp)                              # The absolute amplitude (float)

    def __repr__(self):
        return f"Kernel(kernel={self.kernel}, gradient={self.gradient}, abs_amp={self.abs_amp})"


# Construct a dictionary. Takes kernels as input
def create_dictionary(*kernels):
    dictionary = []
    for i, kernel in enumerate(kernels):
        gradient = np.zeros_like(kernel)
        abs_amp = 0.0
        dictionary.append(Kernel(kernel, gradient, abs_amp))
    return dictionary


def create_gammatone_dictionary(num_filters, fs, f_min, f_max, threshold_value=0.01):
    """
    Create a dictionary of gammatone filters spaced linearly on an ERB-rate scale.

    Parameters:
        num_filters (int): Number of gammatone filters.
        fs (float): Sampling frequency in Hz.
        f_min (float): Minimum center frequency in Hz.
        f_max (float): Maximum center frequency in Hz.
        duration (float): Duration of the filters in seconds.
        threshold_value (float): Threshold value for removing small values from the filters.
    Returns:
        list: A list of Kernel objects representing the gammatone filters.
    """

    # Generate center frequencies spaced linearly on an ERB-rate scale
    erb_min = 21.4 * np.log10(4.37e-3 * f_min + 1)
    erb_max = 21.4 * np.log10(4.37e-3 * f_max + 1)
    erb_space = np.linspace(erb_min, erb_max, num_filters)
    center_frequencies = (10 ** (erb_space / 21.4) - 1) / 4.37e-3

    # Create gammatone filters
    dictionary = []
    for cf in center_frequencies:
        kernel, _ = sig.gammatone(cf, 'fir', 4, fs=fs, numtaps=round(0.1*fs))
        threshold = threshold_value * np.max(np.abs(kernel))
        valid_indices = np.where(np.abs(kernel) > threshold)[0]
        if valid_indices.size > 0:
            kernel = kernel[:valid_indices[-1] + 1]
        kernel = kernel/np.linalg.norm(kernel)
        gradient = np.zeros_like(kernel)
        abs_amp = 0.0
        dictionary.append(Kernel(kernel, gradient, abs_amp))

    return dictionary


def create_dictionary_from_JLD2(filepath):
    """
    Create a dictionary from a JLD2 file containing kernels.
    NOTE: Atm the gradient and abs_amp are not imported from the JLD2 file.
    Parameters:
        filepath (str): Path to the JLD2 file.

    Returns:
        list: A list of Kernel objects representing the kernels in the JLD2 file.
    """

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    dictionary = []
    with h5py.File(filepath, "r") as f:
        kernels = f["kernels"][:]  # Load dataset

        for kernel_entry in kernels:
            kernel_obj = f[kernel_entry]  # First dereference
            kernel_ref = kernel_obj["kernel"]  # Get kernel reference

            if isinstance(kernel_ref, h5py.Reference):  # Ensure it's a reference
                kernel_data = f[kernel_ref][:]  # Second dereference to get actual data
            else:
                kernel_data = kernel_ref[:]  # If it's already data, just extract it

            # Normalize the kernel
            kernel_data = kernel_data / np.linalg.norm(kernel_data)

            # Create a Kernel object
            gradient = np.zeros_like(kernel_data)
            abs_amp = 0.0
            dictionary.append(Kernel(kernel_data, gradient, abs_amp))

    return dictionary


def correlation(x, Di):
    y = sig.correlate(x, Di, mode='valid')
    return y


# Decode (reconstruct) the waveform based on the encoded waveform
def reconstruct(dictionary, encoded_waveform, output_length = None):
    if output_length == None:
        print("Should implement a way to get an output_length here (mp_utils.reconstruct)")
    
    reconstructed_waveform = np.zeros(output_length)
    for (dicElement, amp, indx) in encoded_waveform:
        kernel = dictionary[dicElement].kernel
        reconstructed_waveform[indx:indx+len(kernel)] += amp*kernel
    
    return reconstructed_waveform


# Decode (reconstruct) the waveform based on the encoded waveform
def reconstruct_and_get_norm(dictionary, encoded_waveform, waveform):
    norm_list = []
    reconstructed_waveform = np.zeros(len(waveform))
    for (dicElement, amp, indx) in encoded_waveform:
        kernel = dictionary[dicElement].kernel
        reconstructed_waveform[indx:indx+len(kernel)] += amp*kernel
        norm_list.append(np.linalg.norm(waveform - reconstructed_waveform))

    return reconstructed_waveform, norm_list



# Single iteration of matching pursuit (parallelized with multithreading)
def matching_pursuit_iter(dictionary, x_res):
    def process_element(index, element):
        x_corr = correlation(x_res, element.kernel)  # Compute the cross-correlation
        indxMax_tmp = np.argmax(abs(x_corr))  # Find the location where the max abs value is largest
        ampMax_tmp = x_corr[indxMax_tmp]  # Find the corresponding amplitude
        return ampMax_tmp, indxMax_tmp, index

    results = []
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_element, i, element) for i, element in enumerate(dictionary)]
        for future in futures:
            results.append(future.result())

    # Find the kernel element with the maximum value
    ampMax, indxMax, dicElement = max(results, key=lambda x: abs(x[0]))

    # Subtract the kernel from the residual
    x_res[indxMax:indxMax + len(dictionary[dicElement].kernel)] -= ampMax * dictionary[dicElement].kernel
    return x_res, ampMax, indxMax, dicElement
    
    
def matching_pursuit(dictionary, x, stop_type, stop_condition):
    # Inputs:
    #   stop_type: "amplitude" or "iterations"
    #   stop_condition: the value, i..e. 100 (iterations), 0.1 (amplitude)
    #   dictionary: a list of elements with .kernel containing the kernel
    #   x: the waveform that needs to be encoded
    # Outputs: 
    #   encoded_waveform:   list of (dic_element, time_index, amplitude)
    #   x_res:              the unencoded residual   
    
    x_res = x.copy()
    encoded_waveform = []
    
    count = 1
    while True:
        x_res, ampMax, indxMax, dicElement = matching_pursuit_iter(dictionary, x_res)
        encoded_waveform.append((dicElement, ampMax, indxMax))
        
        # Check stop condition
        count +=1 
        if stop_type == 'amplitude': # Consider 
            if (abs(ampMax) < stop_condition) or (np.linalg.norm(x_res) < 1e-8):
                break
        elif stop_type == 'iterations':
            if (count > stop_condition) or (np.linalg.norm(x_res) < 1e-8): #1e-8 is somewhat arbitrary. Consider adding it as parameter
                break
        else:
            print("Invalid stop condition specified. Breaking the loop")
            break

    return encoded_waveform, x_res    


def plot_dictionary_elements(dictionary):
    fig, axes = plt.subplots(4, 8, figsize=(12, 6))
    plt.subplots_adjust(wspace=0.3, hspace=0.3)
    
    for i, ax in enumerate(axes.flat):
        if i < len(dictionary):
            kernel = dictionary[i].kernel
            ax.plot(kernel)
            
            # Add some text indicating the length of the kernel
            kernel_length = len(kernel)
            ax.text(0.5, 0.9, f"{kernel_length} samples", fontsize=8, color='red', transform=ax.transAxes)
            
        ax.axis('off')  # Remove axis labels and ticks
    
    plt.tight_layout()
    plt.show()


# Construct simple example waveform
def get_simple_waveform(slct=1):
    if slct == 1:
        x = np.zeros(400)

        # The square
        x[30:60] = 1 

        # The sawtooth
        t = np.linspace(0,1,80) 
        x[120:200] = -0.9*sig.sawtooth(0.5*np.pi * t + 0.5*np.pi, 0.5)

        # The sinusoid
        x[300:380] = 1.3*np.sin(10*np.pi*t)
        
    elif slct == 2:
        x = np.zeros(400)

        # The square
        x[30:60] = 0.2 

        # The sawtooth
        t = np.linspace(0,1,80) 
        x[120:200] = -0.9*sig.sawtooth(0.5*np.pi * t + 0.5*np.pi, 0.5)

        # The sinusoid
        x[300:380] = 1.3*np.sin(10*np.pi*t)
        
    elif slct == 3:
        x = np.zeros(500)

        # The square
        x[30:60] = 0.7

        # The sawtooth
        t = np.linspace(0,1,80) 
        x[120:200] = -0.9*sig.sawtooth(0.5*np.pi * t + 0.5*np.pi, 0.5)

        # The sinusoid
        x[300:380] = 1.3*np.sin(10*np.pi*t)
    
        # Extra square 
        x[400:430] = 1
    else:
        print("Selected invalid waveform. Choose from 1 to 3")
    
    return x


# Plotting functions
def plot_simple_dictionary_and_waveform(dictionary_element_1, dictionary_element_2, dictionary_element_3, x):
    # Plot the elements
    fig = plt.figure(figsize=(5, 3))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.2])

    # Top row (3 plots side-by-side)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1], sharey=ax1)
    ax3 = fig.add_subplot(gs[0, 2], sharey=ax1)

    ax1.plot(dictionary_element_1)
    ax2.plot(dictionary_element_2)
    ax3.plot(dictionary_element_3)

    ax1.set_title("Element 1")
    ax2.set_title("Element 2")
    ax3.set_title("Element 3")
    
    # Bottom row (Single plot spanning all columns)
    ax4 = fig.add_subplot(gs[1, :])
    ax4.plot(x)
    ax4.set_title("The waveform (plotted again for comparison)")

    plt.tight_layout()
    plt.show()


def plot_correlations(corr1, corr2, corr3, x, dict_elem1, dict_elem2, dict_elem3, highlighted_positions):
    fig = plt.figure(figsize=(8, 6))
    gs = fig.add_gridspec(4, 2, width_ratios=[5, 1], height_ratios=[1, 1, 1, 1])
    
    # Plotting the waveform at the bottom spanning the full width of the left column
    ax4 = fig.add_subplot(gs[3, 0])
    ax4.plot(x)
    ax4.set_ylabel('Waveform')
    ax4.grid(True)
    ax4.set_xlabel('Index')
    
    # Plotting the correlations with shared x-axis (corr1, corr2, corr3)
    ax1 = fig.add_subplot(gs[0, 0], sharex=ax4)
    ax1.plot(corr1)
    ax1.tick_params('x', labelbottom=False)
    ax1.set_ylim([-10, 10])
    ax1.set_title('Cross-correlations between waveform and dictionary elements')
    ax1.set_ylabel('Elem. 1')
    ax1.grid(True)
    
    ax2 = fig.add_subplot(gs[1, 0], sharex=ax4)
    ax2.plot(corr2)
    ax2.tick_params('x', labelbottom=False)
    ax2.set_ylim([-10, 10])
    ax2.set_ylabel('Elem. 2')
    ax2.grid(True)

    ax3 = fig.add_subplot(gs[2, 0], sharex=ax4)
    ax3.plot(corr3)
    ax3.tick_params('x', labelbottom=False)
    ax3.set_ylim([-10, 10])
    ax3.set_ylabel('Elem. 3')
    ax3.grid(True)

    # Plotting dictionary elements in the right column (small subplots)
    ax1_dict = fig.add_subplot(gs[0, 1])
    ax1_dict.plot(dict_elem1)
    ax1_dict.set_xticks([])
    
    ax2_dict = fig.add_subplot(gs[1, 1])
    ax2_dict.plot(dict_elem2)
    ax2_dict.set_xticks([])
    
    ax3_dict = fig.add_subplot(gs[2, 1])
    ax3_dict.plot(dict_elem3)
    ax3_dict.set_xticks([])

    for ax in [ax1, ax2, ax3, ax4]:
        for pos in highlighted_positions:
            ax.axvline(x=pos, color='red', linestyle='--', linewidth=1)

    plt.tight_layout()
    plt.show()
    
    
def plot_mp_iters(res0, res1, res2, res3, res4, dic1, dic2, dic3, dic4, dictionary):
    fig = plt.figure(figsize=(8, 6))
    gs = fig.add_gridspec(5, 2, width_ratios=[5, 1], height_ratios=[1, 1, 1, 1, 1])
    
    # Plotting the waveform at the bottom spanning the full width of the left column
    ax5 = fig.add_subplot(gs[4, 0])
    ax5.plot(res4)
    ax5.set_ylabel('Res. 4')
    ax5.set_ylim([-1.5, 1.5])
    ax5.grid(True)
    ax5.set_xlabel('Index')
    
    ax1 = fig.add_subplot(gs[0, 0], sharex=ax5)
    ax1.plot(res0)
    ax1.tick_params('x', labelbottom=False)
    ax1.set_title('Iterations of matching pursuit')
    ax1.set_ylabel('Res. 0')
    ax1.set_ylim([-1.5, 1.5])
    ax1.grid(True)
    
    ax2 = fig.add_subplot(gs[1, 0], sharex=ax5)
    ax2.plot(res1)
    ax2.tick_params('x', labelbottom=False)
    ax2.set_ylabel('Res. 1')
    ax2.set_ylim([-1.5, 1.5])
    ax2.grid(True)
    
    ax3 = fig.add_subplot(gs[2, 0], sharex=ax5)
    ax3.plot(res2)
    ax3.tick_params('x', labelbottom=False)
    ax3.set_ylabel('Res. 2')
    ax3.set_ylim([-1.5, 1.5])
    ax3.grid(True)
    
    ax4 = fig.add_subplot(gs[3, 0], sharex=ax5)
    ax4.plot(res3)
    ax4.tick_params('x', labelbottom=False)
    ax4.set_ylabel('Res. 3')
    ax4.set_ylim([-1.5, 1.5])
    ax4.grid(True)

    ax7 = fig.add_subplot(gs[0, 1])
    ax7.plot(dictionary[dic1].kernel)
    ax7.tick_params('x', labelbottom=False)

    ax8 = fig.add_subplot(gs[1, 1])
    ax8.plot(dictionary[dic2].kernel)
    ax8.tick_params('x', labelbottom=False)
    
    ax9 = fig.add_subplot(gs[2, 1])
    ax9.plot(dictionary[dic3].kernel)
    ax9.tick_params('x', labelbottom=False)
    
    ax10 = fig.add_subplot(gs[3, 1])
    ax10.plot(dictionary[dic4].kernel)
    ax10.tick_params('x', labelbottom=False)
    
    plt.tight_layout()
    plt.show()
     
    
def simple_method_to_encode_an_decode_waveform(waveform, dictionary_element_1, dictionary_element_2, dictionary_element_3, length):
    corr_1 = correlation(waveform, dictionary_element_1)
    corr_2 = correlation(waveform, dictionary_element_2)
    corr_3 = correlation(waveform, dictionary_element_3)
    
    indxMax_1 = np.argmax(abs(corr_1))
    ampMax_1 = corr_1[indxMax_1]

    indxMax_2 = np.argmax(abs(corr_2))
    ampMax_2 = corr_2[indxMax_2]

    indxMax_3 = np.argmax(abs(corr_3))
    ampMax_3 = corr_3[indxMax_3]

    rec_waveform = np.zeros(length)
    
    rec_waveform[indxMax_1:indxMax_1+len(dictionary_element_1)] += ampMax_1*dictionary_element_1
    rec_waveform[indxMax_2:indxMax_2+len(dictionary_element_2)] += ampMax_2*dictionary_element_2
    rec_waveform[indxMax_3:indxMax_3+len(dictionary_element_3)] += ampMax_3*dictionary_element_3
    return rec_waveform
    
