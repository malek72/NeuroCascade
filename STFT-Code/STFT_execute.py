import numpy as np
import torch
import math
from mne import create_info
from mne.io import RawArray
from collections import defaultdict
import sys  # Ensure sys is imported
import os

# Import the STFT functions from STFT.py
# Assuming STFT.py is in the same directory or adjust the import path accordingly
# Add the directory containing _stft.py to the Python path
stft_path = "/Users/maleksibai/ESC499 Thesis EEG/mne-python/mne/time_frequency"
sys.path.append(stft_path)

# Import the STFT functions from the specified file
from _stft import stft, istft, stftfreq

def decompose_eeg_into_bands(denoised_data, channel_names, fs=1000, wsize=256, tstep=None):
    """
    Decompose the denoised EEG data into standard frequency bands using STFT.

    Parameters
    ----------
    denoised_data : np.ndarray
        Denoised EEG data of shape (n_channels, n_times).
    channel_names : list
        List of channel names.
    fs : int
        Sampling frequency (default is 1000 Hz).
    wsize : int
        Window size for STFT (default is 256 samples).
    tstep : int or None
        Time step for STFT (default is wsize // 2).

    Returns
    -------
    band_signals : dict
        Dictionary containing the reconstructed signals for each frequency band.
        Keys are band names, and values are NumPy arrays of shape (n_channels, n_times).
    """
    if tstep is None:
        tstep = wsize // 2

    n_channels, n_times = denoised_data.shape

    # Perform STFT on each channel
    X = np.zeros((n_channels, wsize // 2 + 1, int(np.ceil(n_times / tstep))), dtype=np.complex128)
    for ch in range(n_channels):
        X_ch = stft(denoised_data[ch, :], wsize=wsize, tstep=tstep)
        X[ch, :, :] = X_ch[0]

    # Compute the frequency bins
    freqs = stftfreq(wsize, sfreq=fs)

    # Define frequency bands
    frequency_bands = {
        'delta': (0.5, 4),
        'theta': (4, 8),
        'alpha': (8, 12),
        'beta': (12, 30),
        'low_gamma': (30, 50),
        'mid_gamma': (50, 80),
        'high_gamma': (80, 150)
    }

    band_signals = {}

    for band_name, (fmin, fmax) in frequency_bands.items():
        # Identify frequency indices for the band
        band_indices = np.where((freqs >= fmin) & (freqs <= fmax))[0]
        if len(band_indices) == 0:
            print(f"No frequency bins found for band {band_name}.")
            continue

        # Initialize the filtered STFT coefficients
        X_band = np.zeros_like(X)

        # Retain only the coefficients within the frequency band
        X_band[:, band_indices, :] = X[:, band_indices, :]

        # Reconstruct the time-domain signal for the band using inverse STFT
        band_signal = np.zeros((n_channels, n_times))
        for ch in range(n_channels):
            # Reconstruct the signal for each channel
            x_rec = istft(X_band[ch, :, :], tstep=tstep, Tx=n_times)
            band_signal[ch, :] = x_rec[:n_times]

        band_signals[band_name] = band_signal

    return band_signals

def main():
    # Path to the cleaned data files
    # Update this path to where your cleaned data files are saved
    cleaned_data_files = [
        'cleaned_subject_0_batch_1.pth',
        'cleaned_subject_0_batch_2.pth',
        'cleaned_subject_0_batch_3.pth',
        'cleaned_subject_0_batch_4.pth',
        'cleaned_subject_1_batch_1.pth',
        'cleaned_subject_1_batch_2.pth',
        'cleaned_subject_1_batch_3.pth',
        'cleaned_subject_1_batch_4.pth',
        'cleaned_subject_2_batch_1.pth',
        'cleaned_subject_2_batch_2.pth',
        'cleaned_subject_2_batch_3.pth',
        'cleaned_subject_2_batch_4.pth',
        'cleaned_subject_3_batch_1.pth',
        'cleaned_subject_3_batch_2.pth',
        'cleaned_subject_3_batch_3.pth',
        'cleaned_subject_3_batch_4.pth',
        'cleaned_subject_4_batch_1.pth',
        'cleaned_subject_4_batch_2.pth',
        'cleaned_subject_4_batch_3.pth',
        'cleaned_subject_4_batch_4.pth',
        'cleaned_subject_5_batch_1.pth',
        'cleaned_subject_5_batch_2.pth',
        'cleaned_subject_5_batch_3.pth',
        'cleaned_subject_5_batch_4.pth',
        'cleaned_subject_6_batch_1.pth',
        'cleaned_subject_6_batch_2.pth',
        'cleaned_subject_6_batch_3.pth',
        'cleaned_subject_6_batch_4.pth',
        'cleaned_subject_7_batch_1.pth',
        'cleaned_subject_7_batch_2.pth',
        'cleaned_subject_7_batch_3.pth',
        'cleaned_subject_7_batch_4.pth'
    ]


    # Channel names (ensure this list matches your actual channel names)
    channel_names = [
        'Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4',
        'O1', 'O2', 'F7', 'F8', 'T3', 'T4', 'T5', 'T6',
        'Fz', 'Cz', 'Pz', 'Oz', 'M1', 'M2', 'FC1', 'FC2',
        'CP1', 'CP2', 'FC5', 'FC6', 'CP5', 'CP6', 'TP9', 'TP10',
        'AF7', 'AF8', 'AF3', 'AF4', 'F1', 'F2', 'F5', 'F6',
        'FT7', 'FT8', 'FC3', 'FC4', 'C1', 'C2', 'C5', 'C6',
        'CP3', 'CP4', 'P1', 'P2', 'P5', 'P6', 'PO3', 'PO4',
        'PO7', 'PO8', 'O9', 'O10',
        'EOG1', 'EOG2'
    ]

    fs = 1000  # Sampling frequency

    for cleaned_file in cleaned_data_files:
        print(f"Processing file: {cleaned_file}")
        # Load the cleaned data
        cleaned_batch = torch.load(cleaned_file)
        eeg_data_list = [sample['eeg_data'].numpy() for sample in cleaned_batch]
        data = np.concatenate(eeg_data_list, axis=1)  # data shape: (62, 500 * N_samples)

        # Decompose the EEG data into frequency bands
        band_signals = decompose_eeg_into_bands(data, channel_names, fs=fs, wsize=256)

        # Save or process the band signals as needed
        # For example, save each band signal to a file
        for band_name, signal in band_signals.items():
            output_file = f"{cleaned_file.replace('.pth', '')}_{band_name}.npy"
            np.save(output_file, signal)
            print(f"Saved {band_name} band signal to {output_file}")

        print(f"Completed processing for file: {cleaned_file}")

if __name__ == "__main__":
    main()
