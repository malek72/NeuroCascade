import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch
import mne
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s:%(message)s')

# =========================
# Configuration Parameters
# =========================

# Paths to your original and denoised datasets
ORIGINAL_PTH = '/Users/maleksibai/ESC499 Thesis EEG/EEG-ImageNet_2.pth'
DENOISED_PTH = '/Users/maleksibai/ESC499 Thesis EEG/EEG-ImageNet_2_denoised.pth'

# EEG Channel Information
EEG_CHANNELS = [
    'Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4',
    'O1', 'O2', 'F7', 'F8', 'T3', 'T4', 'T5', 'T6',
    'P7', 'P8', 'Fz', 'Cz', 'Pz', 'Oz', 'FC1', 'FC2',
    'CP1', 'CP2', 'FC5', 'FC6', 'CP5', 'CP6', 'TP9',
    'TP10', 'POz', 'PO3', 'PO4', 'F1', 'F2', 'C1',
    'C2', 'P1', 'P2', 'O9', 'O10', 'Fpz', 'CPz',
    'POz', 'F5', 'F6', 'C5', 'C6', 'P5', 'P6',
    'O1', 'O2', 'Fp1', 'Fp2', 'F3', 'F4', 'C3',
    'C4', 'P3', 'P4', 'O1', 'O2'
]

# Sampling Frequency (Assumed based on filter settings; adjust if different)
FS = 200  # in Hz

# Number of samples to visualize
NUM_SAMPLES = 5

# Channels to visualize (indices based on EEG_CHANNELS list)
CHANNEL_INDICES = [0, 1, 2, 3, 4]  # Fp1, Fp2, F3, F4, C3

# =========================
# Function Definitions
# =========================

def load_dataset(pth_path):
    """
    Load the EEG dataset from a .pth file.
    
    Parameters:
    - pth_path: Path to the .pth file.
    
    Returns:
    - A dictionary containing 'dataset', 'labels', and 'images'.
    """
    if not os.path.exists(pth_path):
        logging.error(f"File not found: {pth_path}")
        return None
    data = torch.load(pth_path)
    logging.info(f"Loaded dataset from {pth_path}")
    return data

def visualize_eeg_signals(original_data, denoised_data, sample_index=0, channel_indices=[0]):
    """
    Plot original and denoised EEG signals for specified samples and channels.
    
    Parameters:
    - original_data: Numpy array of shape (n_channels, n_times).
    - denoised_data: Numpy array of shape (n_channels, n_times).
    - sample_index: Index of the sample to visualize.
    - channel_indices: List of channel indices to plot.
    """
    times = np.linspace(0, original_data.shape[1]/FS, original_data.shape[1])
    
    num_channels = len(channel_indices)
    plt.figure(figsize=(15, 3 * num_channels))
    
    for i, ch_idx in enumerate(channel_indices):
        plt.subplot(num_channels, 1, i + 1)
        plt.plot(times, original_data[ch_idx], label='Original', alpha=0.5)
        plt.plot(times, denoised_data[ch_idx], label='Denoised', alpha=0.8)
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.title(f'Sample {sample_index}, Channel {EEG_CHANNELS[ch_idx]}')
        plt.legend()
        plt.tight_layout()
    
    plt.show()
    logging.info(f"Visualized EEG signals for sample {sample_index}.")

def compare_psd(original_data, denoised_data, sample_index=0, channel_indices=[0]):
    """
    Compare the Power Spectral Density (PSD) of original and denoised EEG signals.
    
    Parameters:
    - original_data: Numpy array of shape (n_channels, n_times).
    - denoised_data: Numpy array of shape (n_channels, n_times).
    - sample_index: Index of the sample to analyze.
    - channel_indices: List of channel indices to analyze.
    """
    plt.figure(figsize=(15, 3 * len(channel_indices)))
    
    for i, ch_idx in enumerate(channel_indices):
        freqs_orig, psd_orig = welch(original_data[ch_idx], fs=FS, nperseg=256)
        freqs_clean, psd_clean = welch(denoised_data[ch_idx], fs=FS, nperseg=256)
        
        plt.subplot(len(channel_indices), 1, i + 1)
        plt.semilogy(freqs_orig, psd_orig, label='Original PSD')
        plt.semilogy(freqs_clean, psd_clean, label='Denoised PSD')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('PSD (V²/Hz)')
        plt.title(f'Sample {sample_index}, Channel {EEG_CHANNELS[ch_idx]}: PSD Comparison')
        plt.legend()
        plt.tight_layout()
    
    plt.show()
    logging.info(f"Compared PSD for sample {sample_index}.")

def compute_snr(original, denoised):
    """
    Compute the Signal-to-Noise Ratio (SNR) between original and denoised signals.
    
    Parameters:
    - original: Numpy array of the original signal.
    - denoised: Numpy array of the denoised signal.
    
    Returns:
    - SNR value in decibels (dB).
    """
    signal_power = np.mean(original ** 2)
    noise_power = np.mean((original - denoised) ** 2)
    snr = 10 * np.log10(signal_power / noise_power) if noise_power != 0 else np.inf
    return snr

def compute_variance_reduction(original, denoised):
    """
    Compute the percentage of variance reduction from original to denoised signals.
    
    Parameters:
    - original: Numpy array of the original signal.
    - denoised: Numpy array of the denoised signal.
    
    Returns:
    - Variance reduction percentage.
    """
    var_original = np.var(original)
    var_denoised = np.var(denoised)
    reduction = ((var_original - var_denoised) / var_original) * 100 if var_original != 0 else 0
    return reduction

def plot_statistical_metrics(snr_list, var_reduction_list):
    """
    Plot histograms of SNR and Variance Reduction across samples and channels.
    
    Parameters:
    - snr_list: List of SNR values.
    - var_reduction_list: List of Variance Reduction percentages.
    """
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.hist(snr_list, bins=30, color='skyblue', edgecolor='black')
    plt.xlabel('Signal-to-Noise Ratio (dB)')
    plt.ylabel('Frequency')
    plt.title('Histogram of SNR Across Samples and Channels')
    
    plt.subplot(1, 2, 2)
    plt.hist(var_reduction_list, bins=30, color='salmon', edgecolor='black')
    plt.xlabel('Variance Reduction (%)')
    plt.ylabel('Frequency')
    plt.title('Histogram of Variance Reduction Across Samples and Channels')
    
    plt.tight_layout()
    plt.show()
    logging.info("Plotted statistical metrics.")

# =========================
# Main Execution
# =========================

def main():
    # Load datasets
    original_dataset = load_dataset(ORIGINAL_PTH)
    denoised_dataset = load_dataset(DENOISED_PTH)
    
    if original_dataset is None or denoised_dataset is None:
        logging.error("Failed to load one or both datasets. Exiting.")
        return
    
    num_samples_original = len(original_dataset['dataset'])
    num_samples_denoised = len(denoised_dataset['dataset'])
    
    if num_samples_original != num_samples_denoised:
        logging.warning("Original and denoised datasets have different number of samples.")
    
    # Select samples to visualize
    selected_samples = np.linspace(0, num_samples_original - 1, NUM_SAMPLES, dtype=int)
    
    # Lists to store statistical metrics
    snr_values = []
    var_reduction_values = []
    
    for sample_idx in selected_samples:
        # Extract EEG data
        original_eeg = original_dataset['dataset'][sample_idx]['eeg_data'].numpy()  # Shape: (n_channels, n_times)
        denoised_eeg = denoised_dataset['dataset'][sample_idx]['eeg_data'].numpy()
        
        # Visualize EEG signals
        visualize_eeg_signals(original_eeg, denoised_eeg, sample_index=sample_idx, channel_indices=CHANNEL_INDICES)
        
        # Compare PSD
        compare_psd(original_eeg, denoised_eeg, sample_index=sample_idx, channel_indices=CHANNEL_INDICES)
        
        # Compute statistical metrics for each channel
        for ch_idx in CHANNEL_INDICES:
            snr = compute_snr(original_eeg[ch_idx], denoised_eeg[ch_idx])
            var_reduction = compute_variance_reduction(original_eeg[ch_idx], denoised_eeg[ch_idx])
            snr_values.append(snr)
            var_reduction_values.append(var_reduction)
            logging.info(f"Sample {sample_idx}, Channel {EEG_CHANNELS[ch_idx]}: SNR = {snr:.2f} dB, Variance Reduction = {var_reduction:.2f}%")
    
    # Plot statistical metrics
    plot_statistical_metrics(snr_values, var_reduction_values)
    
    logging.info("Visualization and validation complete.")

if __name__ == "__main__":
    main()
