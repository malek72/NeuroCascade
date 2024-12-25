import torch
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import welch
import os

# ----------------------------- Configuration ----------------------------- #

# Paths to the original and denoised datasets
ORIGINAL_DATASET_PATH = '/Users/maleksibai/ESC499 Thesis EEG/EEG-ImageNet_2.pth'
DENOISED_DATASET_PATH = '/Users/maleksibai/ESC499 Thesis EEG/EEG-ImageNet_2_denoised.pth'

# Sampling frequency (Hz)
FS = 1000  # Adjust this value based on your data's actual sampling rate

# Number of channels and samples (if needed)
NUM_CHANNELS = 62
NUM_SAMPLES = 501

# Channels to visualize (0-based indices)
CHANNEL_INDICES = [0, 1, 2, 3, 4]  # Example: first 5 channels

# Samples to visualize (0-based indices)
SAMPLE_INDICES = [0, 1, 2, 3, 4]  # Example: first 5 samples

# Output directory for saving plots
OUTPUT_DIR = '/Users/maleksibai/ESC499 Thesis EEG/EEG_Comparisons'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------- Helper Functions ---------------------------- #

def load_dataset(pth_path):
    """
    Load a dataset from a .pth file.

    Parameters:
    - pth_path (str): Path to the .pth file.

    Returns:
    - dict: Loaded dataset containing 'dataset', 'labels', 'images', etc.
    """
    try:
        data = torch.load(pth_path, map_location='cpu')
        print(f"Loaded dataset from {pth_path}")
        return data
    except Exception as e:
        print(f"Error loading {pth_path}: {e}")
        return None

def visualize_eeg_signals(original_signal, denoised_signal, fs, sample_idx, ch_idx, channel_name=None):
    """
    Plot original and denoised EEG signals for a specific channel and sample.

    Parameters:
    - original_signal (numpy.ndarray): Original EEG signal of shape (n_times,).
    - denoised_signal (numpy.ndarray): Denoised EEG signal of shape (n_times,).
    - fs (int): Sampling frequency in Hz.
    - sample_idx (int): Index of the sample.
    - ch_idx (int): Index of the channel.
    - channel_name (str, optional): Name of the channel.
    """
    n_times = original_signal.shape[0]
    times = np.linspace(0, n_times / fs, n_times)

    plt.figure(figsize=(15, 5))
    plt.plot(times, original_signal, label='Original', alpha=0.6)
    plt.plot(times, denoised_signal, label='Denoised', alpha=0.8)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    title = f'Sample {sample_idx + 1}, Channel {ch_idx + 1}'
    if channel_name:
        title += f' ({channel_name})'
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    # Save the plot
    plot_filename = f'sample_{sample_idx + 1}_channel_{ch_idx + 1}.png'
    plt.savefig(os.path.join(OUTPUT_DIR, plot_filename))
    plt.show()
    print(f"Saved EEG signal comparison plot: {plot_filename}")

def compare_psd(original_signal, denoised_signal, fs, sample_idx, ch_idx, channel_name=None):
    """
    Plot Power Spectral Density (PSD) of original and denoised EEG signals for a specific channel and sample.

    Parameters:
    - original_signal (numpy.ndarray): Original EEG signal of shape (n_times,).
    - denoised_signal (numpy.ndarray): Denoised EEG signal of shape (n_times,).
    - fs (int): Sampling frequency in Hz.
    - sample_idx (int): Index of the sample.
    - ch_idx (int): Index of the channel.
    - channel_name (str, optional): Name of the channel.
    """
    freqs_orig, psd_orig = welch(original_signal, fs=fs, nperseg=256)
    freqs_clean, psd_clean = welch(denoised_signal, fs=fs, nperseg=256)

    plt.figure(figsize=(10, 6))
    plt.semilogy(freqs_orig, psd_orig, label='Original PSD', alpha=0.7)
    plt.semilogy(freqs_clean, psd_clean, label='Denoised PSD', alpha=0.9)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power Spectral Density (V²/Hz)')
    title = f'PSD Comparison - Sample {sample_idx + 1}, Channel {ch_idx + 1}'
    if channel_name:
        title += f' ({channel_name})'
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    # Save the plot
    plot_filename = f'sample_{sample_idx + 1}_channel_{ch_idx + 1}_PSD.png'
    plt.savefig(os.path.join(OUTPUT_DIR, plot_filename))
    plt.show()
    print(f"Saved PSD comparison plot: {plot_filename}")

def get_channel_names(dataset):
    """
    Retrieve channel names from the dataset if available.

    Parameters:
    - dataset (dict): Loaded dataset.

    Returns:
    - list: List of channel names.
    """
    if 'channel_names' in dataset:
        return dataset['channel_names']
    else:
        # If channel names are not available, return generic names
        return [f'Ch{idx + 1}' for idx in range(NUM_CHANNELS)]

# ------------------------------ Main Script ------------------------------- #

def main():
    # Load datasets
    original_data = load_dataset(ORIGINAL_DATASET_PATH)
    denoised_data = load_dataset(DENOISED_DATASET_PATH)

    if original_data is None or denoised_data is None:
        print("Failed to load one or both datasets. Exiting.")
        return

    # Verify that both datasets have the same number of samples
    num_samples_original = len(original_data['dataset'])
    num_samples_denoised = len(denoised_data['dataset'])

    if num_samples_original != num_samples_denoised:
        print(f"Mismatch in number of samples: Original={num_samples_original}, Denoised={num_samples_denoised}")
        return

    # Retrieve channel names if available
    channel_names = get_channel_names(original_data)

    # Iterate over selected samples and channels
    for sample_idx in SAMPLE_INDICES:
        if sample_idx >= num_samples_original:
            print(f"Sample index {sample_idx} is out of range. Skipping.")
            continue

        for ch_idx in CHANNEL_INDICES:
            if ch_idx >= NUM_CHANNELS:
                print(f"Channel index {ch_idx} is out of range. Skipping.")
                continue

            # Retrieve original and denoised signals
            original_signal_tensor = original_data['dataset'][sample_idx]['eeg_data']
            denoised_signal_tensor = denoised_data['dataset'][sample_idx]['eeg_data']

            # Convert tensors to numpy arrays
            original_signal = original_signal_tensor.numpy()[ch_idx]
            denoised_signal = denoised_signal_tensor.numpy()[ch_idx]

            # Optional: Retrieve channel name
            channel_name = channel_names[ch_idx] if channel_names else None

            # Plot EEG signals
            visualize_eeg_signals(original_signal, denoised_signal, FS, sample_idx, ch_idx, channel_name)

            # Plot PSD comparison
            compare_psd(original_signal, denoised_signal, FS, sample_idx, ch_idx, channel_name)

    print(f"All selected samples and channels have been visualized and saved to {OUTPUT_DIR}.")

if __name__ == "__main__":
    main()
