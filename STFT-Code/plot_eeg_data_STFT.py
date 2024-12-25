import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.signal import welch

# Configuration
CLEANED_DATA_FILES = [
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

SAMPLING_FREQUENCY = 1000  # Hz
SELECTED_CHANNELS = ['Cz', 'Pz', 'Fz', 'Oz']  # Channels to visualize
CHANNEL_NAMES = [
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

def plot_data_and_psd(data, channel_names, sampling_frequency, pdf, file_name):
    """
    Plot time-series data and PSD for selected channels and save to PDF.

    Parameters
    ----------
    data : np.ndarray
        EEG data array of shape (n_channels, n_times).
    channel_names : list
        List of channel names.
    sampling_frequency : int
        Sampling frequency in Hz.
    pdf : PdfPages object
        PdfPages object to save plots.
    file_name : str
        Name of the data file being processed.
    """
    time = np.arange(data.shape[1]) / sampling_frequency
    n_channels = data.shape[0]
    selected_indices = [channel_names.index(ch) for ch in SELECTED_CHANNELS if ch in channel_names]

    # Plot time series
    fig, axs = plt.subplots(len(selected_indices), 1, figsize=(10, 8), sharex=True)
    fig.suptitle(f"Time Series Data from {file_name}")
    for idx, ch_idx in enumerate(selected_indices):
        axs[idx].plot(time[:10000], data[ch_idx, :10000])
        axs[idx].set_ylabel(f"{channel_names[ch_idx]}")
        axs[idx].set_xlim([0, time[9999]])
    axs[-1].set_xlabel("Time (s)")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    pdf.savefig(fig)
    plt.close(fig)

    # Plot PSD
    fig, axs = plt.subplots(len(selected_indices), 1, figsize=(10, 8), sharex=True)
    fig.suptitle(f"PSD of Data from {file_name}")
    for idx, ch_idx in enumerate(selected_indices):
        f, Pxx = welch(data[ch_idx, :], fs=sampling_frequency, nperseg=2048)
        axs[idx].semilogy(f, Pxx)
        axs[idx].set_ylabel(f"{channel_names[ch_idx]}")
        axs[idx].set_xlim([0, 100])  # Limit frequency axis for clarity
    axs[-1].set_xlabel("Frequency (Hz)")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    pdf.savefig(fig)
    plt.close(fig)

def main():
    """
    Main function to generate the PDF report with plots.
    """
    with PdfPages('EEG_Data_Report.pdf') as pdf:
        for file_index, file_path in enumerate(CLEANED_DATA_FILES):
            print(f"Processing file {file_index + 1}/{len(CLEANED_DATA_FILES)}: {file_path}")
            if not os.path.isfile(file_path):
                print(f"File not found: {file_path}. Skipping.")
                continue
            # Load the cleaned data
            cleaned_batch = torch.load(file_path)
            eeg_data_list = [sample['eeg_data'].numpy() for sample in cleaned_batch]
            data = np.concatenate(eeg_data_list, axis=1)  # shape: (n_channels, n_times)
            if data is None or data.size == 0:
                print(f"No data loaded from {file_path}. Skipping.")
                continue
            # Plot data and PSD
            plot_data_and_psd(data, CHANNEL_NAMES, SAMPLING_FREQUENCY, pdf, file_path)
    print("Report generated: EEG_Data_Report.pdf")

if __name__ == "__main__":
    main()
