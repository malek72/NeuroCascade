import os
import glob
import random
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch, find_peaks, spectrogram
from scipy.stats import pearsonr
from matplotlib.backends.backend_pdf import PdfPages
from datetime import datetime


def find_preprocessed_files(directory, pattern='cleaned_subject_*_batch_*.pth'):
    """
    Find all preprocessed .pth files matching the given pattern.
    """
    search_pattern = os.path.join(directory, pattern)
    preprocessed_files = glob.glob(search_pattern)
    return preprocessed_files

def select_random_files(file_list, n=5):
    """
    Randomly select n files from a list.
    """
    if len(file_list) < n:
        n = len(file_list)
    selected_files = random.sample(file_list, n)
    return selected_files

def load_data(pth_file_path):
    """
    Load EEG data from a .pth file.
    """
    data = torch.load(pth_file_path)
    return data  # Assuming data is a list of samples

def extract_eeg_data(samples, channel_names):
    """
    Convert EEG data samples to numpy arrays.
    """
    eeg_data_list = []
    for sample in samples:
        eeg_data = sample['eeg_data']
        if isinstance(eeg_data, torch.Tensor):
            eeg_data = eeg_data.numpy()
        elif isinstance(eeg_data, np.ndarray):
            pass  # Already a NumPy array
        else:
            raise TypeError(f"Unsupported type for 'eeg_data': {type(eeg_data)}")
        eeg_data_list.append(eeg_data)
    
    data = np.concatenate(eeg_data_list, axis=1)  # shape: (n_channels, n_times)
    sfreq = 1000  # Sampling frequency
    return data, sfreq

def generate_pdf_report(pdf_filename, report_content):
    """
    Generate a PDF report with the given content.
    """
    with PdfPages(pdf_filename) as pdf:
        for content in report_content:
            # Add explanation text as a figure with text
            if 'text' in content:
                text_fig = plt.figure(figsize=(8.5, 11))
                plt.axis('off')
                plt.text(0.5, 0.5, content['text'], ha='center', va='center', wrap=True, fontsize=12)
                pdf.savefig(text_fig)
                plt.close(text_fig)
            # Add the plot
            if 'figure' in content:
                fig = content['figure']
                pdf.savefig(fig)
                plt.close(fig)
        # Add metadata
        d = pdf.infodict()
        d['Title'] = 'EEG Signal Denoising Evaluation Report'
        d['Author'] = 'Your Name'
        d['Subject'] = 'Evaluation of EEG Signal Denoising'
        d['Keywords'] = 'EEG, Signal Processing, Denoising, Report'
        d['CreationDate'] = datetime.today()
        d['ModDate'] = datetime.today()

def main():
    # Define directory and pattern
    directory = '/Users/maleksibai/ESC499 Thesis EEG'
    preprocessed_pattern = 'cleaned_subject_*_batch_*.pth'
    raw_dataset_file = '/Users/maleksibai/ESC499 Thesis EEG/EEG-ImageNet_1.pth'  # Update with your raw dataset path

    # Find all preprocessed files
    preprocessed_files = find_preprocessed_files(directory, preprocessed_pattern)
    print(f"Found {len(preprocessed_files)} preprocessed files.")

    if len(preprocessed_files) == 0:
        print("No preprocessed files found. Please check the directory and pattern.")
        return

    # Randomly select 5 preprocessed files
    selected_files = select_random_files(preprocessed_files, n=5)
    print("Selected files:")
    for f in selected_files:
        print(f)

    # Load raw dataset
    print("Loading raw dataset...")
    raw_data_loaded = torch.load(raw_dataset_file)
    raw_dataset = raw_data_loaded['dataset']
    print(f"Loaded raw dataset with {len(raw_dataset)} samples.")

    # Updated channel names list with 62 channels
    channel_names = [
        'Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4',
        'O1', 'O2', 'F7', 'F8', 'T3', 'T4', 'T5', 'T6',
        'Fz', 'Cz', 'Pz', 'Oz', 'M1', 'M2', 'FC1', 'FC2',
        'CP1', 'CP2', 'FC5', 'FC6', 'CP5', 'CP6', 'TP9', 'TP10',
        'AF7', 'AF8', 'AF3', 'AF4', 'F1', 'F2', 'F5', 'F6',
        'FT7', 'FT8', 'FC3', 'FC4', 'C1', 'C2', 'C5', 'C6',
        'CP3', 'CP4', 'P1', 'P2', 'P5', 'P6', 'PO3', 'PO4',
        'PO7', 'PO8', 'O9', 'O10',
        'EOG1', 'EOG2'  # Ensure this list matches your actual channel names
    ]

    # Ensure the channel names list has 62 channels
    assert len(channel_names) == 62, "Channel names list must have 62 channels."

    # Prepare report content
    report_content = []

    # Process each selected file
    for preprocessed_file in selected_files:
        print(f"\nProcessing file: {preprocessed_file}")
        # Load preprocessed samples
        preprocessed_samples = load_data(preprocessed_file)
        # Extract subject ID and batch number from filename
        filename = os.path.basename(preprocessed_file)
        parts = filename.replace('.pth', '').split('_')
        try:
            subject_id = int(parts[2])
            batch_num = int(parts[4])
        except (IndexError, ValueError):
            print("Filename format is incorrect. Skipping this file.")
            continue

        # Filter raw dataset for the same subject
        raw_samples_subject = [sample for sample in raw_dataset if sample.get('subject') == subject_id]

        # Determine batch indices
        batch_size = 1000  # Same as used during preprocessing
        start_idx = (batch_num - 1) * batch_size
        end_idx = start_idx + len(preprocessed_samples)

        # Get corresponding raw samples
        raw_samples = raw_samples_subject[start_idx:end_idx]

        if len(raw_samples) != len(preprocessed_samples):
            print("Warning: Number of raw and preprocessed samples do not match.")
            min_len = min(len(raw_samples), len(preprocessed_samples))
            raw_samples = raw_samples[:min_len]
            preprocessed_samples = preprocessed_samples[:min_len]

        # Convert samples to numpy arrays
        try:
            raw_data, sfreq = extract_eeg_data(raw_samples, channel_names)
            preprocessed_data, _ = extract_eeg_data(preprocessed_samples, channel_names)
        except TypeError as te:
            print(f"Data extraction error: {te}")
            continue

        # Prepare per-file report content
        file_report_content = []

        # Add introductory text
        intro_text = f"**Analysis for {filename}**\nSubject ID: {subject_id}, Batch Number: {batch_num}"
        file_report_content.append({'text': intro_text, 'figure': plt.figure(figsize=(8.5, 1))})

        # Perform evaluations
        # 1. Raw vs. Preprocessed Signal Plots
        n_samples = 5
        duration = 5  # seconds
        random_channels = random.sample(channel_names, n_samples)
        times = np.arange(raw_data.shape[1]) / sfreq

        for ch_name in random_channels:
            if ch_name not in channel_names:
                print(f"Channel {ch_name} not found in channel names. Skipping.")
                continue
            ch_idx = channel_names.index(ch_name)
            fig = plt.figure(figsize=(12, 6))
            plt.plot(times[:int(duration*sfreq)], raw_data[ch_idx, :int(duration*sfreq)], label='Raw')
            plt.plot(times[:int(duration*sfreq)], preprocessed_data[ch_idx, :int(duration*sfreq)], label='Preprocessed')
            plt.title(f'Raw vs. Preprocessed Signal - Channel {ch_name}')
            plt.xlabel('Time (s)')
            plt.ylabel('Amplitude (uV)')
            plt.legend()
            plt.tight_layout()
            # Add figure to report
            explanation = f"Figure: Comparison of raw and preprocessed signals for channel {ch_name}."
            file_report_content.append({'figure': fig, 'text': explanation})

        # 2. Power Spectral Density Comparison
        n_channels = raw_data.shape[0]
        psd_raw = []
        psd_pre = []
        freqs = None

        for ch in range(n_channels):
            f_raw, p_raw = welch(raw_data[ch], fs=sfreq, nperseg=min(2*sfreq, raw_data.shape[1]))
            f_pre, p_pre = welch(preprocessed_data[ch], fs=sfreq, nperseg=min(2*sfreq, preprocessed_data.shape[1]))
            if freqs is None:
                freqs = f_raw
            psd_raw.append(p_raw)
            psd_pre.append(p_pre)

        psd_raw_mean = np.mean(psd_raw, axis=0)
        psd_pre_mean = np.mean(psd_pre, axis=0)

        # Select frequency range
        fmin, fmax = 0.5, 80
        idx_min = np.searchsorted(freqs, fmin)
        idx_max = np.searchsorted(freqs, fmax)
        freqs_plot = freqs[idx_min:idx_max]
        psd_raw_mean_plot = psd_raw_mean[idx_min:idx_max]
        psd_pre_mean_plot = psd_pre_mean[idx_min:idx_max]

        # Plot PSD comparison
        fig = plt.figure(figsize=(10, 6))
        plt.semilogy(freqs_plot, psd_raw_mean_plot, label='Raw')
        plt.semilogy(freqs_plot, psd_pre_mean_plot, label='Preprocessed')
        plt.title('Power Spectral Density Comparison')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('PSD (V^2/Hz)')
        plt.legend()
        plt.tight_layout()
        # Add figure to report
        explanation = "Figure: Comparison of the average Power Spectral Density (PSD) before and after preprocessing."
        file_report_content.append({'figure': fig, 'text': explanation})

        # 3. Signal-to-Noise Ratio Improvement
        noise = raw_data - preprocessed_data
        power_signal = np.mean(preprocessed_data ** 2, axis=1)
        power_noise = np.mean(noise ** 2, axis=1)
        snr = 10 * np.log10(power_signal / power_noise)

        # Plot SNR improvement
        fig = plt.figure(figsize=(12, 6))
        plt.bar(range(len(snr)), snr)
        plt.xticks(range(len(snr)), channel_names, rotation=90)
        plt.title('SNR Improvement Across Channels')
        plt.ylabel('SNR (dB)')
        plt.tight_layout()
        # Add figure to report
        explanation = "Figure: Signal-to-Noise Ratio (SNR) improvement across all channels."
        file_report_content.append({'figure': fig, 'text': explanation})

        # 4. Detection and Removal of Common Artifacts
        # Eye blink detection
        frontal_channels = ['Fp1', 'Fp2', 'AF7', 'AF8', 'AF3', 'AF4']
        frontal_indices = [channel_names.index(ch) for ch in frontal_channels if ch in channel_names]
        if not frontal_indices:
            print("No frontal channels found for eye blink detection.")
        else:
            raw_frontal = raw_data[frontal_indices]
            pre_frontal = preprocessed_data[frontal_indices]
            raw_frontal_sum = np.sum(raw_frontal, axis=0)
            pre_frontal_sum = np.sum(pre_frontal, axis=0)
            times = np.arange(raw_data.shape[1]) / sfreq

            # Find peaks (blinks)
            peaks_raw, _ = find_peaks(raw_frontal_sum, height=np.std(raw_frontal_sum)*5)
            peaks_pre, _ = find_peaks(pre_frontal_sum, height=np.std(pre_frontal_sum)*5)

            # Plot blink detection - Raw
            fig = plt.figure(figsize=(12, 6))
            plt.plot(times, raw_frontal_sum, label='Raw Frontal Sum')
            plt.plot(times[peaks_raw], raw_frontal_sum[peaks_raw], 'rx', label='Detected Blinks')
            plt.title('Eye Blink Detection - Raw Data')
            plt.xlabel('Time (s)')
            plt.ylabel('Amplitude (uV)')
            plt.legend()
            plt.tight_layout()
            explanation = "Figure: Eye blink detection in raw data using frontal channels."
            file_report_content.append({'figure': fig, 'text': explanation})

            # Plot blink detection - Preprocessed
            fig = plt.figure(figsize=(12, 6))
            plt.plot(times, pre_frontal_sum, label='Preprocessed Frontal Sum')
            plt.plot(times[peaks_pre], pre_frontal_sum[peaks_pre], 'rx', label='Detected Blinks')
            plt.title('Eye Blink Detection - Preprocessed Data')
            plt.xlabel('Time (s)')
            plt.ylabel('Amplitude (uV)')
            plt.legend()
            plt.tight_layout()
            explanation = "Figure: Eye blink detection in preprocessed data showing reduced artifacts."
            file_report_content.append({'figure': fig, 'text': explanation})

        # Line noise detection (50 Hz)
        # PSD comparison in 45-55 Hz range
        fmin_line, fmax_line = 45, 55
        idx_min_line = np.searchsorted(freqs, fmin_line)
        idx_max_line = np.searchsorted(freqs, fmax_line)
        freqs_line = freqs[idx_min_line:idx_max_line]
        psd_raw_line = psd_raw_mean[idx_min_line:idx_max_line]
        psd_pre_line = psd_pre_mean[idx_min_line:idx_max_line]

        fig = plt.figure(figsize=(10, 6))
        plt.semilogy(freqs_line, psd_raw_line, label='Raw')
        plt.semilogy(freqs_line, psd_pre_line, label='Preprocessed')
        plt.title('Line Noise (50 Hz) Comparison')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('PSD (V^2/Hz)')
        plt.legend()
        plt.tight_layout()
        explanation = "Figure: Comparison of PSD around 50 Hz to assess line noise removal."
        file_report_content.append({'figure': fig, 'text': explanation})

        # 5. Spectrograms
        ch_name = 'Cz'
        if ch_name not in channel_names:
            print(f"Channel {ch_name} not found in channel names. Skipping spectrograms.")
        else:
            ch_idx = channel_names.index(ch_name)
            duration = 60  # seconds
            end_idx = int(duration * sfreq)
            if end_idx > raw_data.shape[1]:
                end_idx = raw_data.shape[1]
            f_raw, t_raw, Sxx_raw = spectrogram(raw_data[ch_idx, :end_idx], fs=sfreq)
            f_pre, t_pre, Sxx_pre = spectrogram(preprocessed_data[ch_idx, :end_idx], fs=sfreq)

            # Plot raw spectrogram
            fig = plt.figure(figsize=(10, 6))
            plt.pcolormesh(t_raw, f_raw, 10 * np.log10(Sxx_raw), shading='gouraud')
            plt.title(f'Spectrogram - Raw Data ({ch_name})')
            plt.ylabel('Frequency [Hz]')
            plt.xlabel('Time [sec]')
            plt.colorbar(label='Power/Frequency (dB/Hz)')
            plt.tight_layout()
            explanation = f"Figure: Spectrogram of raw data for channel {ch_name}."
            file_report_content.append({'figure': fig, 'text': explanation})

            # Plot preprocessed spectrogram
            fig = plt.figure(figsize=(10, 6))
            plt.pcolormesh(t_pre, f_pre, 10 * np.log10(Sxx_pre), shading='gouraud')
            plt.title(f'Spectrogram - Preprocessed Data ({ch_name})')
            plt.ylabel('Frequency [Hz]')
            plt.xlabel('Time [sec]')
            plt.colorbar(label='Power/Frequency (dB/Hz)')
            plt.tight_layout()
            explanation = f"Figure: Spectrogram of preprocessed data for channel {ch_name}."
            file_report_content.append({'figure': fig, 'text': explanation})

        # 6. Baseline Drift Comparison
        duration = 60  # seconds
        end_idx = int(duration * sfreq)
        if end_idx > raw_data.shape[1]:
            end_idx = raw_data.shape[1]
        mean_raw = np.mean(raw_data[:, :end_idx], axis=1)
        mean_pre = np.mean(preprocessed_data[:, :end_idx], axis=1)

        fig = plt.figure(figsize=(12, 6))
        plt.bar(range(len(mean_raw)), mean_raw, alpha=0.7, label='Raw')
        plt.bar(range(len(mean_pre)), mean_pre, alpha=0.7, label='Preprocessed')
        plt.xticks(range(len(mean_raw)), channel_names, rotation=90)
        plt.title('Baseline Drift Comparison Across Channels')
        plt.xlabel('Channels')
        plt.ylabel('Mean Amplitude (uV)')
        plt.legend()
        plt.tight_layout()
        explanation = "Figure: Comparison of baseline drift before and after preprocessing."
        file_report_content.append({'figure': fig, 'text': explanation})

        # 7. Statistical Comparisons
        n_channels = raw_data.shape[0]
        variance_raw = np.var(raw_data, axis=1)
        variance_pre = np.var(preprocessed_data, axis=1)
        correlations = np.array([pearsonr(raw_data[ch], preprocessed_data[ch])[0] for ch in range(n_channels)])

        # Plot variance comparison
        fig = plt.figure(figsize=(12, 6))
        indices = np.arange(n_channels)
        width = 0.35
        plt.bar(indices - width/2, variance_raw, width, label='Raw')
        plt.bar(indices + width/2, variance_pre, width, label='Preprocessed')
        plt.xticks(indices, channel_names, rotation=90)
        plt.title('Variance Comparison Across Channels')
        plt.ylabel('Variance')
        plt.legend()
        plt.tight_layout()
        explanation = "Figure: Variance of signals across channels before and after preprocessing."
        file_report_content.append({'figure': fig, 'text': explanation})

        # Plot correlation
        fig = plt.figure(figsize=(12, 6))
        plt.bar(indices, correlations, color='lightgreen')
        plt.xticks(indices, channel_names, rotation=90)
        plt.title('Correlation Between Raw and Preprocessed Signals')
        plt.ylabel('Correlation Coefficient')
        plt.ylim([-1, 1])
        plt.tight_layout()
        explanation = "Figure: Correlation between raw and preprocessed signals across channels."
        file_report_content.append({'figure': fig, 'text': explanation})

        # Append file report content to main report
        report_content.extend(file_report_content)

    # Generate PDF report
    pdf_filename = 'EEG_Denoising_Evaluation_Report.pdf'
    generate_pdf_report(pdf_filename, report_content)
    print(f"PDF report generated: {pdf_filename}")

if __name__ == "__main__":
    main()