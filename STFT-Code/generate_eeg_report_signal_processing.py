import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.signal import spectrogram, welch
import mne
from reportlab.lib.pagesizes import letter
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak
)
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from datetime import datetime
import traceback
from math import ceil
import random

# Configuration

# Original list of cleaned data files
ALL_CLEANED_DATA_FILES = [
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

# Select 5 random files from the list
CLEANED_DATA_FILES = random.sample(ALL_CLEANED_DATA_FILES, 5)

OUTPUT_PDF = 'EEG_STFT_Report.pdf'  # Output PDF report filename

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

# Select 5 random channels from CHANNEL_NAMES (excluding EOG channels)
NON_EOG_CHANNELS = [ch for ch in CHANNEL_NAMES if ch not in ['EOG1', 'EOG2']]
SELECTED_CHANNELS = random.sample(NON_EOG_CHANNELS, 5)

SAMPLING_FREQUENCY = 1000  # Hz
STFT_WINDOW = 'hann'
STFT_NPERSEG = 256
STFT_OVERLAP = 128

# Create a directory for temporary images
TEMP_IMG_DIR = './temp_images'
os.makedirs(TEMP_IMG_DIR, exist_ok=True)

# Suppress FutureWarning from torch.load
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, message="You are using `torch.load` with `weights_only=False`")

# STFT functions
from scipy.fft import irfft, rfft, rfftfreq

def stft(x, wsize, tstep=None):
    """Short-Time Fourier Transform using a sine window."""
    if not np.isrealobj(x):
        raise ValueError("x is not a real valued array")

    if x.ndim == 1:
        x = x[None, :]

    n_signals, T = x.shape
    wsize = int(wsize)

    # Errors and warnings
    if wsize % 4:
        raise ValueError("The window length must be a multiple of 4.")

    if tstep is None:
        tstep = wsize / 2

    tstep = int(tstep)

    if (wsize % tstep) or (tstep % 2):
        raise ValueError(
            "The step size must be a multiple of 2 and a "
            "divider of the window length."
        )

    if tstep > wsize / 2:
        raise ValueError("The step size must be smaller than half the window length.")

    n_step = int(ceil(T / float(tstep)))
    n_freq = wsize // 2 + 1

    X = np.zeros((n_signals, n_freq, n_step), dtype=np.complex128)

    if n_signals == 0:
        return X

    # Defining sine window
    win = np.sin(np.arange(0.5, wsize + 0.5) / wsize * np.pi)
    win2 = win ** 2

    swin = np.zeros((n_step - 1) * tstep + wsize)
    for t in range(n_step):
        swin[t * tstep: t * tstep + wsize] += win2
    swin = np.sqrt(wsize * swin)

    # Zero-padding and Pre-processing for edges
    xp = np.zeros((n_signals, wsize + (n_step - 1) * tstep), dtype=x.dtype)
    xp[:, (wsize - tstep) // 2: (wsize - tstep) // 2 + T] = x
    x = xp

    for t in range(n_step):
        # Framing
        wwin = win / swin[t * tstep: t * tstep + wsize]
        frame = x[:, t * tstep: t * tstep + wsize] * wwin[None, :]
        # FFT
        X[:, :, t] = rfft(frame)

    return X

def istft(X, tstep=None, Tx=None):
    """Inverse Short-Time Fourier Transform using a sine window."""
    # Errors and warnings
    X = np.asarray(X)
    if X.ndim < 2:
        raise ValueError(f"X must have ndim >= 2, got {X.ndim}")
    n_win, n_step = X.shape[-2:]
    signal_shape = X.shape[:-2]
    if n_win % 2 == 0:
        raise ValueError("The number of rows of the STFT matrix must be odd.")

    wsize = 2 * (n_win - 1)
    if tstep is None:
        tstep = wsize / 2

    if wsize % tstep:
        raise ValueError(
            "The step size must be a divider of two times the "
            "number of rows of the STFT matrix minus two."
        )

    if wsize % 2:
        raise ValueError("The step size must be a multiple of 2.")

    if tstep > wsize / 2:
        raise ValueError(
            "The step size must be smaller than the number of "
            "rows of the STFT matrix minus one."
        )

    if Tx is None:
        Tx = n_step * tstep

    T = n_step * tstep

    x = np.zeros(signal_shape + (T + wsize - tstep,), dtype=np.float64)

    if np.prod(signal_shape) == 0:
        return x[..., :Tx]

    # Defining sine window
    win = np.sin(np.arange(0.5, wsize + 0.5) / wsize * np.pi)

    # Pre-processing for edges
    swin = np.zeros(T + wsize - tstep, dtype=np.float64)
    for t in range(n_step):
        swin[t * tstep: t * tstep + wsize] += win ** 2
    swin = np.sqrt(swin / wsize)

    for t in range(n_step):
        # IFFT
        frame = irfft(X[..., t], wsize)
        # Overlap-add
        frame *= win / swin[t * tstep: t * tstep + wsize]
        x[..., t * tstep: t * tstep + wsize] += frame

    # Truncation
    x = x[..., (wsize - tstep) // 2: (wsize - tstep) // 2 + T + 1]
    x = x[..., :Tx].copy()
    return x

def stftfreq(wsize, sfreq=None):
    """Compute frequencies of STFT transformation."""
    freqs = rfftfreq(wsize)
    if sfreq is not None:
        freqs *= float(sfreq)
    return freqs

def decompose_eeg_into_bands(denoised_data, fs=1000, wsize=256, tstep=None):
    """
    Decompose the denoised EEG data into standard frequency bands using STFT.

    Parameters
    ----------
    denoised_data : np.ndarray
        Denoised EEG data of shape (n_channels, n_times).
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

def generate_report():
    """
    Main function to generate the PDF report.
    Processes each cleaned data file individually to reduce memory usage.
    """
    try:
        # Initialize report elements
        report_elements = []
        styles = getSampleStyleSheet()
        report_elements.append(Paragraph("EEG STFT Processing Report", styles['Title']))
        report_elements.append(Spacer(1, 12))
        report_elements.append(Paragraph(f"Author: Your Name", styles['Normal']))
        report_elements.append(Paragraph(f"Date: {datetime.now().strftime('%Y-%m-%d')}", styles['Normal']))
        report_elements.append(Spacer(1, 24))

        # Introduction
        report_elements.append(Paragraph("Introduction", styles['Heading1']))
        intro_text = """
        This report presents the results of Short-Time Fourier Transform (STFT) processing on EEG signals. The objective is to decompose the EEG data into standard frequency bands and analyze the effectiveness of the STFT process.
        """
        report_elements.append(Paragraph(intro_text, styles['Normal']))
        report_elements.append(Spacer(1, 12))

        # Methodology
        report_elements.append(Paragraph("Methodology", styles['Heading1']))
        method_text = f"""
        The EEG data was preprocessed and decomposed into the following frequency bands: delta (0.5–4 Hz), theta (4–8 Hz), alpha (8–12 Hz), beta (12–30 Hz), low gamma (30–50 Hz), mid gamma (50–80 Hz), and high gamma (80–150 Hz). The STFT parameters used were window type: {STFT_WINDOW}, window length: {STFT_NPERSEG}, and overlap: {STFT_OVERLAP}. The sampling frequency was {SAMPLING_FREQUENCY} Hz.
        """
        report_elements.append(Paragraph(method_text, styles['Normal']))
        report_elements.append(Spacer(1, 12))

        all_temp_images = []  # Keep track of all generated image paths

        # Process each cleaned data file individually
        for file_index, file_path in enumerate(CLEANED_DATA_FILES):
            print(f"Processing file {file_index + 1}/{len(CLEANED_DATA_FILES)}: {file_path}")
            # Load the cleaned data
            if not os.path.isfile(file_path):
                print(f"File not found: {file_path}. Skipping.")
                continue
            cleaned_batch = torch.load(file_path)
            eeg_data_list = [sample['eeg_data'].numpy() for sample in cleaned_batch]
            data = np.concatenate(eeg_data_list, axis=1)  # shape: (n_channels, n_times)
            if data is None or data.size == 0:
                print(f"No data loaded from {file_path}. Skipping.")
                continue

            # Decompose the EEG data into frequency bands
            print("Decomposing EEG data into frequency bands...")
            band_data = decompose_eeg_into_bands(data, fs=SAMPLING_FREQUENCY, wsize=256)

            # Generate plots and analyses for this file
            print("Generating spectrograms...")
            spectrogram_images = generate_spectrograms(band_data, SELECTED_CHANNELS, TEMP_IMG_DIR)
            print("Generating PSD plots...")
            psd_images = generate_psd_plots(band_data, TEMP_IMG_DIR)
            print("Plotting time-domain waveforms...")
            waveform_images = plot_time_domain_waveforms(band_data, SELECTED_CHANNELS, TEMP_IMG_DIR)
            print("Plotting band power evolution...")
            band_power_images = plot_band_power_evolution(band_data, TEMP_IMG_DIR)

            # Collect all image paths
            file_temp_images = spectrogram_images + psd_images + waveform_images + band_power_images
            all_temp_images.extend(file_temp_images)

            # Results
            report_elements.append(Paragraph(f"Results for {file_path}", styles['Heading1']))

            # Include sampling frequency results
            report_elements.append(Paragraph(f"Sampling Frequency: {SAMPLING_FREQUENCY} Hz", styles['Normal']))
            report_elements.append(Spacer(1, 12))

            # Include plots and analyses
            for img_list, section_title in zip(
                [spectrogram_images, psd_images, waveform_images, band_power_images],
                ["Spectrograms", "Power Spectral Density (PSD) Plots", "Filtered Band Signals", "Time Evolution of Band Power"]
            ):
                report_elements.append(Paragraph(section_title, styles['Heading2']))
                for img_path in img_list:
                    report_elements.append(Image(img_path, width=500, height=250))
                    report_elements.append(Spacer(1, 12))

            # Optionally, add a page break after each file's results
            report_elements.append(PageBreak())

        # Conclusion
        report_elements.append(Paragraph("Conclusion", styles['Heading1']))
        conclusion_text = """
        The STFT processing successfully decomposed the EEG signals into the standard frequency bands. The analyses indicate that the frequency bands were effectively isolated, as evidenced by the spectrograms, PSD plots, and other visualizations. Future work may involve applying these methods to larger datasets or exploring additional signal processing techniques.
        """
        report_elements.append(Paragraph(conclusion_text, styles['Normal']))
        report_elements.append(Spacer(1, 12))

        # References
        report_elements.append(Paragraph("References", styles['Heading1']))
        references_text = """
        - MNE-Python library documentation
        - ReportLab library documentation
        - NumPy and SciPy libraries
        """
        report_elements.append(Paragraph(references_text, styles['Normal']))

        # Compile the PDF report
        print("Compiling the PDF report...")
        compile_pdf_report(report_elements, OUTPUT_PDF)
        print(f"Report generated: {OUTPUT_PDF}")

        # Clean up temporary images directory
        print("Cleaning up temporary images...")
        for img_path in all_temp_images:
            if os.path.exists(img_path):
                os.remove(img_path)
        if os.path.exists(TEMP_IMG_DIR):
            os.rmdir(TEMP_IMG_DIR)

    except Exception as e:
        print("An error occurred during report generation:")
        traceback.print_exc()

def generate_spectrograms(band_data, selected_channels, temp_img_dir):
    """
    Generate spectrograms for selected EEG channels, limited to the frequency range of each band,
    and enhance visualization by adjusting intensity scaling and colormap.

    Parameters
    ----------
    band_data : dict
        Dictionary containing band signals.
    selected_channels : list
        List of channel names to generate spectrograms for.
    temp_img_dir : str
        Directory to save temporary images.

    Returns
    -------
    spectrogram_images : list
        List of file paths to the generated spectrogram images.
    """
    import matplotlib.colors as colors

    spectrogram_images = []
    # Get the frequency bands dictionary
    frequency_bands = {
        'delta': (0.5, 4),
        'theta': (4, 8),
        'alpha': (8, 12),
        'beta': (12, 30),
        'low_gamma': (30, 50),
        'mid_gamma': (50, 80),
        'high_gamma': (80, 150)
    }
    for band_name, data in band_data.items():
        fmin, fmax = frequency_bands[band_name]
        for ch_name in selected_channels:
            if ch_name in CHANNEL_NAMES:
                ch_index = CHANNEL_NAMES.index(ch_name)
                signal = data[ch_index, :]
                f, t, Sxx = spectrogram(signal, fs=SAMPLING_FREQUENCY, window=STFT_WINDOW,
                                        nperseg=STFT_NPERSEG, noverlap=STFT_OVERLAP)
                # Add a small value to avoid log(0)
                Sxx += np.finfo(float).eps
                plt.figure(figsize=(10, 4))
                # Use logarithmic scaling for the spectrogram
                plt.pcolormesh(t, f, Sxx, shading='gouraud', norm=colors.LogNorm(), cmap='viridis')
                plt.title(f'Spectrogram - {band_name} Band - Channel {ch_name}')
                plt.ylabel('Frequency [Hz]')
                plt.xlabel('Time [sec]')
                plt.colorbar(label='Intensity (log scale)')
                plt.ylim(fmin, fmax)  # Limit frequency axis to band range
                img_path = os.path.join(temp_img_dir, f'spectrogram_{band_name}_{ch_name}.png')
                plt.savefig(img_path)
                plt.close()
                spectrogram_images.append(img_path)
    return spectrogram_images

def generate_psd_plots(band_data, temp_img_dir):
    """
    Generate Power Spectral Density (PSD) plots for each frequency band.

    Parameters
    ----------
    band_data : dict
        Dictionary containing band signals.
    temp_img_dir : str
        Directory to save temporary images.

    Returns
    -------
    psd_images : list
        List of file paths to the generated PSD images.
    """
    psd_images = []
    for band_name, data in band_data.items():
        psd_all_channels = []
        for ch_index in range(data.shape[0]):
            signal = data[ch_index, :]
            f, Pxx = welch(signal, fs=SAMPLING_FREQUENCY, nperseg=STFT_NPERSEG)
            psd_all_channels.append(Pxx)
        psd_mean = np.mean(psd_all_channels, axis=0)
        plt.figure()
        plt.semilogy(f, psd_mean)
        plt.title(f'Average PSD - {band_name} Band')
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('PSD [V**2/Hz]')
        img_path = os.path.join(temp_img_dir, f'psd_{band_name}.png')
        plt.savefig(img_path)
        plt.close()
        psd_images.append(img_path)
    return psd_images

def plot_time_domain_waveforms(band_data, selected_channels, temp_img_dir):
    """
    Plot time-domain waveforms for each frequency band.

    Parameters
    ----------
    band_data : dict
        Dictionary containing band signals.
    selected_channels : list
        List of channel names to plot.
    temp_img_dir : str
        Directory to save temporary images.

    Returns
    -------
    waveform_images : list
        List of file paths to the generated waveform images.
    """
    waveform_images = []
    for band_name, data in band_data.items():
        plt.figure(figsize=(12, 6))
        for ch_name in selected_channels:
            if ch_name in CHANNEL_NAMES:
                ch_index = CHANNEL_NAMES.index(ch_name)
                signal = data[ch_index, :]
                time = np.arange(signal.shape[0]) / SAMPLING_FREQUENCY
                plt.plot(time, signal, label=f'Channel {ch_name}')
        plt.title(f'Time-Domain Waveforms - {band_name} Band')
        plt.xlabel('Time [sec]')
        plt.ylabel('Amplitude')
        plt.legend()
        img_path = os.path.join(temp_img_dir, f'waveform_{band_name}.png')
        plt.savefig(img_path)
        plt.close()
        waveform_images.append(img_path)
    return waveform_images

def plot_band_power_evolution(band_data, temp_img_dir):
    """
    Plot the time evolution of band power.

    Parameters
    ----------
    band_data : dict
        Dictionary containing band signals.
    temp_img_dir : str
        Directory to save temporary images.

    Returns
    -------
    band_power_images : list
        List of file paths to the generated band power images.
    """
    band_power_images = []
    for band_name, data in band_data.items():
        power = np.mean(data ** 2, axis=0)
        time = np.arange(power.shape[0]) / SAMPLING_FREQUENCY
        plt.figure()
        plt.plot(time, power)
        plt.title(f'Band Power Evolution - {band_name} Band')
        plt.xlabel('Time [sec]')
        plt.ylabel('Power')
        img_path = os.path.join(temp_img_dir, f'band_power_{band_name}.png')
        plt.savefig(img_path)
        plt.close()
        band_power_images.append(img_path)
    return band_power_images

def compile_pdf_report(report_elements, output_pdf):
    """
    Compile all the elements into a structured PDF report.

    Parameters
    ----------
    report_elements : list
        List of elements to include in the report.
    output_pdf : str
        Output PDF filename.

    Returns
    -------
    None
    """
    doc = SimpleDocTemplate(output_pdf, pagesize=letter)
    doc.build(report_elements)

if __name__ == "__main__":
    generate_report()
