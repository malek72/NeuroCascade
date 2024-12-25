import torch
import numpy as np
import os
import mne
from mne.preprocessing import ICA, create_eog_epochs
import pywt
import logging
from joblib import Parallel, delayed
from tqdm import tqdm
from scipy.signal import butter, filtfilt

# Optional: Configure seaborn for better aesthetics
try:
    import seaborn as sns
    sns.set(style="darkgrid")
except ImportError:
    pass  # Proceed without seaborn if not installed

# ----------------------------- Configuration ----------------------------- #

# Paths to the original and denoised datasets
ORIGINAL_DATASET_PATH = '/Users/maleksibai/ESC499 Thesis EEG/EEG-ImageNet_1.pth'
DENOISED_DATASET_PATH = '/Users/maleksibai/ESC499 Thesis EEG/EEG-ImageNet_1_denoised_new.pth'

# Sampling frequency (Hz)
FS = 1000  # Adjust based on your data's actual sampling rate

# Number of channels and samples
NUM_CHANNELS = 62
NUM_SAMPLES = 501

# Output directory for saving logs and denoised data
OUTPUT_DIR = '/Users/maleksibai/ESC499 Thesis EEG/EEG_Comparisons'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Logging configuration
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Create handlers
file_handler = logging.FileHandler(os.path.join(OUTPUT_DIR, 'denoising_log.log'))
file_handler.setLevel(logging.INFO)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

# Create formatters and add to handlers
formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(message)s')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# Add handlers to the logger
logger.addHandler(file_handler)
logger.addHandler(console_handler)

# ICA parameters
ICA_N_COMPONENTS = 20  # Number of ICA components
ICA_METHOD = 'fastica'  # ICA algorithm
ICA_RANDOM_STATE = 97  # For reproducibility

# Wavelet parameters
WAVELET = 'db4'
WAVELET_LEVEL = 5  # Decomposition level

# EOG channel names (if available)
# Replace with actual EOG channel names present in your dataset
EOG_CHANNELS = ['EOG1', 'EOG2']  # Example names; adjust accordingly

# Number of parallel jobs
N_JOBS = -1  # Use all available CPU cores minus one

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
        # Addressing PyTorch FutureWarning by setting weights_only=True if possible
        try:
            data = torch.load(pth_path, map_location='cpu', weights_only=True)
            logging.info(f"Loaded dataset from {pth_path} with weights_only=True")
        except TypeError:
            # weights_only parameter not available (older PyTorch versions)
            data = torch.load(pth_path, map_location='cpu')
            logging.warning("weights_only=True is not supported in this PyTorch version. Proceeding with weights_only=False.")
        return data
    except Exception as e:
        logging.error(f"Error loading {pth_path}: {e}")
        return None

def get_channel_names(dataset):
    """
    Retrieve channel names from the dataset if available.

    Parameters:
    - dataset (dict): Loaded dataset.

    Returns:
    - list: List of channel names.
    """
    if 'channel_names' in dataset and isinstance(dataset['channel_names'], list):
        return dataset['channel_names']
    else:
        # If channel names are not available, return generic names
        return [f'Ch{idx + 1}' for idx in range(NUM_CHANNELS)]

def average_reference_eeg(eeg_data):
    """
    Apply average reference to EEG data.

    Parameters:
    - eeg_data (numpy.ndarray): EEG data of shape (n_channels, n_times).

    Returns:
    - numpy.ndarray: Average referenced EEG data.
    """
    return eeg_data - np.mean(eeg_data, axis=0, keepdims=True)

def bandpass_filter_scipy(eeg_data, fs, lowcut=1.0, highcut=80.0, order=4):
    """
    Apply Butterworth band-pass filter to EEG data using SciPy.

    Parameters:
    - eeg_data (numpy.ndarray): EEG data of shape (n_channels, n_times).
    - fs (float): Sampling frequency.
    - lowcut (float): Low cutoff frequency.
    - highcut (float): High cutoff frequency.
    - order (int): Filter order.

    Returns:
    - numpy.ndarray: Filtered EEG data.
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    filtered_data = filtfilt(b, a, eeg_data, axis=1)
    return filtered_data

def bandstop_filter_scipy(eeg_data, fs, lowcut=49.0, highcut=51.0, order=4):
    """
    Apply Butterworth band-stop filter to EEG data using SciPy.

    Parameters:
    - eeg_data (numpy.ndarray): EEG data of shape (n_channels, n_times).
    - fs (float): Sampling frequency.
    - lowcut (float): Low cutoff frequency.
    - highcut (float): High cutoff frequency.
    - order (int): Filter order.

    Returns:
    - numpy.ndarray: Band-stop filtered EEG data.
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='bandstop')
    filtered_data = filtfilt(b, a, eeg_data, axis=1)
    return filtered_data

def perform_ica(eeg_data, sfreq, n_components=20, method='fastica', random_state=97):
    """
    Perform ICA on EEG data.

    Parameters:
    - eeg_data (numpy.ndarray): EEG data of shape (n_channels, n_times).
    - sfreq (float): Sampling frequency.
    - n_components (int): Number of ICA components.
    - method (str): ICA algorithm.
    - random_state (int): Seed for reproducibility.

    Returns:
    - ICA object: Fitted ICA instance.
    """
    info = mne.create_info(ch_names=[f'Ch{idx+1}' for idx in range(eeg_data.shape[0])],
                           sfreq=sfreq,
                           ch_types='eeg')
    raw = mne.io.RawArray(eeg_data, info)
    ica = ICA(n_components=n_components, method=method, random_state=random_state, max_iter='auto')
    ica.fit(raw, verbose=False)
    return ica

def identify_artifact_components(ica, raw, eog_channels=None):
    """
    Identify artifact components using EOG channels and ICLabel.

    Parameters:
    - ica (ICA object): Fitted ICA instance.
    - raw (mne.io.Raw): Raw EEG data.
    - eog_channels (list, optional): List of EOG channel names.

    Returns:
    - list: Indices of artifact components.
    """
    artifact_components = []

    # Method 1: Correlation with EOG channels
    if eog_channels:
        try:
            eog_epochs = create_eog_epochs(raw, ch_name=eog_channels, verbose=False)
            eog_average = eog_epochs.average()
            eog_inds, scores = ica.find_bads_eog(eog_average, threshold=0.3)
            artifact_components.extend(eog_inds)
            logging.info(f"Identified artifact components via EOG correlation: {eog_inds}")
        except Exception as e:
            logging.error(f"Error identifying EOG-related artifact components: {e}")

    # Method 2: Using ICLabel (if available)
    try:
        import iclabel
        # ICLabel expects raw data and ICA object
        ica_labels = iclabel.label_components(raw, ica, verbose=False)
        # Define which labels correspond to artifacts
        artifact_labels = ['Eye', 'Muscle', 'Heart', 'Line Noise', 'Channel Noise']
        # Threshold for classification probability
        label_threshold = 0.5  # Adjust as needed
        for idx, label_info in enumerate(ica_labels):
            primary_label = label_info['Label']
            primary_score = max(label_info['Confidence'])
            if primary_label in artifact_labels and primary_score >= label_threshold:
                artifact_components.append(idx)
        artifact_components = list(set(artifact_components))  # Remove duplicates
        logging.info(f"Identified artifact components via ICLabel: {artifact_components}")
    except ImportError:
        logging.warning("ICLabel not installed. Skipping automated ICA component classification.")
    except Exception as e:
        logging.error(f"Error during ICLabel artifact identification: {e}")

    return artifact_components

def apply_ica_artifact_removal(ica, raw, artifact_components):
    """
    Remove artifact components from raw EEG data using ICA.

    Parameters:
    - ica (ICA object): Fitted ICA instance.
    - raw (mne.io.Raw): Raw EEG data.
    - artifact_components (list): Indices of artifact components to remove.

    Returns:
    - numpy.ndarray: Cleaned EEG data.
    """
    raw_clean = raw.copy()
    ica.exclude = artifact_components
    ica.apply(raw_clean, exclude=artifact_components, verbose=False)
    return raw_clean.get_data()

def wavelet_denoise(eeg_data, wavelet='db4', level=5):
    """
    Apply wavelet-based denoising to EEG data.

    Parameters:
    - eeg_data (numpy.ndarray): EEG data of shape (n_channels, n_times).
    - wavelet (str): Wavelet type.
    - level (int): Decomposition level.

    Returns:
    - numpy.ndarray: Wavelet-denoised EEG data.
    """
    denoised_data = np.zeros_like(eeg_data)
    for ch in range(eeg_data.shape[0]):
        coeffs = pywt.wavedec(eeg_data[ch], wavelet, level=level)
        # Thresholding detail coefficients (excluding approximation coefficients)
        sigma = np.median(np.abs(coeffs[-level])) / 0.6745
        uthresh = sigma * np.sqrt(2 * np.log(len(eeg_data[ch])))
        denoised_coeffs = [coeffs[0]]  # Keep approximation coefficients unchanged
        for detail_coeff in coeffs[1:]:
            denoised_detail = pywt.threshold(detail_coeff, value=uthresh, mode='soft')
            denoised_coeffs.append(denoised_detail)
        denoised_signal = pywt.waverec(denoised_coeffs, wavelet)
        # Ensure the denoised signal has the same length
        denoised_data[ch, :eeg_data.shape[1]] = denoised_signal[:eeg_data.shape[1]]
    return denoised_data

def process_sample(sample, ch_names, sfreq, eog_channels=None):
    """
    Process a single EEG sample: filtering, ICA, artifact removal, wavelet denoising.

    Parameters:
    - sample (dict): Single EEG sample containing 'eeg_data'.
    - ch_names (list): List of channel names.
    - sfreq (float): Sampling frequency.
    - eog_channels (list, optional): List of EOG channel names for ICA artifact identification.

    Returns:
    - dict: Denoised EEG sample.
    """
    try:
        eeg_tensor = sample['eeg_data']
        if eeg_tensor.ndim != 2 or eeg_tensor.shape[0] != len(ch_names):
            raise ValueError(f"Unexpected EEG data shape: {eeg_tensor.shape}")
        eeg_data = eeg_tensor.numpy().astype(np.float32)

        # Average reference
        eeg_data = average_reference_eeg(eeg_data)

        # Band-pass filter using SciPy
        eeg_data = bandpass_filter_scipy(eeg_data, sfreq)

        # Band-stop filter (50 Hz noise) using SciPy
        eeg_data = bandstop_filter_scipy(eeg_data, sfreq)

        # Create MNE Raw object for ICA
        info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types='eeg')
        raw = mne.io.RawArray(eeg_data, info)

        # Perform ICA
        ica = perform_ica(eeg_data, sfreq, n_components=ICA_N_COMPONENTS, method=ICA_METHOD, random_state=ICA_RANDOM_STATE)

        # Identify artifact components
        artifact_components = identify_artifact_components(ica, raw, eog_channels=eog_channels)

        # Apply ICA to remove artifacts
        eeg_data_clean = apply_ica_artifact_removal(ica, raw, artifact_components)

        # Wavelet denoising
        eeg_data_denoised = wavelet_denoise(eeg_data_clean, wavelet=WAVELET, level=WAVELET_LEVEL)

        # Update sample with denoised data
        sample['eeg_data'] = torch.from_numpy(eeg_data_denoised)

        return sample

    except Exception as e:
        logging.error(f"Error processing sample: {e}")
        return None

def process_dataset_parallel(input_path, output_path, ch_names, sfreq, eog_channels=None, n_jobs=-1):
    """
    Process the entire dataset in parallel and save the denoised data.

    Parameters:
    - input_path (str): Path to the original dataset.
    - output_path (str): Path to save the denoised dataset.
    - ch_names (list): List of channel names.
    - sfreq (float): Sampling frequency.
    - eog_channels (list, optional): List of EOG channel names for ICA artifact identification.
    - n_jobs (int, optional): Number of parallel jobs. Default is -1 (all available cores).

    Returns:
    - None
    """
    data = load_dataset(input_path)
    if data is None:
        logging.error(f"Failed to load dataset from {input_path}. Skipping.")
        return

    total_samples = len(data['dataset'])
    logging.info(f"Starting processing of {total_samples} samples.")

    denoised_dataset = Parallel(n_jobs=n_jobs)(
        delayed(process_sample)(sample, ch_names, sfreq, eog_channels) for sample in tqdm(data['dataset'], desc=f"Processing {os.path.basename(input_path)}")
    )

    # Filter out None results (failed samples)
    denoised_dataset = [sample for sample in denoised_dataset if sample is not None]
    processed_samples = len(denoised_dataset)
    logging.info(f"Completed processing. Successfully denoised {processed_samples} out of {total_samples} samples.")

    denoised_data = {
        "dataset": denoised_dataset,
        "labels": data.get('labels', []),
        "images": data.get('images', []),
        "channel_names": ch_names  # Preserve channel names
    }

    # Save denoised dataset
    try:
        torch.save(denoised_data, output_path)
        logging.info(f"Denoised data saved to {output_path}")
    except Exception as e:
        logging.error(f"Error saving denoised data to {output_path}: {e}")

# ------------------------------ Main Script ------------------------------- #

def main():
    # Load original dataset to retrieve channel names
    original_data = load_dataset(ORIGINAL_DATASET_PATH)
    if original_data is None:
        logging.error("Failed to load original dataset. Exiting.")
        return
    ch_names = get_channel_names(original_data)

    # Check if specified EOG channels exist in channel names
    if EOG_CHANNELS:
        eog_present = all(eog in ch_names for eog in EOG_CHANNELS)
        if not eog_present:
            logging.warning("Specified EOG channels not found in dataset. Artifact identification will rely solely on ICLabel or manual inspection.")
            eog_channels = None
        else:
            eog_channels = EOG_CHANNELS
    else:
        eog_channels = None

    # Process the dataset (denoising)
    process_dataset_parallel(
        input_path=ORIGINAL_DATASET_PATH,
        output_path=DENOISED_DATASET_PATH,
        ch_names=ch_names,
        sfreq=FS,
        eog_channels=eog_channels,
        n_jobs=N_JOBS
    )

    logging.info("EEG denoising pipeline completed successfully.")

if __name__ == "__main__":
    main()
