import torch
import mne
import numpy as np
import math
from mne import create_info
from mne.io import RawArray
from sklearn.decomposition import FastICA
from scipy.signal import find_peaks
from pywt import wavedec, waverec
from collections import defaultdict

def load_dataset(dataset_file):
    # Load the dataset from the .pth file
    dataset = torch.load(dataset_file)
    return dataset['dataset']

def preprocess_and_save_subject_data(subject_data, channel_names, subject_id, fs=1000):
    # Determine batch size per subject (adjust based on available memory)
    batch_size = 1000
    total_samples = len(subject_data)
    total_batches = math.ceil(total_samples / batch_size)

    for batch_num in range(1, total_batches + 1):
        start_idx = (batch_num - 1) * batch_size
        end_idx = min(batch_num * batch_size, total_samples)
        current_batch = subject_data[start_idx:end_idx]
        print(f"Processing batch {batch_num}/{total_batches} for subject {subject_id} (samples {start_idx} to {end_idx})...")

        # Stack the EEG data along the time axis
        eeg_data_list = [sample['eeg_data'].numpy() for sample in current_batch]
        data = np.concatenate(eeg_data_list, axis=1)  # data shape: (62, 500 * N_samples)

        # Create MNE Raw object
        info = create_info(ch_names=channel_names, sfreq=fs, ch_types='eeg')
        raw = RawArray(data, info)

        # Apply bandpass filter
        raw.filter(l_freq=0.5, h_freq=80.0, fir_design='firwin')

        # Apply notch filter at 50 Hz
        raw.notch_filter(freqs=50.0)

        # Perform ICA decomposition
        n_components = 20  # Adjust based on your data and computational resources
        ica = mne.preprocessing.ICA(n_components=n_components, random_state=97, max_iter=800)
        ica.fit(raw)

        # Identify artifact components
        # Define frontal channels
        frontal_channels = [
            'Fp1', 'Fp2', 'F3', 'F4', 'F7', 'F8',
            'AF3', 'AF4', 'AF7', 'AF8', 'Fz', 'F1', 'F2', 'F5', 'F6'
        ]

        # Get indices of frontal channels
        frontal_picks = mne.pick_channels(raw.info['ch_names'], frontal_channels)

        # Get the data from frontal channels
        frontal_data = raw.get_data(picks=frontal_picks)

        # Get the ICA sources
        sources = ica.get_sources(raw).get_data()  # shape: (n_components, n_times)

        # Calculate correlation between ICA components and frontal channels
        correlations = np.corrcoef(sources, frontal_data)[n_components:, :n_components]

        # Sum the absolute correlations across frontal channels
        abs_correlations = np.abs(correlations)
        sum_correlations = np.sum(abs_correlations, axis=0)

        # Identify components with high correlation to frontal channels
        threshold = np.mean(sum_correlations) + np.std(sum_correlations)
        artifact_components = np.where(sum_correlations > threshold)[0]
        print(f"Identified artifact components: {artifact_components}")

        # Copy sources to modify
        sources_cleaned = sources.copy()
        sfreq = raw.info['sfreq']

        # Process each artifact component
        for idx in artifact_components:
            component = sources_cleaned[idx]
            # Detect peaks in the component
            peaks, _ = find_peaks(np.abs(component), height=np.mean(np.abs(component)), distance=sfreq/2)
            for peak in peaks:
                # Create 1-second epochs around peaks
                start = int(max(0, peak - sfreq/2))
                end = int(min(component.shape[0], peak + sfreq/2))
                epoch = component[start:end]
                # Perform wavelet decomposition
                coeffs = wavedec(epoch, 'sym4', level=5)
                # Zero out the high-frequency detail coefficients
                for i in range(len(coeffs) - 2):
                    coeffs[i] = np.zeros_like(coeffs[i])
                # Reconstruct the cleaned epoch
                cleaned_epoch = waverec(coeffs, 'sym4')
                # Replace the original data with the cleaned data
                sources_cleaned[idx, start:end] = cleaned_epoch[:end - start]

        # Reconstruct the cleaned signal
        # Create a copy of raw data to avoid modifying the original raw object
        raw_clean = raw.copy()

        # Update the ICA object with the cleaned sources
        ica._sources = sources_cleaned

        # Mark the artifact components for exclusion
        ica.exclude = artifact_components.tolist()

        # Apply ICA to the raw data
        ica.apply(raw_clean)

        # Get the cleaned data
        cleaned_data = raw_clean.get_data()

        # Now, split the cleaned data back into individual samples
        sample_length = eeg_data_list[0].shape[1]
        N_samples = len(eeg_data_list)
        # Ensure that the total length is divisible by sample_length
        total_length = cleaned_data.shape[1]
        if total_length != sample_length * N_samples:
            cleaned_data = cleaned_data[:, :sample_length * N_samples]
        cleaned_samples = np.split(cleaned_data, N_samples, axis=1)

        # Save the cleaned data back to the current_batch
        for i, sample in enumerate(current_batch):
            sample['eeg_data'] = torch.tensor(cleaned_samples[i])

        # Save the cleaned data for the batch
        output_file = f'cleaned_subject_{subject_id}_batch_{batch_num}.pth'
        torch.save(current_batch, output_file)
        print(f"Saved cleaned data for subject {subject_id}, batch {batch_num} to {output_file}")

    print(f"Completed processing for subject {subject_id}.")

def main():
    # Define path to the single dataset part
    dataset_file = '/Users/maleksibai/ESC499 Thesis EEG/EEG-ImageNet_1.pth'  # Update with your dataset path

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

    # Load dataset
    print(f"Loading dataset from {dataset_file}...")
    dataset = load_dataset(dataset_file)
    total_samples = len(dataset)
    print(f"Dataset loaded with {total_samples} samples.")

    # Group data by subject
    data_by_subject = defaultdict(list)
    for sample in dataset:
        data_by_subject[sample['subject']].append(sample)

    # Process data per subject
    for subject_id, subject_data in data_by_subject.items():
        print(f"Processing data for subject {subject_id}...")
        preprocess_and_save_subject_data(subject_data, channel_names, subject_id, fs=1000)

    print("All subjects processed and saved successfully.")

if __name__ == "__main__":
    main()