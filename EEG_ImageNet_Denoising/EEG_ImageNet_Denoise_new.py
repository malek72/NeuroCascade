import torch
import mne
import numpy as np
from sklearn.decomposition import FastICA
from scipy.signal import find_peaks
from pywt import wavedec, waverec

# Load the dataset
dataset = torch.load('/Users/maleksibai/ESC499 Thesis EEG/EEG-ImageNet_1.pth', weights_only=False)

# Parameters
n_channels = 62
sfreq = 1000  # Sampling frequency
ch_names = [f'EEG {i+1:03d}' for i in range(n_channels)]

# Prepare to store cleaned data
cleaned_dataset = {
    'dataset': [],
    'labels': dataset['labels'],
    'images': dataset['images'],
}

# Process each EEG sample individually
for sample in dataset['dataset']:
    eeg_data = sample['eeg_data'].numpy()  # Shape: (n_channels, n_times)

    # Skip samples with insufficient length
    if eeg_data.shape[1] < 100:
        continue  # Adjust the threshold as needed

    # Create MNE Info object
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types='eeg')

    # Create Raw object
    raw = mne.io.RawArray(eeg_data, info)

    # Re-reference using the offline linked mastoids method
    raw.set_eeg_reference(ref_channels=['EEG 061', 'EEG 062'])

    # Filtering using IIR filters suitable for short signals
    raw.filter(l_freq=0.5, h_freq=80.0, method='iir')
    raw.notch_filter(freqs=50.0, method='iir')

    # Perform ICA decomposition
    ica = FastICA(n_components=20, random_state=97, max_iter=1000, tol=0.0001)
    data = raw.get_data().T  # Shape: (n_times, n_channels)
    try:
        sources = ica.fit_transform(data).T  # Shape: (n_components, n_times)
    except ValueError:
        continue  # Skip if ICA fails to converge
    mixing_matrix = ica.mixing_.T  # Shape: (n_components, n_channels)

    # Identify and remove artifacts
    frontal_channels = [f'EEG {i+1:03d}' for i in range(16)]
    correlation = []
    for ch in frontal_channels:
        eeg = raw.get_data(picks=ch).flatten()
        component_corr = [np.corrcoef(eeg, source)[0, 1] for source in sources]
        correlation.append(component_corr)
    eog_candidates = [np.argmax(np.abs(corrs)) for corrs in correlation]
    weight_vector = {}
    for idx in eog_candidates:
        weights = np.abs(mixing_matrix[idx, :16])
        avg_weight = np.mean(weights)
        weight_vector[idx] = avg_weight
    weights_list = list(weight_vector.values())
    Q1, Q3 = np.percentile(weights_list, [25, 75])
    IQR = Q3 - Q1
    threshold = Q3 + 1.5 * IQR
    artifact_components = [idx for idx, weight in weight_vector.items() if weight >= threshold]

    # Wavelet denoising on artifact components
    for idx in artifact_components:
        source = sources[idx]
        peaks, _ = find_peaks(np.abs(source), height=np.mean(np.abs(source)), distance=int(sfreq / 2))
        for peak in peaks:
            start = max(0, peak - int(sfreq / 2))
            end = min(source.shape[0], peak + int(sfreq / 2))
            epoch = source[start:end]
            coeffs = wavedec(epoch, 'sym4', level=3)  # Reduced level for short signals
            coeffs[0] = np.zeros_like(coeffs[0])
            cleaned_epoch = waverec(coeffs, 'sym4')[:end - start]
            sources[idx, start:end] = cleaned_epoch

    # Reconstruct the cleaned EEG signals
    cleaned_data = np.dot(mixing_matrix.T, sources).T
    cleaned_data = cleaned_data.T  # Shape: (n_channels, n_times)

    # Update the sample with cleaned data
    sample['eeg_data'] = torch.tensor(cleaned_data)
    cleaned_dataset['dataset'].append(sample)

# Save the cleaned dataset
torch.save(cleaned_dataset, '/Users/maleksibai/ESC499 Thesis EEG/EEG_ImageNet_Denoising/EEG-ImageNet_cleaned.pth')