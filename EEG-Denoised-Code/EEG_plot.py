import torch
import matplotlib.pyplot as plt

# Load the dataset (update the path as needed)
dataset_path = "/Users/maleksibai/ESC499 Thesis EEG/EEG-ImageNet_1.pth"  # Example path
data = torch.load(dataset_path)

# Function to plot all 62 EEG electrodes
def plot_eeg_signals(eeg_data, title="EEG Signal for All Electrodes"):
    n_channels = eeg_data.shape[0]  # Number of electrodes (62 in this dataset)
    time_points = eeg_data.shape[1]  # Number of time points per electrode
    
    # Create a figure with subplots for each channel
    fig, axes = plt.subplots(n_channels, 1, figsize=(15, 2 * n_channels), sharex=True)
    fig.suptitle(title, fontsize=16)
    
    for i in range(n_channels):
        axes[i].plot(eeg_data[i], linewidth=0.8)
        axes[i].set_ylabel(f'Ch {i+1}', fontsize=8)
        axes[i].tick_params(axis='y', labelsize=8)
        axes[i].grid(True, which='both', linestyle='--', linewidth=0.5)
    
    axes[-1].set_xlabel("Time Points", fontsize=12)
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])  # Adjust layout to fit title
    plt.show()

# Example usage: Load one EEG sample and plot
example_entry = data['dataset'][0]  # Accessing the first sample in the dataset
eeg_data_sample = example_entry['eeg_data']  # Assuming this is a 2D tensor (62, time_points)

plot_eeg_signals(eeg_data_sample.numpy())
