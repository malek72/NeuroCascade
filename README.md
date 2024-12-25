# NeuroCascade Project

## Overview

**NeuroCascade** is an innovative project aimed at revolutionizing the diagnosis and understanding of neurological disorders through advanced EEG signal processing and deep learning techniques. By integrating a comprehensive set of EEG features—including phase, amplitude, Phase-to-Phase Coupling (PPC), and Phase-to-Amplitude Coupling (PAC)—NeuroCascade seeks to uncover intricate mechanisms of brain information encoding and discover groundbreaking biomarkers for neurological conditions.

## Table of Contents

- [Background](#background)
- [Problem Statement](#problem-statement)
- [Objectives](#objectives)
  - [Main Objective](#main-objective)
  - [Sub-objectives](#sub-objectives)
- [Methodology](#methodology)
  - [1. Data Cleaning / EEG Denoising](#1-data-cleaning--eeg-denoising)
  - [2. Feature Extraction](#2-feature-extraction)
  - [3. Building the Deep Learning Model](#3-building-the-deep-learning-model)
- [Significance](#significance)
- [Research Question](#research-question)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Background

Neurological disorders stem from abnormalities in the brain's electrical activity, disrupting normal brain functions and essential cognitive processes. Electroencephalography (EEG) is the premier biosignal for capturing and measuring this electrical activity, traditionally used by clinicians to diagnose neurological conditions through manual waveform inspections and basic feature extraction focusing on phase and amplitude.

However, the brain's electrical activity is inherently non-linear, necessitating a deeper exploration of complex interactions among various EEG features such as functional connectivity, amplitude relationships, and phase relationships. Incorporating these intricate features can lead to the discovery of advanced biomarkers, enhancing our understanding of brain information encoding and improving diagnostic approaches for neurological diseases.

Despite the efficacy of EEG in clinical settings, challenges such as the labor-intensive nature of manual inspections and the potential for missed event onsets persist. The advent of deep learning offers a promising solution by enabling automated, comprehensive analysis of EEG data, thereby addressing these challenges and facilitating the extraction of more complex and informative features.

## Problem Statement

Current research in EEG-based neurological disorder diagnosis predominantly focuses on a limited set of features, often overlooking the complex interactions between phase, amplitude, PPC, and PAC. This narrow focus hampers a thorough understanding of how information is encoded in the brain during both healthy and diseased states, limiting the discovery of comprehensive biomarkers essential for accurate diagnosis and monitoring.

## Objectives

### Main Objective

Develop a hierarchical EEG feature framework and construct a state-of-the-art deep learning model capable of systematically evaluating the combined impact of phase, amplitude, PPC, and PAC features. This model aims to provide a robust and nuanced understanding of brain function, uncovering previously unknown aspects of information encoding and identifying novel biomarkers for neurological disorders.

### Sub-objectives

1. **EEG Feature Extraction:**
   - Extract phase, amplitude, PPC, and PAC features from EEG signals using advanced signal processing techniques to capture a comprehensive spectrum of brain function.

2. **Construction of a Deep Learning Model:**
   - Build and optimize a deep learning architecture tailored to analyze the integrated EEG features, facilitating a deeper understanding of brain encoding mechanisms.

## Methodology

### 1. Data Cleaning / EEG Denoising

**EEG Denoising Process:**

- **Preprocessing:**
  - Apply a bandpass filter (0.5 Hz - 80 Hz) to remove low-frequency drifts and high-frequency noise.
  - Use a notch filter at 50 Hz to eliminate electrical mains interference.

- **Artifact Removal:**
  - Perform Independent Component Analysis (ICA) to decompose EEG signals into independent components.
  - Identify and remove components correlated with frontal EEG channels, indicative of artifacts like eye movements.
  - Apply wavelet-based cleaning to attenuate high-frequency noise in artifact components.

- **Reconstruction:**
  - Reconstruct the cleaned EEG signals by reintegrating the processed components.
  - Segment the data into manageable batches for efficient processing and storage.

### 2. Feature Extraction

**EEG Feature Extraction Process:**

- **Data Preparation:**
  - Utilize high-resolution EEG recordings (1000 Hz sampling rate) from multiple subjects and brain regions.

- **Frequency Decomposition:**
  - Apply Short-Time Fourier Transform (STFT) with a window size of 256 samples and a time step of 128 samples to decompose EEG signals into time-frequency components.

- **Band Segmentation:**
  - Segment EEG signals into standard frequency bands:
    - Delta (0.5–4 Hz)
    - Theta (4–8 Hz)
    - Alpha (8–12 Hz)
    - Beta (12–30 Hz)
    - Gamma (30–150 Hz) [further divided into low, mid, high]

- **Signal Reconstruction:**
  - Use Inverse Short-Time Fourier Transform (ISTFT) to reconstruct time-domain signals for each frequency band.

- **Feature Storage:**
  - Organize and save band-specific signals for subsequent integration and deep learning model training.

### 3. Building the Deep Learning Model

- **Pipeline Construction:**
  - Develop a machine learning pipeline tailored to handle the integrated EEG features.

- **Model Architecture:**
  - Design and experiment with various deep learning architectures (e.g., CNNs, RNNs, Transformers) to determine the most effective model for analyzing the comprehensive feature set.

- **Training & Evaluation:**
  - Train models using the extracted features and evaluate their performance to identify the optimal architecture for uncovering brain encoding mechanisms and identifying biomarkers.

## Significance

Integrating a broader set of EEG features allows for the capture of a more complex spectrum of brain functions, leading to a deeper understanding of information encoding in the brain. This comprehensive approach has the potential to uncover novel biomarkers, enhancing diagnostic accuracy and enabling early intervention for neurological disorders. Additionally, a robust deep learning model can serve as an automated alert mechanism, assisting clinicians in monitoring and diagnosing brain diseases more efficiently.

## Research Question

**How does a hierarchical integration of EEG features—phase, amplitude, PPC, PAC—enhance the deep learning algorithm’s ability to uncover mechanisms on how information is encoded in the brain and potentially discover groundbreaking biomarkers?**

## Getting Started

### Prerequisites

- Python 3.7+
- Libraries:
  - NumPy
  - SciPy
  - MNE
  - TensorFlow
  - Scikit-learn
  - Pandas
  - Matplotlib
  - Seaborn

### Installation

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/yourusername/NeuroCascade.git
   cd NeuroCascade
