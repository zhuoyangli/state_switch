# Required imports
from nilearn import datasets, image
from nilearn.input_data import NiftiLabelsMasker
from nilearn.image import math_img
from nilearn.signal import clean
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
import numpy as np
import os
from utils import load_config, load_bold
from matplotlib import pyplot as plt

cfg = load_config()
DATADIR = cfg['DATA_DIR']
STIMDIR = cfg['STIM_DIR']
FIGDIR = cfg['FIG_DIR']

subject = 'sub-001'
task = 'treasureisland'

# Load the BOLD data
bold = load_bold(DATADIR, subject, task)
print(np.min(bold), np.max(bold))

# reshape the data
n_voxels = np.prod(bold.shape[:-1])
n_trs = bold.shape[-1]
bold_reshaped = bold.reshape(n_voxels, n_trs)
print(bold_reshaped.shape)

# Select a random subset of voxels to process (e.g., 1000 voxels)
n_voxels_subset = 1000
voxel_indices = np.random.choice(n_voxels, size=n_voxels_subset, replace=False)
bold_subset = bold_reshaped[voxel_indices, :]

# High-pass filter the data
bold_cleaned_subset = clean(bold_subset, detrend=False, standardize=False, high_pass=0.01, t_r=1.5)
bold_cleaned_with_detrending_subset = clean(bold_subset, detrend=True, standardize=False, high_pass=0.01, t_r=1.5)

# Number of voxels to plot
n_voxels_to_plot = 5

# Randomly select voxel indices from the subset for plotting
plot_voxel_indices = np.random.choice(n_voxels_subset, size=n_voxels_to_plot, replace=False)

# Create subplots for multiple voxels
plt.figure(figsize=(15, 10))
for i, voxel_idx in enumerate(plot_voxel_indices):
    plt.subplot(5, 1, i + 1)
    
    # Increase linewidth and differentiate linestyle
    # plt.plot(bold_cleaned_subset[voxel_idx], label='Cleaned BOLD', color='b', linestyle='--', linewidth=2)
    # plt.plot(bold_cleaned_with_detrending_subset[voxel_idx], label='Cleaned BOLD with Detrending', color='r', linestyle='-', linewidth=2)
    plt.plot(bold_cleaned_subset[voxel_idx] - bold_cleaned_with_detrending_subset[voxel_idx], label='Detrended BOLD', color='g', linestyle='-', linewidth=2)
    plt.title(f'Voxel {voxel_idx} from Subset')
    plt.xlabel('Time (s)')
    plt.ylabel('Intensity')
    plt.legend()

# Adjust layout
plt.tight_layout()
plt.show()