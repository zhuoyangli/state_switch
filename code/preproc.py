# Preprocessing BOLD data after fmriprep pipeline
# uses nilearn.signal.clean to high-pass filter and standardize the data

from nilearn import image, datasets
from nilearn.signal import clean
from nilearn.input_data import NiftiLabelsMasker
import nibabel as nib
import numpy as np
import os
from utils import load_config, load_bold, get_logger, butterworth_highpass
import argparse
from tqdm import tqdm
import time

cfg = load_config()
DATADIR = cfg['DATA_DIR']
log = get_logger(__name__)

def main(subject: str, task: str, atlas: str = 'Schaefer', nrois: int = 200, yeo_networks: int = 17, save: bool = False):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Starting proprocessing {subject}, {task}...")
    # load the BOLD data
    bold = load_bold(DATADIR, subject, task)
    bold_data = bold.get_fdata()

    # reshape the data
    n_voxels = np.prod(bold_data.shape[:-1])
    n_trs = bold_data.shape[-1]
    bold_reshaped = bold_data.reshape(n_voxels, n_trs)

    # high-pass filtering
    # Butterworth high-pass filter with a cutoff frequency of 0.01 Hz
    bold_cleaned = butterworth_highpass(bold_reshaped, 1.5, 0.01)
    # standardization
    bold_cleaned = (bold_cleaned - np.mean(bold_cleaned, axis=1, keepdims=True)) / np.std(bold_cleaned, axis=1, keepdims=True)

    # reshape data back to original 4D shape
    bold_cleaned = bold_cleaned.reshape(bold_data.shape)

    # parcellation
    if atlas == 'Schaefer':
        schaefer_atlas = datasets.fetch_atlas_schaefer_2018(n_rois=nrois, yeo_networks=yeo_networks, resolution_mm=2)
        atlas_filename = schaefer_atlas.maps
        parcel_labels = schaefer_atlas.labels
    
    # get the time series for each ROI
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Starting ROI extraction...")
    masker = NiftiLabelsMasker(labels_img=atlas_filename, standardize=True)
    cleaned_bold_img = image.new_img_like(bold, bold_cleaned, copy_header=True)
    bold_roi = masker.fit_transform(cleaned_bold_img)

    # save the data
    if save:
        save_dir = os.path.join(DATADIR, subject, 'func', 'roi')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        np.savez_compressed(os.path.join(save_dir, f'{subject}_task-{task}_roi.npz'), bold_roi=bold_roi.T, roi_labels=parcel_labels)
        log.info(f"Saved ROI data for subject {subject} on task {task}.")
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Finished saving ROI data...")

    # get the multi-voxel pattern time series for each ROI
    atlas_img = nib.load(atlas_filename)
    atlas_data = atlas_img.get_fdata()
    roi_data = {}
    for roi_idx in tqdm(range(1, nrois + 1), total=nrois, desc="extracting MVPs for ROIs"):
        roi_mask = atlas_data == roi_idx
        roi_mvp = bold_cleaned[roi_mask]
        roi_data[roi_idx] = roi_mvp
    
    # save the data
    if save:
        save_dir = os.path.join(DATADIR, subject, 'func', 'mvp')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        np.savez_compressed(os.path.join(save_dir, f'{subject}_task-{task}_mvp.npz'), roi_data=roi_data, roi_labels=parcel_labels)
        log.info(f"Saved MVP data for subject {subject} on task {task}.")
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Finished saving MVP data...")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocess BOLD data')
    parser.add_argument('--subject', type=str, help='Subject ID')
    parser.add_argument('--task', type=str, help='Task name')
    parser.add_argument('--atlas', type=str, default='Schaefer', help='Atlas name')
    parser.add_argument('--nrois', type=int, default=200, help='Number of ROIs')
    parser.add_argument('--yeo_networks', type=int, default=17, help='Number of Yeo networks')
    parser.add_argument('--save', action='store_true', help='Save the preprocessed data')

    args = parser.parse_args()

    main(args.subject, args.task, args.atlas, args.nrois, args.yeo_networks, args.save)
