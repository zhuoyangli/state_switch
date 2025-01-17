import logging
import os
from datetime import datetime
from pathlib import Path
from random import choices
from typing import List, Optional, Union
import re
import h5py

import numpy as np
import pandas as pd
from nilearn import datasets, image, input_data, masking, surface
import nibabel as nib
from nilearn.signal import clean
from nilearn.input_data import NiftiLabelsMasker
import yaml
from tqdm import tqdm
import librosa
from scipy.signal import hilbert, resample, butter, filtfilt
import time
from typing import Optional, Tuple, Dict


ROOT = Path(__file__).parent
FORMAT = "[%(levelname)s] %(name)s.%(funcName)s - %(message)s"
TEXT_GRID_FORMATS = [
    'File type = "ooTextFile"',
    '"Praat chronological TextGrid text file"',
]


logging.basicConfig(format=FORMAT)


def get_logger(
    name=__name__,
    log_level=logging.INFO,
    log_file: Optional[str] = None,
) -> logging.Logger:
    """Initializes command line logger."""

    logger = logging.getLogger(name)
    logger.setLevel(log_level)

    if log_file is not None:
        formatter = logging.Formatter(FORMAT)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setLevel(log_level)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger


log = get_logger(__name__)

def load_config():
    """Load the configuration from the YAML file."""
    with open(ROOT / "config.yaml", "r") as f:
        config = yaml.safe_load(f)

    return config

cfg = load_config()
DATADIR = cfg['DATA_DIR']
EMBEDDINGS_FILE = os.path.join(DATADIR, 'english1000sm.hf5')

def zscore(v, axis=0):
    return (v-np.mean(v, axis=axis, keepdims=True)) / (np.std(v, axis=axis, keepdims=True) + 1e-6)

# these tokens were defined in:
# https://github.com/HuthLab/deep-fMRI-dataset/blob/eaaa5cd186e0222c374f58adf29ed13ab66cc02a/encoding/ridge_utils/dsutils.py#L5C1-L5C96
SKIP_TOKENS = frozenset(
    ["sentence_start", "sentence_end", "{BR}", "{LG}", "{ls}", "{LS}", "{NS}", "sp"]
)

def load_roi(datadir: str, subject: str, task: str):
    """Load the ROI data for the specified subject and task."""
    roi_file = os.path.join(datadir, subject, 'func', 'roi', f'{subject}_task-{task}_roi.npz')
    with np.load(roi_file, allow_pickle=True) as data:
        roi_data = data['bold_roi']
        roi_labels = data['roi_labels']
    return roi_data, roi_labels

def load_mvp(datadir: str, subject: str, task: str):
    """Load the MVP data for the specified subject and task."""
    mvp_file = os.path.join(datadir, subject, 'func', 'mvp', f'{subject}_task-{task}_mvp.npz')
    with np.load(mvp_file, allow_pickle=True) as data:
        mvp_data = data['roi_data'].item()
        roi_labels = data['roi_labels']
    return mvp_data, roi_labels

def load_surf(data_dir: str, subject: str, task: str, hemi: str = 'L'):
    """Load the surface data for the specified subject, task, and hemisphere."""
    surf_file = os.path.join(data_dir, subject, 'func', f'{subject}_task-{task}_hemi-{hemi}_space-fsaverage6_bold.func.gii')
    if not os.path.exists(surf_file):
        log.error(f"Surface file {surf_file} does not exist.")
        return None
    surf_data = surface.load_surf_data(surf_file)
    return np.array(surf_data, dtype=np.float64)

def load_mp3(dir: str, task: str):
    """Load an MP3 file from the specified directory."""
    audio, sample_rate = librosa.load(os.path.join(dir, f'{task}.mp3'), sr=None)
    return audio, sample_rate

def get_envelope(signal: np.ndarray) -> np.ndarray:
    """Compute the audio envelope"""
    return np.abs(hilbert(signal))  # type: ignore

def downsample(
    signal: np.ndarray, sfreq: float, tr_len: float, n_trs: int
) -> np.ndarray:
    num_samples_uncorrected = signal.shape[0] / (sfreq * tr_len)
    # sometimes the signal samples do not match with the trs
    # either have to correct the number of resulting samples up or down
    # rounding won't work (`undertheinfluence`` vs `naked`)
    if num_samples_uncorrected > n_trs:
        num_samples = int(np.floor(num_samples_uncorrected))
    else:
        num_samples = int(np.ceil(num_samples_uncorrected))
    log.info(f"Downsampling to {num_samples} samples.")
    return resample(signal, num=num_samples)  # type: ignore

def load_bold(datadir: str, subject: str, task: str) -> np.ndarray:
    """Load the preprocessed BOLD data for the specified subject and task."""
    bold_file = os.path.join(datadir, subject, 'func', f'{subject}_task-{task}_space-MNI152NLin6Asym_res-2_desc-preproc_bold.nii.gz')
    bold_img = image.load_img(bold_file)
    log.info(f"Loaded BOLD data for {subject} on task {task}")
    return bold_img

def preproc(subject: str, task: str, atlas: str = 'Schaefer', nrois: int = 200, yeo_networks: int = 17, save: bool = False):
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
    bold_cleaned = clean(bold_reshaped, detrend=False, standardize=False, high_pass=0.01, t_r=1.5)
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

def butterworth_highpass(data, tr_len, cutoff_freq, order=5):
    """
    Apply a Butterworth high-pass filter to fMRI data.
    
    Parameters:
    - data: 2D array (n_voxels, n_trs)
      The fMRI data to be filtered, where n_voxels is the number of voxels and n_trs is the number of time points.
    - tr_len: float
      The TR (time repetition) length of the fMRI data in seconds.
    - cutoff_freq: float
      The cutoff frequency for the high-pass filter in Hz (e.g., 0.01 Hz).
    - order: int, optional
      The order of the Butterworth filter. Default is 5.
    
    Returns:
    - filtered_data: 2D array (n_voxels, n_trs)
      The filtered fMRI data.
    """
    # Compute the Nyquist frequency
    nyquist_freq = 0.5 / tr_len
    
    # Normalize the cutoff frequency by the Nyquist frequency
    normalized_cutoff = cutoff_freq / nyquist_freq
    
    # Design the Butterworth filter
    b, a = butter(order, normalized_cutoff, btype='high', analog=False)
    
    # Initialize array to hold the filtered data
    filtered_data = np.zeros_like(data)
    
    # Apply the filter voxel-wise (row-wise across time points)
    if data.ndim == 1:
        filtered_data = filtfilt(b, a, data)
    else:
        for i in range(data.shape[0]):  # Iterate over each voxel
            filtered_data[i, :] = filtfilt(b, a, data[i, :])
    
    return filtered_data

def get_tier_data(text_grid_lines: list) -> dict:

    lines = [line.strip("\n").strip() for line in text_grid_lines]

    # find what textgrid format it is
    text_grid_ftype = lines[0]  # this is always the first line

    assert (
        text_grid_ftype in TEXT_GRID_FORMATS
    ), f"Unexpected textgrid format: {text_grid_ftype[0]}"

    if text_grid_ftype == TEXT_GRID_FORMATS[0]:
        """
        Example header:

        File type = "ooTextFile"
        Object class = "TextGrid"

        xmin = 0.0124716553288
        xmax = 729.993423837
        tiers? <exists>
        size = 2
        item []:
            item [1]:
                class = "IntervalTier"
                name = "phone"
                xmin = 0.0124716553288
                xmax = 729.993423837
                intervals: size = 6819
                intervals [1]:
                    xmin = 0.0124716553288
                    xmax = 2.82607699773
                    text = "sp"
                intervals [2]:
                    xmin = 2.82607709751
                    xmax = 2.9465576552097734
                    text = "S"
                intervals [3]:
                    xmin = 2.9465576552097734
                    xmax = 3.348726528529025
        """

        def find_tiers(tier_array: np.ndarray):
            starttimes = tier_array[1::4]
            stoptimes = tier_array[2::4]
            textstrings = tier_array[3::4]

            starttimes = [float(e.strip("xmin = ").strip()) for e in starttimes]
            stoptimes = [float(e.strip("xmax = ").strip()) for e in stoptimes]
            textstrings = [e.strip("text = ").strip('"') for e in textstrings]

            assert (
                len(starttimes) == len(stoptimes) == len(textstrings)
            ), f"{len(starttimes)}, {len(stoptimes)}, {len(textstrings)}"

            return starttimes, stoptimes, textstrings

        start = float(text_grid_lines[3].strip("xmin = ").strip())
        stop = float(text_grid_lines[4].strip("xmax = ").strip())

        # find mathc using re.match
        start = float(lines[3].strip("xmin = ").strip())
        stop = float(lines[4].strip("xmax = ").strip())

        assert stop > start

        # find information about the tiers by
        # matching the specific strings in file lines
        interval_tiers = []
        tier_names = {}
        tier_n = []
        tier_starts = []
        for i, line in enumerate(lines):
            if re.match(r'class = "IntervalTier"', line):
                interval_tiers.append(i)
            if re.match(r"name = ", line):
                name = line.split('"')[1]
                tier_names[name] = i
            if re.match(r"intervals: size = ", line):
                tier_n.append(int(line.split(" ")[-1]))
                tier_starts.append(i + 1)

        # find which lines correspond to which tier
        phone_start, word_start = tier_starts
        phone_stop = phone_start + tier_n[0] * 4
        word_stop = word_start + tier_n[1] * 4
        phone_tier = np.array(lines[phone_start:phone_stop])
        word_tier = np.array(lines[word_start:word_stop])

        phones_start, phones_stop, phones = find_tiers(phone_tier)
        phone_dict = {"start": phones_start, "stop": phones_stop, "text": phones}

        words_start, words_stop, words = find_tiers(word_tier)
        word_dict = {"start": words_start, "stop": words_stop, "text": words}

    elif text_grid_ftype == TEXT_GRID_FORMATS[1]:

        """Example header:
        "Praat chronological TextGrid text file"
        0.0124716553288 819.988889088   ! Time domain.
        2   ! Number of tiers.
        "IntervalTier" "phone" 0.0124716553288 819.988889088
        "IntervalTier" "word" 0.0124716553288 819.988889088
        1 0.0124716553288 1.26961451247
        "ns"
        2 0.0124716553288 1.26961451247
        "{NS}"
        1 1.26961451247 1.48948829731937
        "S"
        2 1.26961451247 2.23741496599
        "SO"
        """

        start = float(lines[1].split()[0])
        stop = float(lines[1].split()[1])

        n_tiers = int(lines[2].split()[0])
        tiername_lines = np.arange(3, 3 + n_tiers)
        tier_names = [lines[i].split()[1].strip('"') for i in tiername_lines]

        tier_indicators = np.array([int(line.split()[0]) for line in lines[5::2]])
        tier_indicators = np.repeat(tier_indicators, 2)

        times = lines[5::2]
        words = lines[6::2]
        assert len(times) == len(
            words
        ), f"Mismatch in number of elements ({len(times)}, {len(words)})"

        phone_dict = {"start": [], "stop": [], "text": []}
        word_dict = {"start": [], "stop": [], "text": []}
        for t, w in zip(times, words):

            tier, start, stop = t.split()

            if tier == "1":
                phone_dict["start"].append(float(start))
                phone_dict["stop"].append(float(stop))
                phone_dict["text"].append(w.strip('"'))
            elif tier == "2":
                word_dict["start"].append(float(start))
                word_dict["stop"].append(float(stop))
                word_dict["text"].append(w.strip('"'))

    return {n: d for n, d in zip(tier_names, [phone_dict, word_dict])}


def load_textgrid(story: str) -> dict[str, pd.DataFrame]:

    fn = os.path.join(DATADIR, "TextGrids", f"{story}.TextGrid")

    with open(fn, "r") as f:
        lines = f.readlines()

    tiers_dict = get_tier_data(lines)

    out: dict[str, Optional[pd.DataFrame]] = {n: None for n in tiers_dict.keys()}

    out["phone"] = pd.DataFrame(tiers_dict["phone"], columns=tiers_dict["phone"].keys())
    out["word"] = pd.DataFrame(tiers_dict["word"], columns=tiers_dict["word"].keys())

    return out  # type: ignore

def load_embeddings() -> Tuple[np.ndarray, Dict]:
    """
    Load the embedding vectors and vocabulary from the EMBEDDINGS_FILE (h5py).
    """
    with h5py.File(EMBEDDINGS_FILE, "r") as f:

        # List all groups
        log.info(f"Loading: {EMBEDDINGS_FILE}")

        # Get the data
        data = np.array(f["data"])
        vocab = {e.decode("utf-8"): i for i, e in enumerate(np.array(f["vocab"]))}

        log.info(f"data shape: {data.shape}")
        log.info(f"vocab len: {len(vocab)}")

    return data, vocab

def get_embeddings(story: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load embeddings, vocabulary and word onset/offset times from the textgrid file.
    """
    vecs, vocab = load_embeddings()

    word_grid = load_textgrid(story)["word"]

    tokens = [
        row.text.lower()
        for _, row in word_grid.iterrows()
        if row.text not in SKIP_TOKENS
    ]
    starts = np.array(
        [row.start for _, row in word_grid.iterrows() if row.text not in SKIP_TOKENS]
    )
    stops = np.array(
        [row.stop for _, row in word_grid.iterrows() if row.text not in SKIP_TOKENS]
    )

    exist_tokens = [t for t in tokens if t in vocab]

    log.info(
        f"{len(exist_tokens)}/{len(tokens)} (missing {len(tokens)-len(exist_tokens)}) story tokens found in vocab."
    )

    embs = np.array(
        [vecs[:, vocab[t]] if t in vocab else np.zeros(vecs.shape[0]) for t in tokens]
    )

    return embs, starts, stops

def make_delayed(signal: np.ndarray, delays: np.ndarray, circpad=False) -> np.ndarray:
    """
    Create delayed versions of the 2-D signal.

    Parameters
    -----------
    signal : np.ndarray
        2-D array of shape (n_samples, n_features)
    delays : np.ndarray
        1-D array of delays to apply to the signal
        can be positive or negative; negative values advance the signal (shifting it backward)
    circpad : bool
        If True, use circular padding for delays
        If False, use zero padding for delays

    Returns
    --------
    np.ndarray
        2-D array of shape (n_samples, n_features * n_delays)
    """

    delayed_signals = []

    for delay in delays:
        delayed_signal = np.zeros_like(signal)
        if circpad:
            delayed_signal = np.roll(signal, delay, axis=0)
        else:
            if delay > 0:
                delayed_signal[delay:] = signal[:-delay]
            elif delay < 0:
                delayed_signal[:delay] = signal[-delay:]
            else:
                delayed_signal = signal.copy()
        delayed_signals.append(delayed_signal)

    return np.hstack(delayed_signals)


def sinc(f_c, t):
    """
    Sin function with cutoff frequency f_c.

    Parameters
    -----------
    f_c : float
        Cutoff frequency
    t : np.ndarray or float
        Time

    Returns
    --------
    np.ndarray or float
        Sin function with cutoff frequency f_c
    """
    return np.sin(np.pi * f_c * t) / (np.pi * f_c * t + 1e-6)


def lanczosfun(f_c, t, a=3):
    """
    Lanczos function with cutoff frequency f_c.

    Parameters
    -----------
    f_c : float
        Cutoff frequency
    t : np.ndarray or float
        Time
    a : int
        Number of lobes (window size), typically 2 or 3; only signals within the window will have non-zero weights.

    Returns
    --------
    np.ndarray or float
        Lanczos function with cutoff frequency f_c
    """
    val = sinc(f_c, t) * sinc(f_c, t / a)
    val[t == 0] = 1.0
    val[np.abs(t * f_c) > a] = 0.0

    return val


def lanczosinterp2D(signal, oldtime, newtime, window=3, cutoff_mult=1.0):
    """
    Lanczos interpolation for 2D signals; interpolates [signal] from [oldtime] to [newtime], assuming that the rows of [signal] correspond to [oldtime]. Returns a new signal with rows corresponding to [newtime] and the same number of columns as [signal].

    Parameters
    -----------
    signal : np.ndarray
        2-D array of shape (n_samples, n_features)
    oldtime : np.ndarray
        1-D array of old time points
    newtime : np.ndarray
        1-D array of new time points
    window : int
        Number of lobes (window size) for the Lanczos function
    cutoff_mult : float
        Multiplier for the cutoff frequency

    Returns
    --------
    np.ndarray
        2-D array of shape (len(newtime), n_features)
    """
    # Find the cutoff frequency
    f_c = 1 / (np.max(np.abs(np.diff(newtime)))) * cutoff_mult
    # Build the Lanczos interpolation matrix
    interp_matrix = np.zeros((len(newtime), len(oldtime)))
    for i, t in enumerate(newtime):
        interp_matrix[i, :] = lanczosfun(f_c, t - oldtime, a=window)
    # Interpolate the signal
    newsignal = np.dot(interp_matrix, signal)

    return newsignal

def downsample_embeddings_lanczos(
    embeddings: np.ndarray,
    starts: np.ndarray,
    stops: np.ndarray,
    n_trs: int,
    tr_len: float,
) -> np.ndarray:
    word_times = (starts + stops) / 2
    tr_times = np.arange(n_trs) * tr_len + tr_len / 2.0
    downsampled_embeddings = lanczosinterp2D(embeddings, word_times, tr_times, window=3)
    return downsampled_embeddings

import time
import logging
def counter(iterable, countevery=100, total=None, logger=logging.getLogger("counter")):
    """Logs a status and timing update to [logger] every [countevery] draws from [iterable].
    If [total] is given, log messages will include the estimated time remaining.
    """
    start_time = time.time()

    ## Check if the iterable has a __len__ function, use it if no total length is supplied
    if total is None:
        if hasattr(iterable, "__len__"):
            total = len(iterable)
    
    for count, thing in enumerate(iterable):
        yield thing
        
        if not count%countevery:
            current_time = time.time()
            rate = float(count+1)/(current_time-start_time)

            if rate>1: ## more than 1 item/second
                ratestr = "%0.2f items/second"%rate
            else: ## less than 1 item/second
                ratestr = "%0.2f seconds/item"%(rate**-1)
            
            if total is not None:
                remitems = total-(count+1)
                remtime = remitems/rate
                timestr = ", %s remaining" % time.strftime('%H:%M:%S', time.gmtime(remtime))
                itemstr = "%d/%d"%(count+1, total)
            else:
                timestr = ""
                itemstr = "%d"%(count+1)

            formatted_str = "%s items complete (%s%s)"%(itemstr,ratestr,timestr)
            if logger is None:
                print (formatted_str)
            else:
                logger.info(formatted_str)


def mult_diag(d, mtx, left=True):
    """Multiply a full matrix by a diagonal matrix.
    This function should always be faster than dot.

    Input:
      d -- 1D (N,) array (contains the diagonal elements)
      mtx -- 2D (N,N) array

    Output:
      mult_diag(d, mts, left=True) == dot(diag(d), mtx)
      mult_diag(d, mts, left=False) == dot(mtx, diag(d))
    
    By Pietro Berkes
    From http://mail.scipy.org/pipermail/numpy-discussion/2007-March/026807.html
    """
    if left:
        return (d*mtx.T).T
    else:
        return d*mtx
    
def zs(x: np.ndarray) -> np.ndarray:
    """Returns the z-score of the input array. Z-scores along the first dimension for n-dimensional arrays by default.

    Parameters
    ----------
    x : np.ndarray
        Input array. Can be 1D or n-dimensional.

    Returns
    -------
    zscore : np.ndarray
        Z-score of x.
    """
    return (x - x.mean(axis=0)) / (x.std(axis=0) + 1e-6)


def pearsonr(x1, x2) -> Union[float, np.ndarray]:
    """Returns the pearson correlation between two vectors or two matrices of the same shape (in which case the correlation is computed for each pair of column vectors).

    Parameters
    ----------
    x1 : np.ndarray
        shape = (n_samples,) or (n_samples, n_targets)
    x2 : np.ndarray
        shape = (n_samples,) or (n_samples, n_targets), same shape as x1

    Returns
    -------
    corr: float or np.ndarray
        Pearson correlation between x1 and x2. If x1 and x2 are matrices, returns an array of correlations with shape (n_targets,)
    """
    return np.mean(zs(x1) * zs(x2), axis=0)

