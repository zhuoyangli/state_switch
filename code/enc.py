import numpy as np
from utils import load_roi, load_config, load_mp3, get_envelope, butterworth_highpass, load_surf, get_embeddings, downsample_embeddings_lanczos, make_delayed
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import zscore
from nilearn.signal import clean

cfg = load_config()
DATADIR = cfg['DATA_DIR']
STIMDIR = cfg['STIM_DIR']
FIGDIR = cfg['FIG_DIR']

subjects = ['sub-001', 'sub-003', 'sub-004']
story = 'treasureisland'

embs, starts, stops = get_embeddings(story)

from nilearn import plotting, datasets
import nilearn.surface as surface
import os
import nibabel as nib
datatype = 'roi'
if datatype == 'surf':
    # Fetch fsaverage6 dataset
    fsaverage = datasets.fetch_surf_fsaverage(mesh='fsaverage6')

    surf_data = []

    # load the label
    labels, ctab, names = nib.freesurfer.read_annot('/Users/gio/Downloads/lh.Schaefer2018_200Parcels_17Networks_order.annot')
    labels = labels.astype(int)
    names = [name.decode('utf-8') for name in names]

    tr_len = 1.5

    for subject in subjects:
        surf = load_surf(DATADIR, subject, story, 'L')
        # filtering and z-scoring
        surf_filtered = butterworth_highpass(surf, tr_len, 0.01)
        surf_zscored = (surf_filtered - np.mean(surf_filtered, axis=1, keepdims=True)) / np.std(surf_filtered, axis=1, keepdims=True)

        if subject == 'sub-001':
            if story == 'odetostepfather': # don't trim
                pass
            else:
                surf_zscored = surf_zscored[:, 10:]
        else:
            surf_zscored = surf_zscored[:, 8:]
        surf_data.append(surf_zscored)

    min_length = min([surf.shape[1] for surf in surf_data])
    surf_data = np.array([surf[:, :min_length] for surf in surf_data])
    ydata = surf_data

elif datatype == 'roi':
    roi_data_all = []
    for subject in subjects:
        roi_data, roi_labels = load_roi(DATADIR, subject, story)
        tr_len = 1.5
        # roi_data = zscore(clean(roi_data, detrend=False, standardize=False, high_pass=0.01, t_r=tr_len))
        if subject == 'sub-001':
            if story == 'odetostepfather': # don't trim
                pass
            else:
                roi_data = roi_data[:, 10:]
        else:
            roi_data = roi_data[:, 8:]
        roi_data_all.append(roi_data)

    roi_labels = [label.decode('utf-8') for label in roi_labels]

    # Trim the data to the shortest length and calculate the inter-subject correlation for each VOI
    min_length = min([roi_data.shape[1] for roi_data in roi_data_all])
    roi_data_all = np.array([roi_data[:, :min_length] for roi_data in roi_data_all])
    ydata = roi_data_all
    

X_data = downsample_embeddings_lanczos(embs, starts, stops, ydata.shape[-1], tr_len)

# so now we have X_data and surf_data aligned; trim TRs at the beginning and end that have zero embeddings
trim_start = np.where(np.sum(X_data, axis=1) != 0)[0][0]
trim_end = np.where(np.sum(X_data, axis=1) != 0)[0][-1]
X_data = X_data[trim_start:trim_end, :]
ydata = ydata[:, :, trim_start:trim_end]

# regression
# from regression import cross_validation_ridge_regression, score_correlation
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from scipy.stats import pearsonr
# zscore X_data (surface data is already zscored)
X_data = (X_data - np.mean(X_data, axis=0, keepdims=True)) / np.std(X_data, axis=0, keepdims=True)
# make delayed versions of X_data
n_delays = 4
X_data_delayed = make_delayed(X_data, np.arange(1, n_delays + 1), circpad=False)

# cross-validated ridge regression
from nilearn import datasets, image
schaefer_atlas = datasets.fetch_atlas_schaefer_2018(n_rois=200, yeo_networks=17, resolution_mm=2)
atlas_img = schaefer_atlas.maps
y_data_1 = ydata[0].T
n_splits = 5
alphas = np.logspace(1, 3, 10)
kf = KFold(n_splits=n_splits, shuffle=True)
score_all = np.zeros((n_splits, 200))
for ifold, (train, test) in enumerate(kf.split(X_data_delayed)):
    ridge = RidgeCV(alphas=alphas)
    ridge.fit(X_data_delayed[train], y_data_1[train])
    y_pred = ridge.predict(X_data_delayed[test])
    score = pearsonr(y_pred, y_data_1[test])
    # plt.hist(score.statistic, bins=20)
    # plt.show()
    # print the labels for the 10 highest scoring regions
    score_all[ifold] = score.statistic

# bin plot the scores
# plt.figure(figsize=(10, 5))
mean_score = np.mean(score_all, axis=0)
# sse_score = np.std(score_all, axis=0) / np.sqrt(n_splits)
idx = np.argsort(mean_score)[::-1]
# plt.bar(np.arange(200), mean_score[idx], yerr=sse_score[idx])
# plt.xlabel('Region')
# plt.xticks(np.arange(200), np.array(roi_labels)[idx], rotation=90)
# plt.show()
surf_labels, ctab, surf_names = nib.freesurfer.read_annot('/Users/gio/Downloads/lh.Schaefer2018_200Parcels_17Networks_order.annot')
surf_labels = surf_labels.astype(int)
surf_names = [name.decode('utf-8') for name in surf_names]

surf_score = np.zeros_like(surf_labels, dtype=float)

for ivertex, vertex_label in enumerate(surf_labels):
    if vertex_label == 0:
        continue
    roi_idx = roi_labels.index(surf_names[vertex_label])
    surf_score[ivertex] = mean_score[roi_idx]


fsaverage = datasets.fetch_surf_fsaverage(mesh='fsaverage6')
plotting.plot_surf_stat_map(fsaverage.infl_left, surf_score, hemi='left', title='Encoding model performance', colorbar=True, cmap='coolwarm')
plt.show()