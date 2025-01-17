import numpy as np
from utils import load_roi, load_mvp, load_config, load_mp3, get_envelope, butterworth_highpass
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import zscore
from nilearn.signal import clean
import nibabel as nib
import os

cfg = load_config()
DATADIR = cfg['DATA_DIR']
STIMDIR = cfg['STIM_DIR']
FIGDIR = cfg['FIG_DIR']

subjects = ['sub-001', 'sub-003', 'sub-004']
story = 'treasureisland'

from nilearn import plotting, datasets

# mvp_data, roi_labels = load_mvp(DATADIR, subjects[0], story)

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
roi_data_all = [roi_data[:, :min_length] for roi_data in roi_data_all]

# Calculate the inter-subject correlation for each VOI
roi_corrs_all = np.zeros((len(roi_labels), len(subjects) * (len(subjects) - 1) // 2))
for idx, roi_label in enumerate(roi_labels):
    roi_data_subject = np.array([roi_data[idx] for roi_data in roi_data_all])
    roi_corrs = np.corrcoef(roi_data_subject)
    roi_corrs = [roi_corrs[i, j] for i in range(len(subjects)) for j in range(i + 1, len(subjects))]
    roi_corrs_all[idx] = roi_corrs

# sort the roi_corrs_all and print isc for each roi in descending order
roi_iscs = np.mean(roi_corrs_all, axis=1)
sorted_roi_corrs = np.argsort(roi_iscs)[::-1]

fsaverage = datasets.fetch_surf_fsaverage(mesh='fsaverage6')
surf_labels, ctab, surf_names = nib.freesurfer.read_annot('/Users/gio/Downloads/rh.Schaefer2018_200Parcels_17Networks_order.annot')
surf_labels = surf_labels.astype(int)
surf_names = [name.decode('utf-8') for name in surf_names]

surf_isc = np.zeros_like(surf_labels, dtype=float)
for ivertex, vertex_label in enumerate(surf_labels):
    if vertex_label == 0:
        surf_isc[ivertex] = 0
        continue
    roi_idx = roi_labels.index(surf_names[vertex_label])
    surf_isc[ivertex] = roi_iscs[roi_idx]

plotting.plot_surf_stat_map(fsaverage.infl_left, surf_isc, hemi='right', colorbar=True, cmap='coolwarm', vmax=0.3, vmin=-0.3)
# plt.show()
plt.savefig(os.path.join(FIGDIR, 'surf_isc_roi_R.png'))
