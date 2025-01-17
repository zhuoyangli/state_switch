import numpy as np
from utils import load_roi, load_mvp, load_config, load_mp3, get_envelope, butterworth_highpass
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import zscore
from nilearn.signal import clean

cfg = load_config()
DATADIR = cfg['DATA_DIR']
STIMDIR = cfg['STIM_DIR']
FIGDIR = cfg['FIG_DIR']

subjects = ['sub-001', 'sub-003', 'sub-004']
story = 'swimming'

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
for idx in sorted_roi_corrs:
    print(f"{roi_labels[idx]}: {np.mean(roi_corrs_all[idx])}")

# plot the top 10 regions with the highest inter-subject correlation
from nilearn import plotting, datasets, image
from nilearn.image import mean_img

top_vois = sorted_roi_corrs[:10]

# Load the Schaefer atlas
schaefer_atlas = datasets.fetch_atlas_schaefer_2018(n_rois=200, yeo_networks=17, resolution_mm=2)
atlas_img = schaefer_atlas.maps

# # Plot the top 10 regions, each in a separate plot, with the region name, story name, and the ISC value as the title
# for idx in top_vois:
#     plotting.plot_roi(image.math_img(f'img == {idx + 1}', img=atlas_img), title=f"Story: {story}\n ROI: {roi_labels[idx]}\n ISC: {np.mean(roi_corrs_all[idx]):.2f} ({roi_corrs_all[idx][0]:.2f}, {roi_corrs_all[idx][1]:.2f}, {roi_corrs_all[idx][2]:.2f})")
#     plt.show()
#     # save figure
#     # plt.savefig(f"{FIGDIR}/isc/{story}_{roi_labels[idx]}_isc.png")

# plot the subject-wise time series for the top 10 regions along with the stimulus
offset_unit = 4
story_mp3, sr = load_mp3(STIMDIR, story)
envelope = get_envelope(story_mp3)
envelope = envelope / np.max(envelope) * offset_unit

for idx in top_vois:
    plt.figure(figsize=(16, 8))
    gs = gridspec.GridSpec(2, 1, height_ratios=[5, 3])
    ax1 = plt.subplot(gs[0])
    plotting.plot_roi(image.math_img(f'img == {idx + 1}', img=atlas_img), title=f"Story: {story}\n ROI: {roi_labels[idx]}\n ISC: {np.mean(roi_corrs_all[idx]):.2f} ({roi_corrs_all[idx][0]:.2f}, {roi_corrs_all[idx][1]:.2f}, {roi_corrs_all[idx][2]:.2f})", axes=ax1)

    ax2 = plt.subplot(gs[1])
    # plot the stimulus as the first channel at the bottom
    ax2.plot(np.linspace(0, len(envelope)/sr, len(envelope)), envelope - offset_unit, label='Stimulus envelope')
    for subject in range(len(subjects)):
        ax2.plot(np.linspace(0, min_length * tr_len, min_length), roi_data_all[subject][idx] + subject * offset_unit, label=f'{subjects[subject]}')
    ax2.set_xlabel('Time (s)')
    ax2.set_xlim([0, min_length * tr_len])
    # remove boxes around the plot
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.set_yticks(np.linspace(-offset_unit, offset_unit * (len(subjects) - 1), len(subjects) + 1))
    ax2.set_yticklabels(['Stim envelope'] + subjects)
    # ax2.set_title(f"Story: {story}\n ROI: {roi_labels[idx]}\n ISC: {np.mean(roi_corrs_all[idx]):.2f} ({roi_corrs_all[idx][0]:.2f}, {roi_corrs_all[idx][1]:.2f}, {roi_corrs_all[idx][2]:.2f})")
    
    plt.tight_layout()
    # plt.show()
    # save figure
    plt.savefig(f"{FIGDIR}/isc/{story}_{roi_labels[idx]}_time_series.png")

# # for the top 10 regions, plot the cross-correlation between the time series of each pair of subjects
# intercept = 1
# for idx in top_vois:
#     plt.figure(figsize=(12, 6))
#     for i in range(len(subjects)):
#         for j in range(i + 1, len(subjects)):
#             xcorr = np.correlate(roi_data_all[i][idx], roi_data_all[j][idx], mode='full')
#             xcorr = xcorr/ np.max(xcorr)
#             plt.plot(np.linspace(-len(xcorr) / 2, len(xcorr) / 2, len(xcorr)), xcorr + (i * intercept) + (j * intercept) - intercept, label=f'{subjects[i]} vs {subjects[j]}')
#             plt.vlines(0, -intercept, intercept*3, color='k', linestyle='--')
#     plt.xlabel('Lag')
#     plt.yticks([0, 1, 2], subjects)
#     plt.ylim([-intercept, intercept * len(subjects) * (len(subjects) - 1) / 2])
#     plt.title(f"Story: {story}\n ROI: {roi_labels[idx]}\n ISC: {np.mean(roi_corrs_all[idx]):.2f} ({roi_corrs_all[idx][0]:.2f}, {roi_corrs_all[idx][1]:.2f}, {roi_corrs_all[idx][2]:.2f})")
#     plt.show()
#     # save figure
#     # plt.savefig(f"{FIGDIR}/isc/{story}_{roi_labels[idx]}_xcorr.png")