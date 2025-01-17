# encoding model with audio envelope as regressor
import numpy as np
from scipy.ndimage import convolve1d

from utils import load_config, load_mp3, get_envelope, butterworth_highpass, load_roi, make_delayed
import matplotlib.pyplot as plt
from nilearn.glm.first_level import spm_hrf as hrf

cfg = load_config()
DATADIR = cfg['DATA_DIR']
STIMDIR = cfg['STIM_DIR']

story = 'treasureisland'
subject = 'sub-004'

# load audio data
audio, sr = load_mp3(STIMDIR, story)
audio_envelope = get_envelope(audio)

# average envelope every 0.1 seconds
window = 0.1
audio_envelope_avg = np.mean(audio_envelope[:len(audio_envelope) // int(sr * window) * int(sr * window)].reshape(-1, int(sr * window)), axis=1)
# reshape to 2d
audio_envelope_avg = audio_envelope_avg.reshape(-1, 1)
audio_envelope_zscored = (audio_envelope_avg - np.mean(audio_envelope_avg, axis=0, keepdims=True)) / np.std(audio_envelope_avg, axis=0, keepdims=True)

# load ROI data
roi_data, roi_labels = load_roi(DATADIR, subject, story)
roi_labels = [label.decode('utf-8') for label in roi_labels]

tr = 1.5
# trim the first 8 trs
roi_data = roi_data[:, 8:]

# design matrix
audio_hrf = hrf(tr=1.5, oversampling=int(1/window))
max_lag = audio_hrf.shape[0]
xdata = make_delayed(audio_envelope_avg, np.arange(max_lag))
xdata_zscored = make_delayed(audio_envelope_zscored, np.arange(max_lag))

# select rows of xdata that correspond to the same time points as the ROI data
max_roi = np.floor(audio_envelope_avg.shape[0] * window / tr).astype(int)
roi_data = roi_data[:, :max_roi]
audio_idx = np.arange(0, roi_data.shape[1] * tr / window, tr / window).astype(int)
xdata = xdata[audio_idx]
xdata_zscored = xdata_zscored[audio_idx]
print(roi_data.shape, xdata.shape)

# convolve the audio envelope with the HRF
ydata_hrf = xdata @ audio_hrf[::-1]
ydata_hrf = (ydata_hrf - np.mean(ydata_hrf)) / np.std(ydata_hrf)
ydata_hrf_zscored = xdata_zscored @ audio_hrf[::-1]

fig, ax = plt.subplots(figsize=(20, 4))
plt.plot(np.arange(0, audio_envelope_avg.shape[0] * window, window), audio_envelope_avg / np.max(audio_envelope_avg) * 4 + 8)
plt.title(f'{subject}, Story: {story}')
plt.xlabel('Time (s)')
plt.plot(np.arange(0, ydata_hrf.shape[0] * tr, tr), ydata_hrf + 4)

auditory_rois = ['17Networks_LH_SomMotB_Aud_2', '17Networks_RH_SomMotB_Aud_1', '17Networks_RH_SomMotB_Aud_2', '17Networks_LH_SomMotB_Aud_1']
auditory_roi_idx = np.array([roi_labels.index(roi) for roi in auditory_rois])
offset = 4
for i, idx in enumerate(auditory_roi_idx):
    plt.plot(np.arange(0, roi_data.shape[1] * tr, tr), roi_data[idx] - offset * i)

plt.yticks([-12, -8, -4, 0, 4, 8], [auditory_rois[3], auditory_rois[2], auditory_rois[1], auditory_rois[0], 'Audio envelope convolved with HRF', 'Audio envelope'])
plt.xlim([0, 552 * tr])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.show()

print('Correlation between audio envelope and auditory ROIs:')
for idx in auditory_roi_idx:
    print(f'{roi_labels[idx]}: {np.corrcoef(ydata_hrf, roi_data[idx])[0, 1]}')

# plt.figure(figsize=(20, 10))
# plt.subplot(4, 1, 1)
# plt.plot(np.correlate(ydata_hrf, roi_data[auditory_roi_idx[0]], mode='full'))
# plt.title(f'Cross-correlation between {auditory_rois[0]} and audio envelope')
# plt.subplot(4, 1, 2)
# plt.plot(np.correlate(ydata_hrf, roi_data[auditory_roi_idx[1]], mode='full'))
# plt.title(f'Cross-correlation between {auditory_rois[1]} and audio envelope')
# plt.subplot(4, 1, 3)
# plt.plot(np.correlate(ydata_hrf, roi_data[auditory_roi_idx[2]], mode='full'))
# plt.title(f'Cross-correlation between {auditory_rois[2]} and audio envelope')
# plt.subplot(4, 1, 4)
# plt.plot(np.correlate(ydata_hrf, roi_data[auditory_roi_idx[3]], mode='full'))
# plt.title(f'Cross-correlation between {auditory_rois[3]} and audio envelope')
# plt.tight_layout()
# plt.show()


# # encoding model
# from sklearn.linear_model import RidgeCV
# from sklearn.model_selection import KFold
# from scipy.stats import pearsonr

# xdata = (xdata - np.mean(xdata, axis=0, keepdims=True)) / np.std(xdata, axis=0, keepdims=True)

# n_folds = 5
# kf = KFold(n_splits=n_folds, shuffle=True)
# score_all = np.zeros((roi_data.shape[0], n_folds))
# for ifold, (train, test) in enumerate(kf.split(roi_data.T)):
#     ridge = RidgeCV(alphas=np.logspace(-3, 3, 10))
#     ridge.fit(xdata[train], roi_data[:, train].T)
#     predictions = ridge.predict(xdata[test]).T
#     score = pearsonr(predictions, roi_data[:, test], axis=1)
#     score_all[:, ifold] = score.statistic

# mean_score = np.mean(score_all, axis=1)
# sorted_idx = np.argsort(mean_score)[::-1]
# print('Top 10 ROIs:')
# for idx in sorted_idx[:10]:
#     print(f'{roi_labels[idx]}: {mean_score[idx]}')