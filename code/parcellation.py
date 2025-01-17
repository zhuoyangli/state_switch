import os
import numpy as np
import nilearn as nl
from utils import load_config, preproc

import time

cfg = load_config()
DATADIR = cfg['DATA_DIR']
STORIES = cfg['STORIES']

for subject in ['sub-001']:
    for story in ['treasureisland']: #STORIES:
        # if os.path.exists(os.path.join(DATADIR, subject, 'func', 'roi', f'{subject}_task-{story}_roi.npz')):
        #     print(f"ROI data already exists for subject {subject} on story {story}, skipping...")
        #     continue
        if not os.path.exists(os.path.join(DATADIR, subject, 'func', f'{subject}_task-{story}_space-MNI152NLin6Asym_res-2_desc-preproc_bold.nii.gz')):
            print(f"Preprocessed BOLD data does not exist for subject {subject} on story {story}, skipping...")
            continue
        preproc(subject, story, save=True)
        print(f"Saved MVP data for subject {subject} on story {story}, {time.strftime("%y:%m:%d %H:%M:%S")}")
