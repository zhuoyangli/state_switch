# preprocess functional volumetric data

import os
import numpy as np


import tables

import nilearn

from utils import load_config

cfg = load_config()

# load data
subject = 'sub-004'

stories = cfg['STORIES']

for story in stories:
    