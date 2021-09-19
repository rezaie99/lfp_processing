import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn import mixture
import pandas as pd
import argparse

import video_processing.utils as utils

import os

path = 'F:\Anxiety_ephys'

try:
    os.chdir(path)
    print("Current working directory: {0}".format(os.getcwd()))
except FileNotFoundError:
    print("Directory: {0} does not exist".format(path))
except NotADirectoryError:
    print("{0} is not a directory".format(path))
except PermissionError:
    print("You do not have permissions to change to {0}".format(path))

#%%
ref_file = 'ref_image.npz'
file = '2021-08-02_mBWfus025_EZM_ephys/2021-08-02_mBWfus025_EZM_ephys.mp4'
parser = argparse.ArgumentParser(description='loc finder')
parser.add_argument('--file', help='input file path and name', default=file)
args = parser.parse_args()
file = args.file

mx, my, r1, r2, all_frames, total_count = utils.get_avg(file)
pos = utils.process_mice_loc(mx, my, r1, r2, all_frames, file)
utils.imshow_pos(pos, total_count, all_frames)
utils.save_to(file, pos)
plt.show()


