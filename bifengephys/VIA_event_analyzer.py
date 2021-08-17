import pandas as pd
import numpy as np
import os

path = 'F:\Video_analysis\VIA_manual_annotation\Results'
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
events = pd.read_csv('mBWfus021_2021081111Aug2021_18h39m10s_export.csv', header=1)

#%%
close = events[events['metadata']=='{"EZM":"Close"}'][['temporal_segment_start', 'temporal_segment_end']]
open = events[events['metadata']=='{"EZM":"Open"}'][['temporal_segment_start', 'temporal_segment_end']]

open_close = events[['temporal_segment_start', 'temporal_segment_end']]
#%%
close_start = close['temporal_segment_start'].to_numpy()
close_end = close['temporal_segment_end'].to_numpy()

open_start = open['temporal_segment_start'].to_numpy()
open_end = open['temporal_segment_end'].to_numpy()

open_close = events[['temporal_segment_start', 'temporal_segment_end']]

#%%
five_min_idx = min(np.where(open_end >= 300)[0][0], np.where(open_start >=300)[0][0])
ten_min_idx = min(np.where(open_end >= 600)[0][0], np.where(open_start >=600)[0][0])

open_time_5min = np.sum(open_end[:five_min_idx ] - open_start[:five_min_idx])
open_time_10min = np.sum(open_end[:ten_min_idx] - open_start[:ten_min_idx])
