#%%
import sys
import scipy.stats as stats
from scipy.signal import coherence
from scipy.cluster.hierarchy import fcluster
from sklearn import metrics
from sklearn.linear_model import LinearRegression

import bifengephys.behavior as behav

import numpy as np
from scipy.cluster import hierarchy
from sklearn import cluster

import matplotlib as mpl
import matplotlib.pyplot as plt
# mpl.use('TkAgg')  # or can use 'TkAgg', 'Qt5Agg' whatever you have/prefer
import seaborn as sns
# from lets_plot import *

import glob
import re  # Regular expression operations
import pandas as pd
from scipy import signal
import mne
from tqdm import tqdm
import os
from os.path import exists
import pickle
import statsmodels.stats.api as sms

import bifengephys.utils as utils
import bifengephys.ephys as ephys
import bifengephys.plotting as plotting

plt.rcParams["axes.labelsize"] = 18
plt.rcParams["xtick.labelsize"] = 14
plt.rcParams["ytick.labelsize"] = 14
plt.rcParams["axes.labelsize"] = 18
plt.rcParams["font.size"] = 7
plt.rcParams["font.family"] = "Arial"
plt.rcParams["lines.linewidth"] = 2



sys.path.append('D:\ephys')
#%%
import os

path = 'D:\ephys'

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

mBWfus009 = {
    'arena_0219': '2021-02-19_mBWfus009_arena_ephys',
    'ezm_0219': '2021-02-19_mBWfus009_EZM_ephys',
    'oft_0219': '2021-02-19_mBWfus009_OF_ephys',

    'arena_0226': '2021-02-26_mBWfus009_arena_ephys',
    'ezm_0226': '2021-02-26_mBWfus009_EZM_ephys',
    'oft_0226': '2021-02-26_mBWfus009_OF_ephys',

    'arena_0305_bef': '2021-03-05_mBWfus009_before_arena_ephys',
    'cage_0305': '2021-03-05_mBWfus009_cage_arena_ephys',
    'arena_0305_aft': '2021-03-05_mBWfus009_after_arena_ephys',
    'ezm_0305': '2021-03-05_mBWfus009_EZM_ephys',
    'oft_0305': '2021-03-05_mBWfus009_OF_ephys',

    'arena_0325': '2021-03-25_mBWfus009_arena_ephys',
    'epm_0325': '2021-03-25_mBWfus009_EPM_ephys'
}

mBWfus008 = {
    'arena_0219': '2021-02-19_mBWfus008_arena_ephys',
    'ezm_0219': '2021-02-19_mBWfus008_EZM_ephys',
    'oft_0219': '2021-02-19_mBWfus008_OF_ephys',

    'arena_0226': '2021-02-26_mBWfus008_arena_ephys',
    'ezm_0226': '2021-02-26_mBWfus008_EZM_ephys',
    'oft_0226': '2021-02-26_mBWfus008_OF_ephys',

    'arena_0305_bef': '2021-03-05_mBWfus008_before_arena_ephys',
    'cage_0305': '2021-03-05_mBWfus008_cage_arena_ephys',
    'arena_0305_aft': '2021-03-05_mBWfus008_after_arena_ephys',
    'ezm_0305': '2021-03-05_mBWfus008_EZM_ephys',
    'oft_0305': '2021-03-05_mBWfus008_OF_ephys',
}

mBWfus010 = {
    'arena_0219': '2021-02-19_mBWfus010_arena_ephys',
    'ezm_0219': '2021-02-19_mBWfus010_EZM_ephys',
    'oft_0219': '2021-02-19_mBWfus010_OF_ephys',

    'arena_0301_aft': '2021-03-01_mBWfus010_arena_ephys_after',
    'arena_0301_bef': '2021-03-01_mBWfus010_arena_ephys_before',
    'cage_0301': '2021-03-01_mBWfus010_cage_ephys',
    'oft_0301': '2021-03-01_mBWfus010_OF_ephys',
    'ezm_0301': '2021-03-01_mBWfus010_EZM_ephys',

    'arena_0307_bef': '2021-03-07_mBWfus010_after_arena_ephys',
    'cage_0307': '2021-03-07_mBWfus010_cage_arena_ephys',
    'arena_0307_aft': '2021-03-07_mBWfus010_after_arena_ephys',
    'ezm_0307': '2021-03-07_mBWfus010_EZM_ephys',
    'oft_0307': '2021-03-07_mBWfus010_OF_ephys',
}

mBWfus011 = {
    'arena_0226': '2021-02-26_mBWfus011_arena_ephys',
    'ezm_0226': '2021-02-26_mBWfus011_EZM_ephys',
    'oft_0226': '2021-02-26_mBWfus011_OF_ephys',

    'arena_0305_aft': '2021-03-05_mBWfus011_after_arena_ephys',
    'cage_0305': '2021-03-05_mBWfus011_cage_arena_ephys',
    'arena_0305_bef': '2021-03-05_mBWfus011_before_arena_ephys',
    'oft_0305': '2021-03-05_mBWfus011_OF_ephys',
    'ezm_0305': '2021-03-05_mBWfus011_EZM_ephys',

    'arena_0313_bef': '2021-03-13_mBWfus011_before_arena_ephys',
    'cage_0313': '2021-03-13_mBWfus011_cage_arena_ephys',
    'arena_0313_aft': '2021-03-13_mBWfus011_after_arena_ephys',
    'ezm_0313': '2021-03-13_mBWfus011_EZM_ephys',
    'oft_0313': '2021-03-13_mBWfus011_OF_ephys',
}

mBWfus012 = {
    'arena_0226': '2021-02-26_mBWfus012_arena_ephys',
    'ezm_0226': '2021-02-26_mBWfus012_EZM_ephys',
    'oft_0226': '2021-02-26_mBWfus012_OF_ephys',
}

### Behavioral related processing
#%%
animal = mBWfus009
session = 'ezm_0219'

loc, scorer = behav.load_location(animal[session])
loc = behav.calib_location(loc)
loc = behav.get_locomotion(loc)

rois_stats, transitions = behav.analyze_trajectory_ezm(loc, bd='shoulder', start_time=0, duration=600, fps=50)

#%%
## plot speed of different body parts [head, shoulder, tail]
fps_v = 50
pixel2cm = 0.16 ## 400 pixels = 65 cm
t = np.arange(0, len(loc), 1/fps_v)
start = 60*fps_v
win = 60
end = start + win*fps_v
bds = loc.columns.levels[1].to_list()

for bd in bds:
    plt.plot(t[start:end], loc[scorer, bd, 'avgspd'][start:end]*pixel2cm, label=bd)

plt.legend(loc='upper right')
plt.xlabel('Time sec')
plt.ylabel('Speed cm/s')
plt.show()

#%%
## use the mean speed of three body parts and generate a plot
avgspd = []

for bd in bds:
    avgspd.append(loc[scorer, bd, 'avgspd'])

avgspd = np.array(avgspd)
avgspd = np.mean(avgspd, axis=0)

## plot the mean speed of three body parts: head, shoulder, tails
pixel2cm = 0.16 ## 400 pixels = 65 cm
fps_v = 50
start = 40*fps_v
win = 600
end = start + win*fps_v
d = avgspd[start:end]*pixel2cm
t = np.arange(0, len(loc), 1/fps_v)[start:end]


plt.plot(t, d)
plt.xlabel('Time sec')
plt.ylabel('Speed cm/s')
plt.show()

#%%

#%%


#%%
## plot the distribution of the locomation speed
pixel2cm = 0.16 ## 400 pixels = 65 cm
start = 20*fps_v
win = 600
end = start + win*fps_v
d = avgspd[start:end]*pixel2cm

plt.figure(figsize=(8, 6))
n, x, _ = plt.hist(d, bins =100,
                   histtype=u'step')

plt.xlabel('Speed cm/s')
plt.ylabel('Counts')
plt.xlim(-5, 40)
plt.show()

#%%
## check the statits of EZM-related events
wanted = ['ROI_name', 'cumulative_time_in_roi_sec', 'avg_time_in_roi_sec', 'avg_vel_in_roi']
ezm_stats = {key: rois_stats[key] for key in wanted}
ezm_stats = pd.DataFrame.from_dict(ezm_stats)
# ezm_stats = ezm_stats.drop([1,3])
ezm_stats

#%%
animal = mBWfus009


# arena0218_trigger = 2.5 # LED off, use this for synchronizing video and ephys recording. TTL trigger in the ephys data can be extracted by get_trigger()
# arena0219_trigger = 6.44
# ezm0219_trigger = 14.24

#%%
open_idx, close_idx, OTC_idx, prOTC_idx, prCTO_idx, nosedip_idx = get_events(events_ezm, ezm_trigger, 900)

event_dict = dict(OTC = 1, prOTC= 2, prCTO = 3, nosedip = 4)

OTC = idx_to_events(OTC_idx, 1)
prOTC = idx_to_events(prOTC_idx, 2)
prCTO = idx_to_events(prCTO_idx, 3)
nosedip = idx_to_events(nosedip_idx, 4)

mne_events = merge_events(OTC, prOTC, prCTO, nosedip)

plt.plot(mne_events[:, 0])
plt.show()

##### Ephys analysis

#%%
date = '0219'

# data_arena_bef = load_data(animal['arena_' + date + '_bef'])

# data_arena = load_data(animal['arena_' + date + '_aft'])

data_arena = ephys.load_data(animal['arena_' + date])

data_ezm = ephys.load_data(animal['ezm_' + date])

data_oft = ephys.load_data(animal['oft_' + date])

#%%
### Create MNE raw object
lfp = ephys.column_by_pad(ephys.get_lfp(data_arena, 'all'))
print(lfp.columns)

lfp = ephys.column_by_pad(ephys.get_lfp(data_ezm, 'all'))
print(lfp.columns)

lfp = ephys.column_by_pad(ephys.get_lfp(data_oft, 'all'))
print(lfp.columns)

#%%

corr_matrix = lfp.iloc[50000:100000].corr() # 10 second

plt.imshow(corr_matrix)
cb = plt.colorbar()
plt.xticks(range(corr_matrix.select_dtypes(['number']).shape[1]), corr_matrix.select_dtypes(['number']).columns, fontsize=8, rotation=90)
plt.yticks(range(corr_matrix.select_dtypes(['number']).shape[1]), corr_matrix.select_dtypes(['number']).columns, fontsize=8)
plt.show()

#%%
# bad_ch = [ 23, 24, 25, 26, 27, 28, 31, 38] # mBWfus008
# bad_ch = [ 0, 12, 23, 24, 25, 26, 27, 28, 31, 35, 39, ] mBWfus008


bad_ch = [7, 10, 23, 44, 57, 58, 59, 60, 61, 62] ### mBWfus009 arena_0219
# bad_ch = [6, 7, 10, 23, 44, 57, 58, 59] ### mBWfus009 arena_0226
# bad_ch = [7, 10, 23, 44, 57, 58, 59, 60, 61, 62] ### mBWfus009 arena_0305

# bad_ch = [27, 28, 31, 32, 46, 52] # mBWfus011 arena_0226,0305, 0313

# bad_ch = [3, 9, 14, 15, 16, 17, 18, 19, 20, 21, 22,
#           23, 24, 25, 26, 27, 28, 31, 34, 35, 36,
#           38, 40, 41, 42, 43, 44, 45, 46, 60, 61, 62]

#%%




#%%
lfp = ephys.column_by_pad(ephys.get_lfp(data_arena, 'all'))

bad_ch = [7, 10, 23, 44, 46, 57, 58, 59, 60, 61, 62]

lfp = lfp.drop(labels=bad_ch, axis=1)
print(lfp.shape)

ch_list= [str(el) for el in lfp.columns.tolist()]
print(len(ch_list), ch_list)
n_channels = len(ch_list)
sampling_freq = 500  # in Hertz
info = mne.create_info(n_channels, sfreq=sampling_freq)

ch_types = ['ecog'] * n_channels
info = mne.create_info(ch_list, ch_types=ch_types, sfreq=sampling_freq)
info['bads'] = []  # Names of bad channels
print(info)

lfp_filted = mne.filter.filter_data(data=lfp.T, sfreq=500, l_freq=1, h_freq=None)

raw_arena = mne.io.RawArray(lfp_filted*1e-6, info) ### mne data format (n-channels, n-samples), unit = V

#%%
raw_arena.plot(n_channels = 46, duration=2, scalings='auto')

#%%

lfp = ephys.column_by_pad(ephys.get_lfp(data_ezm, 'all'))

bad_ch = [6, 7, 10, 23, 44, 46, 57, 58, 59, 60, 61, 62]

lfp = lfp.drop(labels=bad_ch, axis=1)
print(lfp.shape)

ch_list= [str(el) for el in lfp.columns.tolist()]
print(len(ch_list), ch_list)
n_channels = len(ch_list)
sampling_freq = 500  # in Hertz
info = mne.create_info(n_channels, sfreq=sampling_freq)

ch_types = ['ecog'] * n_channels
info = mne.create_info(ch_list, ch_types=ch_types, sfreq=sampling_freq)
info['bads'] = []  # Names of bad channels
print(info)

lfp_filted = mne.filter.filter_data(data=lfp.T, sfreq=500, l_freq=1, h_freq=None)

raw_ezm = mne.io.RawArray(lfp_filted*1e-6, info) ### mne data format (n-channels, n-samples)

#%%
lfp = ephys.column_by_pad(ephys.get_lfp(data_oft, 'all'))

bad_ch = [7, 10, 23, 44, 46, 57, 58, 59, 60, 61, 62]

lfp = lfp.drop(labels=bad_ch, axis=1)
print(lfp.shape)

ch_list= [str(el) for el in lfp.columns.tolist()]
print(len(ch_list), ch_list)
n_channels = len(ch_list)
sampling_freq = 500  # in Hertz
info = mne.create_info(n_channels, sfreq=sampling_freq)

ch_types = ['ecog'] * n_channels
info = mne.create_info(ch_list, ch_types=ch_types, sfreq=sampling_freq)
info['bads'] = []  # Names of bad channels
print(info)

lfp_filted = mne.filter.filter_data(data=lfp.T, sfreq=500, l_freq=1, h_freq=None)

raw_oft = mne.io.RawArray(lfp_filted*1e-6, info) ### mne data format (n-channels, n-samples)


#%%
''''
mne.filter.filter_data(data, sfreq, l_freq, h_freq, picks=None, filter_length='auto', 
l_trans_bandwidth='auto', h_trans_bandwidth='auto', n_jobs=1, method='fir', iir_params=None, 
copy=True, phase='zero', fir_window='hamming', fir_design='firwin', pad='reflect_limited', verbose=None)
'''

### extract theta power using Hilbert transform
### epoch the power into 2.6 second segments

sfreq = 500
l_freq_t = 7
h_freq_t = 12

l_freq_g = 40
h_freq_g = 100


# tRaw_arena_bef = raw_arena_bef.filter(l_freq, h_freq)
# tRaw_arena_bef.apply_hilbert(envelope=True)
# tpower_arena_bef_seg = mne.make_fixed_length_epochs(tRaw_arena_bef, duration=2.6)

t_arena = raw_arena.copy()
g_arena = raw_arena.copy()

t_arena = t_arena.filter(l_freq_t, h_freq_t, phase='zero-double')
t_arena.apply_hilbert(envelope=True)
tpower_arena_seg = mne.make_fixed_length_epochs(t_arena, duration=2.6)

g_arena = g_arena.filter(l_freq_g, h_freq_g, phase='zero-double')
g_arena.apply_hilbert(envelope=True)
gpower_arena_seg = mne.make_fixed_length_epochs(g_arena, duration=2.6)

t_ezm = raw_ezm.copy()
g_ezm = raw_ezm.copy()

t_ezm = t_ezm.filter(l_freq_t, h_freq_t, phase='zero-double')
t_ezm.apply_hilbert(envelope=True)
tpower_ezm_seg = mne.make_fixed_length_epochs(t_ezm, duration=2.6)

g_ezm = g_ezm.filter(l_freq_g, h_freq_g, phase='zero-double')
g_ezm.apply_hilbert(envelope=True)
gpower_ezm_seg = mne.make_fixed_length_epochs(g_ezm, duration=2.6)

t_oft = raw_oft.copy()
g_oft = raw_oft.copy()

t_oft = t_oft.filter(l_freq_t, h_freq_t, phase='zero-double')
t_oft.apply_hilbert(envelope=True)
tpower_oft_seg = mne.make_fixed_length_epochs(t_oft, duration=2.6)

g_oft = g_oft.filter(l_freq_g, h_freq_g, phase='zero-double')
g_oft.apply_hilbert(envelope=True)
gpower_oft_seg = mne.make_fixed_length_epochs(g_oft, duration=2.6)

#%%
t_arena.plot(n_channels = 46, duration=5, scalings='auto')

#%%
#mBWfus008 channels in mPFC [0, 21]; ch in the vHPC [21, 42]

#mBWfus009 channels in mPFC [0, 23; ch in the vHPC [23, 45]

#mBWfus011 channels in mPFC [0, 29]; ch in the vHPC [29, 58]

#mBWfus012 channels in mPFC [0, 9]: ch in the vHPC [9, 18]

#%%
### extract theta power from NME.epock to arrays

mpfc_ch = np.arange(0, 23).tolist()
vhipp_ch = np.arange(23, 45).tolist()

# tpower_arena_bef_seg_mpfc = tpower_arena_bef_seg.get_data(picks=mpfc_ch)
# tpower_arena_bef_seg_vhipp = tpower_arena_bef_seg.get_data(picks=vhipp_ch)

tpower_arena_seg_mpfc = tpower_arena_seg.get_data(picks=mpfc_ch)
tpower_arena_seg_vhipp = tpower_arena_seg.get_data(picks=vhipp_ch)

tpower_ezm_seg_mpfc = tpower_ezm_seg.get_data(picks=mpfc_ch)
tpower_ezm_seg_vhipp = tpower_ezm_seg.get_data(picks=vhipp_ch)

tpower_oft_seg_mpfc = tpower_oft_seg.get_data(picks=mpfc_ch)
tpower_oft_seg_vhipp = tpower_oft_seg.get_data(picks=vhipp_ch)

#%%

gpower_arena_seg_mpfc = gpower_arena_seg.get_data(picks=mpfc_ch)
gpower_arena_seg_vhipp = gpower_arena_seg.get_data(picks=vhipp_ch)

gpower_ezm_seg_mpfc = gpower_ezm_seg.get_data(picks=mpfc_ch)
gpower_ezm_seg_vhipp = gpower_ezm_seg.get_data(picks=vhipp_ch)

gpower_oft_seg_mpfc = gpower_oft_seg.get_data(picks=mpfc_ch)
gpower_oft_seg_vhipp = gpower_oft_seg.get_data(picks=vhipp_ch)

#%%
## compute the sum of theta power in 2.6 second segments

# tpower_arena_bef_seg_mpfc_sum = np.sum(tpower_arena_bef_seg_mpfc, axis=2)
# tpower_arena_bef_seg_vhipp_sum = np.sum(tpower_arena_bef_seg_vhipp, axis=2)

tpower_arena_seg_mpfc_sum = np.sum(tpower_arena_seg_mpfc, axis=2)
tpower_arena_seg_vhipp_sum = np.sum(tpower_arena_seg_vhipp, axis=2)

tpower_ezm_seg_mpfc_sum = np.sum(tpower_ezm_seg_mpfc, axis=2)
tpower_ezm_seg_vhipp_sum = np.sum(tpower_ezm_seg_vhipp, axis=2)

tpower_oft_seg_mpfc_sum = np.sum(tpower_oft_seg_mpfc, axis=2)
tpower_oft_seg_vhipp_sum = np.sum(tpower_oft_seg_vhipp, axis=2)

#%%

gpower_arena_seg_mpfc_sum = np.sum(gpower_arena_seg_mpfc, axis=2)
gpower_arena_seg_vhipp_sum = np.sum(gpower_arena_seg_vhipp, axis=2)

gpower_ezm_seg_mpfc_sum = np.sum(gpower_ezm_seg_mpfc, axis=2)
gpower_ezm_seg_vhipp_sum = np.sum(gpower_ezm_seg_vhipp, axis=2)

gpower_oft_seg_mpfc_sum = np.sum(gpower_oft_seg_mpfc, axis=2)
gpower_oft_seg_vhipp_sum = np.sum(gpower_oft_seg_vhipp, axis=2)

#%%
### create dataframe for correlation matrix

df_arena_mpfc = pd.DataFrame(tpower_arena_seg_mpfc_sum)
df_arena_vhipp = pd.DataFrame(tpower_arena_seg_vhipp_sum)

df_ezm_mpfc = pd.DataFrame(tpower_ezm_seg_mpfc_sum)
df_ezm_vhipp = pd.DataFrame(tpower_ezm_seg_vhipp_sum)

# correlate one channel in vHPC with all the channels in the mPFC
corr_mpfc = df_arena_mpfc.corrwith(df_arena_vhipp[8])
print('One vHPC channel against all the channels in mPFC = ', corr_mpfc)

# correlate one channel in mPFC with all the channels in the vHPC
corr_vHPC = df_arena_vhipp.corrwith(df_arena_mpfc[5])
print('One mPFC channel against all the channels in vHPC = ', corr_vHPC)

#%%
##mBWfus009_0219, BWfus009_0305
remv_ch_mpfc = []
remv_ch_vhipp= [0, 1, 2, 3, 4, 5, 6]

##mBWfus008_0219
# rev_ch_mpfc = []
# rev_ch_vhipp= []

##mBWfus010_0226
# rev_ch_mpfc = [27, 28]
# rev_ch_vhipp= [21, 22,23,24,25, 26, 27, 28]


# mBWfus012_0226
# rev_ch_mpfc = []
# rev_ch_vhipp= []

#%%
### remove the channels showing low correlation of theta power (threshold = 0.2 (Pearson coefficient))

print(tpower_arena_seg_mpfc_sum.shape, tpower_arena_seg_vhipp_sum.shape)

# tpower_arena_bef_seg_mpfc_sum = np.delete(tpower_arena_bef_seg_mpfc_sum, rev_ch_mpfc, axis=1)
# tpower_arena_bef_seg_vhipp_sum = np.delete(tpower_arena_bef_seg_vhipp_sum, rev_ch_vhipp, axis=1)

tpower_arena_seg_mpfc_sum = np.delete(tpower_arena_seg_mpfc_sum, remv_ch_mpfc, axis=1)
tpower_arena_seg_vhipp_sum = np.delete(tpower_arena_seg_vhipp_sum, remv_ch_vhipp, axis=1)

tpower_ezm_seg_mpfc_sum = np.delete(tpower_ezm_seg_mpfc_sum, remv_ch_mpfc, axis=1)
tpower_ezm_seg_vhipp_sum = np.delete(tpower_ezm_seg_vhipp_sum, remv_ch_vhipp, axis=1)

tpower_oft_seg_mpfc_sum = np.delete(tpower_oft_seg_mpfc_sum, remv_ch_mpfc, axis=1)
tpower_oft_seg_vhipp_sum = np.delete(tpower_oft_seg_vhipp_sum, remv_ch_vhipp, axis=1)


print(tpower_arena_seg_mpfc_sum.shape, tpower_arena_seg_vhipp_sum.shape)

#%%

### remove the channels showing low correlation of theta power (threshold = 0.2 (Pearson coefficient))

print(gpower_arena_seg_mpfc_sum.shape, gpower_arena_seg_vhipp_sum.shape)

# gpower_arena_bef_seg_mpfc_sum = np.delete(gpower_arena_bef_seg_mpfc_sum, rev_ch_mpfc, axis=1)
# gpower_arena_bef_seg_vhipp_sum = np.delete(gpower_arena_bef_seg_vhipp_sum, rev_ch_vhipp, axis=1)

gpower_arena_seg_mpfc_sum = np.delete(gpower_arena_seg_mpfc_sum, remv_ch_mpfc, axis=1)
gpower_arena_seg_vhipp_sum = np.delete(gpower_arena_seg_vhipp_sum, remv_ch_vhipp, axis=1)

gpower_ezm_seg_mpfc_sum = np.delete(gpower_ezm_seg_mpfc_sum, remv_ch_mpfc, axis=1)
gpower_ezm_seg_vhipp_sum = np.delete(gpower_ezm_seg_vhipp_sum, remv_ch_vhipp, axis=1)

gpower_oft_seg_mpfc_sum = np.delete(gpower_oft_seg_mpfc_sum, remv_ch_mpfc, axis=1)
gpower_oft_seg_vhipp_sum = np.delete(gpower_oft_seg_vhipp_sum, remv_ch_vhipp, axis=1)


print(gpower_arena_seg_mpfc_sum.shape, gpower_arena_seg_vhipp_sum.shape)

#%%
# Plot theta power of individual channel in the mPFc

for i in range(tpower_arena_seg_mpfc_sum.shape[1]):
    plt.plot(tpower_arena_seg_mpfc_sum[20:60, i])

plt.xlabel('Time segments 2.6s')
plt.ylabel('Theta power')
plt.title('All channels in mPFC')
plt.show()

#%%
# Plot theta power of individual channel in the vHPC

for i in range(tpower_arena_seg_vhipp_sum.shape[1]):
    plt.plot(tpower_arena_seg_vhipp_sum[20:60, i])

plt.xlabel('Time segments 2.6s')
plt.ylabel('Theta power')
plt.title('All channels in vHPC')
plt.show()

#%%
# Plot gamma power of individual channel in the mPFc

for i in range(gpower_arena_seg_mpfc_sum.shape[1]):
    plt.plot(gpower_arena_seg_mpfc_sum[20:60, i])

plt.xlabel('Time segments 2.6s')
plt.ylabel('Gamma power')
plt.title('All channels in mPFC')
plt.show()

#%%
# Plot gamma power of individual channel in the vHPC

for i in range(gpower_arena_seg_vhipp_sum.shape[1]):
    plt.plot(gpower_arena_seg_vhipp_sum[20:60, i])

plt.xlabel('Time segments 2.6s')
plt.ylabel('Gamma power')
plt.title('All channels in vHPC')
plt.show()
#%%
# Plot mean theta power of all channels in the vHPC
m = tpower_arena_seg_vhipp_sum.mean(axis=1)[20:60]
sd = np.std(tpower_arena_seg_vhipp_sum, axis=1)[20:60]
x = np.arange(tpower_arena_seg_vhipp_sum.shape[0])[20:60]

plt.plot(x, m)
plt.fill_between(x, m+sd, m-sd, alpha=0.6)
# plt.plot(tpower_arena0218_seg_vhipp_sum.mean(axis=1), label='vHPC_arena')
# plt.legend()
plt.xlabel('Time segments 2.6s')
plt.ylabel('Theta power')
plt.title('All channel in vHPC')
plt.show()

#%%
# Plot mean theta power of all channels in the mPFC

m = tpower_arena_seg_mpfc_sum.mean(axis=1)[20:60]
sd = np.std(tpower_arena_seg_mpfc_sum, axis=1)[20:60]
x = np.arange(tpower_arena_seg_mpfc_sum.shape[0])[20:60]

plt.plot(x, m)
plt.fill_between(x, m+sd, m-sd, alpha=0.6)
# plt.legend()
plt.xlabel('Time segments 2.6s')
plt.ylabel('Theta power')
plt.title('All channel in mPFC')
plt.show()

#%%
# plt.plot(tpower_arena_bef_seg_mpfc_sum.mean(axis=1)[20:120], label='mPFC_arena_bef')
# plt.plot(tpower_arena_bef_seg_vhipp_sum.mean(axis=1)[20:120], label='vHPC_arena_bef')
# plt.legend()
# plt.xlabel('Time segments 2.6s')
# plt.ylabel('Theta power')
# plt.show()

plt.plot(tpower_arena_seg_mpfc_sum.mean(axis=1)[20:120], label='mPFC_Arena')
plt.plot(tpower_arena_seg_vhipp_sum.mean(axis=1)[20:120], label='vHPC_Arena')
plt.legend()
plt.xlabel('Time segments 2.6s')
plt.ylabel('Theta power')
plt.title('Arena')
plt.show()

plt.plot(tpower_ezm_seg_mpfc_sum.mean(axis=1)[20:120], label='mPFC_EZM')
plt.plot(tpower_ezm_seg_vhipp_sum.mean(axis=1)[20:120], label='vHPC_EZM')
plt.xlabel('Time segments 2.6s')
plt.ylabel('Theta power')
plt.title('EZM')
plt.legend()
plt.show()

plt.plot(tpower_oft_seg_mpfc_sum.mean(axis=1)[20:120], label='mPFC_OFT')
plt.plot(tpower_oft_seg_vhipp_sum.mean(axis=1)[20:120], label='vHPC_OFT')
plt.xlabel('Time segments 2.6s')
plt.ylabel('Theta power')
plt.title('OFT')
plt.legend()
plt.show()

#%%

# Plot mean gamma power of all channels in the vHPC
m = gpower_arena_seg_vhipp_sum.mean(axis=1)[20:60]
sd = np.std(gpower_arena_seg_vhipp_sum, axis=1)[20:60]
x = np.arange(gpower_arena_seg_vhipp_sum.shape[0])[20:60]

plt.plot(x, m)
plt.fill_between(x, m+sd, m-sd, alpha=0.6)

plt.xlabel('Time segments 2.6s')
plt.ylabel('Gamma power')
plt.title('All channel in vHPC')
plt.show()

#%%
# Plot mean gamma power of all channels in the mPFC

m = gpower_arena_seg_mpfc_sum.mean(axis=1)[20:60]
sd = np.std(gpower_arena_seg_mpfc_sum, axis=1)[20:60]
x = np.arange(gpower_arena_seg_mpfc_sum.shape[0])[20:60]

plt.plot(x, m)
plt.fill_between(x, m+sd, m-sd, alpha=0.6)

plt.xlabel('Time segments 2.6s')
plt.ylabel('Theta power')
plt.title('All channel in mPFC')
plt.show()

#%%
# plt.plot(gpower_arena_bef_seg_mpfc_sum.mean(axis=1)[20:120], label='mPFC_arena_bef')
# plt.plot(gpower_arena_bef_seg_vhipp_sum.mean(axis=1)[20:120], label='vHPC_arena_bef')
# plt.legend()
# plt.xlabel('Time segments 2.6s')
# plt.ylabel('Theta power')
# plt.show()

plt.plot(gpower_arena_seg_mpfc_sum.mean(axis=1)[20:120], label='mPFC_Arena')
plt.plot(gpower_arena_seg_vhipp_sum.mean(axis=1)[20:120], label='vHPC_Arena')
plt.legend()
plt.xlabel('Time segments 2.6s')
plt.ylabel('Gamma power')
plt.title('Arena')
plt.show()

plt.plot(gpower_ezm_seg_mpfc_sum.mean(axis=1)[20:120], label='mPFC_EZM')
plt.plot(gpower_ezm_seg_vhipp_sum.mean(axis=1)[20:120], label='vHPC_EZM')
plt.xlabel('Time segments 2.6s')
plt.ylabel('Gamma power')
plt.title('EZM')
plt.legend()
plt.show()

plt.plot(gpower_oft_seg_mpfc_sum.mean(axis=1)[20:120], label='mPFC_OFT')
plt.plot(gpower_oft_seg_vhipp_sum.mean(axis=1)[20:120], label='vHPC_OFT')
plt.xlabel('Time segments 2.6s')
plt.ylabel('Gamma power')
plt.title('OFT')
plt.legend()
plt.show()

#%%
### Plot theta power correlation between mPFC and vHPC

def plot_power_corr(x, y, start, end, text_loc=(a, b)):
    corr = np.corrcoef(x, y, rowvar=True)
    plt.figure(figsize=(6, 6))
    plt.scatter(x, y)

    m, b = np.polyfit(x, y, 1)
    plt.plot(x, m * x + b)

    plt.text(text_loc[0], text_loc[1], 'R sequred = {:.2f}'.format(np.square(corr[0][1])))
    plt.title('Theta power correlation mPFC-vHPC arena_bef')
    plt.xlabel('Theta power mPFC')
    plt.ylabel('Theta power vHPC')
    plt.show()
#%%
### Plot theta power correlation between mPFC and vHPC

# x = tpower_arena_bef_seg_mpfc_sum.mean(axis=1)[20:250]
# y = tpower_arena_bef_seg_vhipp_sum.mean(axis=1)[20:250]
#
# corr = np.corrcoef(x, y, rowvar=True)
#
# plt.figure(figsize=(6,6))
# plt.scatter(x, y)
#
# m, b = np.polyfit(x, y, 1)
# plt.plot(x, m*x + b)
#
# plt.text(0.046, 0.115, 'R sequred = {:.2f}'.format(np.square(corr[0][1])))
# plt.title('Theta power correlation mPFC-vHPC arena_bef')
# plt.xlabel('Theta power mPFC')
# plt.ylabel('Theta power vHPC')
# # plt.ylim(0, 0.22)
# plt.show()

x = tpower_arena_seg_mpfc_sum.mean(axis=1)[20:250]
y = tpower_arena_seg_vhipp_sum.mean(axis=1)[20:250]

corr = np.corrcoef(x, y, rowvar=True)

plt.figure(figsize=(6,6))
plt.scatter(x, y)

m, b = np.polyfit(x, y, 1)
plt.plot(x, m*x + b)

plt.gcf().subplots_adjust(bottom=0.15, left=0.18)
plt.text(0.04, 0.23, 'R sequred = {:.2f}'.format(np.square(corr[0][1])))
plt.title('Theta power correlation mPFC-vHPC arena')
plt.xlabel('Theta power mPFC')
plt.ylabel('Theta power vHPC')
# plt.ylim(0, 0.003)
plt.show()


x = tpower_ezm_seg_mpfc_sum.mean(axis=1)[20:250]
y = tpower_ezm_seg_vhipp_sum.mean(axis=1)[20:250]

corr = np.corrcoef(x, y, rowvar=True)

plt.figure(figsize=(6,6))
plt.scatter(x, y)

m, b = np.polyfit(x, y, 1)
plt.plot(x, m*x + b)
plt.gcf().subplots_adjust(bottom=0.15, left=0.18)
plt.text(0.042, 0.23, 'R sequred = {:.2f}'.format(np.square(corr[0][1])))
plt.title('Theta power correlation mPFC-vHPC EZM')
plt.xlabel('Theta power mPFC')
plt.ylabel('Theta power vHPC')
# plt.ylim(0, 0.03)
plt.show()

x = tpower_oft_seg_mpfc_sum.mean(axis=1)[20:250]
y = tpower_oft_seg_vhipp_sum.mean(axis=1)[20:250]

corr = np.corrcoef(x, y, rowvar=True)

plt.figure(figsize=(6,6))
plt.scatter(x, y)

m, b = np.polyfit(x, y, 1)
plt.plot(x, m*x + b)
plt.gcf().subplots_adjust(bottom=0.15, left=0.18)
plt.text(0.045, 0.23, 'R sequred = {:.2f}'.format(np.square(corr[0][1])))
plt.title('Theta power correlation mPFC-vHPC OFT')
plt.xlabel('Theta power mPFC')
plt.ylabel('Theta power vHPC')
# plt.ylim(0, 0.03)
plt.show()

#%%
### Plot gamma power correlation between mPFC and vHPC

# x = gpower_arena_bef_seg_mpfc_sum.mean(axis=1)[20:250]
# y = gpower_arena_bef_seg_vhipp_sum.mean(axis=1)[20:250]
#
# corr = np.corrcoef(x, y, rowvar=True)
#
# plt.figure(figsize=(6,6))
# plt.scatter(x, y)
#
# m, b = np.polyfit(x, y, 1)
# plt.plot(x, m*x + b)
#
# plt.text(0.046, 0.115, 'R sequred = {:.2f}'.format(np.square(corr[0][1])))
# plt.title('Gamma power correlation mPFC-vHPC arena_bef')
# plt.xlabel('Gamma power mPFC')
# plt.ylabel('Gamma power vHPC')
# # plt.ylim(0, 0.22)
# plt.show()

x = gpower_arena_seg_mpfc_sum.mean(axis=1)[20:250]
y = gpower_arena_seg_vhipp_sum.mean(axis=1)[20:250]

corr = np.corrcoef(x, y, rowvar=True)

plt.figure(figsize=(6,6))
plt.scatter(x, y)

m, b = np.polyfit(x, y, 1)
plt.plot(x, m*x + b)

plt.gcf().subplots_adjust(bottom=0.15, left=0.18)
plt.text(0.065, 0.145, 'R sequred = {:.2f}'.format(np.square(corr[0][1])))
plt.title('Gamma power correlation mPFC-vHPC arena')
plt.xlabel('Gamma power mPFC')
plt.ylabel('Gamma power vHPC')
plt.show()


x = gpower_ezm_seg_mpfc_sum.mean(axis=1)[20:250]
y = gpower_ezm_seg_vhipp_sum.mean(axis=1)[20:250]

corr = np.corrcoef(x, y, rowvar=True)

plt.figure(figsize=(6,6))
plt.scatter(x, y)

m, b = np.polyfit(x, y, 1)
plt.plot(x, m*x + b)
plt.gcf().subplots_adjust(bottom=0.15, left=0.18)
plt.text(0.065, 0.14, 'R sequred = {:.2f}'.format(np.square(corr[0][1])))
plt.title('Gamma power correlation mPFC-vHPC EZM')
plt.xlabel('Gamma power mPFC')
plt.ylabel('Gamma power vHPC')
# plt.ylim(0, 0.03)
plt.show()

x = gpower_oft_seg_mpfc_sum.mean(axis=1)[20:250]
y = gpower_oft_seg_vhipp_sum.mean(axis=1)[20:250]

corr = np.corrcoef(x, y, rowvar=True)

plt.figure(figsize=(6,6))
plt.scatter(x, y)

m, b = np.polyfit(x, y, 1)
plt.plot(x, m*x + b)
plt.gcf().subplots_adjust(bottom=0.15, left=0.18)
plt.text(0.065, 0.125, 'R sequred = {:.2f}'.format(np.square(corr[0][1])))
plt.title('Gamma power correlation mPFC-vHPC OFT')
plt.xlabel('Gamma power mPFC')
plt.ylabel('Gamma power vHPC')
# plt.ylim(0, 0.03)
plt.show()

#%%
x = gpower_oft_seg_mpfc_sum.mean(axis=1)[20:250]
y = gpower_oft_seg_vhipp_sum.mean(axis=1)[20:250]

corr = np.corrcoef(x, y, rowvar=True)

plt.figure(figsize=(6,6))
plt.scatter(x, y)

m, b = np.polyfit(x, y, 1)
plt.plot(x, m*x + b)
plt.gcf().subplots_adjust(bottom=0.15, left=0.18)
plt.text(0.065, 0.125, 'R sequred = {:.2f}'.format(np.square(corr[0][1])))
plt.title('Gamma power correlation mPFC-vHPC OFT')
plt.xlabel('Gamma power mPFC')
plt.ylabel('Gamma power vHPC')
# plt.ylim(0, 0.03)
plt.show()

#%%
## generate a bar plot of the theta power in Arena, EZM and OFT
import seaborn as sns
d = {
    #      'Arena_bef_mPFC':tpower_arena_bef_seg_mpfc_sum.mean(axis=1)[20:250],

    'Arena_mPFC': tpower_arena_seg_mpfc_sum.mean(axis=1)[20:250],
    'EZM_mPFC': tpower_ezm_seg_mpfc_sum.mean(axis=1)[20:250],
    'OFT_mPFC': tpower_oft_seg_mpfc_sum.mean(axis=1)[20:250],

    #     'Arena_bef_vHPC': tpower_arena_bef_seg_vhipp_sum.mean(axis=1)[20:250],
    'Arena_vHPC': tpower_arena_seg_vhipp_sum.mean(axis=1)[20:250],
    'EZM_vHPC': tpower_ezm_seg_vhipp_sum.mean(axis=1)[20:250],
    'OFT_vHPC': tpower_oft_seg_vhipp_sum.mean(axis=1)[20:250]
}

df = pd.DataFrame(d)

# 'Group': ['mPFC', 'mPFC', 'mPFC', 'vHPC', 'vHPC','vHPC'],

# dd=pd.melt(df,id_vars=['Group'],value_vars=df.columns,var_name='fruits')

fig, ax = plt.subplots(1, figsize=(10, 4))
ax = sns.barplot(data=df)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
plt.title('Theta power_2.6s_10min')
plt.show()

#%%











def main():
    # %%
    np.random.seed(42)
    animal = mBWfus009
    session = 'ezm_0219'
    behavior_trigger = 14.24
    events = traj_process(animal[session], behavior='ezm', start_time=0, duration=900)
    # # events = pickle.load(open('D:\\ephys\\2021-02-19_mBWfus009_EZM_ephys\ephys_processed\\2021-02-19_mBWfus009_EZM_ephys_results_manually_annotated.pickle',
    # #                           "rb"))
    exclude_keys = ['transitions_per_roi', 'roi_at_each_frame', 'cumulative_time_in_roi', 'avg_time_in_roi']
    rio_stats = {key: events['rois_stats'][key] for key in events['rois_stats'].keys() if key not in exclude_keys}
    rio_stats = pd.DataFrame.from_dict(rio_stats)
    rio_stats = rio_stats.drop([1, 3])
    rio_stats

    f_ephys = 500
    video_duration = 900 # in seconds
    ephys_duration = 950

    # ### extract overall behavioral open/close frame indices
    open_idx, close_idx, OTC_idx, prOTC_idx, prCTO_idx, nosedip_idx = get_events(events, behavior_trigger, 900)

    event_dict = dict(OTC=1, prOTC=2, prCTO=3, nosedip=4)

    OTC = trajectory_process.idx_to_events(OTC_idx, 1)
    prOTC = trajectory_process.idx_to_events(prOTC_idx, 2)
    prCTO = trajectory_process.idx_to_events(prCTO_idx, 3)
    nosedip = trajectory_process.idx_to_events(nosedip_idx, 4)

    mne_events = trajectory_process.merge_events(OTC, prOTC, prCTO, nosedip)

    plt.plot(mne_events[:, 0])
    plt.show()


    # Load ephys data
    dataset = ephys.load_data(animal[session])

    ## arrang the data in the same order as they are in the electrode array

    lfp = ephys.column_by_pad(ephys.get_lfp(dataset, 'all'))

    ### --- cluster analysis - returns relevant cluster channels
    # mpfc_representative_channels = ephys.explore_clusters(dataset, "mpfc", cluster_threshold=1.2, plot=True)
    # vhipp_representative_channels = ephys.explore_clusters(dataset, "vhipp", cluster_threshold=1.0, plot=True)

    lfp_mpfc = ephys.column_by_pad(ephys.get_lfp(dataset, 'mpfc'))
    lfp_vhipp = ephys.column_by_pad(ephys.get_lfp(dataset, 'vhipp'))

    theta_mpfc = ephys.column_by_pad(ephys.get_power(dataset, 'mpfc', 'theta'))
    theta_vhipp = ephys.column_by_pad(ephys.get_power(dataset, 'vhipp', 'theta'))

    ## crosscorrelate the channels within each brain area
    corr_matrix_mpfc = lfp_mpfc.iloc[5000:10000].corr()
    corr_matrix_vhipp = lfp_vhipp.iloc[5000:10000].corr()

    plt.matshow(corr_matrix_mpfc)
    cb = plt.colorbar()
    plt.xticks(range(corr_matrix_mpfc.select_dtypes(['number']).shape[1]),
               corr_matrix_mpfc.select_dtypes(['number']).columns, fontsize=8, rotation=90)
    plt.yticks(range(corr_matrix_mpfc.select_dtypes(['number']).shape[1]),
               corr_matrix_mpfc.select_dtypes(['number']).columns, fontsize=8)
    plt.show()

    plt.matshow(corr_matrix_vhipp)
    cb = plt.colorbar()
    plt.xticks(range(corr_matrix_vhipp.select_dtypes(['number']).shape[1]),
               corr_matrix_vhipp.select_dtypes(['number']).columns, fontsize=8, rotation=90)
    plt.yticks(range(corr_matrix_vhipp.select_dtypes(['number']).shape[1]),
               corr_matrix_vhipp.select_dtypes(['number']).columns, fontsize=8)
    plt.show()

    ### drop the broken channels with low coherence with neighboring channels
    bad_ch_mpfc = [6, 10, 23]
    lfp_mpfc = lfp_mpfc.drop(labels=bad_ch_mpfc, axis=1)
    corr_matrix_mpfc = lfp_mpfc.iloc[5000:10000].corr()

    ## plot corr_matrix
    plt.matshow(corr_matrix_mpfc)
    cb = plt.colorbar()
    plt.xticks(range(corr_matrix_mpfc.select_dtypes(['number']).shape[1]),
               corr_matrix_mpfc.select_dtypes(['number']).columns, fontsize=8, rotation=90)
    plt.yticks(range(corr_matrix_mpfc.select_dtypes(['number']).shape[1]),
               corr_matrix_mpfc.select_dtypes(['number']).columns, fontsize=8)
    plt.show()

    bad_ch_vhipp = [44]
    lfp_vhipp = lfp_vhipp.drop(labels=bad_ch_vhipp, axis=1)
    corr_matrix_vhipp = lfp_vhipp.iloc[5000:10000].corr()

    plt.matshow(corr_matrix_vhipp)
    cb = plt.colorbar()
    plt.xticks(range(corr_matrix_vhipp.select_dtypes(['number']).shape[1]),
               corr_matrix_vhipp.select_dtypes(['number']).columns, fontsize=8, rotation=90)
    plt.yticks(range(corr_matrix_vhipp.select_dtypes(['number']).shape[1]),
               corr_matrix_vhipp.select_dtypes(['number']).columns, fontsize=8)
    plt.show()

    ## create
    lfp = ephys.column_by_pad(ephys.get_lfp(dataset, 'all'))
    bad_ch = [6, 10, 23, 44]
    lfp = lfp.drop(labels=bad_ch, axis=1)
    print(lfp.shape)

    corr_matrix = lfp.iloc[5000:10000].corr()

    plt.imshow(corr_matrix)
    cb = plt.colorbar()
    plt.xticks(range(corr_matrix.select_dtypes(['number']).shape[1]), corr_matrix.select_dtypes(['number']).columns,
               fontsize=8, rotation=90)
    plt.yticks(range(corr_matrix.select_dtypes(['number']).shape[1]), corr_matrix.select_dtypes(['number']).columns,
               fontsize=8)
    plt.show()

    plotting.plot_phase_coh_pairs(dataset, animal, session, savedir='./phase_coh_plots/', band='theta',
                                  srate=f_ephys, tstart=behavior_trigger, twin=video_duration,
                                  nbins=video_duration // 2)











    # mpfc_pad = [2, 17, 25]
    # vhipp_pad = [35, 46, 58]

    ## epoch theta power around certain events
    theta_OTC_mpfc = ephys.epoch_data(theta_mpfc, channels=mpfc_pad, events=OTC_idx)
    theta_OTC_vhipp = ephys.epoch_data(theta_vhipp, channels=vhipp_pad, events=OTC_idx)
    theta_prCTO_mpfc = ephys.epoch_data(theta_mpfc, channels=mpfc_pad, events=prCTO_idx)
    theta_prCTO_vhipp = ephys.epoch_data(theta_vhipp, channels=vhipp_pad, events=prCTO_idx)

    ## epoch LFP around certain events
    lfp_OTC_mpfc = ephys.epoch_data(lfp_mpfc, channels=mpfc_pad, events=OTC_idx)
    lfp_OTC_vhipp = ephys.epoch_data(lfp_vhipp, channels=vhipp_pad, events=OTC_idx)
    lfp_prCTO_mpfc = ephys.epoch_data(lfp_mpfc, channels=mpfc_pad, events=prCTO_idx)
    lfp_prCTO_vhipp = ephys.epoch_data(lfp_vhipp, channels=vhipp_pad, events=prCTO_idx)

    print(theta_OTC_mpfc.shape, theta_OTC_vhipp.shape, theta_prCTO_mpfc.shape, theta_prCTO_vhipp.shape)

    ephys.plot_epochs(lfp_OTC_mpfc)
    ephys.plot_epochs(lfp_OTC_vhipp)
    ephys.plot_epochs(lfp_prCTO_mpfc)
    ephys.plot_epochs(lfp_prCTO_vhipp)


    ## use mne to epoch data



    ### create event dict for mne raw


    frequencies = np.arange(1, 20, 1)
    power = mne.time_frequency.tfr_morlet(OTC_evoked, n_cycles=2, return_itc=False,
                                          freqs=frequencies, decim=3)
    power.plot('A-025')

    a = 'break'



    # mean power of the specified window size
    # TODO: how to remove the outlier ?

    ### time series plots for transition events
    # power_prolonged_close_to_open = ephys.slice_from_arr(power_vhipp,
    #                                                      prolonged_close_to_open_idx,
    #                                                      channels=vhipp_representative_channels,
    #                                                      window=2,
    #                                                      mean=False)
    #
    # power_prolonged_open_to_close = ephys.slice_from_arr(power_vhipp,
    #                                                      prolonged_open_to_close_idx,
    #                                                      channels=vhipp_representative_channels,
    #                                                      window=2,
    #                                                      mean=False)
    #
    # power_open_to_close = ephys.slice_from_arr(power_vhipp,
    #                                            open_to_close_idx,
    #                                            channels=vhipp_representative_channels,
    #                                            window=2,
    #                                            mean=False)
    #
    # conditions = [power_prolonged_close_to_open, power_prolonged_open_to_close, power_open_to_close]
    # titles = ['Power_PlCTO', 'power_PlOTC', 'power_OTC']
    #
    # fig, ax = plt.subplots(len(conditions), len(range(power_prolonged_close_to_open.shape[0])))
    #
    # for condition_idx, condition in enumerate(conditions):
    #     for channel_idx, channel in enumerate(range(power_prolonged_close_to_open.shape[0])):
    #         ax[condition_idx, channel_idx].set_title(titles[condition_idx] + '_ch:' + str(channel_idx))
    #         ax[condition_idx, channel_idx].plot(power_prolonged_close_to_open[channel, :, :].transpose(), alpha=0.2)
    #         ax[condition_idx, channel_idx].plot(power_prolonged_close_to_open.mean(axis=0).mean(axis=0).transpose())
    #
    # plt.show()

    ### mean plots for checking power during different spatial dependencies
    # iter = int(len(power_mpfc[0, :]) / window_samples)
    # idxs = list(range(iter))
    # mean_power_vhipp = slice_from_arr(power_vhipp, idxs, channels=vhipp_representative_channels)
    # mean_power_mpfc = slice_from_arr(power_mpfc, idxs, channels=mpfc_representative_channels)
    #
    # open_power_vhipp = slice_from_arr(power_vhipp, open_idx, channels=vhipp_representative_channels, window=0)
    # # TODO: fix nan
    # closed_power_vhipp = slice_from_arr(power_vhipp, close_idx, channels=vhipp_representative_channels, window=0)
    #
    # power_prolonged_close_to_open = slice_from_arr(power_vhipp,
    #                                                prolonged_close_to_open_idx,
    #                                                channels=vhipp_representative_channels,
    #                                                window=1,
    #                                                mean=True)
    # power_prolonged_open_to_close = slice_from_arr(power_vhipp,
    #                                                prolonged_open_to_close_idx,
    #                                                channels=vhipp_representative_channels,
    #                                                window=1,
    #                                                mean=True)
    # power_open_to_close = slice_from_arr(power_vhipp,
    #                                      open_to_close_idx,
    #                                      channels=vhipp_representative_channels,
    #                                      window=1,
    #                                      mean=True)
    # TODO: extract 'T' timepoints from
    # transition_idxs = all the transition indices calculated from exit to entry
    # transition_power_vhipp = slice_from_arr(power_vhipp, transition_idxs, channels=vhipp_representative_channels, window=0)

    #
    # conditions = [mean_power_vhipp.flatten(),
    #               open_power_vhipp.flatten(),
    #               closed_power_vhipp.flatten(),
    #               transition_power_vhipp.flatten(),
    #               ]

    # conditions = [mean_power_vhipp.flatten(),
    #               open_power_vhipp.flatten(),
    #               closed_power_vhipp.flatten(),
    #               # transition_power_vhipp.flatten(),
    #               power_prolonged_close_to_open.flatten(),
    #               power_prolonged_open_to_close.flatten()]
    # titles = ['mean_power_vhipp', 'open_power_vhipp', 'closed_power_vhipp', 'power_prolonged_close_to_open',
    #           'power_prolonged_open_to_close']

    # import seaborn as sns
    # sns.boxplot(x=titles, y=conditions, )
    #
    # fig, ax = plt.subplots(nrows=len(mpfc_representative_channels),
    #                        ncols=len(vhipp_representative_channels),
    #                        figsize=(10, 13))
    # avg_win = int(50 * 2.6)
    # for ch_mpfc, power_per_ch_mpfc in enumerate(power_mpfc_open):
    #     for ch_vhipp, power_per_ch_vhipp in enumerate(power_vhipp_open):
    #         print(ch_mpfc, ch_vhipp)
    #         new_len_open = int(len(power_per_ch_mpfc) // avg_win)
    #         print(new_len_open)
    #         x = np.mean(np.reshape(power_per_ch_mpfc[:new_len_open * avg_win], (avg_win, new_len_open)),
    #                     axis=0)  # power_vhipp_open[chan_vhipp, :]
    #         print(x.shape, np.isnan(x).any())
    #         y = np.mean(np.reshape(power_per_ch_vhipp[:new_len_open * avg_win], (avg_win, new_len_open)),
    #                     axis=0)  # power_mpfc_close[chan_mpfc, :]
    #         print(y.shape, np.isnan(y).any())
    #         # # TODO: zscore here? how to remove the outlier ?
    #         x = scipy.stats.zscore(x)
    #         y = scipy.stats.zscore(y)
    #         slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x, y)
    #         ax[ch_mpfc, ch_vhipp].scatter(x, y)
    #         ax[ch_mpfc, ch_vhipp].set_title('R-squared = %0.2f' % r_value ** 2)
    #         ax[ch_mpfc, ch_vhipp].set_xlabel(
    #             'Z-scored theta power' + '\n' + 'mpfc ch' + str(mpfc_representative_channels[ch_mpfc]))
    #         ax[ch_mpfc, ch_vhipp].set_ylabel(
    #             'Z-scored theta power' + '\n' + 'vhipp ch' + str(vhipp_representative_channels[ch_vhipp]))
    #         sns.regplot(x=x, y=y, ax=ax[ch_mpfc, ch_vhipp])
    #         # ax[ch_mpfc, ch_vhipp].plot(np.unique(x), np.poly1d(np.polyfit(x, y, 1))(np.unique(x)))
    # fig.tight_layout()
    # fig.show()
    #
    # fig, ax = plt.subplots(nrows=len(mpfc_representative_channels),
    #                        ncols=len(vhipp_representative_channels),
    #                        figsize=(10, 13))
    # avg_win = int(50 * 2.6)
    # for ch_mpfc, power_per_ch_mpfc in enumerate(power_mpfc_close):
    #     for ch_vhipp, power_per_ch_vhipp in enumerate(power_vhipp_close):
    #         print(ch_mpfc, ch_vhipp)
    #         new_len_close = int(len(power_per_ch_mpfc) // avg_win)
    #         print(new_len_close)
    #         x = np.mean(np.reshape(power_per_ch_mpfc[:new_len_close * avg_win], (avg_win, new_len_close)),
    #                     axis=0)  # power_vhipp_open[chan_vhipp, :]
    #         print(x.shape, np.isnan(x).any())
    #         y = np.mean(np.reshape(power_per_ch_vhipp[:new_len_close * avg_win], (avg_win, new_len_close)),
    #                     axis=0)  # power_mpfc_close[chan_mpfc, :]
    #         print(y.shape, np.isnan(y).any())
    #         # # TODO: zscore here? how to remove the outlier ?
    #         x = scipy.stats.zscore(x)
    #         y = scipy.stats.zscore(y)
    #         slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x, y)
    #         ax[ch_mpfc, ch_vhipp].scatter(x, y)
    #         ax[ch_mpfc, ch_vhipp].set_title('R-squared = %0.2f' % r_value ** 2)
    #         ax[ch_mpfc, ch_vhipp].set_xlabel(
    #             'Z-scored theta power' + '\n' + 'mpfc ch' + str(mpfc_representative_channels[ch_mpfc]))
    #         ax[ch_mpfc, ch_vhipp].set_ylabel(
    #             'Z-scored theta power' + '\n' + 'vhipp ch' + str(vhipp_representative_channels[ch_vhipp]))
    #         sns.regplot(x=x, y=y, ax=ax[ch_mpfc, ch_vhipp])
    #         # ax[ch_mpfc, ch_vhipp].plot(np.unique(x), np.poly1d(np.polyfit(x, y, 1))(np.unique(x)))
    # fig.tight_layout()
    # fig.show()

    ### --- overall coherence analysis
    # coherence between areas

    lfp_mpfc = ephys.get_lfp(dataset, brain_area='mpfc')
    lfp_vhipp = ephys.get_lfp(dataset, brain_area='vhipp')

    ### crop the ephys data prior to trigger
    lfp_mpfc = lfp_mpfc[:, int(f_ephys * ephys_trigger):]
    lfp_vhipp = lfp_vhipp[:, int(f_ephys * ephys_trigger):]

    coherence_mpfc_to_vhipp = np.zeros((len(vhipp_representative_channels), len(vhipp_representative_channels)))
    coherence_bands = []
    correlation_vals = []

    new_freq = 50
    old_freq = 500

    for vhipp_id, vhipp_channel in enumerate(vhipp_representative_channels):
        vhipp_data = lfp_vhipp[vhipp_channel, :old_freq * 100]
        for mpfc_id, mpfc_channel in enumerate(vhipp_representative_channels):
            mpfc_data = lfp_mpfc[mpfc_channel, :old_freq * 100]
            this_coherence = coherence(x=vhipp_data, y=mpfc_data, fs=old_freq)[1][:20]
            coherence_mpfc_to_vhipp[vhipp_id, mpfc_id] = this_coherence.mean()
            coherence_bands.append(this_coherence)
            correlate = signal.correlate(vhipp_data, mpfc_data, mode='same')
            # resample to 50 hz
            samples = int(len(correlate) * (new_freq / old_freq))
            correlate = signal.resample(correlate, samples)
            correlation_vals.append(correlate)

    plt.imshow(coherence_mpfc_to_vhipp)
    plt.colorbar()
    plt.show()

    fig, ax = plt.subplots(len(correlation_vals), 1)
    for val_id, val in enumerate(correlation_vals):
        ax[val_id].plot(val)

    plt.show()

    print('done')


if __name__ == '__main__':
    main()
