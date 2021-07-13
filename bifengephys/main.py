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

plt.rcParams['axes.titlesize'] = 18
plt.rcParams["axes.labelsize"] = 18
plt.rcParams["xtick.labelsize"] = 14
plt.rcParams["ytick.labelsize"] = 14
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
session = 'ezm_0226'

loc, scorer = behav.load_location(animal[session])
loc = behav.calib_location(loc)
loc = behav.get_locomotion(loc)

rois_stats, transitions = behav.analyze_trajectory_ezm(loc, bd='head', start_time=0, duration=600, fps=50)

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
avgspd = loc[scorer, bd, 'avgspd'].to_numpy()

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
animal = mBWfus009
date = '0219'

#
# data_arena_bef = ephys.load_data(animal['arena_' + date + '_bef'])
# sdir_arena_bef = animal['arena_' + date + '_bef'] + '/figures/'
# if not os.path.exists(sdir_arena_bef):
#     os.makedirs(sdir_arena_bef)
# print(sdir_arena_bef)
#
# data_arena = ephys.load_data(animal['arena_' + date + '_aft'])

data_arena = ephys.load_data(animal['arena_' + date])
sdir_arena = animal['arena_' + date] + '/figures/'
if not os.path.exists(sdir_arena):
    os.makedirs(sdir_arena)
print(sdir_arena)

data_ezm = ephys.load_data(animal['ezm_' + date])
sdir_ezm = animal['ezm_' + date] + '/figures/'
if not os.path.exists(sdir_ezm):
    os.makedirs(sdir_ezm)
print(sdir_ezm)


data_oft = ephys.load_data(animal['oft_' + date])
sdir_oft = animal['oft_' + date] + '/figures/'
if not os.path.exists(sdir_oft):
    os.makedirs(sdir_oft)
print(sdir_oft)
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

bad_ch_arena = [7, 10, 23, 44, 57, 58, 59, 60, 61, 62] ### mBWfus009 arena_0219
bad_ch_ezm = [6, 7, 10, 23, 44, 57, 58, 59, 60, 61, 62] ### mBWfus009 ezm_0219
bad_ch_oft = [ 7, 10, 23, 44, 57, 58, 59, 60, 61, 62] ### mBWfus009 oft_0219


# bad_ch_arena = [6, 7, 10, 23, 44, 57, 58, 59] ### mBWfus009 arena_0226
# bad_ch_ezm = [6, 7, 10, 23, 44, 57, 58, 59] ### mBWfus009 ezm_0226
# bad_ch_oft = [6, 7, 10, 23, 44] ### mBWfus009 oft_0226
# bad_ch = [7, 10, 23, 44, 57, 58, 59, 60, 61, 62] ### mBWfus009 arena_0305

# bad_ch_arena = [7, 10, 23, 44, 57, 58, 59, 60, 61, 62] ### mBWfus009 arena_0305
# bad_ch_ezm = [7, 10, 23, 44, 57, 58, 59, 60, 61, 62] ### mBWfus009 ezm_0305
# bad_ch_oft = [7, 10, 23, 44, 57, 58, 59, 60, 61, 62] ### mBWfus009 oft_0305

# bad_ch = [27, 28, 31, 32, 46, 52] # mBWfus011 arena_0226,0305, 0313

# bad_ch = [3, 9, 14, 15, 16, 17, 18, 19, 20, 21, 22,
#           23, 24, 25, 26, 27, 28, 31, 34, 35, 36,
#           38, 40, 41, 42, 43, 44, 45, 46, 60, 61, 62]

#%%
#%%
lfp = ephys.column_by_pad(ephys.get_lfp(data_arena_bef, 'all'))

# bad_ch = [7, 10, 23, 44, 46, 57, 58, 59, 60, 61, 62]

lfp = lfp.drop(labels=bad_ch_arena, axis=1)
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

raw_arena_bef = mne.io.RawArray(lfp_filted*1e-6, info) ### mne data format (n-channels, n-samples), unit = V




#%%
lfp = ephys.column_by_pad(ephys.get_lfp(data_arena, 'all'))

# bad_ch = [7, 10, 23, 44, 46, 57, 58, 59, 60, 61, 62]

lfp = lfp.drop(labels=bad_ch_arena, axis=1)
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

raw_arena = mne.io.RawArray(lfp_filted, info) ### mne data format (n-channels, n-samples), unit = V

#%%
raw_arena.plot(n_channels = 46, duration=2, scalings='auto')

#%%

lfp = ephys.column_by_pad(ephys.get_lfp(data_ezm, 'all'))

# bad_ch = [6, 7, 10, 23, 44, 46, 57, 58, 59, 60, 61, 62]

lfp = lfp.drop(labels=bad_ch_ezm, axis=1)
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

lfp = lfp.drop(labels=bad_ch_oft, axis=1)
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
raw_arena.plot_psd(fmin=1, fmax=20, tmin=None, tmax=None, proj=False, n_fft=1024,
                   n_overlap=0.9, reject_by_annotation=True, picks=None, ax=None, color='black',
                   xscale='linear', area_mode='std', area_alpha=0.33, dB=True,
                   estimate='power', show=True, n_jobs=1, average=False, line_alpha=None, spatial_colors=True,
                   sphere=None, window='hamming', verbose=None)

#%%
### compute power spectra using the Welch method
test = raw_arena.copy()
psds, freqs = mne.time_frequency.psd_welch(test, fmin=0, fmax=40, tmin=None, tmax=None, n_fft=2048, n_overlap=180,
                                  n_per_seg=200, picks=None, proj=False, n_jobs=1, reject_by_annotation=True,
                                  average='mean', window='hamming', verbose=None)

#%%
for _ in range(psds.shape[0]):
    plt.plot(freqs, psds[_, :])
plt.show()

#%%
### compute power spectra using the multitaper method
''''
mne.time_frequency.psd_multitaper(inst, fmin=0, fmax=inf, tmin=None, tmax=None, bandwidth=None, adaptive=False, 
low_bias=True, normalization='length', picks=None, proj=False, n_jobs=1, reject_by_annotation=False, verbose=None)
'''

test = raw_arena.copy()
psds, freqs = mne.time_frequency.psd_multitaper(test, fmin=1, fmax=20, tmin=20, tmax=620, bandwidth=2.5, n_jobs=20)


#%%
for _ in range(psds.shape[0]):
    plt.plot(freqs, psds[_, :])

plt.title('PSD using multitaper')
plt.show()

#%%
''''
mne.time_frequency.tfr_multitaper(inst, freqs, n_cycles, time_bandwidth=4.0, use_fft=True, return_itc=True, 
decim=1, n_jobs=1, picks=None, average=True, verbose=None)

freqs : ndarray, shape (n_freqs,)
        The frequencies in Hz.
'''

test = raw_arena.copy()
test = mne.make_fixed_length_epochs(test, duration=2.6)
freqs = np.array([4, 5, 6, 7, 8, 9, 10, 11, 12])
power = mne.time_frequency.tfr_multitaper(test, freqs=freqs, n_cycles=7,
                                          return_itc=False, n_jobs=20, picks=None, average=False)

#%%
p = power.data

#%%
p = np.mean(p, axis=3).sum(axis=2)
#%%
for _ in range(p.shape[1]):
    plt.plot(p[:, _])

plt.show()

#%%
# ch_mpfc = [1:16]
# ch_vhipp = [10:16]

pwr_mpfc = p[:, [1, 2, 3, 4, 5, 7, 8, 9, 10, 11]]
pwr_vhipp = p[:, 34:40]

#%%

#%%
# correlate all channels in vHPC with all the channels in the mPFC
df_arena_mpfc = pd.DataFrame(pwr_mpfc)
df_arena_vhipp = pd.DataFrame(pwr_vhipp)

corr_matrix = []
for column in df_arena_vhipp:
    corr = df_arena_mpfc.corrwith(df_arena_vhipp[column])
    corr_matrix.append(corr)

corr_matrix = np.array(corr_matrix)

#%%
plt.imshow(corr_matrix)
plt.xlabel('Pads in the mPFC')
plt.ylabel('Pads in the vHPC')
plt.colorbar(label='Pearson coefficient')
plt.show()

#%%
fig, ax = plt.subplots()
for _ in range(corr_matrix.shape[0]):
    ax.plot(corr_matrix[_, :])

plt.xlabel('Pads in the mPFC')
plt.ylabel('Pearson coefficient')
plt.show()

#%%
fig, ax = plt.subplots()
for _ in range(corr_matrix.shape[0]):
    ax.plot(corr_matrix[:, _])

plt.xlabel('Pads in the vHPC')
plt.ylabel('Pearson coefficient')
plt.show()

#%%
''''
mne.filter.filter_data(data, sfreq, l_freq, h_freq, picks=None, filter_length='auto', 
l_trans_bandwidth='auto', h_trans_bandwidth='auto', n_jobs=1, method='fir', iir_params=None, 
copy=True, phase='zero', fir_window='hamming', fir_design='firwin', pad='reflect_limited', verbose=None)
'''

''''
If envelope=False, the analytic signal for the channels defined in picks is computed and the data of 
the Raw object is converted to a complex representation (the analytic signal is complex valued).

If envelope=True, the absolute value of the analytic signal for the channels defined in picks is computed, 
resulting in the envelope signal.

If envelope=False, more memory is required since the original raw data as well as the analytic signal have 
temporarily to be stored in memory. If n_jobs > 1, more memory is required as len(picks) * n_times additional 
time points need to be temporaily stored in memory.

Also note that the n_fft parameter will allow you to pad the signal with zeros before performing the Hilbert transform. 
This padding is cut off, but it may result in a slightly different result (particularly around the edges). 

'''

### extract theta power using Hilbert transform
### epoch the power into 2.6 second segments

sfreq = 500
l_freq_t = 4
h_freq_t = 12

l_freq_g = 40
h_freq_g = 100

#
# tRaw_arena_bef = raw_arena_bef.filter(l_freq_t, h_freq_t, phase='zero-double')
# tRaw_arena_bef.apply_hilbert(envelope=True)
# tpower_arena_bef_epoch = mne.make_fixed_length_epochs(tRaw_arena_bef, duration=2.6)

t_arena = raw_arena.copy()
g_arena = raw_arena.copy()

t_arena = t_arena.filter(l_freq_t, h_freq_t, phase='zero-double')
t_arena.apply_hilbert(envelope=True)
tpower_arena_epoch = mne.make_fixed_length_epochs(t_arena, duration=2.6)

g_arena = g_arena.filter(l_freq_g, h_freq_g, phase='zero-double')
g_arena.apply_hilbert(envelope=True)
gpower_arena_epoch = mne.make_fixed_length_epochs(g_arena, duration=2.6)

t_ezm = raw_ezm.copy()
g_ezm = raw_ezm.copy()

t_ezm = t_ezm.filter(l_freq_t, h_freq_t, phase='zero-double')
t_ezm.apply_hilbert(envelope=True)
tpower_ezm_epoch = mne.make_fixed_length_epochs(t_ezm, duration=2.6)

g_ezm = g_ezm.filter(l_freq_g, h_freq_g, phase='zero-double')
g_ezm.apply_hilbert(envelope=True)
gpower_ezm_epoch = mne.make_fixed_length_epochs(g_ezm, duration=2.6)

t_oft = raw_oft.copy()
g_oft = raw_oft.copy()

t_oft = t_oft.filter(l_freq_t, h_freq_t, phase='zero-double')
t_oft.apply_hilbert(envelope=True)
tpower_oft_epoch = mne.make_fixed_length_epochs(t_oft, duration=2.6)

g_oft = g_oft.filter(l_freq_g, h_freq_g, phase='zero-double')
g_oft.apply_hilbert(envelope=True)
gpower_oft_epoch = mne.make_fixed_length_epochs(g_oft, duration=2.6)

#%%
t_arena.plot(n_channels = 46, duration=5, scalings='auto')

#%%
#mBWfus008 channels in mPFC [0, 21]; ch in the vHPC [21, 42]

#mBWfus009 channels in mPFC [0, 23; ch in the vHPC [23, 45]

#mBWfus011 channels in mPFC [0, 29]; ch in the vHPC [29, 58]

#mBWfus012 channels in mPFC [0, 9]: ch in the vHPC [9, 18]

#%%
### 1, extract theta power from NME.epock to arrays
## 2, compute the sum theta power of 2.6 second

mpfc_ch = np.arange(0, 23).tolist()
vhipp_ch = np.arange(24, 45).tolist()

# tpower_arena_bef_mpfc = tpower_arena_bef_epoch.get_data(picks=mpfc_ch).sum(axis=2)
# tpower_arena_bef_vhipp = tpower_arena_bef_epoch.get_data(picks=vhipp_ch).sum(axis=2)

tpower_arena_mpfc = tpower_arena_epoch.get_data(picks=mpfc_ch).sum(axis=2)
tpower_arena_vhipp = tpower_arena_epoch.get_data(picks=vhipp_ch).sum(axis=2)

tpower_ezm_mpfc = tpower_ezm_epoch.get_data(picks=mpfc_ch).sum(axis=2)
tpower_ezm_vhipp = tpower_ezm_epoch.get_data(picks=vhipp_ch).sum(axis=2)

tpower_oft_mpfc = tpower_oft_epoch.get_data(picks=mpfc_ch).sum(axis=2)
tpower_oft_vhipp = tpower_oft_epoch.get_data(picks=vhipp_ch).sum(axis=2)



#%%

gpower_arena_mpfc = gpower_arena_epoch.get_data(picks=mpfc_ch).sum(axis=2)
gpower_arena_vhipp = gpower_arena_epoch.get_data(picks=vhipp_ch).sum(axis=2)

gpower_ezm_mpfc = gpower_ezm_epoch.get_data(picks=mpfc_ch).sum(axis=2)
gpower_ezm_vhipp = gpower_ezm_epoch.get_data(picks=vhipp_ch).sum(axis=2)

gpower_oft_mpfc = gpower_oft_epoch.get_data(picks=mpfc_ch).sum(axis=2)
gpower_oft_vhipp = gpower_oft_epoch.get_data(picks=vhipp_ch).sum(axis=2)



#%%


#%%


#%%
### create dataframe for correlation matrix

df_arena_mpfc = pd.DataFrame(tpower_arena_mpfc)
df_arena_vhipp = pd.DataFrame(tpower_arena_vhipp)

df_ezm_mpfc = pd.DataFrame(tpower_ezm_mpfc)
df_ezm_vhipp = pd.DataFrame(tpower_ezm_vhipp)

# correlate one channel in vHPC with all the channels in the mPFC
corr_mpfc = df_arena_mpfc.corrwith(df_arena_vhipp[8])
print('One vHPC channel against all the channels in mPFC = ', corr_mpfc)

# correlate one channel in mPFC with all the channels in the vHPC
corr_vHPC = df_arena_vhipp.corrwith(df_arena_mpfc[5])
print('One mPFC channel against all the channels in vHPC = ', corr_vHPC)


#%%
##mBWfus009_0219, BWfus009_0305
remv_ch_mpfc = []
remv_ch_vhipp= [0, 1, 2, 3, 4, 5, 6, 7, 8]

ch_mpfc = [1:16]
ch_vhipp = [10:16]

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

print(tpower_arena_mpfc.shape, tpower_arena_vhipp.shape)
#
# tpower_arena_bef_mpfc = np.delete(tpower_arena_bef_mpfc, remv_ch_mpfc, axis=1)
# tpower_arena_bef_vhipp = np.delete(tpower_arena_bef_vhipp, remv_ch_vhipp, axis=1)

tpower_arena_mpfc = np.delete(tpower_arena_mpfc, remv_ch_mpfc, axis=1)
tpower_arena_vhipp = np.delete(tpower_arena_vhipp, remv_ch_vhipp, axis=1)

tpower_ezm_mpfc = np.delete(tpower_ezm_mpfc, remv_ch_mpfc, axis=1)
tpower_ezm_vhipp = np.delete(tpower_ezm_vhipp, remv_ch_vhipp, axis=1)

tpower_oft_mpfc = np.delete(tpower_oft_mpfc, remv_ch_mpfc, axis=1)
tpower_oft_vhipp = np.delete(tpower_oft_vhipp, remv_ch_vhipp, axis=1)


print(tpower_arena_mpfc.shape, tpower_arena_vhipp.shape)

#%%
### remove the channels showing low correlation of theta power (threshold = 0.2 (Pearson coefficient))

print(tpower_arena_mpfc.shape, tpower_arena_vhipp.shape)
#
# tpower_arena_bef_mpfc = tpower_arena_bef_mpfc, remv_ch_mpfc, axis=1)
# tpower_arena_bef_vhipp = tpower_arena_bef_vhipp, remv_ch_vhipp, axis=1)

tpower_arena_mpfc = tpower_arena_mpfc[:, 1:16]
tpower_arena_vhipp = tpower_arena_vhipp[:, 10:16]

tpower_ezm_mpfc = tpower_ezm_mpfc[:, 1:16]
tpower_ezm_vhipp = tpower_ezm_vhipp[:, 10:16]

tpower_oft_mpfc = tpower_oft_mpfc[:, 1:16]
tpower_oft_vhipp = tpower_oft_vhipp[:, 10:16]

print(tpower_arena_mpfc.shape, tpower_arena_vhipp.shape)
#%%
# correlate all channels in vHPC with all the channels in the mPFC
df_arena_mpfc = pd.DataFrame(tpower_arena_mpfc)
df_arena_vhipp = pd.DataFrame(tpower_arena_vhipp)

corr_matrix = []
for column in df_arena_vhipp:
    corr = df_arena_mpfc.corrwith(df_arena_vhipp[column])
    corr_matrix.append(corr)

corr_matrix = np.array(corr_matrix)

#%%
plt.imshow(corr_matrix)
plt.xlabel('Pads in the mPFC')
plt.ylabel('Pads in the vHPC')
plt.colorbar(label='Pearson coefficient')
plt.show()

#%%
fig, ax = plt.subplots()
for _ in range(corr_matrix.shape[0]):
    ax.plot(corr_matrix[_, :])

plt.xlabel('Pads in the mPFC')
plt.ylabel('Pearson coefficient')
plt.show()

#%%
fig, ax = plt.subplots()
for _ in range(corr_matrix.shape[0]):
    ax.plot(corr_matrix[:, _])

plt.xlabel('Pads in the vHPC')
plt.ylabel('Pearson coefficient')
plt.show()


#%%
### remove the channels showing low correlation of theta power (threshold = 0.2 (Pearson coefficient))

print(gpower_arena_mpfc.shape, gpower_arena_vhipp.shape)

# gpower_arena_bef_mpfc = np.delete(gpower_arena_bef_mpfc, rev_ch_mpfc, axis=1)
# gpower_arena_bef_vhipp = np.delete(gpower_arena_bef_vhipp, rev_ch_vhipp, axis=1)

gpower_arena_mpfc = np.delete(gpower_arena_mpfc, remv_ch_mpfc, axis=1)
gpower_arena_vhipp = np.delete(gpower_arena_vhipp, remv_ch_vhipp, axis=1)

gpower_ezm_mpfc = np.delete(gpower_ezm_mpfc, remv_ch_mpfc, axis=1)
gpower_ezm_vhipp = np.delete(gpower_ezm_vhipp, remv_ch_vhipp, axis=1)

gpower_oft_mpfc = np.delete(gpower_oft_mpfc, remv_ch_mpfc, axis=1)
gpower_oft_vhipp = np.delete(gpower_oft_vhipp, remv_ch_vhipp, axis=1)


print(gpower_arena_mpfc.shape, gpower_arena_vhipp.shape)

#%%
# Plot theta power of individual channel in the mPFc
title = 'Theta power mPFC'
xlabel = 'Time segments 2.6s'
ylabel = 'Theta power sum of 2.6s'

fig, ax = plt.subplots()

for i in range(tpower_arena_mpfc.shape[1]):
    ax.plot(tpower_arena_mpfc[20:60, i])

plt.xlabel(xlabel)
plt.ylabel(ylabel)
plt.title(title)
plt.savefig(sdir_arena + title + '.png')
plt.show()

#%%
# Plot theta power of individual channel in the vHPC
title = 'Theta power vHPC'
xlabel = 'Time segments 2.6s'
ylabel = 'Theta power sum of 2.6s'

fig, ax = plt.subplots()

for i in range(tpower_arena_vhipp.shape[1]):
    ax.plot(tpower_arena_vhipp[20:60, i])

plt.xlabel(xlabel)
plt.ylabel(ylabel)
plt.title(title)
plt.savefig(sdir_arena + title + '.png')
plt.show()

#%%
# Plot gamma power of individual channel in the mPFc

for i in range(gpower_arena_mpfc.shape[1]):
    plt.plot(gpower_arena_mpfc[20:60, i])

plt.xlabel('Time segments 2.6s')
plt.ylabel('Gamma power')
plt.title('All channels in mPFC')
plt.show()

#%%
# Plot gamma power of individual channel in the vHPC

for i in range(gpower_arena_vhipp.shape[1]):
    plt.plot(gpower_arena_vhipp[20:60, i])

plt.xlabel('Time segments 2.6s')
plt.ylabel('Gamma power')
plt.title('All channels in vHPC')
plt.show()
#%%
# Plot mean theta power of all channels in the vHPC
m = tpower_arena_vhipp.mean(axis=1)[20:60]
sd = np.std(tpower_arena_vhipp, axis=1)[20:60]
x = np.arange(tpower_arena_vhipp.shape[0])[20:60]

plt.plot(x, m)
plt.fill_between(x, m+sd, m-sd, alpha=0.6)
# plt.plot(tpower_arena0218_vhipp.mean(axis=1), label='vHPC_arena')
# plt.legend()
plt.xlabel('Time segments 2.6s')
plt.ylabel('Theta power')
plt.title('All channel in vHPC')
plt.show()

#%%
# Plot mean theta power of all channels in the mPFC

m = tpower_arena_mpfc.mean(axis=1)[20:60]
sd = np.std(tpower_arena_mpfc, axis=1)[20:60]
x = np.arange(tpower_arena_mpfc.shape[0])[20:60]

plt.plot(x, m)
plt.fill_between(x, m+sd, m-sd, alpha=0.6)
# plt.legend()
plt.xlabel('Time segments 2.6s')
plt.ylabel('Theta power')
plt.title('All channel in mPFC')
plt.show()

#%%
# plt.plot(tpower_arena_bef_mpfc.mean(axis=1)[20:120], label='mPFC_arena_bef')
# plt.plot(tpower_arena_bef_vhipp.mean(axis=1)[20:120], label='vHPC_arena_bef')
# plt.legend()
# plt.xlabel('Time segments 2.6s')
# plt.ylabel('Theta power')
# plt.show()

plt.plot(tpower_arena_mpfc.mean(axis=1)[20:120], label='mPFC_Arena')
plt.plot(tpower_arena_vhipp.mean(axis=1)[20:120], label='vHPC_Arena')
plt.legend()
plt.xlabel('Time segments 2.6s')
plt.ylabel('Theta power')
plt.title('Arena')
plt.show()

plt.plot(tpower_ezm_mpfc.mean(axis=1)[20:120], label='mPFC_EZM')
plt.plot(tpower_ezm_vhipp.mean(axis=1)[20:120], label='vHPC_EZM')
plt.xlabel('Time segments 2.6s')
plt.ylabel('Theta power')
plt.title('EZM')
plt.legend()
plt.show()

plt.plot(tpower_oft_mpfc.mean(axis=1)[20:120], label='mPFC_OFT')
plt.plot(tpower_oft_vhipp.mean(axis=1)[20:120], label='vHPC_OFT')
plt.xlabel('Time segments 2.6s')
plt.ylabel('Theta power')
plt.title('OFT')
plt.legend()
plt.show()

#%%

# Plot mean gamma power of all channels in the vHPC
m = gpower_arena_vhipp.mean(axis=1)[20:60]
sd = np.std(gpower_arena_vhipp, axis=1)[20:60]
x = np.arange(gpower_arena_vhipp.shape[0])[20:60]

plt.plot(x, m)
plt.fill_between(x, m+sd, m-sd, alpha=0.6)

plt.xlabel('Time segments 2.6s')
plt.ylabel('Gamma power')
plt.title('All channel in vHPC')
plt.show()

#%%
# Plot mean gamma power of all channels in the mPFC

m = gpower_arena_mpfc.mean(axis=1)[20:60]
sd = np.std(gpower_arena_mpfc, axis=1)[20:60]
x = np.arange(gpower_arena_mpfc.shape[0])[20:60]

plt.plot(x, m)
plt.fill_between(x, m+sd, m-sd, alpha=0.6)

plt.xlabel('Time segments 2.6s')
plt.ylabel('Theta power')
plt.title('All channel in mPFC')
plt.show()

#%%
# plt.plot(gpower_arena_bef_mpfc.mean(axis=1)[20:120], label='mPFC_arena_bef')
# plt.plot(gpower_arena_bef_vhipp.mean(axis=1)[20:120], label='vHPC_arena_bef')
# plt.legend()
# plt.xlabel('Time segments 2.6s')
# plt.ylabel('Theta power')
# plt.show()

plt.plot(gpower_arena_mpfc.mean(axis=1)[20:120], label='mPFC_Arena')
plt.plot(gpower_arena_vhipp.mean(axis=1)[20:120], label='vHPC_Arena')
plt.legend()
plt.xlabel('Time segments 2.6s')
plt.ylabel('Gamma power')
plt.title('Arena')
plt.show()

plt.plot(gpower_ezm_mpfc.mean(axis=1)[20:120], label='mPFC_EZM')
plt.plot(gpower_ezm_vhipp.mean(axis=1)[20:120], label='vHPC_EZM')
plt.xlabel('Time segments 2.6s')
plt.ylabel('Gamma power')
plt.title('EZM')
plt.legend()
plt.show()

plt.plot(gpower_oft_mpfc.mean(axis=1)[20:120], label='mPFC_OFT')
plt.plot(gpower_oft_vhipp.mean(axis=1)[20:120], label='vHPC_OFT')
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
xlabel = 'Theta power mPFC'
ylabel = 'Theta power vHPC'

# x = tpower_arena_bef_mpfc.mean(axis=1)[20:250]
# y = tpower_arena_bef_vhipp.mean(axis=1)[20:250]
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

## plot arena
x = tpower_arena_mpfc.mean(axis=1)[20:250]
y = tpower_arena_vhipp.mean(axis=1)[20:250]

corr = np.corrcoef(x, y, rowvar=True)

plt.figure(figsize=(6,6))
plt.scatter(x, y)

m, b = np.polyfit(x, y, 1)
plt.plot(x, m*x + b)

title = 'Theta power correlation mPFC-vHPC arena' + '\n' + 'R sequred = {:.2f}'.format(np.square(corr[0][1]))
plt.gcf().subplots_adjust(bottom=0.15, left=0.18)
plt.xlabel(xlabel)
plt.ylabel(ylabel)
plt.title(title)
title = 'Theta power correlation mPFC-vHPC arena'
plt.savefig(sdir_arena + title + '.png')
plt.show()

## plot ezm
x = tpower_ezm_mpfc.mean(axis=1)[20:250]
y = tpower_ezm_vhipp.mean(axis=1)[20:250]

corr = np.corrcoef(x, y, rowvar=True)

plt.figure(figsize=(6,6))
plt.scatter(x, y)

m, b = np.polyfit(x, y, 1)
plt.plot(x, m*x + b)

title = 'Theta power correlation mPFC-vHPC ezm' + '\n' + 'R sequred = {:.2f}'.format(np.square(corr[0][1]))
plt.gcf().subplots_adjust(bottom=0.15, left=0.18)
plt.xlabel(xlabel)
plt.ylabel(ylabel)
plt.title(title)
title = 'Theta power correlation mPFC-vHPC ezm'
plt.savefig(sdir_ezm + title + '.png')
plt.show()

## plot oft
x = tpower_oft_mpfc.mean(axis=1)[20:250]
y = tpower_oft_vhipp.mean(axis=1)[20:250]

corr = np.corrcoef(x, y, rowvar=True)

plt.figure(figsize=(6,6))
plt.scatter(x, y)

m, b = np.polyfit(x, y, 1)
plt.plot(x, m*x + b)

title = 'Theta power correlation mPFC-vHPC oft' + '\n' + 'R sequred = {:.2f}'.format(np.square(corr[0][1]))
plt.gcf().subplots_adjust(bottom=0.15, left=0.18)
plt.xlabel(xlabel)
plt.ylabel(ylabel)
plt.title(title)
title = 'Theta power correlation mPFC-vHPC oft'
plt.savefig(sdir_oft + title + '.png')
plt.show()

#%%
### Plot gamma power correlation between mPFC and vHPC

# x = gpower_arena_bef_mpfc.mean(axis=1)[20:250]
# y = gpower_arena_bef_vhipp.mean(axis=1)[20:250]
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

x = gpower_arena_mpfc.mean(axis=1)[20:250]
y = gpower_arena_vhipp.mean(axis=1)[20:250]

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


x = gpower_ezm_mpfc.mean(axis=1)[20:250]
y = gpower_ezm_vhipp.mean(axis=1)[20:250]

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

x = gpower_oft_mpfc.mean(axis=1)[20:250]
y = gpower_oft_vhipp.mean(axis=1)[20:250]

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
x = gpower_oft_mpfc.mean(axis=1)[20:250]
y = gpower_oft_vhipp.mean(axis=1)[20:250]

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
    #      'Arena_bef_mPFC':tpower_arena_bef_mpfc.mean(axis=1)[20:250],

    'Arena_mPFC': tpower_arena_mpfc.mean(axis=1)[20:250],
    'EZM_mPFC': tpower_ezm_mpfc.mean(axis=1)[20:250],
    'OFT_mPFC': tpower_oft_mpfc.mean(axis=1)[20:250],

    #     'Arena_bef_vHPC': tpower_arena_bef_vhipp.mean(axis=1)[20:250],
    'Arena_vHPC': tpower_arena_vhipp.mean(axis=1)[20:250],
    'EZM_vHPC': tpower_ezm_vhipp.mean(axis=1)[20:250],
    'OFT_vHPC': tpower_oft_vhipp.mean(axis=1)[20:250]
}

df = pd.DataFrame(d)

# 'Group': ['mPFC', 'mPFC', 'mPFC', 'vHPC', 'vHPC','vHPC'],

# dd=pd.melt(df,id_vars=['Group'],value_vars=df.columns,var_name='fruits')
mean_sd = df.agg([np.mean, np.std])
ms = mean_sd.T
ms.plot(kind = "bar", legend = False,
          yerr = "std", color='gray', rot=45)
plt.ylabel('Theta power')
plt.title('Theta power_2.6s_10min')
plt.gcf().subplots_adjust(bottom=0.25, left=0.15)
plt.show()

#%%

## generate a bar plot of the gamma power in Arena, EZM and OFT

d = {
    #      'Arena_bef_mPFC':gpower_arena_bef_mpfc.mean(axis=1)[20:250],

    'Arena_mPFC': gpower_arena_mpfc.mean(axis=1)[20:250],
    'EZM_mPFC': gpower_ezm_mpfc.mean(axis=1)[20:250],
    'OFT_mPFC': gpower_oft_mpfc.mean(axis=1)[20:250],

    #     'Arena_bef_vHPC': gpower_arena_bef_vhipp.mean(axis=1)[20:250],
    'Arena_vHPC': gpower_arena_vhipp.mean(axis=1)[20:250],
    'EZM_vHPC': gpower_ezm_vhipp.mean(axis=1)[20:250],
    'OFT_vHPC': gpower_oft_vhipp.mean(axis=1)[20:250]
}

df = pd.DataFrame(d)

# 'Group': ['mPFC', 'mPFC', 'mPFC', 'vHPC', 'vHPC','vHPC'],

# dd=pd.melt(df,id_vars=['Group'],value_vars=df.columns,var_name='fruits')
mean_sd = df.agg([np.mean, np.std])
ms = mean_sd.T
ms.plot(kind = "bar", legend = False,
          yerr = "std", color='gray', rot=45)
plt.ylabel('Gamma power')
plt.title('Theta power_2.6s_10min')
plt.gcf().subplots_adjust(bottom=0.25, left=0.15)
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
