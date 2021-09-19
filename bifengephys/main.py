#%%
import sys

import scipy.signal
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
plt.rcParams["lines.linewidth"] = 1.8


# sys.path.append('D:\ephys')

import os

# path = 'D:\ephys'
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

mBWfus025 = {
    'arena_0802_bef': '2021-08-02_mBWfus025_bef_arena_ephys',
    'cage_0802': '2021-08-02_mBWfus025_cage_arena_ephys',
    'recover_0802': '2021-08-02_mBWfus025_recover_arena_ephys',
    'arena_0802_aft': '2021-08-02_mBWfus025_aft_arena_ephys',
    'ezm_0802': '2021-08-02_mBWfus025_EZM_ephys',
    'oft_0802': '2021-08-02_mBWfus025_OFT_ephys',

    'arena_0806_bef': '2021-08-06_mBWfus025_bef_arena_ephys',
    'cage_0806': '2021-08-06_mBWfus025_cage_arena_ephys',
    'recover_0806': '2021-08-06_mBWfus025_recover_arena_ephys',
    'arena_0806_aft': '2021-08-06_mBWfus025_aft_arena_ephys',
    'ezm_0806': '2021-08-06_mBWfus025_EZM_ephys',
    'oft_0806': '2021-08-06_mBWfus025_OFT_ephys',
                 }

mBWfus026 ={
    'arena_0804_bef': '2021-08-04_mBWfus026_bef_arena_ephys',
    'cage_0804': '2021-08-04_mBWfus026_cage_arena_ephys',
    'recover_0804': '2021-08-04_mBWfus026_recover_arena_ephys',
    'arena_0804_aft': '2021-08-04_mBWfus026_aft_arena_ephys',
    'ezm_0804': '2021-08-04_mBWfus026_EZM_ephys',
    'oft_0804': '2021-08-04_mBWfus026_OFT_ephys',

    'arena_0807_bef': '2021-08-07_mBWfus026_bef_arena_ephys',
    'cage_0807': '2021-08-07_mBWfus026_cage_arena_ephys',
    'recover_0807': '2021-08-07_mBWfus026_recover_arena_ephys',
    'arena_0807_aft': '2021-08-07_mBWfus026_aft_arena_ephys',
    'ezm_0807': '2021-08-07_mBWfus026_EZM_ephys',
    'oft_0807': '2021-08-07_mBWfus026_OFT_ephys',}

mBWfus027  ={
    'arena_0804_bef': '2021-08-04_mBWfus027_bef_arena_ephys',
    'cage_0804': '2021-08-04_mBWfus027_cage_arena_ephys',
    'recover_0804': '2021-08-04_mBWfus027_recover_arena_ephys',
    'arena_0804_aft': '2021-08-04_mBWfus027_aft_arena_ephys',
    'ezm_0804': '2021-08-04_mBWfus027_EZM_ephys',
    'oft_0804': '2021-08-04_mBWfus027_OFT_ephys',

    'arena_0807_bef': '2021-08-07_mBWfus027_bef_arena_ephys',
    'cage_0807': '2021-08-07_mBWfus027_cage_arena_ephys',
    'recover_0807': '2021-08-07_mBWfus027_recover_arena_ephys',
    'arena_0807_aft': '2021-08-07_mBWfus027_aft_arena_ephys',
    'ezm_0807': '2021-08-07_mBWfus027_EZM_ephys',
    'oft_0807': '2021-08-07_mBWfus027_OFT_ephys',}

mBWfus028  ={
    'arena_0805_bef': '2021-08-05_mBWfus028_bef_arena_ephys',
    'cage_0805': '2021-08-05_mBWfus028_cage_arena_ephys',
    'recover_0805': '2021-08-05_mBWfus028_recover_arena_ephys',
    'arena_0805_aft': '2021-08-05_mBWfus028_aft_arena_ephys',
    'ezm_0805': '2021-08-05_mBWfus028_EZM_ephys',
    'oft_0805': '2021-08-05_mBWfus028_OFT_ephys',

    'arena_0808_bef': '2021-08-08_mBWfus028_bef_arena_ephys',
    'cage_0808': '2021-08-08_mBWfus028_cage_arena_ephys',
    'recover_0808': '2021-08-08_mBWfus028_recover_arena_ephys',
    'arena_0808_aft': '2021-08-08_mBWfus028_aft_arena_ephys',
    'ezm_0808': '2021-08-08_mBWfus028_EZM_ephys',
    'oft_0808': '2021-08-08_mBWfus028_OFT_ephys',}

mBWfus029  ={
    'arena_0803_bef': '2021-08-03_mBWfus029_bef_arena_ephys',
    'cage_0803': '2021-08-03_mBWfus029_cage_arena_ephys',
    'recover_0803': '2021-08-03_mBWfus029_recover_arena_ephys',
    'arena_0803_aft': '2021-08-03_mBWfus029_aft_arena_ephys',
    'ezm_0803': '2021-08-03_mBWfus029_EZM_ephys',
    'oft_0803': '2021-08-03_mBWfus029_OFT_ephys',

    'arena_0808_bef': '2021-08-08_mBWfus029_bef_arena_ephys',
    'cage_0808': '2021-08-08_mBWfus029_cage_arena_ephys',
    'recover_0808': '2021-08-08_mBWfus029_recover_arena_ephys',
    'arena_0808_aft': '2021-08-08_mBWfus029_aft_arena_ephys',
    'ezm_0808': '2021-08-08_mBWfus029_EZM_ephys',
    'oft_0808': '2021-08-08_mBWfus029_OFT_ephys',}

mBWfus031  ={
    'arena_0805_bef': '2021-08-05_mBWfus031_bef_arena_ephys',
    'cage_0805': '2021-08-05_mBWfus031_cage_arena_ephys',
    'recover_0805': '2021-08-05_mBWfus031_recover_arena_ephys',
    'arena_0805_aft': '2021-08-05_mBWfus031_aft_arena_ephys',
    'ezm_0805': '2021-08-05_mBWfus031_EZM_ephys',
    'oft_0805': '2021-08-05_mBWfus031_OFT_ephys',

    'arena_0809_bef': '2021-08-09_mBWfus031_bef_arena_ephys',
    'cage_0809': '2021-08-09_mBWfus031_cage_arena_ephys',
    'recover_0809': '2021-08-09_mBWfus031_recover_arena_ephys',
    'arena_0809_aft': '2021-08-09_mBWfus031_aft_arena_ephys',
    'ezm_0809': '2021-08-09_mBWfus031_EZM_ephys',
    'oft_0809': '2021-08-09_mBWfus031_OFT_ephys',}

mBWfus032  ={
    'arena_0806_bef': '2021-08-06_mBWfus032_bef_arena_ephys',
    'cage_0806': '2021-08-06_mBWfus032_cage_arena_ephys',
    'recover_0806': '2021-08-06_mBWfus032_recover_arena_ephys',
    'arena_0806_aft': '2021-08-06_mBWfus032_aft_arena_ephys',
    'ezm_0806': '2021-08-06_mBWfus032_EZM_ephys',
    'oft_0806': '2021-08-06_mBWfus032_OFT_ephys',

    'arena_0809_bef': '2021-08-09_mBWfus032_bef_arena_ephys',
    'cage_0809': '2021-08-09_mBWfus032_cage_arena_ephys',
    'recover_0809': '2021-08-09_mBWfus032_recover_arena_ephys',
    'arena_0809_aft': '2021-08-09_mBWfus032_aft_arena_ephys',
    'ezm_0809': '2021-08-09_mBWfus032_EZM_ephys',
    'oft_0809': '2021-08-09_mBWfus032_OFT_ephys',}


#%%
### for updating infomation
# animal = mBWfus032
#
# for key in animal:
#     session = animal[key]
#     dir_save = session + '/ephys_processed/'
#     data = ephys.load_data(session)
#     wanted_ch= [ 1,  2,  4,  5,  8,  9, 10, 11, 12, 13, 14, 15, 17, 18, 20, 21,
#                 22, 23, 24, 25, 26, 28, 37, 38, 41, 42, 43, 44, 45, 46, 47, 48, 49,
#                 50] ## mBWfus032
#     data['info']['wanted_channel'] = wanted_ch
#     with open(dir_save + session + '_dataset.pkl', 'wb') as f:
#         pickle.dump(data, f)
#         f.close()


##### Ephys analysis

#%%
animal = mBWfus025
date = '0802'


data_cage = ephys.load_data(animal['cage_' + date])
sdir_cage = animal['cage_' + date] + '/results/'
if not os.path.exists(sdir_cage):
    os.makedirs(sdir_cage)
print(sdir_cage)

# # # #
data_arena_bef = ephys.load_data(animal['arena_' + date + '_bef'])
sdir_arena_bef = animal['arena_' + date + '_bef'] + '/results/'
if not os.path.exists(sdir_arena_bef):
    os.makedirs(sdir_arena_bef)
print(sdir_arena_bef)

data_recover = ephys.load_data(animal['recover_' + date])
sdir_recover = animal['recover_' + date] + '/results/'
if not os.path.exists(sdir_recover):
    os.makedirs(sdir_recover)
print(sdir_recover)
# # # #
data_arena = ephys.load_data(animal['arena_' + date + '_aft'])
sdir_arena = animal['arena_' + date + '_aft'] + '/results/'
if not os.path.exists(sdir_arena):
    os.makedirs(sdir_arena)
print(sdir_arena)

# data_arena = ephys.load_data(animal['arena_' + date])
# sdir_arena = animal['arena_' + date] + '/figures/'
# if not os.path.exists(sdir_arena):
#     os.makedirs(sdir_arena)
# print(sdir_arena)

data_ezm = ephys.load_data(animal['ezm_' + date])
sdir_ezm = animal['ezm_' + date] + '/results/'
if not os.path.exists(sdir_ezm):
    os.makedirs(sdir_ezm)
print(sdir_ezm)
#
data_oft = ephys.load_data(animal['oft_' + date])
sdir_oft = animal['oft_' + date] + '/results/'
if not os.path.exists(sdir_oft):
    os.makedirs(sdir_oft)
print(sdir_oft)

wanted_ch = data_cage['info']['wanted_channel'] ## the wanted_ch are the pads ordered by the depth
mpfc_ch = [str(el) for el in wanted_ch if el < 32]
vhipp_ch = [str(el) for el in wanted_ch if el >= 32]

#%%

lfp = ephys.column_by_pad(ephys.get_lfp(data_oft))
print(lfp.columns)

corr_matrix = lfp.iloc[10000:200000].corr() # 10 second

plt.imshow(corr_matrix)
cb = plt.colorbar()
plt.xticks(range(corr_matrix.select_dtypes(['number']).shape[1]), corr_matrix.select_dtypes(['number']).columns, fontsize=8, rotation=90)
plt.yticks(range(corr_matrix.select_dtypes(['number']).shape[1]), corr_matrix.select_dtypes(['number']).columns, fontsize=8)
plt.show()

#%%
# bad_ch_arena = [23, 24, 25, 26, 27, 28, 31, 38] # mBWfus008 arena_0219
# bad_ch_ezm = [0, 23, 24, 25, 26, 27, 28, 31] # mBWfus008 ezm_0219
# bad_ch_oft = [ 23, 24, 25, 26, 27, 28, 31, 38] # mBWfus008 oft_0219

# wanted_ch= [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20,  21, 22,
#             33, 34, 36, 37, 40, 41, 42, 43, 44, 45, 46, 47, 48 ,49, 50] # mBWfus008

# bad_ch_arena = [7, 10, 23, 44, 57, 58, 59, 60, 61, 62] ### mBWfus009 arena_0219
# bad_ch_ezm = [6, 7, 10, 23, 44, 57, 58, 59, 60, 61, 62] ### mBWfus009 ezm_0219
# bad_ch_oft = [ 7, 10, 23, 44, 57, 58, 59, 60, 61, 62] ### mBWfus009 oft_0219
# wanted_ch= [0, 1, 2, 4, 5, 8, 9, 11, 12, 13, 14, 15, 16, 17, 18, 19,
#             32, 33, 34, 36, 37, 38, 40] #, 20,  21, 22, 45, 46, 47, 48 ,49, 50] # mBWfus009

# bad_ch_arena = [6, 7, 10, 23, 44, 57, 58, 59] ### mBWfus009 arena_0226
# bad_ch_ezm = [6, 7, 10, 23, 44, 57, 58, 59] ### mBWfus009 ezm_0226
# bad_ch_oft = [6, 7, 10, 23, 44] ### mBWfus009 oft_0226

# bad_ch_arena = [7, 10, 23, 44, 57, 58, 59, 60, 61, 62] ### mBWfus009 arena_0305
# bad_ch_ezm = [7, 10, 23, 44, 57, 58, 59, 60, 61, 62] ### mBWfus009 ezm_0305
# bad_ch_oft = [7, 10, 23, 44, 57, 58, 59, 60, 61, 62] ### mBWfus009 oft_0305

# bad_ch = [27, 28, 31, 32, 46, 52] # mBWfus011 arena_0226,0305, 0313
# wanted_ch= [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
#            35, 36, 37, 38, 39, 40, 41, 42, 44, 45, 48 ,49, 50, 51, 53,
#             54, 55, 56, 57, 58, 59, 60] # mBWfus011

# wanted_ch= [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 14, 15, 16,
#            35, 36, 38, 48 ,49, 50, 51, 53,54] # mBWfus012

# wanted_ch = [ 0,  1,  2, 5,  7,  8, 10, 11, 12, 14, 16, 17, 18, 19, 20, 21,
#             22, 24, 25, 26, 27, 28, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44,
#             45, 46, 47, 48, 50, 54, 55, 58, 60, 61, 62] ## mBWfus025

# wanted_ch = [ 1, 5, 9, 53, 55, 56, 57] ## mBWfus026

# wanted_ch = [ 1,  2,  4,  5,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 19, 20, 21,
#             24, 25, 26, 27, 28, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 45, 46,
#             48, 49, 50, 54, 55, 57] ## mBWfus027

# wanted_ch = [ 0,  1,  4,  5,  8,  9, 10, 11, 13, 14, 15, 16, 18, 19, 20, 21,
#             22, 24, 28, 34, 35, 36, 37, 38, 39, 40, 41, 44, 45, 46, 47, 48,
#             49, 50, 51, 55, 57, 58] ## mBWfus028

# wanted_ch = [2, 3, 4, 5, 9, 10, 11, 12, 13, 14, 15, 17, 19, 24, 26] ## mBWfus029
#
# wanted_ch= [ 0,  1,  2,  4,  5,  7,  8,  9, 10, 11, 12, 14, 15, 16, 17, 18, 19,
#             22, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46,
#             47, 48, 49, 50, 51, 53, 54, 57, 58, 60, 61, 62, 63] ## mBWfus031
# #
# wanted_ch= [ 1,  2,  4,  5,  8,  9, 10, 11, 12, 13, 14, 15, 17, 18, 20, 21,
#             22, 23, 24, 25, 26, 28, 37, 38, 41, 42, 43, 44, 45, 46, 47, 48, 49,
#             50] ## mBWfus032



#%%
lfp = ephys.column_by_pad(ephys.get_lfp(data_cage))
lfp = lfp[wanted_ch]
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

lfp_filted = mne.filter.filter_data(data=lfp.T, sfreq=500, l_freq=0.5, h_freq=None) ## high-pass filter at 0.5Hz
raw_cage = mne.io.RawArray(lfp_filted, info) ### mne data format (n-channels, n-samples), unit = V

print('Mission Completed')


#%%
lfp = ephys.column_by_pad(ephys.get_lfp(data_arena_bef))
lfp = lfp[wanted_ch]
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

lfp_filted = mne.filter.filter_data(data=lfp.T, sfreq=500, l_freq=0.5, h_freq=None) ## high-pass filter at 0.5Hz
raw_arena_bef = mne.io.RawArray(lfp_filted, info) ### mne data format (n-channels, n-samples), unit = V

print('Mission Completed')

#%%
lfp = ephys.column_by_pad(ephys.get_lfp(data_recover))
lfp = lfp[wanted_ch]
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

lfp_filted = mne.filter.filter_data(data=lfp.T, sfreq=500, l_freq=0.5, h_freq=None) ## high-pass filter at 0.5Hz
raw_recover = mne.io.RawArray(lfp_filted, info) ### mne data format (n-channels, n-samples), unit = V

print('Mission Completed')

#%%
lfp = ephys.column_by_pad(ephys.get_lfp(data_arena))
lfp = lfp[wanted_ch]
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

lfp_filted = mne.filter.filter_data(data=lfp.T, sfreq=500, l_freq=0.5, h_freq=None)

raw_arena = mne.io.RawArray(lfp_filted, info) ### mne data format (n-channels, n-samples), unit = mV

print('Mission Completed')

#%%
lfp = ephys.column_by_pad(ephys.get_lfp(data_ezm))
lfp = lfp[wanted_ch]
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

lfp_filted = mne.filter.filter_data(data=lfp.T, sfreq=500, l_freq=0.5, h_freq=None)

raw_ezm = mne.io.RawArray(lfp_filted, info) ### mne data format (n-channels, n-samples) V
print('Mission Completed')
#%%
lfp = ephys.column_by_pad(ephys.get_lfp(data_oft))
lfp = lfp[wanted_ch]
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

lfp_filted = mne.filter.filter_data(data=lfp.T, sfreq=500, l_freq=0.5, h_freq=None)

raw_oft = mne.io.RawArray(lfp_filted, info) ### mne data format (n-channels, n-samples) mV
print('Mission Completed')

#%%

raw_cage.plot(n_channels = 46, duration=2, scalings='auto')

#%%
raw_cage.plot_psd(fmin=1, fmax=100, tmin=0, tmax=300, proj=False, n_fft=2048,
                   n_overlap=256, reject_by_annotation=True, picks=None, ax=None, color='black',
                   xscale='linear', area_mode='std', area_alpha=0.15, dB=False,
                   estimate='power', show=True, n_jobs=20, average=False, line_alpha=None, spatial_colors=True,
                   sphere=None, window='hamming', verbose=None)

#%%
title = 'Power Spectrum'
xlabel = 'Frequency (Hz)'
ylabel = 'Power (uV**2/Hz)'
power_spectrum = {}
workers=20

#%%
pad_name = [str(el) for el in np.arange(0, 64)]
depth = np.arange(2170, -1, -70)
depth_list = np.concatenate([depth, depth])
pad_depth =dict(zip(pad_name, depth_list))
#%%
ch_depth = {}
#%%
data = raw_cage.copy()
psds_cage, freqs = mne.time_frequency.psd_welch(data, fmin=0, fmax=100, tmin=0, tmax=600, n_fft=1024, n_overlap=128,
                                  n_per_seg=256, picks=None, proj=False, n_jobs=workers, reject_by_annotation=True,
                                  average='mean', window='hamming', verbose=None)
# # Convert power to dB scale.
# psds_cage = 10 * np.log10(psds_cage)
#%%
import seaborn as sns

psds_cage_df = pd.DataFrame(psds_cage.T, columns=ch_list)
psds_cage_mpfc = psds_cage_df[mpfc_ch]
psds_cage_vhpc = psds_cage_df[vhipp_ch]
depth = [pad_depth[key] for key in mpfc_ch]
psds_cage_mpfc.loc['pad_depth']= depth
psds_cage_mpfc['freqs'] = freqs
psds_cage_vhpc['freqs'] = freqs

psds_cage_mpfc.to_csv(sdir_cage + 'mPFC_spectrum.csv')
psds_cage_vhpc.to_csv(sdir_cage + 'vHPC_spectrum.csv')


palette = sns.color_palette("rocket_r")
#%%
sns.histplot(psds_cage_vhpc.iloc[:-1])
plt.xlim(-20, 500)
plt.show()

#%%
sns.set_theme(style="ticks")
dots = sns.load_dataset("dots")

# Define the palette as a list to specify exact values
palette = sns.color_palette("rocket_r")

# Plot the lines on two facets
sns.relplot(
    data=psds_cage_mpfc,
    x="freqs", y="1",
    hue="coherence", size="choice", col="align",
    kind="line", size_order=["T1", "T2"], palette=palette,
    height=5, aspect=.75, facet_kws=dict(sharex=False),
)

plt.show()


#%%


#%%
for _ in range(psds_cage.shape[0]):
    depth = pad_depth[str(_)]
    print(depth)

    #%%
    plt.plot(freqs, psds_cage[_, :])

plt.xlabel(xlabel)
plt.ylabel(ylabel)
plt.title(title + '_cage')
plt.gcf().subplots_adjust(bottom=0.15, left=0.18)
plt.ticklabel_format(axis='both', style="sci", scilimits=(0,3))
plt.savefig(sdir_cage + title + '.png', transparent=True)
plt.show()

#%%

for _ in range(psds_cage.shape[0]):
    plt.plot(freqs, psds_cage[_, :])

plt.xlabel(xlabel)
plt.ylabel(ylabel)
plt.title(title + '_cage')
plt.gcf().subplots_adjust(bottom=0.15, left=0.18)
plt.ticklabel_format(axis='both', style="sci", scilimits=(0, 3))
plt.savefig(sdir_cage + title + '.png', transparent=True)
plt.show()

mean_power_mPFC = psds_cage[:len(mpfc_ch), :].mean(axis=0)
power_spectrum['frequencies'] = freqs
power_spectrum['mean_power_mPFC_cage'] = mean_power_mPFC
sd_power_mPFC = psds_cage[:len(mpfc_ch), :].std(axis=0)
sd_power_mPFC_cage = sd_power_mPFC
plt.plot(freqs, mean_power_mPFC, label='mPFC')
plt.fill_between(freqs, mean_power_mPFC - sd_power_mPFC, mean_power_mPFC + sd_power_mPFC, alpha=0.15)

mean_power_vHPC = psds_cage[len(mpfc_ch):, :].mean(axis=0)
power_spectrum['mean_power_vHPC_cage'] = mean_power_vHPC
sd_power_vHPC = psds_cage[len(mpfc_ch):, :].std(axis=0)
sd_power_vHPC_cage = sd_power_vHPC
plt.plot(freqs, mean_power_vHPC, label='vHPC')
plt.fill_between(freqs, mean_power_vHPC - sd_power_vHPC, mean_power_vHPC + sd_power_vHPC,  alpha=0.15)

plt.xlabel(xlabel)
plt.ylabel(ylabel)
plt.title(title + '_cage')
plt.gcf().subplots_adjust(bottom=0.15, left=0.18)
plt.ticklabel_format(axis='both', style="sci", scilimits=(0,3))
plt.legend()
plt.savefig(sdir_cage + title + '_mean.png')
plt.show()

print('Mission Completed')

#%%

### compute power spectra using the Welch method
data = raw_arena_bef.copy()
psds_arena_bef, freqs = mne.time_frequency.psd_welch(data, fmin=0, fmax=100, tmin=00, tmax=600, n_fft=1024, n_overlap=128,
                                  n_per_seg=256, picks=None, proj=False, n_jobs=workers, reject_by_annotation=True,
                                  average='mean', window='hamming', verbose=None)
# # Convert power to dB scale.
# psds_arena_bef = 10 * np.log10(psds_arena_bef)

for _ in range(psds_arena_bef.shape[0]):
    plt.plot(freqs, psds_arena_bef[_, :])
plt.xlabel(xlabel)
plt.ylabel(ylabel)
plt.title(title + '_Arena_bef')
plt.gcf().subplots_adjust(bottom=0.15, left=0.18)
plt.ticklabel_format(axis='both', style="sci", scilimits=(0,3))
plt.savefig(sdir_arena_bef + title + '.png')
plt.show()

mean_power_mPFC = psds_arena_bef[:len(mpfc_ch), :].mean(axis=0)
power_spectrum['mean_power_mPFC_arena1'] = mean_power_mPFC
sd_power_mPFC = psds_arena_bef[:len(mpfc_ch), :].std(axis=0)
plt.plot(freqs, mean_power_mPFC, label='mPFC')
plt.fill_between(freqs, mean_power_mPFC - sd_power_mPFC, mean_power_mPFC + sd_power_mPFC, alpha=0.15)

mean_power_vHPC = psds_arena_bef[len(mpfc_ch):, :].mean(axis=0)
power_spectrum['mean_power_vHPC_arena1'] = mean_power_vHPC
sd_power_vHPC = psds_arena_bef[len(mpfc_ch):, :].std(axis=0)
plt.plot(freqs, mean_power_vHPC, label='vHPC')
plt.fill_between(freqs, mean_power_vHPC - sd_power_vHPC, mean_power_vHPC + sd_power_vHPC,  alpha=0.15)

plt.xlabel(xlabel)
plt.ylabel(ylabel)
plt.title(title + '_Arena_bef')
plt.gcf().subplots_adjust(bottom=0.15, left=0.18)
plt.ticklabel_format(axis='both', style="sci", scilimits=(0,3))
plt.legend()
plt.savefig(sdir_arena_bef + title + '_mean.png')
plt.show()

print('Mission Completed')


#%%

### compute power spectra using the Welch method
data = raw_recover.copy()
psds_recover_10, freqs = mne.time_frequency.psd_welch(data, fmin=0, fmax=100, tmin=00, tmax=600, n_fft=1024, n_overlap=128,
                                  n_per_seg=256, picks=None, proj=False, n_jobs=workers, reject_by_annotation=True,
                                  average='mean', window='hamming', verbose=None)
# psds_recover_10 = 10 * np.log10(psds_recover_10)

psds_recover_20, freqs = mne.time_frequency.psd_welch(data, fmin=0, fmax=100, tmin=600, tmax=1200, n_fft=1024, n_overlap=128,
                                  n_per_seg=256, picks=None, proj=False, n_jobs=workers, reject_by_annotation=True,
                                  average='mean', window='hamming', verbose=None)
# psds_recover_20 = 10 * np.log10(psds_recover_20)

psds_recover_30, freqs = mne.time_frequency.psd_welch(data, fmin=0, fmax=100, tmin=1200, tmax=1800, n_fft=1024, n_overlap=128,
                                  n_per_seg=256, picks=None, proj=False, n_jobs=workers, reject_by_annotation=True,
                                  average='mean', window='hamming', verbose=None)
# psds_recover_30 = 10 * np.log10(psds_recover_30)

psds_recover_40, freqs = mne.time_frequency.psd_welch(data, fmin=0, fmax=100, tmin=1800, tmax=2400, n_fft=1024, n_overlap=128,
                                  n_per_seg=256, picks=None, proj=False, n_jobs=workers, reject_by_annotation=True,
                                  average='mean', window='hamming', verbose=None)

power_spectrum['mean_power_mPFC_recover_10min'] = psds_recover_10[:len(mpfc_ch), :].mean(axis=0)
power_spectrum['mean_power_vHPC_recover_10min'] = psds_recover_10[len(mpfc_ch):, :].mean(axis=0)

power_spectrum['mean_power_mPFC_recover_20min'] = psds_recover_20[:len(mpfc_ch), :].mean(axis=0)
power_spectrum['mean_power_vHPC_recover_20min'] = psds_recover_20[len(mpfc_ch):, :].mean(axis=0)

power_spectrum['mean_power_mPFC_recover_30min'] = psds_recover_30[:len(mpfc_ch), :].mean(axis=0)
power_spectrum['mean_power_vHPC_recover_30min'] = psds_recover_30[len(mpfc_ch):, :].mean(axis=0)

power_spectrum['mean_power_mPFC_recover_40min'] = psds_recover_40[:len(mpfc_ch), :].mean(axis=0)
power_spectrum['mean_power_vHPC_recover_40min'] = psds_recover_40[len(mpfc_ch):, :].mean(axis=0)


print('Mission Completed')

#%%
data = raw_arena.copy()
psds_arena, freqs = mne.time_frequency.psd_welch(data, fmin=0, fmax=100, tmin=0, tmax=600, n_fft=1024, n_overlap=128,
                                  n_per_seg=256, picks=None, proj=False, n_jobs=workers, reject_by_annotation=True,
                                  average='mean', window='hamming', verbose=None)

# # Convert power to dB scale.
# psds_arena = 10 * np.log10(psds_arena)

for _ in range(psds_arena.shape[0]):
    plt.plot(freqs, psds_arena[_, :])
plt.xlabel(xlabel)
plt.ylabel(ylabel)
plt.title(title + '_Arena')
plt.gcf().subplots_adjust(bottom=0.15, left=0.18)
plt.ticklabel_format(axis='both', style="sci", scilimits=(0,3))
plt.savefig(sdir_arena + title + '.png')
plt.show()

mean_power_mPFC = psds_arena[:len(mpfc_ch), :].mean(axis=0)
power_spectrum['mean_power_mPFC_arena2']= mean_power_mPFC
sd_power_mPFC = psds_arena[:len(mpfc_ch), :].std(axis=0)
plt.plot(freqs, mean_power_mPFC, label='mPFC')
plt.fill_between(freqs, mean_power_mPFC - sd_power_mPFC, mean_power_mPFC + sd_power_mPFC, alpha=0.15)

mean_power_vHPC = psds_arena[len(mpfc_ch):, :].mean(axis=0)
power_spectrum['mean_power_vHPC_arena2'] = mean_power_vHPC
sd_power_vHPC = psds_arena[len(mpfc_ch):, :].std(axis=0)
plt.plot(freqs, mean_power_vHPC, label='vHPC')
plt.fill_between(freqs, mean_power_vHPC - sd_power_vHPC, mean_power_vHPC + sd_power_vHPC,  alpha=0.15)

plt.xlabel(xlabel)
plt.ylabel(ylabel)
plt.title(title + '_Arena')
plt.gcf().subplots_adjust(bottom=0.15, left=0.18)
plt.ticklabel_format(axis='both', style="sci", scilimits=(0,3))
plt.legend()
plt.savefig(sdir_arena + title + '_mean.png')
plt.show()

print('Mission Completed')

#%%
data = raw_ezm.copy()
psds_ezm, freqs = mne.time_frequency.psd_welch(data, fmin=0, fmax=100, tmin=0, tmax=600, n_fft=1024, n_overlap=128,
                                  n_per_seg=256, picks=None, proj=False, n_jobs=workers, reject_by_annotation=True,
                                  average='mean', window='hamming', verbose=None)

# # Convert power to dB scale.
# psds_ezm = 10 * np.log10(psds_ezm)

for _ in range(psds_ezm.shape[0]):
    plt.plot(freqs, psds_ezm[_, :])
plt.xlabel(xlabel)
plt.ylabel(ylabel)
plt.title(title + '_EZM')
plt.gcf().subplots_adjust(bottom=0.15, left=0.18)
plt.ticklabel_format(axis='both', style="sci", scilimits=(0,3))
plt.savefig(sdir_ezm + title + '.png')
plt.show()

mean_power_mPFC = psds_ezm[:len(mpfc_ch), :].mean(axis=0)
power_spectrum['mean_power_mPFC_ezm'] = mean_power_mPFC
sd_power_mPFC = psds_ezm[:len(mpfc_ch), :].std(axis=0)
plt.plot(freqs, mean_power_mPFC, label='mPFC')
plt.fill_between(freqs, mean_power_mPFC - sd_power_mPFC, mean_power_mPFC + sd_power_mPFC, alpha=0.15)

mean_power_vHPC = psds_ezm[len(mpfc_ch):, :].mean(axis=0)
power_spectrum['mean_power_vHPC_ezm'] = mean_power_vHPC
sd_power_vHPC = psds_ezm[len(mpfc_ch):, :].std(axis=0)
plt.plot(freqs, mean_power_vHPC, label='vHPC')
plt.fill_between(freqs, mean_power_vHPC - sd_power_vHPC, mean_power_vHPC + sd_power_vHPC,  alpha=0.15)

plt.xlabel(xlabel)
plt.ylabel(ylabel)
plt.title(title + '_EZM')
plt.gcf().subplots_adjust(bottom=0.15, left=0.18)
plt.ticklabel_format(axis='both', style="sci", scilimits=(0,3))
plt.legend()
plt.savefig(sdir_ezm + title + '_mean.png')
plt.show()

print('Mission Completed')

#%%
data = raw_oft.copy()
psds_oft, freqs = mne.time_frequency.psd_welch(data, fmin=0, fmax=100, tmin=0, tmax=600, n_fft=1024, n_overlap=128,
                                  n_per_seg=256, picks=None, proj=False, n_jobs=workers, reject_by_annotation=True,
                                  average='mean', window='hamming', verbose=None)

# # Convert power to dB scale.
# psds_oft = 10 * np.log10(psds_oft)

for _ in range(psds_oft.shape[0]):
    plt.plot(freqs, psds_oft[_, :])
plt.xlabel(xlabel)
plt.ylabel(ylabel)
plt.title(title + '_OFT')
plt.gcf().subplots_adjust(bottom=0.15, left=0.18)
plt.ticklabel_format(axis='both', style="sci", scilimits=(0,3))
plt.savefig(sdir_oft + title + '.png')
plt.show()

mean_power_mPFC = psds_oft[:len(mpfc_ch), :].mean(axis=0)
power_spectrum['mean_power_mPFC_oft'] = mean_power_mPFC
sd_power_mPFC = psds_oft[:len(mpfc_ch), :].std(axis=0)
plt.plot(freqs, mean_power_mPFC, label='mPFC')
plt.fill_between(freqs, mean_power_mPFC - sd_power_mPFC, mean_power_mPFC + sd_power_mPFC, alpha=0.15)

mean_power_vHPC = psds_oft[len(mpfc_ch):, :].mean(axis=0)
power_spectrum['mean_power_vHPC_oft'] = mean_power_vHPC
sd_power_vHPC = psds_oft[len(mpfc_ch):, :].std(axis=0)
plt.plot(freqs, mean_power_vHPC, label='vHPC')
plt.fill_between(freqs, mean_power_vHPC - sd_power_vHPC, mean_power_vHPC + sd_power_vHPC,  alpha=0.15)

plt.xlabel(xlabel)
plt.ylabel(ylabel)
plt.title(title + '_OFT')
plt.gcf().subplots_adjust(bottom=0.15, left=0.18)
plt.ticklabel_format(axis='both', style="sci", scilimits=(0,3))
plt.legend()
plt.savefig(sdir_oft + title + '_mean.png')
plt.show()


print('Mission Completed')

#%%
power_spectrum1 = pd.DataFrame(power_spectrum)
power_spectrum1.to_csv(sdir_cage + 'power_spectrum_all_tasks.csv' )

low_theta_power = power_spectrum1[(power_spectrum1['frequencies']>=3.5) & (power_spectrum1['frequencies']<=6.5)].sum()
high_theta_power = power_spectrum1[(power_spectrum1['frequencies']>=6.5) & (power_spectrum1['frequencies']<=12.5)].sum()
low_theta_power.to_csv(sdir_cage + 'low_theta_power_all_tasks.csv' )
high_theta_power.to_csv(sdir_cage + 'high_theta_power_all_tasks.csv' )
#%%
power_spectrum1.iloc[:40].plot(x = 'frequencies',
                    y = ['mean_power_mPFC_cage',
                         'mean_power_mPFC_arena1',
                         'mean_power_mPFC_recover_40min',
                         'mean_power_mPFC_arena2',
                         'mean_power_mPFC_ezm',
                         'mean_power_mPFC_oft',
                         ])
plt.xlabel(xlabel)
plt.ylabel(ylabel)
plt.title(title + '_mPFC')
plt.gcf().subplots_adjust(bottom=0.15, left=0.18)
plt.ticklabel_format(axis='y', style="sci", scilimits=(0,0))
plt.savefig(sdir_cage + title + '_mPFC_all_tasks.png')
plt.show()

#%%
power_spectrum1.iloc[:40].plot(x = 'frequencies',
                    y = ['mean_power_vHPC_cage',
                         'mean_power_vHPC_arena1',
                         'mean_power_vHPC_recover_40min',
                         'mean_power_vHPC_arena2',
                         'mean_power_vHPC_ezm',
                         'mean_power_vHPC_oft'])
plt.xlabel(xlabel)
plt.ylabel(ylabel)
plt.title(title + '_vHPC')
plt.gcf().subplots_adjust(bottom=0.15, left=0.18)
plt.ticklabel_format(axis='y', style="sci", scilimits=(0,0))
plt.savefig(sdir_cage + title + '_vHPC_all_tasks.png')
plt.show()


#%%
power_spectrum1.iloc[:40].plot(x = 'frequencies',
                    y = ['mean_power_mPFC_cage',
                        'mean_power_mPFC_recover_10min',
                         'mean_power_mPFC_recover_20min',
                         'mean_power_mPFC_recover_30min',
                         'mean_power_mPFC_recover_40min'])

plt.xlabel(xlabel)
plt.ylabel(ylabel)
plt.title(title + '_mPFC')
plt.gcf().subplots_adjust(bottom=0.15, left=0.18)
plt.ticklabel_format(axis='y', style="sci", scilimits=(0,0))
plt.savefig(sdir_cage + title + '_mPFC_recovery.png')
plt.show()
#%%
power_spectrum1.iloc[:40].plot(x = 'frequencies',
                    y = ['mean_power_vHPC_cage',
                        'mean_power_vHPC_recover_10min',
                         'mean_power_vHPC_recover_20min',
                         'mean_power_vHPC_recover_30min',
                         'mean_power_vHPC_recover_40min'])
plt.xlabel(xlabel)
plt.ylabel(ylabel)
plt.title(title + '_vHPC')
plt.gcf().subplots_adjust(bottom=0.15, left=0.18)
plt.ticklabel_format(axis='y', style="sci", scilimits=(0,0))
plt.savefig(sdir_cage + title + '_vHPC_recovery.png')
plt.show()

#%%
'''
    Return Area under the curve (AUC) for Gaussain Function
    Input: xdata: frquency or independent varaible
    Input: ydata: the value of spectrum at the given frequency

    Output: Area Under the Curve
    '''

# theta_power = ephys.exg_auc(freqs, psds_arena[:22,:].mean(axis=0))

#%%
# import importlib
# importlib.reload(bifengephys.ephys)
#
# theta_power = ephys.exg_auc(freqs, psds_arena[:22,:].mean(axis=0))

#%%

# xdata = freqs
# ydata = psds_ezm[len(mpfc_ch):,:].mean(axis=0)
# import pandas as pd
# pd.DataFrame.from_dict({'xdata':xdata, 'ydata':ydata}).to_csv('F:/data.csv')
# popt, pcov = ephys.fit_exg(xdata, ydata, bounds=(0, [3., 1., 0.5]))
#
# plt.plot(xdata, ydata, 'b-', label='data')
# plt.plot(xdata, ephys.exg_fun(xdata, *popt), 'g--',
#          label='fit: a=%5.3f, b=%5.3f, c=%5.3f, u=%5.3f, s=%5.3f, b2=%5.3f, c2=%5.3f, u2=%5.3f' % tuple(popt))
# plt.legend()
#
# print(ephys.exg_auc(xdata, ydata))
#
# plt.show()


#%%
### compute power spectra using the multitaper method
''''
mne.time_frequency.psd_multitaper(inst, fmin=0, fmax=inf, tmin=None, tmax=None, bandwidth=None, adaptive=False, 
low_bias=True, normalization='length', picks=None, proj=False, n_jobs=1, reject_by_annotation=False, verbose=None)

Example:
    f, ax = plt.subplots()
    psds, freqs = psd_multitaper(epochs, fmin=2, fmax=40, n_jobs=1)
    psds = 10. * np.log10(psds)
    psds_mean = psds.mean(0).mean(0)
    psds_std = psds.mean(0).std(0)
    
    ax.plot(freqs, psds_mean, color='k')
    ax.fill_between(freqs, psds_mean - psds_std, psds_mean + psds_std,
                    color='k', alpha=.5)
    ax.set(title='Multitaper PSD (gradiometers)', xlabel='Frequency (Hz)',
           ylabel='Power Spectral Density (dB)')
    plt.show()
'''

# data1 = raw_arena.copy()
# psds, freqs = mne.time_frequency.psd_multitaper(data1, fmin=1, fmax=20, tmin=20, tmax=620, bandwidth=2.5, n_jobs=workers)
#
# for _ in range(psds.shape[0]):
#     plt.plot(freqs, psds[_, :])
#
# plt.title('PSD using multitaper')
# plt.show()

#%%
### compute time-frequency representation using multitaper method, return power-time
''''
mne.time_frequency.tfr_multitaper(inst, freqs, n_cycles, time_bandwidth=4.0, use_fft=True, return_itc=True, 
decim=1, n_jobs=1, picks=None, average=True, verbose=None)

freqs : ndarray, shape (n_freqs,)
        The frequencies in Hz.
        
elif method == 'multitaper':
        Ws = _make_dpss(sfreq, freqs, n_cycles=n_cycles,
                        time_bandwidth=time_bandwidth, zero_mean=zero_mean)
                        
                        
return power-time
if dB:
        data = 10 * np.log10((data * data.conj()).real)
        

mne.time_frequency.psd_multitaper(inst, fmin=0, fmax=inf, tmin=None, tmax=None, bandwidth=None, 
adaptive=False, low_bias=True, normalization='length', picks=None, proj=False, n_jobs=1, verbose=None

'''
#%%
# #
# data = raw_arena_bef.copy()
# sdir = sdir_arena_bef
# # # #
# data = raw_arena.copy()
# sdir = sdir_arena
# #
data = raw_ezm.copy()
sdir = sdir_ezm
# #
# data = raw_oft.copy()
# sdir = sdir_oft

epochs = mne.make_fixed_length_epochs(data, duration=2.6)
freqs = np.arange(3.5, 13., 0.2)
n_cycles = freqs / 2.
power = mne.time_frequency.tfr_multitaper(epochs, freqs=freqs, n_cycles=n_cycles, time_bandwidth=4.0,
                                          return_itc=False, n_jobs=workers, picks=None, average=False)

print('Mission Completed')


#%%
tpower = power.data
tpower = np.mean(tpower, axis=3)# take the averaged power of 2.6 second
tpower_46hz = np.mean(tpower[:, :, :16], axis=2) #(epoch, channel, freqs)
tpower_712hz = np.mean(tpower[:, :, 16:], axis=2)

tpower_46hz_all_ch = np.mean(tpower_46hz, axis=0)
tpower_712hz_all_ch = np.mean(tpower_712hz, axis=0)
plt.plot(tpower_46hz_all_ch)
plt.plot(tpower_712hz_all_ch)
plt.show()

#%%
## select channels for further analysis, based on how well the theta correlate between two brain areas
## threashold coefficient >= 0.4


# tp_mpfc = tp[:, :21, :] ## mBWfus008
# tp_vhipp = tp[:, 21:, :]

# tp_mpfc = tp[:, :16, :] ## mBWfus009
# tp_vhipp = tp[:, 16:, :]

# tp_mpfc = tp[:, :21, :] ## mBFuss011
# tp_vhipp = tp[:, 21:, :]

# tp_mpfc = tp[:, :12, :] ## mBWfus012
# tp_vhipp = tp[:, 12:, :]

#
# tp_mpfc_46hz = tpower_46hz[20:250, :len(mpfc_ch)] ## mBWfus025, ## choose 230 epochs [epochs, channels, freqs]
# tp_mpfc_712hz = tpower_712hz[20:250, :len(mpfc_ch)]
# tp_vhipp_46hz = tpower_46hz[20:250, len(mpfc_ch):]
# tp_vhipp_712hz = tpower_712hz[20:250, len(mpfc_ch):]

tp_mpfc_46hz = tpower_46hz[20:250, :len(mpfc_ch)] ## mBWfus026, ## choose 230 epochs [epochs, channels, freqs]
tp_mpfc_712hz = tpower_712hz[20:250, :len(mpfc_ch)]
tp_vhipp_46hz = tpower_46hz[20:250, len(mpfc_ch):]
tp_vhipp_712hz = tpower_712hz[20:250, len(mpfc_ch):]


## plot the theta power of the selected channels in the mPFC
for _ in range(tp_mpfc_46hz.shape[1]):
    plt.plot(tp_mpfc_46hz[:40, _])
title = 'Power 4-6 Hz' + ' all ch in the mPFC'
plt.title(title)
plt.savefig(sdir + title + '.png')
plt.show()

for _ in range(tp_mpfc_712hz.shape[1]):
    plt.plot(tp_mpfc_712hz[:40, _])
title = 'Power 7-12 Hz' + ' all ch in the mPFC'
plt.title(title)
plt.savefig(sdir + title + '.png')
plt.show()

## plot the theta power of the selected channels in the vHPC, deep channels
for _ in range(tp_vhipp_46hz.shape[1]):
    plt.plot(tp_vhipp_46hz[:40, _])
title = 'Power 4-6 Hz' + ' all ch in the vHPC'
plt.title(title)
plt.savefig(sdir + title + '.png')
plt.show()

for _ in range(tp_vhipp_712hz.shape[1]):
    plt.plot(tp_vhipp_712hz[:40, _])
title = 'Power 7-12 Hz' + ' all ch in the vHPC'
plt.title(title)
plt.savefig(sdir + title + '.png')
plt.show()


#%%
tp_mPFC_mean_46hz = tp_mpfc_46hz.mean(axis=1) ## mean power of the selected channels, as their values are similar
tp_vHPC_mean_46hz = tp_vhipp_46hz.mean(axis=1) ## mean power of the selected channels

tp_mPFC_mean_712hz = tp_mpfc_712hz.mean(axis=1) ## mean power of the selected channels, as their values are similar
tp_vHPC_mean_712hz = tp_vhipp_712hz.mean(axis=1) ## mean power of the selected channels

## plot and correlate the power of individual frequencies (4-12)

xlabel = 'mPFC Power'
ylabel = 'vHPC Power'

x = tp_mPFC_mean_46hz
y = tp_vHPC_mean_46hz
corr = np.corrcoef(x,y)[0][1]
title = 'Power correlation 4-6 Hz'
plotting.power_correlation_plot(x, y, corr, sdir, title, xlabel, ylabel)

x = tp_mPFC_mean_712hz
y = tp_vHPC_mean_712hz
corr = np.corrcoef(x,y)[0][1]
title = 'Power correlation 7-12 Hz'
plotting.power_correlation_plot(x, y, corr, sdir, title, xlabel, ylabel)

print('Mission Completed')




#%%

## plot and correlate the power of individual frequencies (4-12)
corr_freqs=[]

col_mPFC = tp_mpfc_shallow_vHPC.columns.tolist()[:9]
col_vHPC = tp_mpfc_shallow_vHPC.columns.tolist()[9:]

xlabel = 'mPFC Theta Power'
ylabel = 'vHPC Theta Power'

for i, freq in enumerate(col_mPFC):
    x = tp_mpfc_shallow_vHPC[col_mPFC[i]]
    y = tp_mpfc_shallow_vHPC[col_vHPC[i]]
    corr = x.corr(y, method='pearson')
    corr_freqs.append(corr)
    title = 'Power correlation shallow_' + freq.split('_')[1]
    plotting.power_correlation_plot(x, y, corr, sdir, title, xlabel, ylabel)

## save the correlation efficients into a dataframe
cols = []
for i in np.arange(4, 13):
    cols.append('Power correlation_' + str(i) + '_Hz')

r_sqr = np.array(corr_freqs)**2

corr_freqs = pd.DataFrame(data=[r_sqr], columns=cols)
corr_freqs['Mean across all frequency'] = corr_freqs.mean(axis=0)

corr_freqs.to_excel(sdir + 'Power_correlation_shallow.xlsx')

print('Mission Completed')





#%%
## plot and correlate the power of two frequency band (4-6, 7-12)
mPFC = tp_df.columns.tolist()[:9]
vHPC = tp_df.columns.tolist()[9:]

xlabel = 'mPFC power 4-6Hz'
ylabel = 'vHPC power 4-6Hz'

x = tp_df[mPFC[:3]].mean(axis=1)
y = tp_df[vHPC[:3]].mean(axis=1)
corr = x.corr(y, method='pearson')
title = 'Power correlation 4-6 Hz'
plotting.power_correlation_plot(x, y, corr, sdir, title, xlabel, ylabel)

#%%
## plot and correlate the power of two frequency band (4-6, 7-12)
mPFC = tp_df.columns.tolist()[:9]
vHPC = tp_df.columns.tolist()[9:]

xlabel = 'mPFC power 7-9Hz'
ylabel = 'vHPC power 7-9Hz'

x = tp_df[mPFC[3:6]].mean(axis=1)
y = tp_df[vHPC[3:6]].mean(axis=1)
corr = x.corr(y, method='pearson')
title = 'Power correlation 7-9 Hz'
plotting.power_correlation_plot(x, y, corr, sdir, title, xlabel, ylabel)

#%%
## plot and correlate the power of two frequency band (4-6, 7-12)
mPFC = tp_df.columns.tolist()[:9]
vHPC = tp_df.columns.tolist()[9:]

xlabel = 'mPFC power 10-12Hz'
ylabel = 'vHPC power 10-12Hz'

x = tp_df[mPFC[6:]].mean(axis=1)
y = tp_df[vHPC[6:]].mean(axis=1)
corr = x.corr(y, method='pearson')
title = 'Power correlation 10-12 Hz'
plotting.power_correlation_plot(x, y, corr, sdir, title, xlabel, ylabel)


#%%
animal = mBWfus025
task_date = 'ezm_0802'
session = animal[task_date]
sdir = session + '/results/'
if not os.path.exists(sdir):
    os.makedirs(sdir)
print(sdir)
#%%
event_file = os.path.join(session, session + '.csv')
print(event_file)
events = pd.read_csv(event_file, header=1)
close = events[events['metadata']=='{"EZM":"Close"}'][['temporal_segment_start', 'temporal_segment_end']]
open = events[events['metadata']=='{"EZM":"Open"}'][['temporal_segment_start', 'temporal_segment_end']]
LED_off = events[events['metadata']=='{"EZM":"LED_off"}']['temporal_segment_start'].to_numpy()

close_start = close['temporal_segment_start'].to_numpy()
close_end = close['temporal_segment_end'].to_numpy()

open_start = open['temporal_segment_start'].to_numpy()
open_end = open['temporal_segment_end'].to_numpy()

open_time_5min = 0
mouse_in = close_start[0]
five_min = 300 + mouse_in
ten_min = 600 + mouse_in
trans_5min = []
trans_10min = []

for i in range(len(open_end)):
    if open_end[i] <= five_min  and open_start[i] <= five_min:
        dtime =open_end[i] - open_start[i]
        open_time_5min += dtime
    elif open_end[i] > five_min and open_start[i] <= five_min:
        dtime = five_min - open_start[i]
        open_time_5min += dtime
    else:
        break

open_time_10min = 0
for i in range(len(open_end)):
    if open_end[i] <= ten_min and open_start[i] <=ten_min:
        dtime =open_end[i] - open_start[i]
        open_time_10min += dtime
    elif open_end[i] > ten_min and open_start[i] <=ten_min:
        dtime = ten_min - open_start[i]
        open_time_10min += dtime
    else:
        break

close_time_5min = 0
for i in range(len(close_end)):
    if close_end[i] <= five_min  and close_start[i] <= five_min:
        dtime =close_end[i] - close_start[i]
        close_time_5min += dtime
        trans_5min.append(i)
    elif close_end[i] > five_min and close_start[i] <= five_min:
        dtime = five_min - close_start[i]
        close_time_5min += dtime
        trans_5min.append(i)
    else:
        break

close_time_10min = 0
for i in range(len(close_end)):
    if close_end[i] <= ten_min and close_start[i] <=ten_min:
        dtime =close_end[i] - close_start[i]
        close_time_10min += dtime
        trans_10min.append(i)

    elif close_end[i] > ten_min and close_start[i] <=ten_min:
        dtime = ten_min - close_start[i]
        close_time_10min += dtime
        trans_10min.append(i)
    else:
        break

print('open_time_5min = ', open_time_5min, '\n',
      'close_time_5min = ', close_time_5min, '\n',
      'open_time_10min = ', open_time_10min, '\n',
      'close_time_10min = ', close_time_10min)

print('transition in 5 min ', trans_5min[-1], '\n',
      'transition in 10 min ', trans_10min[-1],)

ret = {}
ret['open_time_5min'] = open_time_5min
ret['close_time_5min'] = close_time_5min
ret['transition_5min'] = trans_5min[-1]

ret['open_time_10min'] = open_time_10min
ret['close_time_10min'] = close_time_10min
ret['transition_10min'] = trans_10min[-1]

result = pd.DataFrame(ret, index=[0])
print(open_end - open_start)

#%%
#%%
duration = 900
fps_v = 25
pixel2cm = 0.186 ## 350 pixels = 65 cm
crop_from = int(LED_off*fps_v)
crop_till = int((LED_off + duration)*fps_v)

loc, scorer = behav.load_location(session)
loc = behav.calib_location(loc, sdir, task='ezm')
loc = behav.get_locomotion(loc, fps=fps_v, move_cutoff=5, avgspd_win=9)
# loc = behav.get_locomotion(loc)
# loc, scorer = behav.load_location(animal[session])

# loc, events = behav.loc_analyzer(session, start, duration, task='ezm', bp='head', fps=25) ## (rois_stats, transitions)

#%%
bps = loc.columns.levels[1].to_list()
spd = []
loc_x = []
loc_y = []
for bp in bps:
    spd.append(loc[scorer, bp, 'speed_filtered'])
    loc_x.append(loc[scorer, bp, 'x'])
    loc_y.append(loc[scorer, bp, 'y'])

## take the mean speed, location of head, shoulder, tail
spd = np.array(spd).mean(axis=0)
loc_x = np.array(loc_x).mean(axis=0)
loc_y = np.array(loc_y).mean(axis=0)
loc_x_alinged = loc_x[crop_from:]
loc_y_alinged = loc_y[crop_from:]

plt.figure(figsize=(8, 8))
plt.scatter(loc_x_alinged, loc_y_alinged, alpha=0.2)
plt.xlim(0, 450)
plt.ylim(0, 450)
plt.show()

## plot the locomotion speed vs time
plt.figure(figsize=(8, 6))
spd_aligned = spd[crop_from:]*pixel2cm ## crop be time before LED_off, convert speed from pixel to cm
t = np.arange(0, len(spd_aligned)/fps_v, 1/fps_v)

plt.plot(t, spd_aligned)
plt.xlabel('Time sec')
plt.ylabel('Speed cm/s')
plt.title('locomotion speed')
plt.savefig(sdir_ezm + 'locomotion speed.png')
plt.show()
#%%
## plot the distribution of the locomotion speed
plt.figure(figsize=(8, 6))
n, x, _ = plt.hist(spd_aligned, bins='auto',
                   density=True,
                   cumulative=False,
                   histtype='step',
                   linewidth=2.5)

plt.xlabel('Speed cm/s')
plt.ylabel('Percentage of time')
plt.xlim(-1, 30)
plt.title('Distribution of speed in EZM')
plt.savefig(sdir_ezm + 'Speed distribution.png')
plt.show()

#%%
from scipy import signal

num_sample = int(len(spd_aligned)*500/fps_v)
spd_upsampled = signal.resample(spd_aligned, num_sample)
plt.plot(spd_upsampled)
plt.show()

#%%
## check the statits of EZM-related events
wanted = ['ROI_name', 'cumulative_time_in_roi_sec', 'avg_time_in_roi_sec', 'avg_vel_in_roi']
ezm_stats = {key: events['rois_stats'][key] for key in wanted}
ezm_stats = pd.DataFrame.from_dict(ezm_stats)
# ezm_stats = ezm_stats.drop([1,3])
ezm_stats


# open_idx, close_idx, OTC_idx, prOTC_idx, prCTO_idx, nosedip_idx = behav.get_events(events, LED_off, 900)
#
# event_dict = dict(OTC = 1, prOTC= 2, prCTO = 3, nosedip = 4)
#
# OTC = behav.create_mne_events(OTC_idx, 1)
# prOTC = behav.create_mne_events(prOTC_idx, 2)
# prCTO = behav.create_mne_events(prCTO_idx, 3)
# nosedip = behav.create_mne_events(nosedip_idx, 4)
#
# mne_events = behav.merge_events(OTC, prOTC, prCTO, nosedip)
#
# plt.plot(mne_events[:, 0])
# plt.show()


#%%
srate = 500
open_start_idx = np.round((open_start - LED_off) * srate)
open_end_idx = np.round((open_end - LED_off) * srate)

close_start_idx = np.round((close_start - LED_off) * srate)
close_end_idx = np.round((close_end - LED_off) * srate)


#%%
lfp = ephys.column_by_pad(ephys.get_lfp(data_ezm)) ## crop the time series before LED_off
lfp = lfp[wanted_ch]
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

lfp_filtered = mne.filter.filter_data(data=lfp.T, sfreq=500, l_freq=0.5, h_freq=None)

#%%
open_lfp = []
close_lfp = []
open_spd = []
close_spd = []

## slice and concatenate LFP while the mouse was in the close area
for i in range(len(close_end_idx)):
    seg_lfp = lfp_filtered[:, int(close_start_idx[i]):int(close_end_idx[i])]
    seg_spd = spd_upsampled[int(close_start_idx[i]):int(close_end_idx[i])]
    close_lfp.append(seg_lfp)
    close_spd.append(seg_spd)

close_lfp = np.concatenate(close_lfp, axis=1 )
close_spd = np.concatenate(close_spd)

spd_1 = np.where(close_spd<=1)[0]
close_lfp_1 = close_lfp[:, spd_1]

spd_5 = np.where((close_spd>1) & (close_spd<=5))[0]
close_lfp_5 = close_lfp[:, spd_5]

spd_10 = np.where((close_spd>5) & (close_spd<=10))[0]
close_lfp_10 = close_lfp[:, spd_10]

spd_15 = np.where((close_spd>10) & (close_spd<=15))[0]
close_lfp_15 = close_lfp[:, spd_15]

spd_20 = np.where(close_spd>15)[0]
close_lfp_20 = close_lfp[:, spd_20]

## slice and concatenate LFP while the mouse was in the open area
for i in range(len(open_end_idx)):
    seg_lfp = lfp_filtered[:, int(open_start_idx[i]):int(open_end_idx[i])]
    seg_spd = spd_upsampled[int(open_start_idx[i]):int(open_end_idx[i])]
    open_lfp.append(seg_lfp)
    open_spd.append(seg_spd)

open_lfp = np.concatenate(open_lfp, axis=1 )
open_spd = np.concatenate(open_spd)

spd_1 = np.where(open_spd<=1)[0]
open_lfp_1 = open_lfp[:, spd_1]

spd_5 = np.where((open_spd>1) & (open_spd<=5))[0]
open_lfp_5 = open_lfp[:, spd_5]

spd_10 = np.where((open_spd>5) & (open_spd<=10))[0]
open_lfp_10 = open_lfp[:, spd_10]

spd_15 = np.where((open_spd>10) & (open_spd<=15))[0]
open_lfp_15 = open_lfp[:, spd_15]

spd_20 = np.where(open_spd>15)[0]
open_lfp_20 = open_lfp[:, spd_20]



raw_close = mne.io.RawArray(close_lfp, info)
raw_open = mne.io.RawArray(open_lfp, info)

#%%
## plot the distribution of the locomotion speed
plt.figure(figsize=(8, 6))
plt.hist([close_spd, open_spd],
         bins=50,
         density=True,
         # cumulative=True,
         histtype='step',
         linewidth=2.5,
         label=['Close', 'Open'])

plt.xlabel('Speed cm/s')
plt.ylabel('Percentage of time')
plt.xlim(-1, 30)
plt.title('Distribution of speed in EZM')
plt.legend(loc='upper right')
plt.savefig(sdir_ezm + 'Speed distribution open vs close.png')
plt.show()
#%%
workers=20

#%%
data =close_lfp_1
psds_close_1, freqs = mne.time_frequency.psd_array_welch(data, sfreq=500, fmin=0, fmax=20, n_fft=1024, n_overlap=128,
                                  n_per_seg=256, n_jobs=workers,
                                  average='mean', window='hamming', verbose=None)

data =close_lfp_5
psds_close_5, freqs = mne.time_frequency.psd_array_welch(data, sfreq=500, fmin=0, fmax=20, n_fft=1024, n_overlap=128,
                                  n_per_seg=256, n_jobs=workers,
                                  average='mean', window='hamming', verbose=None)

data =close_lfp_10
psds_close_10, freqs = mne.time_frequency.psd_array_welch(data, sfreq=500, fmin=0, fmax=20, n_fft=1024, n_overlap=128,
                                  n_per_seg=256, n_jobs=workers,
                                  average='mean', window='hamming', verbose=None)

data =close_lfp_15
psds_close_15, freqs = mne.time_frequency.psd_array_welch(data, sfreq=500, fmin=0, fmax=20, n_fft=1024, n_overlap=128,
                                  n_per_seg=256, n_jobs=workers,
                                  average='mean', window='hamming', verbose=None)

data =close_lfp_20
psds_close_20, freqs = mne.time_frequency.psd_array_welch(data, sfreq=500, fmin=0, fmax=20, n_fft=1024, n_overlap=128,
                                  n_per_seg=256, n_jobs=workers,
                                  average='mean', window='hamming', verbose=None)

#%% Compute the power spectrum against speed in the open area
data =open_lfp_1
psds_open_1, freqs = mne.time_frequency.psd_array_welch(data, sfreq=500, fmin=0, fmax=20, n_fft=1024, n_overlap=128,
                                  n_per_seg=256, n_jobs=workers,
                                  average='mean', window='hamming', verbose=None)

data =open_lfp_5
psds_open_5, freqs = mne.time_frequency.psd_array_welch(data, sfreq=500, fmin=0, fmax=20, n_fft=1024, n_overlap=128,
                                  n_per_seg=256, n_jobs=workers,
                                  average='mean', window='hamming', verbose=None)

data =open_lfp_10
psds_open_10, freqs = mne.time_frequency.psd_array_welch(data, sfreq=500, fmin=0, fmax=20, n_fft=1024, n_overlap=128,
                                  n_per_seg=256, n_jobs=workers,
                                  average='mean', window='hamming', verbose=None)

data =open_lfp_15
psds_open_15, freqs = mne.time_frequency.psd_array_welch(data, sfreq=500, fmin=0, fmax=20, n_fft=1024, n_overlap=128,
                                  n_per_seg=256, n_jobs=workers,
                                  average='mean', window='hamming', verbose=None)

data =open_lfp_20
psds_open_20, freqs = mne.time_frequency.psd_array_welch(data, sfreq=500, fmin=0, fmax=20, n_fft=1024, n_overlap=128,
                                  n_per_seg=256, n_jobs=workers,
                                  average='mean', window='hamming', verbose=None)

#%% plot power spectrum against locomotion speed in the close area
plt.plot(freqs, psds_close_1[:len(mpfc_ch), :].mean(axis=0), label='Close_mPFC_1cm/s')
plt.plot(freqs, psds_close_5[:len(mpfc_ch), :].mean(axis=0), label='Close_mPFC_5cm/s')
plt.plot(freqs, psds_close_10[:len(mpfc_ch), :].mean(axis=0), label='Close_mPFC_10cm/s')
plt.plot(freqs, psds_close_15[:len(mpfc_ch), :].mean(axis=0), label='Close_mPFC_15cm/s')
plt.plot(freqs, psds_close_20[:len(mpfc_ch), :].mean(axis=0), label='Close_mPFC_20cm/s')
plt.xlabel('Freqs (Hz)')
plt.ylabel('Power (uV**2/Hz)')
plt.legend(loc='upper right')
plt.savefig(sdir_ezm + 'Power spectrum_different speed in close.png')
plt.show()

#%% plot power spectrum against locomotion speed in the open area
plt.plot(freqs, psds_open_1[:len(mpfc_ch), :].mean(axis=0), label='Open_mPFC_1cm/s')
plt.plot(freqs, psds_open_5[:len(mpfc_ch), :].mean(axis=0), label='Open_mPFC_5cm/s')
plt.plot(freqs, psds_open_10[:len(mpfc_ch), :].mean(axis=0), label='Open_mPFC_10cm/s')
plt.plot(freqs, psds_open_15[:len(mpfc_ch), :].mean(axis=0), label='Open_mPFC_15cm/s')
plt.plot(freqs, psds_open_20[:len(mpfc_ch), :].mean(axis=0), label='Open_mPFC_20cm/s')
plt.xlabel('Freqs (Hz)')
plt.ylabel('Power (uV**2/Hz)')
plt.legend(loc='upper right')
plt.savefig(sdir_ezm + 'Power spectrum_different speed in open.png')
plt.show()

#%%
plt.plot(freqs, psds_close_1[:len(mpfc_ch), :].mean(axis=0), label='Close_mPFC_1cm/s')
plt.plot(freqs, psds_open_1[:len(mpfc_ch), :].mean(axis=0), label='Open_mPFC_1cm/s')
plt.xlabel('Freqs (Hz)')
plt.ylabel('Power (uV**2/Hz)')
plt.legend(loc='upper right')
plt.savefig(sdir_ezm + 'Power spectrum_1cmperS.png')
plt.show()

plt.plot(freqs, psds_close_5[:len(mpfc_ch), :].mean(axis=0), label='Close_mPFC_5cm/s')
plt.plot(freqs, psds_open_5[:len(mpfc_ch), :].mean(axis=0), label='Open_mPFC_5cm/s')
plt.xlabel('Freqs (Hz)')
plt.ylabel('Power (uV**2/Hz)')
plt.legend(loc='upper right')
plt.savefig(sdir_ezm + 'Power spectrum_5cmperS.png')
plt.show()


plt.plot(freqs, psds_close_10[:len(mpfc_ch), :].mean(axis=0), label='Close_mPFC_10cm/s')
plt.plot(freqs, psds_open_10[:len(mpfc_ch), :].mean(axis=0), label='Open_mPFC_10cm/s')
plt.xlabel('Freqs (Hz)')
plt.ylabel('Power (uV**2/Hz)')
plt.legend(loc='upper right')
plt.savefig(sdir_ezm + 'Power spectrum_10cmperS.png')
plt.show()

plt.plot(freqs, psds_close_15[:len(mpfc_ch), :].mean(axis=0), label='Close_mPFC_15cm/s')
plt.plot(freqs, psds_open_15[:len(mpfc_ch), :].mean(axis=0), label='Open_mPFC_15cm/s')
plt.xlabel('Freqs (Hz)')
plt.ylabel('Power (uV**2/Hz)')
plt.legend(loc='upper right')
plt.savefig(sdir_ezm + 'Power spectrum_15cmperS.png')
plt.show()


plt.plot(freqs, psds_close_20[:len(mpfc_ch), :].mean(axis=0), label='Close_mPFC_20cm/s')
plt.plot(freqs, psds_open_20[:len(mpfc_ch), :].mean(axis=0), label='Open_mPFC_20cm/s')
plt.xlabel('Freqs (Hz)')
plt.ylabel('Power (uV**2/Hz)')
plt.legend(loc='upper right')
plt.savefig(sdir_ezm + 'Power spectrum_20cmperS.png')
plt.show()

#%%
data =close_lfp
psds_close, freqs = mne.time_frequency.psd_array_welch(data, sfreq=500, fmin=0, fmax=20, n_fft=1024, n_overlap=128,
                                  n_per_seg=256, n_jobs=workers,
                                  average='mean', window='hamming', verbose=None)

# # Convert power to dB scale.
# psds_close = 10 * np.log10(psds_close)

title = 'Power Spectrum'
xlabel = 'Frequency (Hz)'
ylabel = 'Power (uV**2/Hz)'

#%%
for _ in range(psds_close.shape[0]):
    plt.plot(freqs, psds_close[_, :])
plt.xlabel(xlabel)
plt.ylabel(ylabel)
plt.title(title + '_close')
plt.gcf().subplots_adjust(bottom=0.15, left=0.18)
plt.ticklabel_format(axis='both', style="sci", scilimits=(0,15))
plt.savefig(sdir_ezm + title + '_close.svg', dpi=300, transparent=True)
plt.show()

mean_power_mPFC = psds_close[:len(mpfc_ch), :].mean(axis=0)
sd_power_mPFC = psds_close[:len(mpfc_ch), :].std(axis=0)
plt.plot(freqs, mean_power_mPFC, label='mPFC')
plt.fill_between(freqs, mean_power_mPFC - sd_power_mPFC, mean_power_mPFC + sd_power_mPFC, alpha=0.15)

mean_power_vHPC = psds_close[len(mpfc_ch):, :].mean(axis=0)
sd_power_vHPC = psds_close[len(mpfc_ch):, :].std(axis=0)
plt.plot(freqs, mean_power_vHPC, label='vHPC')
plt.fill_between(freqs, mean_power_vHPC - sd_power_vHPC, mean_power_vHPC + sd_power_vHPC,  alpha=0.15)

plt.xlabel(xlabel)
plt.ylabel(ylabel)
plt.title(title + '_close')
plt.gcf().subplots_adjust(bottom=0.15, left=0.18)
plt.ticklabel_format(axis='both', style="sci", scilimits=(0,3))
plt.legend(fontsize=10)
plt.savefig(sdir_ezm + title + '_mean_close.svg', dpi=300, transparent=True)
plt.show()
#%%
data =open_lfp
psds_open, freqs = mne.time_frequency.psd_array_welch(data, sfreq=500, fmin=0, fmax=20, n_fft=1024, n_overlap=128,
                                  n_per_seg=256, n_jobs=workers,
                                  average='mean', window='hamming', verbose=None)

# psds_open = 10 * np.log10(psds_open)


for _ in range(psds_open.shape[0]):
    plt.plot(freqs, psds_open[_, :])
plt.xlabel(xlabel)
plt.ylabel(ylabel)
plt.title(title + '_open')
plt.gcf().subplots_adjust(bottom=0.15, left=0.18)
plt.savefig(sdir_ezm + title + '_open.svg', dpi=300, transparent=True)
plt.show()

mean_power_mPFC = psds_open[:len(mpfc_ch), :].mean(axis=0)
sd_power_mPFC = psds_open[:len(mpfc_ch), :].std(axis=0)
plt.plot(freqs, mean_power_mPFC, label='mPFC')
plt.fill_between(freqs, mean_power_mPFC - sd_power_mPFC, mean_power_mPFC + sd_power_mPFC, alpha=0.15)

mean_power_vHPC = psds_open[len(mpfc_ch):, :].mean(axis=0)
sd_power_vHPC = psds_open[len(mpfc_ch):, :].std(axis=0)
plt.plot(freqs, mean_power_vHPC, label='vHPC')
plt.fill_between(freqs, mean_power_vHPC - sd_power_vHPC, mean_power_vHPC + sd_power_vHPC,  alpha=0.15)

plt.xlabel(xlabel)
plt.ylabel(ylabel)
plt.title(title + '_open')
plt.gcf().subplots_adjust(bottom=0.15, left=0.18)
plt.ticklabel_format(axis='both', style="sci", scilimits=(0,15))
plt.legend(fontsize=14)
plt.savefig(sdir_ezm + title + '_mean_open.svg', dpi=300, transparent=True)
plt.show()
#%%
## plot the power spectrum open area vs close area
mean_power_mPFC = psds_open[:len(mpfc_ch), :].mean(axis=0)
sd_power_mPFC = psds_open[:len(mpfc_ch), :].std(axis=0)
plt.plot(freqs, mean_power_mPFC, label='mPFC_open')
plt.fill_between(freqs, mean_power_mPFC - sd_power_mPFC, mean_power_mPFC + sd_power_mPFC, alpha=0.15)

mean_power_mPFC = psds_close[:len(mpfc_ch), :].mean(axis=0)
sd_power_mPFC = psds_close[:len(mpfc_ch), :].std(axis=0)
plt.plot(freqs, mean_power_mPFC, label='mPFC_enclosed')
plt.fill_between(freqs, mean_power_mPFC - sd_power_mPFC, mean_power_mPFC + sd_power_mPFC, alpha=0.15)

plt.xlabel(xlabel)
plt.ylabel(ylabel)
plt.ylim(0, 1000)
# plt.title(title + '_mPFC')
plt.gcf().subplots_adjust(bottom=0.15, left=0.18)
# plt.ticklabel_format(axis='both', style="sci", scilimits=(0,3))
plt.legend(fontsize=13)
plt.savefig(sdir_ezm + title + '_mPFC_open_vs_close.svg', dpi=300, transparent=True)
plt.show()

mean_power_vHPC = psds_open[len(mpfc_ch):, :].mean(axis=0)
sd_power_vHPC = psds_open[len(mpfc_ch):, :].std(axis=0)
plt.plot(freqs, mean_power_vHPC, label='vHPC_open')
plt.fill_between(freqs, mean_power_vHPC - sd_power_vHPC, mean_power_vHPC + sd_power_vHPC,  alpha=0.15)

mean_power_vHPC = psds_close[len(mpfc_ch):, :].mean(axis=0)
sd_power_vHPC = psds_close[len(mpfc_ch):, :].std(axis=0)
plt.plot(freqs, mean_power_vHPC, label='vHPC_enclosed')
plt.fill_between(freqs, mean_power_vHPC - sd_power_vHPC, mean_power_vHPC + sd_power_vHPC,  alpha=0.15)

plt.xlabel(xlabel)
plt.ylabel(ylabel)
# plt.ylim(0, 2000)
# plt.title(title + '_vHPC')
plt.gcf().subplots_adjust(bottom=0.15, left=0.18)
# plt.ticklabel_format(axis='both', style="sci", scilimits=(0,3))
plt.legend(fontsize=13)
plt.savefig(sdir_ezm + title + '_vHPC_open_vs_close.svg', dpi=300, transparent=True)
plt.show()


#%%
data =close_lfp
psds_close, freqs = mne.time_frequency.psd_array_welch(data, sfreq=500, fmin=30, fmax=100, n_fft=1024, n_overlap=128,
                                  n_per_seg=256, n_jobs=workers,
                                  average='mean', window='hamming', verbose=None)


title = 'Power Spectrum Welch'
xlabel = 'Frequency'
ylabel = 'Power'

for _ in range(psds_close.shape[0]):
    plt.plot(freqs, psds_close[_, :])
plt.xlabel(xlabel)
plt.ylabel(ylabel)
plt.title(title + '_gamma_close')
plt.gcf().subplots_adjust(bottom=0.15, left=0.18)
plt.ticklabel_format(axis='both', style="sci", scilimits=(0,3))
plt.savefig(sdir_ezm + title + '_gamma_close.png', transparent=True)
plt.show()

mean_power_mPFC = psds_close[:len(mpfc_ch), :].mean(axis=0)
sd_power_mPFC = psds_close[:len(mpfc_ch), :].std(axis=0)
plt.plot(freqs, mean_power_mPFC, label='mPFC')
plt.fill_between(freqs, mean_power_mPFC - sd_power_mPFC, mean_power_mPFC + sd_power_mPFC, alpha=0.15)

mean_power_vHPC = psds_close[len(mpfc_ch):, :].mean(axis=0)
sd_power_vHPC = psds_close[len(mpfc_ch):, :].std(axis=0)
plt.plot(freqs, mean_power_vHPC, label='vHPC')
plt.fill_between(freqs, mean_power_vHPC - sd_power_vHPC, mean_power_vHPC + sd_power_vHPC,  alpha=0.15)

plt.xlabel(xlabel)
plt.ylabel(ylabel)
plt.title(title + '_gamma_close')
plt.gcf().subplots_adjust(bottom=0.15, left=0.18)
plt.ticklabel_format(axis='both', style="sci", scilimits=(0,3))
plt.legend()
plt.savefig(sdir_ezm + title + '_gamma_mean_close.png', transparent=True)
plt.show()

print('Mission Completed')

data =open_lfp
psds_open, freqs = mne.time_frequency.psd_array_welch(data, sfreq=500, fmin=30, fmax=100, n_fft=1024, n_overlap=128,
                                  n_per_seg=256, n_jobs=workers,
                                  average='mean', window='hamming', verbose=None)


title = 'Power Spectra Welch'
xlabel = 'Frequency'
ylabel = 'Power'

for _ in range(psds_open.shape[0]):
    plt.plot(freqs, psds_open[_, :])
plt.xlabel(xlabel)
plt.ylabel(ylabel)
plt.title(title + '_gamma_open')
plt.gcf().subplots_adjust(bottom=0.15, left=0.18)
plt.ticklabel_format(axis='both', style="sci", scilimits=(0,3))
plt.savefig(sdir_ezm + title + '_gamma_open.png', transparent=True)
plt.show()

mean_power_mPFC = psds_open[:len(mpfc_ch), :].mean(axis=0)
sd_power_mPFC = psds_open[:len(mpfc_ch), :].std(axis=0)
plt.plot(freqs, mean_power_mPFC, label='mPFC')
plt.fill_between(freqs, mean_power_mPFC - sd_power_mPFC, mean_power_mPFC + sd_power_mPFC, alpha=0.15)

mean_power_vHPC = psds_open[len(mpfc_ch):, :].mean(axis=0)
sd_power_vHPC = psds_open[len(mpfc_ch):, :].std(axis=0)
plt.plot(freqs, mean_power_vHPC, label='vHPC')
plt.fill_between(freqs, mean_power_vHPC - sd_power_vHPC, mean_power_vHPC + sd_power_vHPC,  alpha=0.15)

plt.xlabel(xlabel)
plt.ylabel(ylabel)
plt.title(title + '_gamma_open')
plt.gcf().subplots_adjust(bottom=0.15, left=0.18)
plt.ticklabel_format(axis='both', style="sci", scilimits=(0,3))
plt.legend()
plt.savefig(sdir_ezm + title + '_gamma_mean_open.png', transparent=True)
plt.show()

print('Mission Completed')


mean_power_mPFC = psds_open[:len(mpfc_ch), :].mean(axis=0)
sd_power_mPFC = psds_open[:len(mpfc_ch), :].std(axis=0)
plt.plot(freqs, mean_power_mPFC, label='mPFC_open')
plt.fill_between(freqs, mean_power_mPFC - sd_power_mPFC, mean_power_mPFC + sd_power_mPFC, alpha=0.15)

mean_power_mPFC = psds_close[:len(mpfc_ch), :].mean(axis=0)
sd_power_mPFC = psds_close[:len(mpfc_ch), :].std(axis=0)
plt.plot(freqs, mean_power_mPFC, label='mPFC_close')
plt.fill_between(freqs, mean_power_mPFC - sd_power_mPFC, mean_power_mPFC + sd_power_mPFC, alpha=0.15)

plt.xlabel(xlabel)
plt.ylabel(ylabel)
plt.title(title + '_mPFC')
plt.gcf().subplots_adjust(bottom=0.15, left=0.18)
plt.ticklabel_format(axis='both', style="sci", scilimits=(0,3))
plt.legend()
plt.savefig(sdir_ezm + title + '_gamma_mPFC_open_vs_close.png', transparent=True)
plt.show()

mean_power_vHPC = psds_open[len(mpfc_ch):, :].mean(axis=0)
sd_power_vHPC = psds_open[len(mpfc_ch):, :].std(axis=0)
plt.plot(freqs, mean_power_vHPC, label='vHPC_open')
plt.fill_between(freqs, mean_power_vHPC - sd_power_vHPC, mean_power_vHPC + sd_power_vHPC,  alpha=0.15)

mean_power_vHPC = psds_close[len(mpfc_ch):, :].mean(axis=0)
sd_power_vHPC = psds_close[len(mpfc_ch):, :].std(axis=0)
plt.plot(freqs, mean_power_vHPC, label='vHPC_close')
plt.fill_between(freqs, mean_power_vHPC - sd_power_vHPC, mean_power_vHPC + sd_power_vHPC,  alpha=0.15)

plt.xlabel(xlabel)
plt.ylabel(ylabel)
plt.title(title + '_gamma_vHPC')
plt.gcf().subplots_adjust(bottom=0.15, left=0.18)
plt.ticklabel_format(axis='both', style="sci", scilimits=(0,3))
plt.legend()
plt.savefig(sdir_ezm + title + '_gamma_vHPC_open_vs_close.png', transparent=True)
plt.show()

#%%
freqs = np.arange(3.5, 13., 0.2)
n_low_theta = len(np.where(freqs <= 6.5)[0])
n_low_theta = len(np.where(freqs > 6.5)[0])
n_cycles = freqs / 2.

epochs = mne.make_fixed_length_epochs(raw_close, duration=2.6)
power_close = mne.time_frequency.tfr_multitaper(epochs, freqs=freqs, n_cycles=n_cycles, time_bandwidth=4.0,
                                          return_itc=False, n_jobs=workers, picks=None, average=False)

## power = [epochs, channels, freqs, time_points]
#%%
epochs = mne.make_fixed_length_epochs(raw_open, duration=2.6)
power_open = mne.time_frequency.tfr_multitaper(epochs, freqs=freqs, n_cycles=n_cycles, time_bandwidth=4.0,
                                          return_itc=False, n_jobs=workers, picks=None, average=False)

#%% ## power correlation when the mouse is in the closed area
tpower = power_close.data                        ## power = [epochs, channels, freqs, time_points]
tpower = np.mean(tpower, axis=3)          # take the averaged power of each epoch
tpower_low = np.mean(tpower[:, :, :n_low_theta], axis=2)    #(epoch, channel, freqs) 4-6 hz
tpower_high = np.mean(tpower[:, :, n_low_theta:], axis=2)   # 7-12 hz

tp_low_mPFC  = tpower_low[:, :len(mpfc_ch)] ##[epochs, channels]
tp_high_mPFC  = tpower_high[:, :len(mpfc_ch)]
tp_low_vHPC  = tpower_low[:, len(mpfc_ch):]
tp_high_vHPC = tpower_high[:, len(mpfc_ch):]


## plot the theta power of the selected channels in the mPFC
for _ in range(tp_low_mPFC.shape[1]):
    plt.plot(tp_low_mPFC[:40, _])
title = 'Close_Low_theta_power_all_ch_mPFC'
plt.title(title)
plt.savefig(sdir_ezm + title + '.png')
plt.show()

for _ in range(tp_high_mPFC.shape[1]):
    plt.plot(tp_high_mPFC[:40, _])
title = 'Close_High_theta_power_all_ch_mPFC'
plt.title(title)
plt.savefig(sdir_ezm + title + '.png')
plt.show()

## plot the theta power of the selected channels in the vHPC
for _ in range(tp_low_vHPC.shape[1]):
    plt.plot(tp_low_vHPC[:40, _])
title = 'Close_Low_theta_power_all_ch_vHPC'
plt.title(title)
plt.savefig(sdir_ezm + title + '.png')
plt.show()

for _ in range(tp_high_vHPC.shape[1]):
    plt.plot(tp_high_vHPC[:40, _])
title = 'Close_High_theta_power_all_ch_vHPC'
plt.title(title)
plt.savefig(sdir_ezm + title + '.png')
plt.show()

tp_low_mPFC = tp_low_mPFC.mean(axis=1) ## mean power of the selected channels, as their values are similar
tp_low_vHPC = tp_low_vHPC.mean(axis=1) ## mean power of the selected channels

tp_high_mPFC = tp_high_mPFC.mean(axis=1) ## mean power of the selected channels, as their values are similar
tp_high_vHPC = tp_high_vHPC.mean(axis=1) ## mean power of the selected channels

## plot and correlate the power of individual frequencies (4-12)

xlabel = 'mPFC Power'
ylabel = 'vHPC Power'

x = tp_low_mPFC[:100]
y = tp_low_vHPC[:100]
corr = np.corrcoef(x,y)[0][1]
title = 'Close_low_theta_Power correlation'
plotting.power_correlation_plot(x, y, corr, sdir, title, xlabel, ylabel)

x = tp_high_mPFC[:100]
y = tp_high_vHPC[:100]
corr = np.corrcoef(x,y)[0][1]
title = 'Close_high_theta_Power correlation'
plotting.power_correlation_plot(x, y, corr, sdir, title, xlabel, ylabel)

print('Mission Completed')


#%% ## power correlation when the mouse is in the open area
tpower = power_open.data                        ## power = [epochs, channels, freqs, time_points]
tpower = np.mean(tpower, axis=3)          # take the averaged power of each epoch
tpower_low = np.mean(tpower[:, :, :n_low_theta], axis=2)    #(epoch, channel, freqs) 4-6 hz
tpower_high = np.mean(tpower[:, :, n_low_theta:], axis=2)   # 7-12 hz

tp_low_mPFC  = tpower_low[:, :len(mpfc_ch)] ##[epochs, channels]
tp_high_mPFC  = tpower_high[:, :len(mpfc_ch)]
tp_low_vHPC  = tpower_low[:, len(mpfc_ch):]
tp_high_vHPC = tpower_high[:, len(mpfc_ch):]


## plot the theta power of the selected channels in the mPFC
for _ in range(tp_low_mPFC.shape[1]):
    plt.plot(tp_low_mPFC[:40, _])
title = 'Open_Low_theta_power_all_ch_mPFC'
plt.title(title)
plt.savefig(sdir_ezm + title + '.png')
plt.show()

for _ in range(tp_high_mPFC.shape[1]):
    plt.plot(tp_high_mPFC[:40, _])
title = 'OPen_High_theta_power_all_ch_mPFC'
plt.title(title)
plt.savefig(sdir_ezm + title + '.png')
plt.show()

## plot the theta power of the selected channels in the vHPC
for _ in range(tp_low_vHPC.shape[1]):
    plt.plot(tp_low_vHPC[:40, _])
title = 'Open_Low_theta_power_all_ch_vHPC'
plt.title(title)
plt.savefig(sdir_ezm + title + '.png')
plt.show()

for _ in range(tp_high_vHPC.shape[1]):
    plt.plot(tp_high_vHPC[:40, _])
title = 'Open_High_theta_power_all_ch_vHPC'
plt.title(title)
plt.savefig(sdir_ezm + title + '.png')
plt.show()

tp_low_mPFC = tp_low_mPFC.mean(axis=1) ## mean power of the selected channels, as their values are similar
tp_low_vHPC = tp_low_vHPC.mean(axis=1) ## mean power of the selected channels

tp_high_mPFC = tp_high_mPFC.mean(axis=1) ## mean power of the selected channels, as their values are similar
tp_high_vHPC = tp_high_vHPC.mean(axis=1) ## mean power of the selected channels

## plot and correlate the power of individual frequencies (4-12)

xlabel = 'mPFC Power'
ylabel = 'vHPC Power'

x = tp_low_mPFC[:100]
y = tp_low_vHPC[:100]
corr = np.corrcoef(x,y)[0][1]
title = 'Open_low_theta_Power correlation'
plotting.power_correlation_plot(x, y, corr, sdir, title, xlabel, ylabel)

x = tp_high_mPFC[:100]
y = tp_high_vHPC[:100]
corr = np.corrcoef(x,y)[0][1]
title = 'Open_high_theta_Power correlation'
plotting.power_correlation_plot(x, y, corr, sdir, title, xlabel, ylabel)

print('Mission Completed')

#%% Compute the coherence matrix of fixed-length epochs
epochs = mne.make_fixed_length_epochs(raw_close, duration=2.6)

#%%
epochs_data = epochs.get_data()


#%%

con, freqs, times, n_epochs, n_tapers = mne.connectivity.spectral_connectivity(epochs_data[0, :, :], method='coh', indices=None, sfreq=500,
                                       mode='multitaper', fmin=1.0, fmax=20.0, fskip=0,
                                       faverage=False, tmin=None, tmax=None, mt_bandwidth=None,
                                       mt_adaptive=False, mt_low_bias=True, cwt_freqs=None, cwt_n_cycles=7,
                                       block_size=1000, n_jobs=workers, verbose=None)



#%%
plt.imshow(con.mean(axis=-1))
plt.colorbar()
plt.show()



#%%
raw = raw_ezm.copy()
sdir = sdir_ezm

event_dict = dict(OTC = 1, CTO = 2) #, nosedip = 3)
OTC = behav.create_mne_events(close_start_idx[1:], 1)
# prOTC = behav.create_mne_events(prOTC_idx, 2)
CTO = behav.create_mne_events(close_end_idx, 2)
# nosedip = behav.create_mne_events(nosedip_idx, 4)

events = np.concatenate((OTC, CTO), axis=0)
mne_events = events[events[:,0].argsort()]

plt.plot(mne_events[:, 0])
plt.show()

ch_mPFC = [str(el) for el in mpfc_ch]
ch_vHPC = [str(el) for el in vhipp_ch]

epochs_mPFC = mne.Epochs(raw, mne_events, event_id=event_dict, tmin=-1.0, tmax=2.0, baseline=(-1, 0), picks=ch_mPFC,
                    preload=True)
epochs_vHPC = mne.Epochs(raw, mne_events, event_id=event_dict, tmin=-1.0, tmax=2.0, baseline=(-1, 0), picks=ch_vHPC,
                    preload=True)

# fig = epochs_mPFC.plot()

#%%
CTO_vHPC = epochs_vHPC['CTO']
CTO_mPFC = epochs_mPFC['CTO']

OTC_vHPC = epochs_vHPC['OTC']
OTC_mPFC = epochs_mPFC['OTC']
#%%
# CTO_vHPC.plot_image(combine='gfp')
CTO_mPFC.plot_image(combine='gfp')
#%%
CTO_vHPC.plot_psd(fmin=0.5, fmax=20, average=True)
OTC_vHPC.plot_psd(fmin=0.5, fmax=20, average=True)
CTO_mPFC.plot_psd(fmin=0.5, fmax=20, average=True)
OTC_mPFC.plot_psd(fmin=0.5, fmax=20, average=True)
#%%
freqs = np.arange(1., 20., 0.2)
# vmin, vmax = -3., 3.  # Define our color limits.

n_cycles = freqs / 2.
time_bandwidth = 4.0 # small value, higher

#%%
## return container for time-freqency data (n_channels, n_feqs, n_times)
power = mne.time_frequency.tfr_multitaper(CTO_vHPC, freqs=freqs, n_cycles=n_cycles,
                                          time_bandwidth=time_bandwidth,
                                          return_itc=False,
                                          n_jobs=workers,
                                          average=True)

fig = power.plot(baseline=(-1, 0), mode='mean', tmin=None, tmax=None, fmin=None, fmax=None,
           dB=False, colorbar=True, show=True, title=None,
           axes=None, layout=None, yscale='auto', mask=None, mask_style=None, mask_cmap='Greys',
           mask_alpha=0.1, combine=None, exclude=[], verbose=None)

ax = fig.axes[0]
ax.set_ylim(0, 20)
fig.savefig(sdir_ezm +'Spectrogram_close_to_open_vHPC.svg', transparent=True)

#%%
power = mne.time_frequency.tfr_multitaper(CTO_mPFC, freqs=freqs, n_cycles=n_cycles,
                                          time_bandwidth=time_bandwidth,
                                          return_itc=False,
                                          n_jobs=workers,
                                          average=True)

fig = power.plot(baseline=(-1.0, 0), mode='mean', tmin=None, tmax=None, fmin=None, fmax=None,
           dB=False, colorbar=True, show=True, title=None,
           axes=None, layout=None, yscale='auto', mask=None, mask_style=None, mask_cmap='Greys',
           mask_alpha=1, combine=None, exclude=[], verbose=None)

ax = fig.axes[0]
ax.set_ylim(0, 20)
fig.savefig(sdir_ezm +'Spectrogram_close_to_open_mPFC.svg', transparent=True)


#%%
power = mne.time_frequency.tfr_multitaper(OTC_vHPC, freqs=freqs, n_cycles=n_cycles,
                                          time_bandwidth=time_bandwidth,
                                          return_itc=False,
                                          n_jobs=workers,
                                          average=True)

fig = power.plot(baseline=(-1, 0), mode='mean', tmin=None, tmax=None, fmin=None, fmax=None,
           dB=False, colorbar=True, show=True, title=None,
           axes=None, layout=None, yscale='auto', mask=None, mask_style=None, mask_cmap='Greys',
           mask_alpha=0.1, combine=None, exclude=[], verbose=None)

ax = fig.axes[0]
ax.set_ylim(0, 20)
fig.savefig(sdir_ezm +'Spectrogram_open_to_close_vHPC.svg', transparent=True)

#%%
power = mne.time_frequency.tfr_multitaper(OTC_mPFC, freqs=freqs, n_cycles=n_cycles,
                                          time_bandwidth=time_bandwidth,
                                          return_itc=False,
                                          n_jobs=workers,
                                          average=True)

fig = power.plot(baseline=(-1, 0), mode='mean', tmin=None, tmax=None, fmin=None, fmax=None,
           dB=False, colorbar=True, show=True, title=None,
           axes=None, layout=None, yscale='auto', mask=None, mask_style=None, mask_cmap='Greys',
           mask_alpha=0.1, combine=None, exclude=[], verbose=None)

ax = fig.axes[0]
ax.set_ylim(0, 20)
fig.savefig(sdir_ezm +'Spectrogram_open_to_close_mPFC.svg', transparent=True)

#%%
type(fig)
ax = fig.axes[0]
ax.set_ylim(0, 20)
fig.show()
fig.gca().set_title('Test')
fig.show()






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
## plot arena



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
