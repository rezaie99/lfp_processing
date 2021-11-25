#%%
import sys

import scipy.signal
import scipy.stats as stats
from scipy.signal import coherence
from scipy.cluster.hierarchy import fcluster
from sklearn import metrics
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm

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
from shapely.geometry import Point, Polygon

import bifengephys.utils as utils
import bifengephys.ephys as ephys
import bifengephys.plotting as plotting

plt.rcParams['axes.titlesize'] = 18
plt.rcParams["axes.labelsize"] = 18
plt.rcParams["xtick.labelsize"] = 14
plt.rcParams["ytick.labelsize"] = 14
plt.rcParams["font.size"] = 7
plt.rcParams["font.family"] = "Arial"
plt.rcParams["lines.linewidth"] = 1.0


# sys.path.append('D:\ephys')
#%%
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
# animal = mBWfus008
# task_date = 'oft_0305'
# session = animal[task_date]
# data = ephys.load_data(session)
#
# lfp = ephys.column_by_pad(ephys.get_lfp(data))
# print(lfp.columns)
#
# corr_matrix = lfp.iloc[10000:200000].corr() # 10 second
#
# plt.imshow(corr_matrix)
# cb = plt.colorbar()
# plt.xticks(range(corr_matrix.select_dtypes(['number']).shape[1]), corr_matrix.select_dtypes(['number']).columns, fontsize=8, rotation=90)
# plt.yticks(range(corr_matrix.select_dtypes(['number']).shape[1]), corr_matrix.select_dtypes(['number']).columns, fontsize=8)
# plt.show()

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

# wanted_ch = data_cage['info']['wanted_channel'] ## the wanted_ch are the pads ordered by the depth
# mpfc_ch = [str(el) for el in wanted_ch if el < 32]
# vhipp_ch = [str(el) for el in wanted_ch if el >= 32]


lfp = ephys.column_by_pad(ephys.get_lfp(data_cage))
print(lfp.columns)

corr_matrix = lfp.iloc[10000:200000].corr() # 10 second

plt.imshow(corr_matrix)
cb = plt.colorbar()
plt.xticks(range(corr_matrix.select_dtypes(['number']).shape[1]), corr_matrix.select_dtypes(['number']).columns, fontsize=8, rotation=90)
plt.yticks(range(corr_matrix.select_dtypes(['number']).shape[1]), corr_matrix.select_dtypes(['number']).columns, fontsize=8)
plt.show()

#%%
## generate a dict of the depth of each electrode pad
pad_name = [str(el) for el in np.arange(0, 64)]
depth = np.arange(2170, -1, -70)
depth_list = np.concatenate([depth, depth])
pad_depth =dict(zip(pad_name, depth_list))

#%%
wanted_ch = [11, 12, 14, 16, 17, 35, 36, 37, 38, 39, 45, 46, 47, 48, 50]
mpfc_ch = [11, 12, 14, 16, 17]
vvHPC_ch = [ 35, 36, 37, 38, 39]
dvHPC_ch = [45, 46, 47, 48, 50]


#%%
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



# [str(el) for el in wanted_ch if el >= 32]

#%%
lfp = ephys.column_by_pad(ephys.get_lfp(data_cage))
lfp = lfp[wanted_ch]
print(lfp.shape)
## normalize the data to the root mean square of the entire recording session
lfp_rms = ephys.get_rms(lfp)
lfp_norm = lfp / lfp_rms

ch_list= [str(el) for el in lfp.columns.tolist()]
print(len(ch_list), ch_list)
n_channels = len(ch_list)
sampling_freq = 500  # in Hertz
info = mne.create_info(n_channels, sfreq=sampling_freq)

ch_types = ['ecog'] * n_channels
info = mne.create_info(ch_list, ch_types=ch_types, sfreq=sampling_freq)
info['bads'] = []  # Names of bad channels
print(info)

lfp_filted = mne.filter.filter_data(data=lfp_norm.T, sfreq=500, l_freq=0.5, h_freq=None) ## high-pass filter at 0.5Hz
raw_cage = mne.io.RawArray(lfp_filted, info) ### mne data format (n-channels, n-samples), unit = V

print('Mission Completed')

#%%
lfp = ephys.column_by_pad(ephys.get_lfp(data_arena_bef))
lfp = lfp[wanted_ch]
print(lfp.shape)
## normalize the data to the root mean square of the entire recording session
lfp_rms = ephys.get_rms(lfp)
lfp_norm = lfp / lfp_rms

ch_list= [str(el) for el in lfp.columns.tolist()]
print(len(ch_list), ch_list)
n_channels = len(ch_list)
sampling_freq = 500  # in Hertz
info = mne.create_info(n_channels, sfreq=sampling_freq)

ch_types = ['ecog'] * n_channels
info = mne.create_info(ch_list, ch_types=ch_types, sfreq=sampling_freq)
info['bads'] = []  # Names of bad channels
print(info)

lfp_filted = mne.filter.filter_data(data=lfp_norm.T, sfreq=500, l_freq=0.5, h_freq=None) ## high-pass filter at 0.5Hz
raw_arena_bef = mne.io.RawArray(lfp_filted, info) ### mne data format (n-channels, n-samples), unit = V

print('Mission Completed')

#%%
lfp = ephys.column_by_pad(ephys.get_lfp(data_recover))
lfp = lfp[wanted_ch]
print(lfp.shape)
## normalize the data to the root mean square of the entire recording session
lfp_rms = ephys.get_rms(lfp)
lfp_norm = lfp / lfp_rms

ch_list= [str(el) for el in lfp.columns.tolist()]
print(len(ch_list), ch_list)
n_channels = len(ch_list)
sampling_freq = 500  # in Hertz
info = mne.create_info(n_channels, sfreq=sampling_freq)

ch_types = ['ecog'] * n_channels
info = mne.create_info(ch_list, ch_types=ch_types, sfreq=sampling_freq)
info['bads'] = []  # Names of bad channels
print(info)

lfp_filted = mne.filter.filter_data(data=lfp_norm.T, sfreq=500, l_freq=0.5, h_freq=None) ## high-pass filter at 0.5Hz
raw_recover = mne.io.RawArray(lfp_filted, info) ### mne data format (n-channels, n-samples), unit = V

print('Mission Completed')

#%%
lfp = ephys.column_by_pad(ephys.get_lfp(data_arena))
lfp = lfp[wanted_ch]
print(lfp.shape)
## normalize the data to the root mean square of the entire recording session
lfp_rms = ephys.get_rms(lfp)
lfp_norm = lfp / lfp_rms

ch_list= [str(el) for el in lfp.columns.tolist()]
print(len(ch_list), ch_list)
n_channels = len(ch_list)
sampling_freq = 500  # in Hertz
info = mne.create_info(n_channels, sfreq=sampling_freq)

ch_types = ['ecog'] * n_channels
info = mne.create_info(ch_list, ch_types=ch_types, sfreq=sampling_freq)
info['bads'] = []  # Names of bad channels
print(info)

lfp_filted = mne.filter.filter_data(data=lfp_norm.T, sfreq=500, l_freq=0.5, h_freq=None)

raw_arena = mne.io.RawArray(lfp_filted, info) ### mne data format (n-channels, n-samples), unit = mV

print('Mission Completed')

#%%
lfp = ephys.column_by_pad(ephys.get_lfp(data_ezm))
lfp = lfp[wanted_ch]
print(lfp.shape)
## normalize the data to the root mean square of the entire recording session
lfp_rms = ephys.get_rms(lfp)
lfp_norm = lfp / lfp_rms

ch_list= [str(el) for el in lfp.columns.tolist()]
print(len(ch_list), ch_list)
n_channels = len(ch_list)
sampling_freq = 500  # in Hertz
info = mne.create_info(n_channels, sfreq=sampling_freq)

ch_types = ['ecog'] * n_channels
info = mne.create_info(ch_list, ch_types=ch_types, sfreq=sampling_freq)
info['bads'] = []  # Names of bad channels
print(info)

lfp_filted = mne.filter.filter_data(data=lfp_norm.T, sfreq=500, l_freq=0.5, h_freq=None)

raw_ezm = mne.io.RawArray(lfp_filted, info) ### mne data format (n-channels, n-samples) V
print('Mission Completed')
#%%
lfp = ephys.column_by_pad(ephys.get_lfp(data_oft))
lfp = lfp[wanted_ch]
print(lfp.shape)
## normalize the data to the root mean square of the entire recording session
lfp_rms = ephys.get_rms(lfp)
lfp_norm = lfp / lfp_rms

ch_list= [str(el) for el in lfp.columns.tolist()]
print(len(ch_list), ch_list)
n_channels = len(ch_list)
sampling_freq = 500  # in Hertz
info = mne.create_info(n_channels, sfreq=sampling_freq)

ch_types = ['ecog'] * n_channels
info = mne.create_info(ch_list, ch_types=ch_types, sfreq=sampling_freq)
info['bads'] = []  # Names of bad channels
print(info)

lfp_filted = mne.filter.filter_data(data=lfp_norm.T, sfreq=500, l_freq=0.5, h_freq=None)

raw_oft = mne.io.RawArray(lfp_filted, info) ### mne data format (n-channels, n-samples) mV
print('Mission Completed')

#%%

raw_cage.plot(n_channels = 46, duration=2, scalings='auto')

#%%
# title = 'Power Spectrum'
xlabel = 'Frequency (Hz)'
ylabel = 'Power (uV**2/Hz)'
power_spectrum = {}
workers = 1

f_min = 0.5
f_max = 20



#%%
# #%%
title = 'Power spectrum in cage'
data = raw_cage.copy()
psds, freqs = mne.time_frequency.psd_welch(data, fmin=f_min, fmax=f_max, tmin=0, tmax=300, n_fft=1024, n_overlap=128,
                                      n_per_seg=256, picks=None, proj=False, n_jobs=workers, reject_by_annotation=True,
                                      average='mean', window='hamming', verbose=None)

#%%
psds_df = pd.DataFrame(psds.T, columns=ch_list)
## Organize the dataframe into a long-form table
psds_long = ephys.data_to_long(psds_df, mpfc_ch, vvHPC_ch, freqs, pad_depth)

sns.set_theme(style="ticks")
# Define the palette as a list to specify exact values
palette = sns.color_palette("rocket_r", n_colors=len(mpfc_ch))
# Plot the lines on two facets
sns.relplot(
    data=psds_long[psds_long.Area == 'mPFC'],
    x="Freq(Hz)", y="Amp",
    hue="Depth", col="Area",
    kind="line", palette=palette,
    height=5, aspect=.75, facet_kws=dict(sharex=False))

plt.show()
#
# palette = sns.color_palette("rocket_r", n_colors=len(vvHPC_ch))
# # Plot the lines on two facets
# sns.relplot(
#     data=psds_long[psds_long.Area == 'vHPC'],
#     x="Freq(Hz)", y="Amp",
#     hue="Depth", col="Area",
#     kind="line", palette=palette,
#     height=5, aspect=.75, facet_kws=dict(sharex=False),
# )
#
# plt.show()



#%%
epochs = {'0-2min':(0, 120),
         '1-3min': (60, 180),
         '2-4min': (120, 240),
         '5-7min': (300, 420),
          '6-8min': (360, 480),
          '7-9min': (420, 540),
          '8-10min': (480, 600),
          '0-5min': (0, 300),
          '5-10min': (300, 600),
          '0-10min': (0, 600)}


tasks = {'cage': (raw_cage.copy(), sdir_cage),
        'arena_bef': (raw_arena_bef.copy(), sdir_arena_bef),
        # 'recovery': (raw_recover.copy(), sdir_recover),
        'arena_aft': (raw_arena.copy(), sdir_arena),
        'ezm': (raw_ezm.copy(), sdir_ezm),
        'oft': (raw_oft.copy(), sdir_oft)}

#%%
tasks_psds = {}

for task in tasks:
    data, sdir = tasks[task]
    epochs_psds = {}
    for key in epochs:
        tmin, tmax = epochs[key]
        psds, freqs = mne.time_frequency.psd_welch(data, fmin=f_min, fmax=f_max, tmin=tmin, tmax=tmax, n_fft=1024, n_overlap=128,
                                      n_per_seg=256, picks=None, proj=False, n_jobs=workers, reject_by_annotation=True,
                                      average='mean', window='hamming', verbose=None)

        epochs_psds[key] = psds

    tasks_psds[task] = epochs_psds

#%%
spectrum_all_tasks = pd.DataFrame(tasks_psds)
spectrum_all_tasks.to_csv(sdir_cage + 'power_spectrum_all_tasks.csv' )
#%%

for task in tasks:
    _, sdir = tasks[task]
    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(18, 6), sharey=False)

    for key in epochs_psds:
        data = tasks_psds[task][key]
        mPFC = data[:len(mpfc_ch), :].mean(axis=0)
        vvHPC = data[len(mpfc_ch):len(mpfc_ch)+len(vvHPC_ch), :].mean(axis=0)
        dvHPC = data[len(dvHPC_ch):, :].mean(axis=0)

        axs[0].plot(freqs, mPFC, label=key)
        axs[1].plot(freqs, dvHPC, label=key)
        axs[2].plot(freqs, dvHPC, label=key)

    axs[0].legend(loc='upper right')
    axs[1].legend(loc='upper right')
    axs[2].legend(loc='upper right')

    axs[0].set_title('Power spectrum mPFC in ' + task)
    axs[0].set_xlabel('Frequency (Hz)')
    axs[0].set_ylabel('Power density (Normalized)')

    axs[1].set_title('Power spectrum vvHPC in ' + task)
    axs[1].set_xlabel('Frequency (Hz)')
    axs[1].set_ylabel('Power density (Normalized)')

    axs[2].set_title('Power spectrum dvHPC in ' + task)
    axs[2].set_xlabel('Frequency (Hz)')
    axs[2].set_ylabel('Power density (Normalized)')

    plt.tight_layout()
    plt.savefig(sdir + 'Spectrum_2min_epochs.png', dpi=300, transparent=True)
    plt.show()


#%%
epochs = {'0-5min':(0, 300),
         '5-10min': (300, 600),
         '10-15min': (600, 900),
         '15-20min': (900, 1200),
          '20-25min': (1200, 1500),
          '25-30min': (1500, 1800),
          '30-35min': (1800, 2100),
          '35-40min': (2100, 2400)
          }

data= raw_recover.copy()
sdir = sdir_recover

epochs_psds_recovery = {}
for key in epochs:
    tmin, tmax = epochs[key]
    psds, freqs = mne.time_frequency.psd_welch(data, fmin=f_min, fmax=f_max, tmin=tmin, tmax=tmax, n_fft=1024, n_overlap=128,
                                  n_per_seg=256, picks=None, proj=False, n_jobs=workers, reject_by_annotation=True,
                                  average='mean', window='hamming', verbose=None)

    epochs_psds_recovery[key] = psds

fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(18, 6), sharey=False)

for key in epochs_psds_recovery:
    data = epochs_psds_recovery[key]
    mPFC = data[:len(mpfc_ch), :].mean(axis=0)
    vvHPC = data[len(mpfc_ch):len(mpfc_ch)+len(vvHPC_ch), :].mean(axis=0)
    dvHPC = data[len(mpfc_ch)+len(vvHPC_ch):, :].mean(axis=0)

    axs[0].plot(freqs, mPFC, label=key)
    axs[1].plot(freqs, vvHPC, label=key)
    axs[2].plot(freqs, dvHPC, label=key)


axs[0].legend(loc='upper right')
axs[1].legend(loc='upper right')
axs[2].legend(loc='upper right')

axs[0].set_title('Power spectrum mPFC')
axs[0].set_xlabel('Frequency (Hz)')
axs[0].set_ylabel('Power density (Normalized)')

axs[1].set_title('Power spectrum vvHPC')
axs[1].set_xlabel('Frequency (Hz)')
axs[1].set_ylabel('Power density (Normalized)')

axs[2].set_title('Power spectrum dvHPC')
axs[2].set_xlabel('Frequency (Hz)')
axs[2].set_ylabel('Power density (Normalized)')

plt.tight_layout()
plt.savefig(sdir_recover + 'Spectrum_5min_epochs.png', dpi=300, transparent=True)
plt.show()


#%%
power_ratio = {}

power_ratio['Arena_bef_cage_mPFC'] = low_theta_power.Power_mPFC_arena_bef/low_theta_power.Power_mPFC_cage
power_ratio['Arena_aft_cage_mPFC'] = low_theta_power.Power_mPFC_arena_aft/low_theta_power.Power_mPFC_cage

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
data = raw_cage.copy()
sdir = sdir_cage
#
# data = raw_arena_bef.copy()
# sdir = sdir_arena_bef
# # # #
# data = raw_arena.copy()
# sdir = sdir_arena
# #
# data = raw_ezm.copy()
# sdir = sdir_ezm
# #
# data = raw_oft.copy()
# sdir = sdir_oft

epochs = mne.make_fixed_length_epochs(data, duration=2.6)
freqs = np.arange(4, 12, 0.1)
n_cycles = freqs / 2.
n_freq_low_theta = len(np.where(freqs<=6)[0])

power = mne.time_frequency.tfr_multitaper(epochs, freqs=freqs, n_cycles=n_cycles, time_bandwidth=4.0,
                                          return_itc=False, n_jobs=workers, picks=None, average=False)
#%%
tpower = power.data ## (epochs, n_channel, freqs, n_time)
tpower = np.mean(tpower, axis=3)# take the averaged power of 2.6 second
tpower_low = np.mean(tpower[:, :, :n_freq_low_theta], axis=-1) #(epoch, channel, freqs)
tpower_high = np.mean(tpower[:, :, n_freq_low_theta:], axis=-1)

## select channels for further analysis, based on how well the theta correlate between two brain areas
## threashold coefficient >= 0.4

tp_mpfc_low = tpower_low[20:250, :len(mpfc_ch)] ## mBWfus026, ## choose 230 epochs [epochs, channels, freqs]
tp_mpfc_high = tpower_high[20:250, :len(mpfc_ch)]
tp_vhipp_low = tpower_low[20:250, len(mpfc_ch):]
tp_vhipp_high = tpower_high[20:250, len(mpfc_ch):]

## plot the theta power of the selected channels in the mPFC
for _ in range(tp_mpfc_low.shape[1]):
    plt.plot(tp_mpfc_low[:, _])
title = 'Power low_theta' + ' all ch in the mPFC'
plt.title(title)
plt.savefig(sdir + title + '.png', dpi=300, transparent=True)
plt.show()

for _ in range(tp_mpfc_high.shape[1]):
    plt.plot(tp_mpfc_high[:, _])
title = 'Power high_theta' + ' all ch in the mPFC'
plt.title(title)
plt.savefig(sdir + title + '.png', dpi=300, transparent=True)
plt.show()

## plot the theta power of the selected channels in the vHPC, deep channels
for _ in range(tp_vhipp_low.shape[1]):
    plt.plot(tp_vhipp_low[:, _])
title = 'Power low_theta' + ' all ch in the vHPC'
plt.title(title)
plt.savefig(sdir + title + '.png', dpi=300, transparent=True)
plt.show()

for _ in range(tp_vhipp_high.shape[1]):
    plt.plot(tp_vhipp_high[:, _])
title = 'Power high_theta' + ' all ch in the vHPC'
plt.title(title)
plt.savefig(sdir + title + '.png', dpi=300, transparent=True)
plt.show()


tp_mPFC_mean_low = tp_mpfc_low.mean(axis=1) ## mean power of the selected channels, as their values are similar
# tp_mPFC_mean_low = np.delete(tp_mPFC_mean_low, np.where(tp_mPFC_mean_low==max(tp_mPFC_mean_low)))
# tp_mPFC_mean_low = np.delete(tp_mPFC_mean_low, np.where(tp_mPFC_mean_low==max(tp_mPFC_mean_low)))
tp_vHPC_mean_low = tp_vhipp_low.mean(axis=1) ## mean power of the selected channels
# tp_vHPC_mean_low = np.delete(tp_vHPC_mean_low, np.where(tp_vHPC_mean_low==max(tp_vHPC_mean_low)))
# tp_vHPC_mean_low = np.delete(tp_vHPC_mean_low, np.where(tp_vHPC_mean_low==max(tp_vHPC_mean_low)))

tp_mPFC_mean_high = tp_mpfc_high.mean(axis=1) ## mean power of the selected channels, as their values are similar
tp_vHPC_mean_high = tp_vhipp_high.mean(axis=1) ## mean power of the selected channels

## plot and correlate the power of individual frequencies (4-12)

xlabel = 'mPFC Power'
ylabel = 'vHPC Power'

x = tp_mPFC_mean_low
y = tp_vHPC_mean_low
corr = np.corrcoef(x,y)[0][1]
title = 'Power correlation low_theta'
plotting.power_correlation_plot(x, y, corr, sdir, title, xlabel, ylabel)

x = tp_mPFC_mean_high
y = tp_vHPC_mean_high
corr = np.corrcoef(x,y)[0][1]
title = 'Power correlation high_theta'
plotting.power_correlation_plot(x, y, corr, sdir, title, xlabel, ylabel)

print('Mission Completed')




#%%
animal = mBWfus025
task_date = 'ezm_0802'
session = animal[task_date]
sdir_ezm = session + '/results/'
if not os.path.exists(sdir_ezm):
    os.makedirs(sdir_ezm)
print(sdir_ezm)

#%%  For EZM
data = ephys.load_data(session)
# wanted_ch = data['info']['wanted_channel'] ## the wanted_ch are the pads ordered by the depth
# mpfc_ch = [str(el) for el in wanted_ch if el < 32]
# vhipp_ch = [str(el) for el in wanted_ch if el >= 32]

lfp = ephys.column_by_pad(ephys.get_lfp(data)) ## crop the time series before LED_off
lfp = lfp[wanted_ch]
print(lfp.shape)
lfp_rms = ephys.get_rms(lfp) ## get the root mean square of individual channel from the entire recording
lfp_norm = lfp / lfp_rms ## normalize LFP to the rms

ch_list= [str(el) for el in lfp.columns.tolist()]
print(len(ch_list), ch_list)
n_channels = len(ch_list)
sampling_freq = 500  # in Hertz
# info = mne.create_info(n_channels, sfreq=sampling_freq)

ch_types = ['ecog'] * n_channels
info = mne.create_info(ch_list, ch_types=ch_types, sfreq=sampling_freq)
info['bads'] = []  # Names of bad channels
print(info)

lfp_filtered = mne.filter.filter_data(data=lfp_norm.T, sfreq=500, l_freq=0.5, h_freq=None)
raw_ezm = mne.io.RawArray(lfp_filtered, info)

#%% for manually annotated Elevated plus maze
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
    if open_start[i] <= five_min and open_end[i] <= five_min:
        dtime =open_end[i] - open_start[i]
        open_time_5min += dtime
    elif open_start[i] <= five_min and  open_end[i] > five_min:
        dtime = five_min - open_start[i]
        open_time_5min += dtime
    else:
        break

open_time_10min = 0
for i in range(len(open_end)):
    if open_start[i] <= ten_min and open_end[i] <=ten_min:
        dtime =open_end[i] - open_start[i]
        open_time_10min += dtime
    elif open_start[i] <=ten_min and open_end[i] > ten_min:
        dtime = ten_min - open_start[i]
        open_time_10min += dtime
    else:
        break

close_time_5min = 0
for i in range(len(close_end)):
    if close_end[i] <= five_min and close_start[i] <= five_min:
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
result.to_csv(sdir_ezm + 'EZM_behavioral_result.csv')

# For DLC EZM
duration = 900
fps_v = 25
pixel2cm = 0.186 ## 350 pixels = 65 cm
crop_from = int(LED_off*fps_v)
crop_till = int((LED_off + duration)*fps_v)

loc, scorer = behav.load_location(session)
loc = behav.calib_location(loc, sdir_ezm, xymax=400)
loc = behav.get_locomotion(loc, fps=fps_v, move_cutoff=5, avgspd_win=9)
# loc = behav.get_locomotion(loc)
# loc, scorer = behav.load_location(animal[session])

# loc, events = behav.loc_analyzer(session, start, duration, task='ezm', bp='head', fps=25) ## (rois_stats, transitions)

#%%
bps = loc.columns.levels[1].to_list()
spd = []
loc_x = []
loc_y = []

## take the mean speed, location of head, shoulder, tail
for bp in bps:
    spd.append(loc[scorer, bp, 'speed_filtered'])
    loc_x.append(loc[scorer, bp, 'x'])
    loc_y.append(loc[scorer, bp, 'y'])

spd = np.array(spd).mean(axis=0)
loc_x = np.array(loc_x).mean(axis=0)
loc_y = np.array(loc_y).mean(axis=0)
loc_x_aligned = loc_x[crop_from:]
loc_y_aligned = loc_y[crop_from:]
spd_aligned = spd[crop_from:]

plt.figure(figsize=(8, 8))
plt.plot(loc_x_aligned, loc_y_aligned, alpha=0.8)
# plt.xlim(0, 450)
# plt.ylim(0, 450)

plt.title('Locomotion Trajectory \n'
          'Open time in 5min = {:.2f}'.format(open_time_5min) + ' sec' + '\n'
          'Open time in 10min = {:.2f}'.format(open_time_10min) + ' sec' )
plt.savefig(sdir_ezm + 'Locomotion Trajectory.png', dpi=300, transparent=True)
plt.show()

## plot the locomotion speed vs time
plt.figure(figsize=(8, 6))
spd_aligned = spd_aligned*pixel2cm ## crop be time before LED_off, convert speed from pixel to cm
t = np.arange(0, len(spd_aligned)/fps_v, 1/fps_v)

plt.plot(t, spd_aligned)
plt.xlabel('Time (second)')
plt.ylabel('Speed (cm/s)')
plt.title('Locomotion speed')
plt.savefig(sdir_ezm + 'Locomotion speed.png', dpi=300, transparent=True)
plt.show()
#%%
## plot the distribution of the locomotion speed
plt.figure(figsize=(8, 6))
n, x, _ = plt.hist(spd_aligned,
                   bins='auto',
                   density=True,
                   cumulative=False,
                   histtype='step',
                   linewidth=2.5)

plt.xlabel('Speed cm/s')
plt.ylabel('Density')
plt.xlim(-1, 30)
plt.title('Distribution of speed in EZM')
plt.savefig(sdir_ezm + 'Speed distribution.png')
plt.show()

from scipy import signal
num_sample = int(len(spd_aligned)*500/fps_v)
spd_upsampled = signal.resample(spd_aligned, num_sample)
plt.figure(figsize=(8, 6))
plt.plot(spd_upsampled)
plt.title('Speed_upsampled_500Hz')
plt.show()

n_sample = int(len(loc_x_aligned)*500/fps_v)
loc_x_upsampled = signal.resample(loc_x_aligned, n_sample)
loc_y_upsampled = signal.resample(loc_y_aligned, n_sample)
plt.figure(figsize=(8, 8))
plt.plot(loc_x_upsampled, loc_y_upsampled)
plt.show()


#%%
## check the statits of EZM-related events
wanted = ['ROI_name', 'cumulative_time_in_roi_sec', 'avg_time_in_roi_sec', 'avg_vel_in_roi']
ezm_stats = {key: events['rois_stats'][key] for key in wanted}
ezm_stats = pd.DataFrame.from_dict(ezm_stats)
# ezm_stats = ezm_stats.drop([1,3])
ezm_stats

#%% For EZM open vs close
srate = 500

close_start_idx = np.round((close_start - LED_off) * srate) ## close_start, end, LED_off are time in second
close_end_idx = np.round((close_end - LED_off) * srate)

close_idx = []
for i in range(len(close_end_idx)):
    idx = np.arange(close_start_idx[i], close_end_idx[i])
    close_idx.append(idx)
close_idx = np.concatenate(close_idx).astype(int)

open_start_idx = np.round((open_start - LED_off) * srate)
open_end_idx = np.round((open_end - LED_off) * srate)
open_idx = []
for i in range(len(open_end_idx)):
    idx = np.arange(open_start_idx[i], open_end_idx[i])
    open_idx.append(idx)
open_idx = np.concatenate(open_idx).astype(int)

lfp_close = lfp_filtered[:, close_idx]
spd_close = spd_upsampled[close_idx]
loc_x_close = loc_x_upsampled[close_idx]
loc_y_close = loc_y_upsampled[close_idx]

lfp_open = lfp_filtered[:, open_idx]
spd_open = spd_upsampled[open_idx]
loc_x_open = loc_x_upsampled[open_idx]
loc_y_open = loc_y_upsampled[open_idx]

spd_1 = np.where(spd_close<=1)[0]
spd_5 = np.where((spd_close>1) & (spd_close<=5))[0]
spd_10 = np.where((spd_close>5) & (spd_close<=10))[0]
spd_15 = np.where((spd_close>10) & (spd_close<=15))[0]
spd_20 = np.where(spd_close>15)[0]
spd_215 = np.where((spd_close>=2) & (spd_close<=15))[0]

spd_class = {
            'less than 1cm/s': spd_1,
            '1-5cm/s': spd_5,
            '5-10cm/s': spd_10,
            '10-15cm/s': spd_15,
            'faster than 15cm/s': spd_20,
            '2-15cm/s': spd_215}

lfp_close_spd = {}
loc_x_close_spd = {}
loc_y_close_spd = {}
for spd in spd_class:
    lfp_close_spd[spd] = lfp_close[:, spd_class[spd]]
    loc_x_close_spd[spd] = loc_x_close[spd_class[spd]]
    loc_y_close_spd[spd] = loc_y_close[spd_class[spd]]

spd_1 = np.where(spd_open<=1)[0]
spd_5 = np.where((spd_open>1) & (spd_open<=5))[0]
spd_10 = np.where((spd_open>5) & (spd_open<=10))[0]
spd_15 = np.where((spd_open>10) & (spd_open<=15))[0]
spd_20 = np.where(spd_open>15)[0]
spd_215 = np.where((spd_open>=2) & (spd_open<=15))[0]

spd_class = {
            'less than 1cm/s': spd_1,
            '1-5cm/s': spd_5,
            '5-10cm/s': spd_10,
            '10-15cm/s': spd_15,
            'faster than 15cm/s': spd_20,
            '2-15cm/s': spd_215}

lfp_open_spd = {}
loc_x_open_spd = {}
loc_y_open_spd = {}
for spd in spd_class:
    lfp_open_spd[spd] = lfp_open[:, spd_class[spd]]
    loc_x_open_spd[spd] = loc_x_open[spd_class[spd]]
    loc_y_open_spd[spd] = loc_y_open[spd_class[spd]]

raw_close = mne.io.RawArray(lfp_close, info)
raw_open = mne.io.RawArray(lfp_open, info)

raw_close_spd = {}
for spd in spd_class.keys():
    lfp = lfp_close_spd[spd]
    raw_close_spd[spd] = mne.io.RawArray(lfp, info)

raw_open_spd = {}
for spd in spd_class.keys():
    lfp = lfp_open_spd[spd]
    raw_open_spd[spd] = mne.io.RawArray(lfp, info)


#%%
## plot the distribution of the locomotion speed open area vs close area
plt.figure(figsize=(8, 6))
plt.hist([spd_close, spd_open],
         bins=50,
         density=True,
         # cumulative=True,
         histtype='step',
         linewidth=2.5,
         label=['Close', 'Open'])

plt.xlabel('Speed cm/s')
plt.ylabel('Density')
plt.xlim(-1, 30)
plt.title('Distribution of speed in EZM')
plt.legend(loc='upper right')
plt.tight_layout()
plt.savefig(sdir_ezm + 'Speed distribution open vs close.png')
plt.show()

#%% speed distribution again location
duration = 600
n_sample = 600*25 ## 25 frames/s

loc_x = loc_x_aligned[:n_sample]
loc_y = loc_y_aligned[:n_sample]

fig, ax = plt.subplots(figsize=(9, 8))
im1 = ax.scatter(loc_x, loc_y, c=spd_aligned[:n_sample], s=5, cmap="RdBu_r",)# vmin=-1.5, vmax=4.0)
fig.colorbar(im1, ax=ax, label='Speed cm/s')

plt.title('Motion speed vs location')
fig.tight_layout()
fig.subplots_adjust(top=0.88)
plt.savefig(sdir_ezm + 'Location_vs_speed_EZM.png', dpi=300, transparent=True)
plt.show()

#%% Plot the power spectrum of 2 minutes time segments with movement information
workers= 1
f_min = 0.5
f_max= 20.0

areas = {'Closed arm': (raw_close.copy(), loc_x_close, loc_y_close, spd_close),
         'Open arm': (raw_open.copy(), loc_x_open, loc_y_open, spd_open)}


epochs = {'0-2min':(0, 120),
         '1-3min': (60, 180),
         '2-4min': (120, 240),
          '3-5min': (180, 300),
          'All': (0, 600)}

psds_ezm_area = {}
spectrum_area_mPFC = {}
spectrum_area_vvHPC = {}
spectrum_area_dvHPC = {}

for area in areas:
    data, loc_x, loc_y, speed = areas[area]
    sfreq = data.info['sfreq']
    data_length = data._raw_lengths[0]/sfreq

    psds_ezm_area[area] = {}
    spectrum_area_mPFC[area] = {}
    spectrum_area_vvHPC[area] = {}
    spectrum_area_dvHPC[area] = {}

    for i, key in enumerate(epochs):
        tmin, tmax = epochs[key]
        if tmax <= data_length:
            print(i)
            psds, freqs = mne.time_frequency.psd_welch(data, fmin=f_min, fmax=f_max, tmin=tmin, tmax=tmax, n_fft=1024, n_overlap=128,
                                          n_per_seg=256, picks=None, proj=False, n_jobs=workers, reject_by_annotation=True,
                                          average='mean', window='hamming', verbose=None)

            psds_ezm_area[area][key] = psds
            spectrum_area_mPFC[area][key] = psds[:len(mpfc_ch), :].mean(axis=0)
            spectrum_area_vvHPC[area][key] = psds[len(mpfc_ch):len(mpfc_ch)+len(vvHPC_ch), :].mean(axis=0)
            spectrum_area_dvHPC[area][key] = psds[len(mpfc_ch)+len(vvHPC_ch):, :].mean(axis=0)
        else:
            break

    ## plot location of the mouse and the distribution of speed
#%%

# area, loc_x, loc_y, speed = 'Closed arm', loc_x_close, loc_y_close, spd_close
area, loc_x, loc_y, speed = 'Open arm', loc_x_open, loc_y_open, spd_open


epochs = {'0-2min':(0, 120),
         '1-3min': (60, 180),
         '2-4min': (120, 240),
          '3-5min': (180, 300),
          'All': (0, 600)}

fig, axes = plt.subplots(5, 2, figsize=(12, 24))
for i, key in enumerate(epochs):
    start = epochs[key][0] * srate
    end = epochs[key][1] * srate
    if end <= len(loc_x):
        axes[i, 0].scatter(loc_x[start:end], loc_y[start:end], s=2, label=key, alpha=0.1)
        axes[i, 0].set_xlim(0, 400)
        axes[i, 0].set_ylim(0, 400)
        axes[i, 0].legend(loc='center')
        axes[i, 1].hist(speed[start:end])
    else:
        break

axes[0, 0].set_title('Location in the ' + area)
axes[0, 1].set_title('Speed distribution in the ' + area)
axes[3, 0].set_xlabel('X-Y in the EZM')
axes[3, 1].set_xlabel('Speed (cm/s)')

plt.tight_layout()
plt.savefig(sdir_ezm + 'Movement 2min_epochs in the ' + area + '.png', dpi=300, transparent=True)
plt.show()



#%%
workers= 1
f_min = 0.5
f_max= 20.0

spectrum_close_spd = {}

for spd in lfp_close_spd:
    data = lfp_close_spd[spd].copy()
    psds, freqs = mne.time_frequency.psd_array_welch(data, sfreq=500, fmin=f_min, fmax=f_max, n_fft=1024, n_overlap=128,
                                  n_per_seg=256, n_jobs=workers,
                                  average='mean', window='hamming', verbose=None)
    spectrum_close_spd[spd] = psds

spectrum_open_spd = {}
for spd in lfp_open_spd:
    data = lfp_open_spd[spd].copy()
    psds, freqs = mne.time_frequency.psd_array_welch(data, sfreq=500, fmin=f_min, fmax=f_max, n_fft=1024, n_overlap=128,
                                  n_per_seg=256, n_jobs=workers,
                                  average='mean', window='hamming', verbose=None)
    spectrum_open_spd[spd] = psds
#%%
spectrum_close_spd_mPFC = {}
spectrum_close_spd_vvHPC = {}
spectrum_close_spd_dvHPC = {}

for key in spectrum_close_spd:
    dat = spectrum_close_spd[key]
    mean_mPFC = dat[:len(mpfc_ch), :].mean(axis=0)
    mean_vvHPC = dat[len(mpfc_ch):len(mpfc_ch)+len(vvHPC_ch), :].mean(axis=0)
    mean_dvHPC = dat[len(mpfc_ch)+len(vvHPC_ch):, :].mean(axis=0)
    spectrum_close_spd_mPFC[key] = mean_mPFC
    spectrum_close_spd_vvHPC[key] = mean_vvHPC
    spectrum_close_spd_dvHPC[key] = mean_dvHPC

spectrum_open_spd_mPFC = {}
spectrum_open_spd_vvHPC = {}
spectrum_open_spd_dvHPC = {}
for key in spectrum_open_spd:
    dat = spectrum_open_spd[key]
    mean_mPFC = dat[:len(mpfc_ch), :].mean(axis=0)
    mean_vvHPC = dat[len(mpfc_ch):len(mpfc_ch)+len(vvHPC_ch), :].mean(axis=0)
    mean_dvHPC = dat[len(mpfc_ch)+len(vvHPC_ch):, :].mean(axis=0)
    spectrum_open_spd_mPFC[key] = mean_mPFC
    spectrum_open_spd_vvHPC[key] = mean_vvHPC
    spectrum_open_spd_dvHPC[key] = mean_dvHPC

# spectrum_spd_mPFC = pd.DataFrame(spectrum_spd_mPFC)
# spectrum_spd_mPFC['freqs'] = freqs
# close_cols = [col for col in spectrum_spd_mPFC.columns if 'closed' in col]
# open_cols = [col for col in spectrum_spd_mPFC.columns if 'open' in col]
#%% plot power spectrum against locomotion speed
# mPFC
title = 'Power spectrum'
xlabel = 'Frequency (Hz)'
ylabel = 'Power density (a.u)'

fig, axes = plt.subplots(3, 2, figsize=(12, 12))

for i, spd in enumerate(spd_class):
    axes[0, 0].plot(freqs, spectrum_close_spd_mPFC[spd], label=spd, lw=1.5)
    axes[0, 1].plot(freqs, spectrum_open_spd_mPFC[spd], label=spd, lw=1.5)
    axes[1, 0].plot(freqs, spectrum_close_spd_vvHPC[spd], label=spd, lw=1.5)
    axes[1, 1].plot(freqs, spectrum_open_spd_vvHPC[spd], label=spd, lw=1.5)
    axes[2, 0].plot(freqs, spectrum_close_spd_dvHPC[spd], label=spd, lw=1.5)
    axes[2, 1].plot(freqs, spectrum_open_spd_dvHPC[spd], label=spd, lw=1.5)

axes[2, 0].set_xlabel(xlabel)
axes[2, 1].set_xlabel(xlabel)
axes[0, 0].set_ylabel(ylabel + '_mPFC')
axes[1, 0].set_ylabel(ylabel+ '_vvHPC')
axes[2, 0].set_ylabel(ylabel+ '_dvHPC')

axes[0, 0].set_title('Closed arm')
axes[0, 1].set_title('Open arm')

for i, row in enumerate(axes):
    for j, ax in enumerate(row):
       ax.legend(loc='upper right')

plt.tight_layout()
plt.savefig(sdir_ezm + 'Power spectrum_speed.png', dpi=300, transparent=True)
plt.show()


#%% Plot / compare power spectrum in the open arm vs the closed arm in different speed
fig, axes = plt.subplots(6, 3, figsize=(18, 36))

for i, spd in enumerate(spd_class):
        axes[i, 0].plot(freqs, spectrum_close_spd_mPFC[spd], label='Close_' + spd, lw=2.5)
        axes[i, 0].plot(freqs, spectrum_open_spd_mPFC[spd], label='Open_' + spd, lw=2.5)
        axes[i, 1].plot(freqs, spectrum_close_spd_vvHPC[spd], label='Close_' + spd, lw=2.5)
        axes[i, 1].plot(freqs, spectrum_open_spd_vvHPC[spd], label='Open_' + spd, lw=2.5)
        axes[i, 2].plot(freqs, spectrum_close_spd_dvHPC[spd], label='Close_' + spd, lw=2.5)
        axes[i, 2].plot(freqs, spectrum_open_spd_dvHPC[spd], label='Open_' + spd, lw=2.5)

for i, row in enumerate(axes):
    for j, ax in enumerate(row):
       ax.legend(loc='upper right')

axes[0, 0].set_title('mPFC')
axes[0, 1].set_title('vvHPC')
axes[0, 2].set_title('dvHPC')
plt.tight_layout()
plt.savefig(sdir_ezm + title + 'Power spectrum_speed_closed_vs_open.png', dpi=300, transparent=True)
plt.show()

#%%
freqs = np.arange(3.5, 13., 0.2)
n_low_theta = len(np.where(freqs <= 6.5)[0])
n_high_theta = len(np.where(freqs > 6.5)[0])
n_cycles = freqs / 2.


# data_close = raw_close_1.copy()
# data_open = raw_open_1
# title_close_low_theta = 'Power correlation in close low-theta 1cmps'
# title_close_high_theta = 'Power correlation in close high-theta 1cmps'
# title_open_low_theta = 'Power correlation in open low-theta 1cmps'
# title_open_high_theta = 'Power correlation in open high-theta 1cmps'

# data_close = raw_close_5.copy()
# data_open = raw_open_5
# title_close_low_theta = 'Power correlation in close low-theta 5cmps'
# title_close_high_theta = 'Power correlation in close high-theta 5cmps'
# title_open_low_theta = 'Power correlation in open low-theta 5cmps'
# title_open_high_theta = 'Power correlation in open high-theta 5cmps'
#
# data_close = raw_close_10.copy()
# data_open = raw_open_10
# title_close_low_theta = 'Power correlation in close low-theta 10cmps'
# title_close_high_theta = 'Power correlation in close high-theta 10cmps'
# title_open_low_theta = 'Power correlation in open low-theta 10cmps'
# title_open_high_theta = 'Power correlation in open high-theta 10cmps'
#
# data_close = raw_close_15.copy()
# data_open = raw_open_15
# title_close_low_theta = 'Power correlation in close low-theta 15cmps'
# title_close_high_theta = 'Power correlation in close high-theta 15cmps'
# title_open_low_theta = 'Power correlation in open low-theta 15cmps'
# title_open_high_theta = 'Power correlation in open high-theta 15cmps'

# data_close = raw_close_515.copy()
# data_open = raw_open_515
# title_close_low_theta = 'Power correlation in close low-theta 515cmps'
# title_close_high_theta = 'Power correlation in close high-theta 515cmps'
# title_open_low_theta = 'Power correlation in open low-theta 515cmps'
# title_open_high_theta = 'Power correlation in open high-theta 515cmps'
#
data_close = raw_close.copy()
data_open = raw_open
title_close_low_theta = 'Power correlation in close low-theta'
title_close_high_theta = 'Power correlation in close high-theta'
title_open_low_theta = 'Power correlation in open low-theta'
title_open_high_theta = 'Power correlation in open high-theta'



epochs = mne.make_fixed_length_epochs(data_close, duration=2.6)
power_close = mne.time_frequency.tfr_multitaper(epochs, freqs=freqs, n_cycles=n_cycles, time_bandwidth=4.0,
                                          return_itc=False, n_jobs=workers, picks=None, average=False)

## power = [epochs, channels, freqs, time_points]
epochs = mne.make_fixed_length_epochs(data_open, duration=2.6)
power_open = mne.time_frequency.tfr_multitaper(epochs, freqs=freqs, n_cycles=n_cycles, time_bandwidth=4.0,
                                          return_itc=False, n_jobs=workers, picks=None, average=False)

## power correlation when the mouse is in the closed area
tpower = power_close.data                        ## power = [epochs, channels, freqs, time_points]
tpower = np.mean(tpower, axis=3)                 ## take the averaged power of each epoch
tpower_low = np.mean(tpower[:, :, :n_low_theta], axis=2)    #(epoch, channel, freqs) 4-6 hz
tpower_high = np.mean(tpower[:, :, n_low_theta:], axis=2)   # 7-12 hz

tp_low_mPFC  = tpower_low[:, :len(mpfc_ch)] ##[epochs, channels]
tp_high_mPFC  = tpower_high[:, :len(mpfc_ch)]
tp_low_vHPC  = tpower_low[:, len(mpfc_ch):]
tp_high_vHPC = tpower_high[:, len(mpfc_ch):]


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
title = title_close_low_theta
plotting.power_correlation_plot(x, y, corr, sdir_ezm, title, xlabel, ylabel)

x = tp_high_mPFC[:100]
y = tp_high_vHPC[:100]
corr = np.corrcoef(x,y)[0][1]
title = title_close_high_theta
plotting.power_correlation_plot(x, y, corr, sdir_ezm, title, xlabel, ylabel)

print('Mission Completed')


## power correlation when the mouse is in the open area
tpower = power_open.data                        ## power = [epochs, channels, freqs, time_points]
tpower = np.mean(tpower, axis=3)          # take the averaged power of each epoch
tpower_low = np.mean(tpower[:, :, :n_low_theta], axis=2)    #(epoch, channel, freqs) 4-6 hz
tpower_high = np.mean(tpower[:, :, n_low_theta:], axis=2)   # 7-12 hz

tp_low_mPFC  = tpower_low[:, :len(mpfc_ch)] ##[epochs, channels]
tp_high_mPFC  = tpower_high[:, :len(mpfc_ch)]
tp_low_vHPC  = tpower_low[:, len(mpfc_ch):]
tp_high_vHPC = tpower_high[:, len(mpfc_ch):]

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
title = title_open_low_theta
plotting.power_correlation_plot(x, y, corr, sdir_ezm, title, xlabel, ylabel)

x = tp_high_mPFC[:100]
y = tp_high_vHPC[:100]
corr = np.corrcoef(x,y)[0][1]
title = title_open_high_theta
plotting.power_correlation_plot(x, y, corr, sdir_ezm, title, xlabel, ylabel)

print('Mission Completed')

#%% compute coherence matrix across all the channles
##  average over all epochs

epochs = mne.make_fixed_length_epochs(raw_close, duration=2.6)
con, freqs, times, n_epochs, n_tapers = mne.connectivity.spectral_connectivity(epochs,
                                                                               method='coh',
                                                                               indices=None,
                                                                               sfreq=500,
                                                                               mode='multitaper',
                                                                               fmin=1.0, fmax=13.0, fskip=0,
                                                                               faverage=True,
                                                                               tmin=None, tmax=None,
                                                                               mt_bandwidth=4,
                                                                               mt_adaptive=False,
                                                                               mt_low_bias=True,
                                                                               cwt_freqs=None, cwt_n_cycles=7,
                                                                               block_size=1000, n_jobs=workers, verbose=None)

## out: con: [n_ch, n_ch, freqs]

plt.imshow(con)
plt.colorbar()
plt.title('Coherence closed area')
plt.savefig(sdir_ezm + 'Coherence closed area.png')
plt.show()


#%%
raw = raw_ezm.copy()
event_dict = dict(OTC = 1, CTO = 2) #, nosedip = 3)
OTC = behav.create_mne_events(close_start_idx[1:], 1)
CTO = behav.create_mne_events(close_end_idx, 2)
# nosedip = behav.create_mne_events(nosedip_idx, 4)
events = np.concatenate((OTC, CTO), axis=0)
mne_events = events[events[:,0].argsort()]
plt.plot(mne_events[:, 0])
plt.show()

ch_mPFC = [str(el) for el in mpfc_ch]
ch_vHPC = [str(el) for el in vvHPC_ch]

epochs_mPFC = mne.Epochs(raw, mne_events, event_id=event_dict, tmin=-3.0, tmax=3.0, baseline=None, picks=ch_mPFC,
                    preload=True)
epochs_vHPC = mne.Epochs(raw, mne_events, event_id=event_dict, tmin=-3.0, tmax=3.0, baseline=None, picks=ch_vHPC,
                    preload=True)
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
freqs = np.arange(0.5, 20., 0.1)
n_cycles = freqs / 2.
time_bandwidth = 4.0 # small value, higher

#%% close to open ventral hippocampus
## return container for time-freqency data (n_channels, n_feqs, n_times)
power = mne.time_frequency.tfr_multitaper(CTO_vHPC,
                                          freqs=freqs,
                                          n_cycles=n_cycles,
                                          time_bandwidth=time_bandwidth,
                                          return_itc=False,
                                          n_jobs=workers,
                                          average=True)

fig = power.plot(baseline=(-3, -1),
                 mode='zscore',
                 tmin=None, tmax=None, fmin=None, fmax=None,
                 dB=False, colorbar=True, show=True,
                 title=None,
                 axes=None, layout=None, yscale='auto',
                 mask=None, mask_style=None, mask_cmap='Greys', mask_alpha=0.1,
                 combine='mean', exclude=[], verbose=None)

ax = fig.axes[0]
ax.set_ylim(0, 20) ## only work on the saved plot
fig.savefig(sdir_ezm +'Spectrogram_close_to_open_vHPC.png', dpi=300, transparent=True)

#%% Close to open mPFC
power = mne.time_frequency.tfr_multitaper(CTO_mPFC,
                                          freqs=freqs, n_cycles=n_cycles,
                                          time_bandwidth=time_bandwidth,
                                          return_itc=False,
                                          n_jobs=workers,
                                          average=True)

fig = power.plot(baseline=(-3.0, -1), mode='zscore', tmin=None, tmax=None, fmin=None, fmax=None,
           dB=False, colorbar=True, show=True, title=None,
           axes=None, layout=None, yscale='auto', mask=None, mask_style=None, mask_cmap='Greys',
           mask_alpha=1, combine='mean', exclude=[], verbose=None)

ax = fig.axes[0]
ax.set_ylim(0, 20)
fig.savefig(sdir_ezm +'Spectrogram_close_to_open_mPFC.png', dpi=300, transparent=True)


#%% open to close ventral hippocampus
power = mne.time_frequency.tfr_multitaper(OTC_vHPC, freqs=freqs, n_cycles=n_cycles,
                                          time_bandwidth=time_bandwidth,
                                          return_itc=False,
                                          n_jobs=workers,
                                          average=True)

fig = power.plot(baseline=(-3, 0), mode='zscore', tmin=None, tmax=None, fmin=None, fmax=None,
           dB=False, colorbar=True, show=True, title=None,
           axes=None, layout=None, yscale='auto', mask=None, mask_style=None, mask_cmap='Greys',
           mask_alpha=0.1, combine='mean', exclude=[], verbose=None)

ax = fig.axes[0]
ax.set_ylim(0, 20)
fig.savefig(sdir_ezm +'Spectrogram_open_to_close_vHPC.png', dpi=300, transparent=True)

#%% open to close medial prefrontal cortex
power = mne.time_frequency.tfr_multitaper(OTC_mPFC, freqs=freqs, n_cycles=n_cycles,
                                          time_bandwidth=time_bandwidth,
                                          return_itc=False,
                                          n_jobs=workers,
                                          average=True)

fig = power.plot(baseline=(-3, 0), mode='zscore', tmin=None,
             tmax=None, fmin=None, fmax=None, vmin=None, vmax=None,
             cmap='RdBu_r', dB=False, colorbar=True, show=True, title=None,
             axes=None, layout=None, yscale='auto', mask=None,
             mask_style=None, mask_cmap="Greys", mask_alpha=0.1, combine='mean',
             exclude=[], verbose=None)


ax = fig.axes[0]
ax.set_ylim(0, 20)
fig.savefig(sdir_ezm +'Spectrogram_open_to_close_mPFC.png', dpi=300, transparent=True)


#%%
raw = raw_ezm.copy()

bands = {'delta': (1.5, 4.0),
         'theta': (4.0, 7.0),
         'alpha': (7.0, 10.0),
         'beta': (10.0, 30.0),
         'gamma': (30.0, 100.0),
         'mua': (200, None)}

ezm_hilbert_power = {}
power_close = {}
Power_open = {}

for band in bands:
    freq_l, freq_h = bands[band]
    band_filtered = raw.copy().filter(l_freq=freq_l, h_freq = freq_h)
    amp = band_filtered.apply_hilbert(envelope=True).get_data() # if envelope==True return np.abs(out), else return analytic signal (128 complex)
    power = np.square(amp)     ## square the amplitude to get power
    ezm_hilbert_power[band] = power
    power_close[band] = power[:, close_idx]
    Power_open[band] = power[:, open_idx]


# low_theta_close.apply_hilbert(envelope=False) # if envelope==True return np.abs(out), else return analytic signal (128 complex)
# analytic_signal = low_theta_close.get_data()
# low_theta_amp_close = np.abs(analytic_signal)
# low_theta_phase_close = np.unwrap(np.angle(analytic_signal))

#%%

fig, axes = plt.subplots(6, 1, figsize=(6, 24))

for i, band in enumerate(bands):
    axes[i].plot(ezm_hilbert_power[band][0, 5000:6000], label='mPFC')
    axes[i].plot(ezm_hilbert_power[band][5, 5000:6000], label='vHPC')
    axes[i].legend(loc='upper right')
    axes[i].set_title(band)

plt.savefig(sdir_ezm + 'hilbert power.png', dpi=300)
plt.show()

#%% Find the MUA peaks using 2 SD
epsilon_zscored = stats.zscore(ezm_hilbert_power['mua'], axis=1) ## z-score the power to mean and sd

epsilon_close = epsilon_zscored[:, close_idx]
epsilon_open = epsilon_zscored[:, open_idx]


mua = []
for ch in range(epsilon_zscored.shape[0]):
    mua.append(signal.find_peaks(epsilon_zscored[0, :], height=2.0))


#%%
fig, axes = plt.subplots(3, 5, figsize=(16, 12))
# ax.imshow(epsilon_zscored[:, 5500:6000],cmap="RdBu_r")
# ax.plot(epsilon_zscored[0, 10000:20000])
axes = axes.reshape(-1)
for i in range(epsilon_zscored.shape[0]):
    axes[i].hist(epsilon_close[i, :], bins=50, range=(0, 6), density=True, label='Closed arm', alpha=0.6)
    axes[i].hist(epsilon_open[i, :], bins=50, range=(0, 6), density=True, label='Open arm',  alpha=0.6)
    axes[i].legend(loc='upper right')

for i in range(10,15):
    axes[i].set_xlabel('MUA power (z-scored)')

fig.suptitle('MUA power distribution (z-scored)', fontsize=32)
plt.tight_layout()
plt.savefig(sdir_ezm + 'distribution of mua power.png', dpi=300)
plt.show()




#%% correlate power of different bands with speed
xlabel = 'Locomotion speed (cm/s)'
ylabel = 'Power'

areas = {'EZM': (ezm_hilbert_power, spd_upsampled),
         'Closed arm': (power_close, spd_close),
         'Open arm': (Power_open, spd_open),
        }

for area in areas:
    power_, speed_ = areas[area]

    epoch_length = 2.0 # second
    n_epochs = int(speed_.shape[0]//(srate*epoch_length))
    crop_till = int(epoch_length * n_epochs * srate)

    fig, axes = plt.subplots(6, 3, figsize=(12, 18))
    for i, band in enumerate(bands):
        power = power_[band]
        power_mPFC = np.mean(power[:len(mpfc_ch), :], axis=0)  ## take the average of 5 channles
        powerp_vvHPC = np.mean(power[len(mpfc_ch):len(mpfc_ch)+len(vvHPC_ch), :], axis=0)  ## take the average of 5 channles
        power_dvHPC = np.mean(power[len(mpfc_ch)+len(vvHPC_ch):, :], axis=0)  ## take the average of 5 channles

        speed = np.array(np.split(speed_[:crop_till], n_epochs)).mean(axis=1)
        power_mPFC = np.array(np.split(power_mPFC[:crop_till], n_epochs)).mean(axis=1) #take the mean of each epoch
        powerp_vvHPC = np.array(np.split(powerp_vvHPC[:crop_till], n_epochs)).mean(axis=1)
        power_dvHPC = np.array(np.split(power_dvHPC[:crop_till], n_epochs)).mean(axis=1)

        x = speed
        y = power_mPFC
        corr = np.corrcoef(x,y)[0][1]
        label = 'R sequred = {:.2f}'.format(np.square(corr))
        title = 'Speed vs Power_mPFC in Closed ' + band
        axes[i, 0].scatter(x, y, s=10, label=band + '\n' + label)
        m, b = np.polyfit(x, y, 1)
        axes[i, 0].plot(x, m*x + b)
        axes[i, 0].legend(loc='upper right')

        y = powerp_vvHPC
        corr = np.corrcoef(x,y)[0][1]
        label = 'R sequred = {:.2f}'.format(np.square(corr))
        title = 'Speed vs Power_vvHPC in Closed ' + band
        axes[i, 1].scatter(x, y, s=10, label= band + '\n' + label)
        m, b = np.polyfit(x, y, 1)
        axes[i, 1].plot(x, m*x + b)
        axes[i, 1].legend(loc='upper right')

        y = power_dvHPC
        corr = np.corrcoef(x,y)[0][1]
        label = 'R sequred = {:.2f}'.format(np.square(corr))
        title = 'Speed vs Power_dvHPC in Closed ' + band
        axes[i, 2].scatter(x, y, s=10, label= band + '\n' + label)
        m, b = np.polyfit(x, y, 1)
        axes[i, 2].plot(x, m*x + b)
        axes[i, 2].legend(loc='upper right')

    axes[5, 0].set_xlabel(xlabel)
    axes[5, 1].set_xlabel(xlabel)
    axes[5, 2].set_xlabel(xlabel)

    axes[0, 0].set_ylabel(ylabel)
    axes[1, 0].set_ylabel(ylabel)
    axes[2, 0].set_ylabel(ylabel)
    axes[3, 0].set_ylabel(ylabel)
    axes[4, 0].set_ylabel(ylabel)
    axes[5, 0].set_ylabel(ylabel)

    axes[0, 0].set_title('mPFC')
    axes[0, 1].set_title('vvHPC')
    axes[0, 2].set_title('dvHPC')

    fig.suptitle('Motion speed vs Power across bands in the closed arm \n '
                 'mPFC, vvHPC, dvHPC \n '
                 'delta(1.5-4.0 Hz), theta(4-7 Hz), alpha(7-10 Hz), delta(10-30 Hz), gamma(30 - 80 Hz), fast(80-200 Hz)')
    fig.tight_layout()
    fig.subplots_adjust(top=0.88)
    plt.savefig(sdir_ezm + 'band_power_correlate_speed_' + area + '.png', dpi=300, transparent=True)
    plt.show()


#%%
duration = 600
n_sample = 600*500


loc_x = loc_x_upsampled[:n_sample]
loc_y = loc_y_upsampled[:n_sample]

fig, axes = plt.subplots(6, 3, figsize=(14, 24))

for i, band in enumerate(bands):
    power = ezm_hilbert_power[band]
    power_mPFC = stats.zscore(np.mean(power[:len(mpfc_ch), :], axis=0)) ## take the average of 5 channles, z-scored to mean and sd
    power_vvHPC = stats.zscore(np.mean(power[len(mpfc_ch):len(mpfc_ch)+len(vvHPC_ch), :], axis=0))  ## take the average of 5 channles
    power_dvHPC = stats.zscore(np.mean(power[len(mpfc_ch)+len(vvHPC_ch):, :], axis=0))  ## take the average of 5 channles

    im1 = axes[i, 0].scatter(loc_x, loc_y, c=power_mPFC[:n_sample], s=2, cmap="RdBu_r", vmin=-1.5, vmax=4.0)
    fig.colorbar(im1, ax=axes[i, 0])
    im2 = axes[i, 1].scatter(loc_x, loc_y, c=power_vvHPC[:n_sample], s=2, cmap="RdBu_r",  vmin=-1.5, vmax=4.0)
    fig.colorbar(im2, ax=axes[i, 1])
    im3 = axes[i, 2].scatter(loc_x, loc_y, c=power_dvHPC[:n_sample], s=2, cmap="RdBu_r",  vmin=-1.5, vmax=4.0)
    fig.colorbar(im3, ax=axes[i, 2])

fig.suptitle('Motion speed vs Power across bands \n '
             'mPFC, vvHPC, dvHPC \n '
             'delta(1.5-4.0 Hz), theta(4-7 Hz), alpha(7-10 Hz), delta(10-30 Hz), gamma(30 - 80 Hz), fast(80-200 Hz)')
fig.tight_layout()
fig.subplots_adjust(top=0.88)
plt.savefig(sdir_ezm + 'Location_vs_band_power_EZM.png', dpi=300, transparent=True)
plt.show()

 #%%
## TODO: band_power correlation between mPFC and vHPC at different speed
spd_1_idx = np.where(spd_upsampled <=1)[0]
spd_5_idx = np.where((spd_upsampled>1) & (spd_upsampled<=5))[0]
spd_10_idx = np.where((spd_upsampled>5) & (spd_upsampled<=10))[0]
spd_15_idx = np.where((spd_upsampled>10) & (spd_upsampled<=15))[0]
spd_20_idx = np.where(spd_upsampled>15)[0]
spd_215_idx = np.where((spd_upsampled>=2) & (spd_upsampled<=15))[0]

spd_class = {'less than 1cm/s': spd_1_idx,
            '1-5cm/s': spd_5_idx,
            '5-10cm/s': spd_10_idx,
            '10-15cm/s': spd_15_idx,
            'faster than 15cm/s': spd_20_idx,
            '2-15cm/s': spd_215_idx}

hilber_power_spd = {}

for spd in spd_class:
    hilber_power_spd[spd] = {}
    for band in bands:
        hilber_power_spd[spd][band] = ezm_hilbert_power[band][:, spd_class[spd]]


#%% power correlation between mPFC and vHPC at different speeds

for epoch_length in [0.5, 1.0, 1.5, 2.0]:

    fig, axes = plt.subplots(6, 6, figsize=(24, 24))
    for j, spd in enumerate(spd_class):
        speed_ = spd_upsampled[spd_class[spd]]
        # epoch_length = epoch_length  # second
        n_epochs = int(len(speed_) // (srate * epoch_length))
        crop_till = int(epoch_length * n_epochs * srate)

        for i, band in enumerate(bands):
            power = hilber_power_spd[spd][band]
            power_mPFC = np.mean(power[:len(mpfc_ch), :], axis=0)  ## take the average of 5 channles
            # powerp_vvHPC = np.mean(power[5:10, :], axis=0)  ## take the average of 5 channles
            power_dvHPC = np.mean(power[len(mpfc_ch):, :], axis=0)  ## take the average of 5 channles

            power_mPFC = np.array(np.split(power_mPFC[:crop_till], n_epochs)).mean(axis=1)  # take the mean of each epoch
            # powerp_vvHPC = np.array(np.split(powerp_vvHPC[:crop_till], n_epochs)).mean(axis=1)
            power_dvHPC = np.array(np.split(power_dvHPC[:crop_till], n_epochs)).mean(axis=1)

            x = power_mPFC
            y = power_dvHPC
            corr = np.corrcoef(x, y)[0][1]
            label = 'R sequred = {:.2f}'.format(np.square(corr))
            title = 'Power_mPFC vs Power_dvHPC ' + band
            axes[j, i].scatter(x, y, s=10, label= spd +'_' + band + '\n' + label)
            m, b = np.polyfit(x, y, 1)
            axes[j, i].plot(x, m * x + b)
            axes[j, i].legend(loc='upper right')

            # y = powerp_vvHPC
            # corr = np.corrcoef(x,y)[0][1]
            # label = 'R sequred = {:.2f}'.format(np.square(corr))
            # title = 'Speed vs Power_vvHPC in Closed ' + band
            # axes[i, 1].scatter(x, y, s=10, label= band + '\n' + label)
            # m, b = np.polyfit(x, y, 1)
            # axes[i, 1].plot(x, m*x + b)
            # axes[i, 1].legend(loc='upper right')

        #
        # axes[5, 0].set_xlabel(xlabel)
        # axes[5, 1].set_xlabel(xlabel)
        # axes[5, 2].set_xlabel(xlabel)
        #
        # axes[0, 0].set_ylabel(ylabel)
        # axes[1, 0].set_ylabel(ylabel)
        # axes[2, 0].set_ylabel(ylabel)
        # axes[3, 0].set_ylabel(ylabel)
        # axes[4, 0].set_ylabel(ylabel)
        # axes[5, 0].set_ylabel(ylabel)
        #
        # axes[0, 0].set_title('mPFC')
        # axes[0, 1].set_title('vvHPC')
        # axes[0, 2].set_title('dvHPC')

        fig.suptitle('Power correlate between mPFC and vHPC with different speed \n '
                     'Speed = ' + spd + '\n'
                    'Epoch length = ' + str(epoch_length) + ' seconds' + '\n '
                    'Number of epochs = ' + str(n_epochs) + '\n'
                    'mPFC, dvHPC \n '
                    'delta(1.5-4.0 Hz), theta(4-7 Hz), alpha(7-10 Hz), delta(10-30 Hz), gamma(30 - 80 Hz), fast(80-200 Hz)')
        fig.tight_layout()
        fig.subplots_adjust(top=0.88)

    plt.title('Power correlation between mPFC and vHPC')
    plt.savefig(sdir_ezm + 'Power correlate between mPFC and vHPC with different speed ' + str(epoch_length) + ' s.png', dpi=300, transparent=True)
    plt.show()







#%% For OFT analysis
animal = mBWfus026
task_date = 'oft_0804'
session = animal[task_date]
sdir_oft = session + '/results/'
if not os.path.exists(sdir_oft):
    os.makedirs(sdir_oft)
print(sdir_oft)

#%%
data = ephys.load_data(session)
wanted_ch = data['info']['wanted_channel'] ## the wanted_ch are the pads ordered by the depth
mpfc_ch = [str(el) for el in wanted_ch if el < 32]
vhipp_ch = [str(el) for el in wanted_ch if el >= 32]

lfp = ephys.column_by_pad(ephys.get_lfp(data)) ## crop the time series before LED_off
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
raw_oft = mne.io.RawArray(lfp_filtered, info)


## read LED_off time from manually annotated csv file
event_file = os.path.join(session, session + '.csv')
print(event_file)
events = pd.read_csv(event_file, header=1)
LED_off = events[events['metadata']=='{"OFT_ephys":"LED_off"}']['temporal_segment_start'].to_numpy() # in second

duration = 600
fps_v = 25                           ## video acquiring speed
pixel2cm = 0.15                    ## 300 pixels = 45 cm
crop_from = int(LED_off*fps_v)
crop_till = int((LED_off + duration)*fps_v)

# load and process data from DLC
loc, scorer = behav.load_location(session)
loc = behav.calib_location(loc, sdir_oft, xymax=350)
loc = behav.get_locomotion(loc, fps=fps_v, move_cutoff=5, avgspd_win=9)

## take the mean speed, location of head, shoulder, tail
bps = loc.columns.levels[1].to_list()
spd = []
loc_x = []
loc_y = []
for bp in bps:
    spd.append(loc[scorer, bp, 'speed_filtered'])
    loc_x.append(loc[scorer, bp, 'x'])
    loc_y.append(loc[scorer, bp, 'y'])

spd = np.array(spd).mean(axis=0)
loc_x = np.array(loc_x).mean(axis=0)
loc_y = np.array(loc_y).mean(axis=0)
loc_x_aligned = loc_x[crop_from:]
loc_y_aligned = loc_y[crop_from:]

plt.figure(figsize=(8, 8))
plt.scatter(loc_x_aligned, loc_y_aligned, s=1.0, alpha=0.8)
plt.title('Locomotion Trajectory')
plt.savefig(sdir_oft + 'Locomotion Trajectory.png', dpi=300, transparent=True)
plt.show()

## plot the locomotion speed vs time
plt.figure(figsize=(8, 6))
spd_aligned = spd[crop_from:]*pixel2cm ## crop be time before LED_off, convert speed from pixel to cm
t = np.arange(0, len(spd_aligned)/fps_v, 1/fps_v)

plt.plot(t, spd_aligned)
plt.xlabel('Time sec')
plt.ylabel('Speed cm/s')
plt.title('Locomotion speed')
plt.savefig(sdir_oft + 'Locomotion speed.png', dpi=300, transparent=True)
plt.show()

## plot the distribution of the locomotion speed
plt.figure(figsize=(8, 6))
n, x, _ = plt.hist(spd_aligned,
                   bins='auto',
                   density=True,
                   cumulative=False,
                   histtype='step',
                   linewidth=2.5)

plt.xlabel('Speed cm/s')
plt.ylabel('Density')
plt.xlim(-1, 30)
plt.title('Distribution of speed in OFT')
plt.savefig(sdir_oft + 'Speed distribution.png')
plt.show()

#%% upsample the speed array to 500 Hz to match the LFP data
from scipy import signal
num_sample = int(len(spd_aligned)*500/fps_v)
spd_upsampled = signal.resample(spd_aligned, num_sample)
plt.plot(spd_upsampled)
plt.show()

n_sample = int(len(loc_x_aligned)*500/fps_v)
loc_x_upsampled = signal.resample(loc_x_aligned, n_sample)
loc_y_upsampled = signal.resample(loc_y_aligned, n_sample)
plt.figure(figsize=(8, 8))
plt.scatter(loc_x_upsampled, loc_y_upsampled, s=0.1)
plt.show()
#%%
# import geopandas as gpd
srate = 500
open_field = Polygon([(5,5), (330,5), (330, 330), (5,330)])
center_area = Polygon([(105, 95), (255, 95), (255, 245), (105, 245)])
corner1 = Polygon([(20, 245), (105, 245), (105, 325), (20, 325)])
corner2 = Polygon([(255, 245), (340, 245), (340, 325), (255, 325)])
corner3 = Polygon([(255, 20), (340, 20), (340, 95), (255, 95)])
corner4 = Polygon([(20, 15), (105, 15), (105, 95), (20, 95)])
corners = [corner1, corner2, corner3, corner4]

time_in_center, time_in_peri, percentage_in_center_area, center_idx_25hz, peri_idx_25hz, corner_idx_25hz= behav.evaluate_OF(loc_x_aligned, loc_y_aligned, 25, center_area, corners)
_, _, _, center_idx_500hz, peri_idx_500hz, corner_idx_500hz= behav.evaluate_OF(loc_x_upsampled, loc_y_upsampled, srate, center_area, corners)

#%% plot trajectory with region of interested
plt.figure(figsize=(8, 8))
plt.plot(loc_x_aligned, loc_y_aligned, alpha=0.8)
plt.title('Locomotion Trajectory')
# plt.plot(*open_field.exterior.xy)
plt.plot(*center_area.exterior.xy, lw=3)
plt.plot(*corner1.exterior.xy, lw=3)
plt.plot(*corner2.exterior.xy, lw=3)
plt.plot(*corner3.exterior.xy, lw=3)
plt.plot(*corner4.exterior.xy, lw=3)
title = 'Time in center % : {:.2f}'.format(percentage_in_center_area*100)
plt.title(title)
plt.savefig(sdir_oft + 'Locomotion Trajectory with rois.png', dpi=300, transparent=True)
plt.show()


#%%  ##'Speed distribution peripheral vs center'
spd_center = spd_upsampled[center_idx_500hz]
spd_peri = spd_upsampled[peri_idx_500hz]
spd_corner = spd_upsampled[corner_idx_500hz]

plt.figure(figsize=(8, 6))
plt.hist([spd_peri, spd_center, spd_corner],
         bins='auto',
         density=True,
         # cumulative=True,
         histtype='step',
         linewidth=1.5,
         label=['Peripheral', 'Center', 'Corner'])

plt.xlabel('Speed cm/s')
plt.ylabel('Density')
plt.xlim(-1, 30)
plt.title('Distribution of speed in open field')
plt.legend(loc='upper right')
plt.savefig(sdir_oft + 'Speed distribution peripheral vs center.png', dpi=300, transparent=True)
plt.show()

#%%
duration = 600
n_sample = 600*25 ## 25 frames/s

loc_x = loc_x_aligned[:n_sample]
loc_y = loc_y_aligned[:n_sample]

fig, ax = plt.subplots(figsize=(8, 8))
# speed_z = stats.zscore(spd_aligned) ## take the average of 5 channles, z-scored to mean and sd
im1 = ax.scatter(loc_x, loc_y, c=spd_aligned[:n_sample], s=5, cmap="RdBu_r", vmin=0, vmax=40)
fig.colorbar(im1, ax=ax, label=('Speed (cm/s)'))

fig.suptitle('Motion speed vs location')
fig.tight_layout()
fig.subplots_adjust(top=0.88)
plt.savefig(sdir_oft + 'Location_vs_speed_OFT.png', dpi=300, transparent=True)
plt.show()

#%% For OFT slice the LFP data based on different moving speed
spd_1 = np.where(spd_upsampled<=1)[0]
spd_5 = np.where((spd_upsampled>1) & (spd_upsampled<=5))[0]
spd_10 = np.where((spd_upsampled>5) & (spd_upsampled<=10))[0]
spd_15 = np.where((spd_upsampled>10) & (spd_upsampled<=15))[0]
spd_20 = np.where(spd_upsampled>15)[0]
spd_215 = np.where((spd_upsampled>=2) & (spd_upsampled<=15))[0]

spd_class = {'less than 1cm/s': spd_1,
                 '1-5cm/s': spd_5,
                 '5-10cm/s':spd_10,
                 '10-15cm/s': spd_15,
                 'faster than 15cm/s': spd_20,
                 '2-15cm/s': spd_215}
lfp_by_spd = {}
loc_x_spd = {}
loc_y_spd = {}
for spd in spd_class:
    lfp_by_spd[spd] = lfp_filtered[:, spd_class[spd]]
    loc_x_spd[spd] = loc_x_upsampled[spd_class[spd]]
    loc_y_spd[spd] = loc_y_upsampled[spd_class[spd]]



#%%
fmin=0.5
fmax=20
workers=1

spectrum_spd = {}
spectrum_spd_mPFC = {}
# spectrum_spd_vvHPC = {}
spectrum_spd_dvHPC = {}

for spd in lfp_by_spd:
    data = lfp_by_spd[spd]
    psds, freqs = mne.time_frequency.psd_array_welch(data, sfreq=500, fmin=fmin, fmax=fmax, n_fft=1024, n_overlap=128,
                                      n_per_seg=256, n_jobs=workers,
                                      average='mean', window='hamming', verbose=None)
    spectrum_spd[spd] = psds
    spectrum_spd_mPFC[spd] = np.mean(psds[:len(mpfc_ch), :], axis=0)
    # spectrum_spd_vvHPC[spd] = np.mean(psds[5:10, :], axis=0)
    spectrum_spd_dvHPC[spd] = np.mean(psds[len(mpfc_ch):, :], axis=0)

#%% Plot / compare power spectrum in different speed
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

for i, spd in enumerate(spd_class):
        axes[0].plot(freqs, spectrum_spd_mPFC[spd], label= spd, lw=1.5)
        # axes[1].plot(freqs, spectrum_spd_vvHPC[spd], label= spd, lw=2.5)
        axes[2].plot(freqs, spectrum_spd_dvHPC[spd], label= spd, lw=1.5)

axes[0].set_title('mPFC')
axes[1].set_title('vvHPC')
axes[2].set_title('dvHPC')

for i in range(3):
    axes[i].legend(loc='upper right')
    axes[i].set_xlabel('Frequency (Hz)')
    axes[i].set_ylabel('Power density (uV**2/Hz)')

plt.tight_layout()
plt.savefig(sdir_oft + title + 'Power spectrum_speed.png', dpi=300, transparent=True)
plt.show()


#%% Compare power spectrum in ROIs_OFT
lfp_peri = lfp_filtered[:, peri_idx_500hz]
lfp_center = lfp_filtered[:, center_idx_500hz]
lfp_corner = lfp_filtered[:, corner_idx_500hz]

lfp_areas = {
    'peri': lfp_peri,
    'center': lfp_center,
    'corner': lfp_corner
}

spectrum_rio = {}
spectrum_rio_mPFC = {}
spectrum_rio_vvHPC = {}
spectrum_rio_dvHPC = {}

for area in lfp_areas:
    data = lfp_areas[area]
    psds, freqs = mne.time_frequency.psd_array_welch(data, sfreq=500, fmin=fmin, fmax=fmax, n_fft=1024, n_overlap=128,
                                  n_per_seg=256, n_jobs=workers,
                                  average='mean', window='hamming', verbose=None)

    spectrum_rio[area] = psds
    spectrum_rio_mPFC[area] = np.mean(psds[:len(mpfc_ch), :], axis=0)
    # spectrum_rio_vvHPC[area] = np.mean(psds[5:10, :], axis=0)
    spectrum_rio_dvHPC[area] = np.mean(psds[len(mpfc_ch):, :], axis=0)


#%%
xlabel = 'Frequency (Hz)'
ylabel = 'Power density (uV**2/Hz)'
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
for area in lfp_areas:
    axes[0].plot(spectrum_rio_mPFC[area], label=area, lw=1.5)
    # axes[1].plot(spectrum_rio_vvHPC[area],  label=area, lw=2.5)
    axes[2].plot(spectrum_rio_dvHPC[area],  label=area, lw=1.5)

for i in range(len(axes)):
    axes[i].legend(loc='upper right')
    axes[i].set_ylabel(ylabel)

axes[2].set_xlabel(xlabel)
axes[0].set_title('Power spectrum in the mPFC_vvHPC_dvHPC')
fig.tight_layout()
plt.savefig(sdir_oft + 'Power spectrum' + 'center vs peripheral.png', dpi=300, transparent=True)
plt.show()


#%%
raw_peri = mne.io.RawArray(lfp_peri, info)
raw_center = mne.io.RawArray(lfp_center, info)

freqs = np.arange(3.5, 13., 0.2)
n_low_theta = len(np.where(freqs <= 6.5)[0])
n_high_theta = len(np.where(freqs > 6.5)[0])
n_cycles = freqs / 2.

#%%
data_peri = raw_peri.copy()
data_center = raw_center.copy()
title_peri_low_theta = 'Power correlation in periphery low-theta'
title_peri_high_theta = 'Power correlation in periphery high-theta'
title_center_low_theta = 'Power correlation in center low-theta'
title_center_high_theta = 'Power correlation in center high-theta'

epochs = mne.make_fixed_length_epochs(data_peri, duration=2.6)
power_peri = mne.time_frequency.tfr_multitaper(epochs, freqs=freqs, n_cycles=n_cycles, time_bandwidth=4.0,
                                          return_itc=False, n_jobs=workers, picks=None, average=False)

## power = [epochs, channels, freqs, time_points]
epochs = mne.make_fixed_length_epochs(data_center, duration=2.6)
power_center = mne.time_frequency.tfr_multitaper(epochs, freqs=freqs, n_cycles=n_cycles, time_bandwidth=4.0,
                                          return_itc=False, n_jobs=workers, picks=None, average=False)

#%%
## power correlation when the mouse is in the closed area
tpower = power_peri.data                        ## power = [epochs, channels, freqs, time_points]
tpower = np.mean(tpower, axis=3)                 ## take the averaged power of each epoch
tpower_low = np.mean(tpower[:, :, :n_low_theta], axis=2)    #(epoch, channel, freqs) 4-6 hz
tpower_high = np.mean(tpower[:, :, n_low_theta:], axis=2)   # 7-12 hz

tp_low_mPFC  = tpower_low[:, :len(mpfc_ch)] ##[epochs, channels]
tp_high_mPFC  = tpower_high[:, :len(mpfc_ch)]
tp_low_vHPC  = tpower_low[:, len(mpfc_ch):]
tp_high_vHPC = tpower_high[:, len(mpfc_ch):]

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
title = title_peri_low_theta
plotting.power_correlation_plot(x, y, corr, sdir_oft, title, xlabel, ylabel)

x = tp_high_mPFC[:100]
y = tp_high_vHPC[:100]
corr = np.corrcoef(x,y)[0][1]
title = title_peri_high_theta
plotting.power_correlation_plot(x, y, corr, sdir_oft, title, xlabel, ylabel)

print('Mission Completed')


## power correlation when the mouse is in the open area
tpower = power_center.data                        ## power = [epochs, channels, freqs, time_points]
tpower = np.mean(tpower, axis=3)          # take the averaged power of each epoch
tpower_low = np.mean(tpower[:, :, :n_low_theta], axis=2)    #(epoch, channel, freqs) 4-6 hz
tpower_high = np.mean(tpower[:, :, n_low_theta:], axis=2)   # 7-12 hz

tp_low_mPFC  = tpower_low[:, :len(mpfc_ch)] ##[epochs, channels]
tp_high_mPFC  = tpower_high[:, :len(mpfc_ch)]
tp_low_vHPC  = tpower_low[:, len(mpfc_ch):]
tp_high_vHPC = tpower_high[:, len(mpfc_ch):]

tp_low_mPFC = tp_low_mPFC.mean(axis=1) ## mean power of the selected channels, as their values are similar
tp_low_vHPC = tp_low_vHPC.mean(axis=1) ## mean power of the selected channels

tp_high_mPFC = tp_high_mPFC.mean(axis=1) ## mean power of the selected channels, as their values are similar
tp_high_vHPC = tp_high_vHPC.mean(axis=1) ## mean power of the selected channels

## plot theta power correlation

xlabel = 'mPFC Power'
ylabel = 'vHPC Power'

x = tp_low_mPFC[:100]
y = tp_low_vHPC[:100]
corr = np.corrcoef(x,y)[0][1]
title = title_center_low_theta
plotting.power_correlation_plot(x, y, corr, sdir_oft, title, xlabel, ylabel)

x = tp_high_mPFC[:100]
y = tp_high_vHPC[:100]
corr = np.corrcoef(x,y)[0][1]
title = title_center_high_theta
plotting.power_correlation_plot(x, y, corr, sdir_oft, title, xlabel, ylabel)

print('Mission Completed')

#%%
raw = raw_oft.copy()

bands = {'delta': (1.5, 4.0),
         'theta': (4.0, 7.0),
         'alpha': (7.0, 10.0),
         'beta': (10.0, 30.0),
         'gamma': (30.0, 80.0),
         'fast': (80.0, 200.0)}

oft_hilbert_power = {}

for band in bands:
    freq_l, freq_h = bands[band]
    band_filtered = raw.copy().filter(l_freq=freq_l, h_freq = freq_h)
    amp = band_filtered.apply_hilbert(envelope=True).get_data() # if envelope==True return np.abs(out), else return analytic signal (128 complex)
    oft_hilbert_power[band] = np.square(amp)

#%% Split the hilbert power into periphery and center

peri_power = {}
center_power = {}
corner_power = {}
## slice and concatenate Hilbert amplitude while the mouse was in the close area
for band in bands:
    amp = oft_hilbert_power[band]
    peri_data = amp[:, peri_idx_500hz]
    center_data = amp[:, center_idx_500hz]
    corner_data = amp[:, corner_idx_500hz]

    peri_power[band] = peri_data
    center_power[band] = center_data
    corner_power[band] = corner_data

#%%
xlabel = 'Locomotion speed (cm/s)'
ylabel = 'Power'
# spd_center = spd_upsampled[center_idx_500hz]
# spd_peri = spd_upsampled[peri_idx_500hz]



areas = {'OFT': (oft_hilbert_power, spd_upsampled),
         'Periphery': (peri_power, spd_peri),
         'Center': (center_power, spd_center),
         'Corner': (corner_power, spd_corner)}

for area in areas:
    power, speed = areas[area]

    # epoch_length = 2.0 # second
    n_epochs = int(speed.shape[0]//(srate*epoch_length))
    crop_till = int(epoch_length * n_epochs * srate)

    fig, axes = plt.subplots(6, 3, figsize=(12, 18))
    for i, band in enumerate(bands):
        amp = power[band]
        amp_mPFC = np.mean(amp[:len(mpfc_ch), :], axis=0)  ## take the average of 5 channles
        # amp_vvHPC = np.mean(amp[5:10, :], axis=0)  ## take the average of 5 channles
        amp_dvHPC = np.mean(amp[len(mpfc_ch):, :], axis=0)  ## take the average of 5 channles

        spd = np.array(np.split(speed[:crop_till], n_epochs)).mean(axis=1)  # take the mean of each epoch
        amp_mPFC = np.array(np.split(amp_mPFC[:crop_till], n_epochs)).mean(axis=1)
        # amp_vvHPC = np.array(np.split(amp_vvHPC[:crop_till], n_epochs)).mean(axis=1)
        amp_dvHPC = np.array(np.split(amp_dvHPC[:crop_till], n_epochs)).mean(axis=1)

        x = spd
        y = amp_mPFC
        corr = np.corrcoef(x,y)[0][1]
        label = 'R sequred = {:.2f}'.format(np.square(corr))
        title = 'Speed vs mPFC in ' + area + ' ' + band
        axes[i, 0].scatter(x, y, s=10, label=label)
        m, b = np.polyfit(x, y, 1)
        axes[i, 0].plot(x, m*x + b)
        axes[i, 0].legend(loc='upper right')
        #
        # y = amp_vvHPC
        # corr = np.corrcoef(x,y)[0][1]
        # label = 'R sequred = {:.2f}'.format(np.square(corr))
        # title = 'Speed vs vvHPC in ' + area + ' ' + band
        # axes[i, 1].scatter(x, y, s=10,label=label)
        # m, b = np.polyfit(x, y, 1)
        # axes[i, 1].plot(x, m*x + b)
        # axes[i, 1].legend(loc='upper right')

        y = amp_dvHPC
        corr = np.corrcoef(x,y)[0][1]
        label = 'R sequred = {:.2f}'.format(np.square(corr))
        title = 'Speed vs dvHPC in ' + area + ' ' + band
        axes[i, 2].scatter(x, y, s=10, label=label)
        m, b = np.polyfit(x, y, 1)
        axes[i, 2].plot(x, m*x + b)
        axes[i, 2].legend(loc='upper right')

    axes[5, 0].set_xlabel(xlabel)
    axes[5, 1].set_xlabel(xlabel)
    axes[5, 2].set_xlabel(xlabel)

    axes[0, 0].set_ylabel(ylabel)
    axes[1, 0].set_ylabel(ylabel)
    axes[2, 0].set_ylabel(ylabel)
    axes[3, 0].set_ylabel(ylabel)
    axes[4, 0].set_ylabel(ylabel)
    axes[5, 0].set_ylabel(ylabel)

    # axes[0, 0].set_title('mPFC')
    # axes[0, 1].set_title('vvHPC')
    # axes[0, 2].set_title('dvHPC')

    fig.suptitle('Power and locomotion speed correlation \n '
                 'mPFC, vvHPC, dvHPC \n '
                 'delta(1.5-4.0 Hz), theta(4-7 Hz), alpha(7-10 Hz), delta(10-30 Hz), gamma(30 - 80 Hz), fast(80-200 Hz)')
    fig.tight_layout()
    fig.subplots_adjust(top=0.88)
    plt.savefig(sdir_oft + 'band_power_correlate_speed_' + area + '.png', dpi=300, transparent=True)
    plt.show()

#%%
## TODO: band_power correlation between mPFC and vHPC at different speed
spd_1_idx = np.where(spd_upsampled <=1)[0]
spd_5_idx = np.where((spd_upsampled>1) & (spd_upsampled<=5))[0]
spd_10_idx = np.where((spd_upsampled>5) & (spd_upsampled<=10))[0]
spd_15_idx = np.where((spd_upsampled>10) & (spd_upsampled<=15))[0]
spd_20_idx = np.where(spd_upsampled>15)[0]
spd_215_idx = np.where((spd_upsampled>=2) & (spd_upsampled<=15))[0]

spd_class = {'less than 1cm/s': spd_1_idx,
            '1-5cm/s': spd_5_idx,
            '5-10cm/s': spd_10_idx,
            '10-15cm/s': spd_15_idx,
            'faster than 15cm/s': spd_20_idx,
            '2-15cm/s': spd_215_idx}

oft_hilber_power_spd = {}

for spd in spd_class:
    oft_hilber_power_spd[spd] = {}
    for band in bands:
        oft_hilber_power_spd[spd][band] = oft_hilbert_power[band][:, spd_class[spd]]


#%% power correlation between mPFC and vHPC at different speeds
para = []
for epoch_length in [0.5, 1.0, 1.5, 2.0]:
    para.append(epoch_length)
    fig, axes = plt.subplots(6, 6, figsize=(24, 24))
    for j, spd in enumerate(spd_class):
        print(spd)
        speed_ = spd_upsampled[spd_class[spd]]
        # epoch_length = epoch_length  # second
        n_epochs = int(len(speed_) // (srate * epoch_length))
        para.append(n_epochs)
        crop_till = int(epoch_length * n_epochs * srate)

        for i, band in enumerate(bands):
            para.append(band)
            power = oft_hilber_power_spd[spd][band]
            power_mPFC = np.mean(power[:len(mpfc_ch), :], axis=0)  ## take the average of 5 channles
            # powerp_vvHPC = np.mean(power[5:10, :], axis=0)  ## take the average of 5 channles
            power_dvHPC = np.mean(power[len(mpfc_ch):, :], axis=0)  ## take the average of 5 channles

            power_mPFC = np.array(np.split(power_mPFC[:crop_till], n_epochs)).mean(axis=1)  # take the mean of each epoch
            # powerp_vvHPC = np.array(np.split(powerp_vvHPC[:crop_till], n_epochs)).mean(axis=1)
            power_dvHPC = np.array(np.split(power_dvHPC[:crop_till], n_epochs)).mean(axis=1)

            x = power_mPFC
            y = power_dvHPC
            corr = np.corrcoef(x, y)[0][1]
            label = 'R sequred = {:.2f}'.format(np.square(corr))
            title = 'Power_mPFC vs Power_dvHPC ' + band
            axes[j, i].scatter(x, y, s=10, label= spd +'_' + band + '\n' + label)
            m, b = np.polyfit(x, y, 1)
            axes[j, i].plot(x, m * x + b)
            axes[j, i].legend(loc='upper right')

            # y = powerp_vvHPC
            # corr = np.corrcoef(x,y)[0][1]
            # label = 'R sequred = {:.2f}'.format(np.square(corr))
            # title = 'Speed vs Power_vvHPC in Closed ' + band
            # axes[i, 1].scatter(x, y, s=10, label= band + '\n' + label)
            # m, b = np.polyfit(x, y, 1)
            # axes[i, 1].plot(x, m*x + b)
            # axes[i, 1].legend(loc='upper right')

        #
        # axes[5, 0].set_xlabel(xlabel)
        # axes[5, 1].set_xlabel(xlabel)
        # axes[5, 2].set_xlabel(xlabel)
        #
        # axes[0, 0].set_ylabel(ylabel)
        # axes[1, 0].set_ylabel(ylabel)
        # axes[2, 0].set_ylabel(ylabel)
        # axes[3, 0].set_ylabel(ylabel)
        # axes[4, 0].set_ylabel(ylabel)
        # axes[5, 0].set_ylabel(ylabel)
        #
        # axes[0, 0].set_title('mPFC')
        # axes[0, 1].set_title('vvHPC')
        # axes[0, 2].set_title('dvHPC')

        fig.suptitle('Power correlate between mPFC and vHPC with different speed \n '
                     'Speed = ' + spd + '\n'
                     'Epoch length = ' + str(epoch_length) + ' seconds' + '\n '
                     'Number of epochs = ' + str(n_epochs) + '\n'
                     'mPFC, dvHPC \n '
                     'delta(1.5-4.0 Hz), theta(4-7 Hz), alpha(7-10 Hz), delta(10-30 Hz), gamma(30 - 80 Hz), fast(80-200 Hz)')
        fig.tight_layout()
        fig.subplots_adjust(top=0.88)

    plt.title('Power correlation between mPFC and vHPC')
    plt.savefig(sdir_oft + 'Power correlate between mPFC and vHPC with different speed ' + str(epoch_length) + ' s.png', dpi=300, transparent=True)
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
