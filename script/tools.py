
from intanutil.load_intan_rhd_format import read_data

import numpy as np
import matplotlib.pyplot as plt
import glob
import re # Regular expression operations
import pandas as pd
from scipy import signal
import mne
from tqdm import tqdm
import os
from shapely.geometry import Point, Polygon
import geopandas as gpd
from matplotlib.mlab import specgram
from scipy.signal import medfilt
import seaborn as sb

from pactools import Comodulogram, REFERENCES
from pactools import simulate_pac
from pactools.dar_model import DAR, extract_driver

all_pac_methods = [
    'ozkurt', 'canolty', 'tort', 'penny', 'vanwijk', 'duprelatour', 'colgin',
    'sigl', 'bispectrum'
    ]
    


def init_ezm_gem():
    open_area1 = Polygon([(78, 40), (128, 110), (298, 119), (360, 60), (280, 0),(156, 0)])
    open_area2 = Polygon([(280, 291), (338, 369), (260, 400), (106, 400), (49, 335), (113, 277)])
    close_area1 = Polygon([(49, 335), (113, 277), (128, 110), (78, 40), (0,100), (0, 270)])
    close_area2 = Polygon([(360, 60), (298, 119), (280, 291), (338, 369), (400, 280), (400, 160)])
    return [open_area1, open_area2, close_area1, close_area2]

def init_electrod_gem():
    '''
    The logic of the id array is that the indices from 0-63 correspond to the
    electrode pads from top to bottom in their geometrical positions, and
    the corresponding entries show which Intan amplifier channel belongs to
    that electrode pad. (e.g. Ch 51 in Intan software is the deepest
    electrode, Ch 28 in Intan software is the most superficial one).
    '''

    id = {0:[51], 1:[53], 2:[48], 3:[37], 4:[61], 5:[59], 6:[36], 7:[44],
          8:[63], 9:[58], 10:[55], 11:[50], 12:[47], 13:[42], 14:[39], 15:[34],
          16:[62], 17:[57], 18:[54], 19:[49], 20:[46], 21:[41], 22:[38], 23:[33],
          24:[52], 25:[60], 26:[40], 27:[35], 28:[56], 29:[43], 30:[45], 31:[32],
          32:[3], 33:[4], 34:[5], 35:[16], 36:[13], 37:[0], 38:[24], 39:[19],
          40:[30], 41:[25], 42:[22], 43:[17], 44:[14], 45:[9], 46:[6], 47:[2],
          48:[31], 49:[26], 50:[23], 51:[18], 52:[15], 53:[10], 54:[8], 55:[1],
          56:[7], 57:[11], 58:[21], 59:[20], 60:[12], 61:[27], 62:[29], 63:[28]
         }

    vertical_distance = 25 ## um
    horizontal_distance = 5 ## um

    geometry = {}

    for i in range(64):
        geometry[id[63-i][0]] = (i*horizontal_distance, i*vertical_distance)

    return [geometry]

def load_all_data(dataset):
            
    dataset['brain_area_map'] = {}
    for key in dataset['brain_area']:
        dataset['brain_area_map'].update({key:
                                          [dataset['channel_map_dict'][ch]
                                           for ch in dataset['brain_area'][key]]})
    
    data     = read_data  (dataset['lfp_dir']
                           +'/'
                           +dataset['lfp_file_list'][0])
    location = pd.read_hdf(dataset['loc_file'])
    
    ch_name = [d['native_channel_name'] for d in data['amplifier_channels']]
    ch_impedence = np.array([int(d['electrode_impedance_magnitude']) for d in data['amplifier_channels']])/1000
    
    amplifier_data = []
    event_raw      = []
    for file in dataset['lfp_file_list']:
        print('>>'+file)
        data = read_data  (dataset['lfp_dir']+'/'+file)
        amplifier_data.append(data['amplifier_data'])
        event_raw.append(data['board_dig_in_data'])

    amplifier_data = np.concatenate(amplifier_data, axis = 1)
    event_raw      = np.concatenate(event_raw, axis = 1)
    data.update({'amplifier_data'   : amplifier_data,
                 'board_dig_in_data': event_raw})
    
    return {'data_info'    : dataset,
            'lfp'          : data,
            'loc'          : location,
            'ch_name'      : ch_name,
            'ch_impedence' : ch_impedence,
            'event_raw'    : event_raw,
            'sampling_rate': data['frequency_parameters']['board_adc_sample_rate']}

def get_brain_area_by_name(dataset, brain_area_name):
    name = brain_area_name
    try:
        if brain_area_name in dataset['data_info']['brain_area_map']:
             brain_area_name = dataset['data_info']['brain_area_map'][brain_area_name]
        else:
            name = 'None'
    except:
        name = 'None'

    return [brain_area_name, name]

def is_in_range(data_array, range_list):
    data_array = np.array(data_array)
    result = np.zeros(np.shape(data_array)).astype(np.bool_)
    for index in range_list:
        result |=  (data_array >= index[0]) & (data_array <= index[1])
    return result
    
def process_get_spike_train(dataset):
    print("Escape Spike Train.")
    return dataset

def process_downsample_by_50(dataset):
    ephys_2000Hz = signal.decimate(dataset['lfp']['amplifier_data'], 10, ftype='fir', axis= -1, zero_phase=True)
    ephys_400hz = signal.decimate(ephys_2000Hz, 5, ftype='fir', axis= -1, zero_phase=True)
    lfp         = dataset['lfp']
    lfp['amplifier_data'] =  ephys_400hz

    dataset.update({'lfp':lfp})
    dataset.update({'sampling_rate':400})
    return dataset

def process_band_pass_filer(dataset):
    freqs = {
    'Raw':0,
    'delta':{'low': 1,      'high':4},
    'theta':{'low': 4,      'high':8},
    'alpha':{'low': 8,      'high':13},
    'beta' :{'low': 13,     'high':30},
    'gamma':{'low': 30,     'high':70}
    }
    sampling_rate = dataset['sampling_rate']
  
    theta = np.zeros(np.shape(dataset['lfp']['amplifier_data']))
    beta  = np.zeros(np.shape(dataset['lfp']['amplifier_data']))
    gamma = np.zeros(np.shape(dataset['lfp']['amplifier_data']))
    
    bands = {'theta':{'value':theta},
             'beta' :{'value':beta},
             'gamma':{'value':gamma}
            }
    
    for i in tqdm(range(dataset['lfp']['amplifier_data'].shape[0])):
        for frq in bands:
            bands[frq]['value'][i, :] = mne.filter.filter_data(dataset['lfp']['amplifier_data'][i, :],
                                               sfreq  = sampling_rate,
                                               l_freq = freqs[frq]['low'],
                                               h_freq = freqs[frq]['high'],
                                               n_jobs=1)
    dataset['bands'] = bands
    return dataset
    

def process_hilbert_tranform(dataset):

    for frq in dataset['bands']:
        power = np.zeros(np.shape(dataset['bands'][frq]['value']))
        phase = np.zeros(np.shape(dataset['bands'][frq]['value']))

        for i in range(dataset['lfp']['amplifier_data'].shape[0]):
            # convert uV to mV
            hilbert_trans = signal.hilbert(dataset['bands'][frq]['value'][i, :]*10e-3)
            ### square the amplitude to get the power
            power[i, :] = np.abs(hilbert_trans)**2
            phase[i, :] = np.angle(hilbert_trans)

        dataset['bands'][frq].update({'power':power})
        dataset['bands'][frq].update({'phase':phase})
    return dataset
def process_align_lfp_vs_loc(dataset):
    return dataset

def plot_location(dataset, with_filter = True, axs = None):
    fig = plt.figure(figsize=(11, 11))
    score = dataset['loc'].columns[0][0];
    x_data = dataset['loc'][score, 'shoulder', 'x'][:]
    y_data = dataset['loc'][score, 'shoulder', 'y'][:]
    # Use median filter
    if with_filter:
        kernel_size = 5
        x_data = medfilt(x_data, kernel_size)
        y_data = medfilt(y_data, kernel_size)

    # Plot
    sb.scatterplot(x=x_data, y=y_data, s=5, color=".15")
    sb.histplot   (x=x_data, y=y_data, bins=100, pthresh=.1, cmap="mako")
    sb.kdeplot    (x=x_data, y=y_data, levels=5, color="w", linewidths=1)

def plot_hist(dataset, band_frq, axs = None):

    working_ch_hipp = range(30) + 30
    if axs == None:
        fig, axs = plt.subplots(len(working_ch_hipp), 1, figsize=(4, 4*8))

    for i, ele in enumerate(working_ch_hipp):
        phase_diff1 = dataset['bands'][band_frq][0,:] - dataset['bands'][band_frq][ele-30,:]
        axs[i].hist(theta_phase_diff1, bins=round(len(theta_phase_diff1)/srate*0.5), alpha=1) ### bin size = 0.5 second
        axs[i].set_title('Theta phase lag / Ch64 vs. Ch' + str(ele))
        axs[i].grid(True)

    fig.tight_layout()
    plt.show()

def plot_event(dataset, axs = None):
    fig = plt.figure(figsize=(11, 4))
    sampling_rate = dataset['sampling_rate']
    time = np.arange(0, dataset['event_raw'].shape[-1]/sampling_rate, 1/sampling_rate)
    plt.plot(time, dataset['event_raw'][0], '--k', alpha=.4)
    plt.plot(time, dataset['event_raw'][1], 'b', alpha=.2)

    plt.title('Trigger + video_frame')
    plt.xlabel('sec')
    plt.show()

def plot_spectrum(dataset, channel_list = [0], axs = None):

    try:
        if channel_list in dataset['brain_area_map']:
            channel_list = dataset['brain_area_map'][channel_list]
    except:
        pass

    if axs == None:
        fig, axs = plt.subplots(len(channel_list))

    for i, index in enumerate(channel_list):
        Pxx, frq, t = specgram(ezm_data['lfp']['amplifier_data'][index,:],
                               NFFT=1000,
                               Fs=400,
                               noverlap=900,
                               pad_to=8192)
        axs[i].pcolormesh(t, frq, 10*np.log10(Pxx), cmap='hot', shading='auto')


def plot_travel_distance_over_time(datasets, axs = None):
    fig, axs = plt.subplots(1)
    for dataset in datasets:
        score = dataset['loc'].columns[0][0]
        x = dataset['loc'][score, 'shoulder', 'x'][:]* dataset['data_info']['pixel_to_mm']
        y = dataset['loc'][score, 'shoulder', 'y'][:]* dataset['data_info']['pixel_to_mm']
        time = np.arange(len(x)) / 40
        cum_dis = np.sqrt(np.cumsum(x**2) + np.cumsum(y**2))

        values = cum_dis
        plt.plot(time, values)

def plot_compare(dataset1, dataset2, grouby_brain_area=[0], axs = None): #two dataset compare power, speed ~ power
    # caclucalte the speed

    speed_list = []
    window_size = 40 # in 5 second
    for dataset in [dataset1,dataset2]:
        score = dataset['loc'].columns[0][0]
        x = dataset['loc'][score, 'shoulder', 'x'][:]
        y = dataset['loc'][score, 'shoulder', 'y'][:]
        speed = np.sqrt(np.diff(x)**2 + np.diff(y)**2) * dataset['data_info']['video_frame_rate'] * dataset['data_info']['pixel_to_mm']
        speed = pd.Series(speed).rolling(window=window_size).mean().iloc[window_size-1:].values
        speed_list.append(speed)

    # caculate the traveling distance
    # speed ~ bandpower
    for speed, dataset,data_name in zip(speed_list,[dataset1,dataset2], ['set1', 'set2']):
        data_list = []
        for band in dataset['bands'].keys():
            power = signal.decimate(dataset['bands'][band]['power']
                                    , 10
                                    , ftype='fir'
                                    , axis= -1
                                    , zero_phase=True)
            x_len = len(speed)
            theta_len = np.shape(power)[1]
            min_len = min(x_len, theta_len)
            
            [grouby_brain_area_map,_] = get_brain_area_by_name(dataset, grouby_brain_area)
            power = 10*np.log10(np.mean(power[grouby_brain_area_map,:min_len],axis=0))
            speed = 20*np.log10(speed)

            data = pd.DataFrame.from_dict({
               'Speed_dB':speed,
               'Power_dB':power,
               'Band' : band,
               'Dataset': data_name
            })
            data_list.append(data)

        data = pd.concat(data_list)

        g = sb.jointplot(
        data=data,
        x="Speed_dB", y="Power_dB", hue="Band",
        kind="kde")
        plt.title(dataset['Name'])

def plot_compare_lfp_power_vs_loc(dataset, brain_area_1 = [0], brain_area_2 = [0], axs = None):

    [brain_area1, name1] = get_brain_area_by_name(dataset, brain_area_1)
    [brain_area2, name2] = get_brain_area_by_name(dataset, brain_area_2)

    for brain_area, name in zip([brain_area1, brain_area2],[name1, name2]):
        data_list = []
        for band in dataset['bands'].keys():

            score  = dataset['loc'].columns[0][0]
            x = dataset['loc'][score, 'shoulder', 'x'][:]
            y = dataset['loc'][score, 'shoulder', 'y'][:]
            power = signal.decimate(dataset['bands'][band]['power']
                                    , 10
                                    , ftype='fir'
                                    , axis= -1
                                    , zero_phase=True)
            x_len = len(x)
            theta_len = np.shape(power)[1]
            min_len = min(x_len, theta_len)

            x = x[:min_len]
            y = y[:min_len]
            power = 10*np.log10(np.mean(power[brain_area,:min_len],axis=0))
            angel = np.arctan2(y-np.median(y),x-np.median(x))/np.pi*180
            data = pd.DataFrame.from_dict({
               'Angel':angel,
               'Power_dB':power,
               'Band' : band
            })
            data_list.append(data)

        data = pd.concat(data_list)

        plt.figure()
        g = sb.jointplot(
        data=data,
        x="Angel", y="Power_dB", hue="Band",
        kind="kde")
        plt.title(name)

    
def stat_compare_lfp_power_vs_loc(dataset, brain_area_1 = [0], brain_area_2 = [0], axs = None):

    [brain_area1, name1] = get_brain_area_by_name(dataset, brain_area_1)
    [brain_area2, name2] = get_brain_area_by_name(dataset, brain_area_2)

    for brain_area, name in zip([brain_area1, brain_area2],[name1, name2]):
        data_list = []
        for band in dataset['bands'].keys():

            score  = dataset['loc'].columns[0][0]
            x = dataset['loc'][score, 'shoulder', 'x'][:]
            y = dataset['loc'][score, 'shoulder', 'y'][:]
            power = signal.decimate(dataset['bands'][band]['power']
                                    , 10
                                    , ftype='fir'
                                    , axis= -1
                                    , zero_phase=True)
            x_len = len(x)
            theta_len = np.shape(power)[1]
            min_len = min(x_len, theta_len)

            x = x[:min_len]
            y = y[:min_len]
            power = 10*np.log10(np.mean(power[brain_area,:min_len],axis=0))
            angel = np.arctan2(y-np.median(y),x-np.median(x))/np.pi*180
            data = pd.DataFrame.from_dict({
               'Open_Close':map(lambda x: 'Closed' if x else 'Open',
                                is_in_range(angel, [[45,135],[-135,45]])),
               'Power_dB':power,
               'Band' : band,
               'Brain_Area' : name
            })
            data_list.append(data)

        data = pd.concat(data_list)
        plt.figure()
        sb.boxplot(data = data,
                   x = 'Band',
                   y ='Power_dB',
                   hue ='Open_Close')
        plt.title(name)

    
    
def plot_compare_lfp_power_cross_frq_coh_vs_loc(dataset, brain_area_1 = [0], brain_area_2 = [0], axs = None):


    [brain_area1, _] = get_brain_area_by_name(dataset, brain_area_1)
    [brain_area2, _] = get_brain_area_by_name(dataset, brain_area_2)


def plot_compare_lfp_power_coh_vs_loc(dataset, brain_area_1 = [0], brain_area_2 = [0], axs = None, method = 'duprelatour'):

    if ~(method in all_pac_methods):
        method = 'duprelatour'
        
    bands ={
    'theta':{'low': 4,      'high':8},
    'beta' :{'low': 13,     'high':30},
    'gamma':{'low': 30,     'high':70}}


    [brain_area1, name1] = get_brain_area_by_name(dataset, brain_area_1)
    [brain_area2, name2] = get_brain_area_by_name(dataset, brain_area_2)
    
    n_columns = len(bands.keys())
    n_lines   = 2
    frq_res   = 50
    fig, axs = plt.subplots(
    n_lines, n_columns, figsize=(4 * n_columns, 3 * n_lines))
    axs = axs.ravel()
    
    fs = dataset['sampling_rate']
    
    for i, (area, name) in enumerate(zip([brain_area1,brain_area1], [name1,name2])):
        for index, band in enumerate(bands):
            low_fq_range = np.linspace(bands[band]['low'], bands[band]['high'], frq_res)
            low_fq_width = 1.0
            signal = np.mean(dataset['lfp']['amplifier_data'][area,:], axis = 0)
            estimator = Comodulogram(fs=fs, low_fq_range=low_fq_range,
                                     low_fq_width=low_fq_width, method=method,
                                     progress_bar=False)
            estimator.fit(signal)
            estimator.plot(titles= [band +'@'+name], axs=[axs[index + i * n_columns]])


def stat_compare_lfp_power_coh_vs_loc(dataset, brain_area_1 = [0], brain_area_2 = [0], axs = None, method = 'duprelatour'):

    if ~(method in all_pac_methods):
        method = 'duprelatour'

    [brain_area1, _] = get_brain_area_by_name(dataset, brain_area_1)
    [brain_area2, _] = get_brain_area_by_name(dataset, brain_area_2)

def plot_compare_lfp_phase_coh_vs_loc(dataset, brain_area_1 = [0], brain_area_2 = [0], axs = None):

    [brain_area1, _] = get_brain_area_by_name(dataset, brain_area_1)
    [brain_area2, _] = get_brain_area_by_name(dataset, brain_area_2)

def stat_compare_lfp_phase_coh_vs_loc(dataset, brain_area_1 = [0], brain_area_2 = [0], axs = None):

    [brain_area1, _] = get_brain_area_by_name(dataset, brain_area_1)
    [brain_area2, _] = get_brain_area_by_name(dataset, brain_area_2)

def plot_compare_fire_rate_vs_loc(dataset, brain_area_1 = [0], brain_area_2 = [0], axs = None):

    [brain_area1, _] = get_brain_area_by_name(dataset, brain_area_1)
    [brain_area2, _] = get_brain_area_by_name(dataset, brain_area_2)

def stat_compare_fire_rate_vs_loc(dataset, brain_area_1 = [0], brain_area_2 = [0], axs = None):

    [brain_area1, _] = get_brain_area_by_name(dataset, brain_area_1)
    [brain_area2, _] = get_brain_area_by_name(dataset, brain_area_2)
    
def plot_copula_connectivity_loc(dataset, brain_area_1 = [0], brain_area_2 = [0], axs = None):

    [brain_area1, _] = get_brain_area_by_name(dataset, brain_area_1)
    [brain_area2, _] = get_brain_area_by_name(dataset, brain_area_2)

def stat_copula_connectivity_loc(dataset, brain_area_1 = [0], brain_area_2 = [0], axs = None):

    [brain_area1, _] = get_brain_area_by_name(dataset, brain_area_1)
    [brain_area2, _] = get_brain_area_by_name(dataset, brain_area_2)
