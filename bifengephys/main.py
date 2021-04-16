import sys
import scipy.stats
from scipy.signal import coherence
from scipy.cluster.hierarchy import fcluster
from sklearn import metrics
from trajectory_process import traj_process

sys.path.append('D:\ephys')
import numpy as np
from scipy.cluster import hierarchy
from sklearn import cluster
import matplotlib.pyplot as plt
# %matplotlib notebook
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

import utils
import ephys
import plotting

plt.rcParams["axes.labelsize"] = 12
plt.rcParams["font.size"] = 7
plt.rcParams["font.family"] = "Arial"

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


def slice_from_arr(arr, idxs, channels=None, window=1, f_ephys=500, f_behavior=50, mean=True):
    window_samples = window * f_ephys
    f_ratio = f_ephys / f_behavior

    if not channels:
        channels = arr.shape[0]

    ret = []
    for channel in channels:
        power_per_channel = []
        for idx in idxs:
            idx_in_ephys = f_ratio * idx
            window_from = np.max([int(idx_in_ephys - window_samples), 0])
            window_to = np.min([int(idx_in_ephys + window_samples), arr.shape[-1]-1])
            if window_to==0:
                window_to=1
            bit_power = arr[channel, window_from:window_to]
            if mean:
                power_per_channel.append(bit_power.mean())
            else:
                power_per_channel.append(bit_power)
        ret.append(np.vstack(power_per_channel))

    if mean:
        return np.stack(ret)[:, :, 0]
    else:
        return np.stack(ret)


def main():
    # %%
    np.random.seed(42)
    animal = mBWfus009
    session = 'ezm_0226'
    behavior_trigger = 14.24
    events = traj_process(animal[session], behavior='ezm', start_time=0, duration=10)
    # events = pickle.load(open('D:\\ephys\\2021-02-19_mBWfus009_EZM_ephys\ephys_processed\\2021-02-19_mBWfus009_EZM_ephys_results_manually_annotated.pickle',
    #                           "rb"),)
    f_behavior = 50
    f_ephys = 500
    behavior_window_duration = 600
    ephys_window_duration = 600

    ### extract overall behavioral open/close frame indices
    open_idx = [i for i, el in enumerate(
        events['rois_stats']['roi_at_each_frame'][
        int(f_behavior * behavior_trigger):int(f_behavior * behavior_trigger) + f_behavior * behavior_window_duration])
                if
                el == 'open']
    close_idx = [i for i, el in enumerate(
        events['rois_stats']['roi_at_each_frame'][
        int(f_behavior * behavior_trigger):int(f_behavior * behavior_trigger) + f_behavior * behavior_window_duration])
                 if
                 el == 'closed']
    open_to_close_idx = events['transitions']['open_closed_exittime']
    prolonged_close_to_open_idx = events['transitions']['prolonged_open_closed_exittime']
    prolonged_open_to_close_idx = events['transitions']['prolonged_closed_open_exittime']

    dataset = ephys.load_data(animal[session])

    ### --- cluster analysis - returns relevant cluster channels
    # Todo: the cluster_threshold is sensitive. Can cause error

    vhipp_channels = explore_clusters(dataset, "vhipp", cluster_threshold=2., plot=True)
    mpfc_channels = explore_clusters(dataset, "mpfc", cluster_threshold=1.5, plot=True)

    # coherence between areas

    lfp_mpfc = ephys.get_lfp(dataset, brain_area='mpfc')
    lfp_vhipp = ephys.get_lfp(dataset, brain_area='vhipp')

    ### --- aligning data
    ephys_trigger = dataset['info']['ephys_trigger']

    ### - crop out ephys trigger
    lfp_mpfc = lfp_mpfc[:, int(f_ephys * ephys_trigger):]
    lfp_vhipp = lfp_vhipp[:, int(f_ephys * ephys_trigger):]
    power_mpfc = ephys.get_power(dataset, 'mpfc')
    power_vhipp = ephys.get_power(dataset, 'vhipp')

    power_mpfc = power_mpfc[:, int(f_ephys * ephys_trigger):f_ephys * ephys_window_duration]
    power_vhipp = power_vhipp[:, int(f_ephys * ephys_trigger):f_ephys * ephys_window_duration]

    # downsample to the same srate as video
    # power_mpfc = signal.decimate(power_mpfc[:, int(500 * ephys_trigger):], 10, ftype='fir', zero_phase=True)
    # power_vhipp = signal.decimate(power_vhipp[:, int(500 * ephys_trigger):], 10, ftype='fir', zero_phase=True)

    ### --- windowed power calculation
    # TODO: cleanup - fcn
    power_vhipp_open = []
    power_mpfc_open = []

    # power_vhipp_open = [power_vhipp[:, 10 * idx] for idx in open_idx]
    # power_mpfc_open = [power_mpfc[:, 10 * idx] for idx in open_idx]
    # power_vhipp_open = np.array(power_vhipp_open).T
    # power_mpfc_open = np.array(power_mpfc_open).T
    #
    # power_vhipp_close = [power_vhipp[:, 10 * idx] for idx in open_idx]
    # power_mpfc_close = [power_vhipp[:, 10 * idx] for idx in open_idx]
    # power_vhipp_close = np.array(power_vhipp_close).T
    # power_mpfc_close = np.array(power_mpfc_close).T

    mean_power_mpfc = []
    mean_power_vhipp = []

    window_samples = 500  # 2 second

    # mean power of the specified window size
    # how to remove the outlier ?

    ### time series plots for transition events
    power_prolonged_close_to_open = slice_from_arr(power_vhipp,
                                                   prolonged_close_to_open_idx,
                                                   channels=vhipp_channels,
                                                   window=1,
                                                   mean=False)
    power_prolonged_open_to_close = slice_from_arr(power_vhipp,
                                                   prolonged_open_to_close_idx,
                                                   channels=vhipp_channels,
                                                   window=1,
                                                   mean=False)
    power_open_to_close = slice_from_arr(power_vhipp,
                                                   open_to_close_idx,
                                                   channels=vhipp_channels,
                                                   window=1,
                                                   mean=False)
    conditions = [power_prolonged_close_to_open, power_prolonged_open_to_close, power_open_to_close]
    titles = ['power_prolonged_close_to_open','power_prolonged_open_to_close','power_open_to_close']

    fig, ax = plt.subplots(len(conditions),len(range(power_prolonged_close_to_open.shape[0])))

    for condition_idx, condition in enumerate(conditions):
        for channel_idx, channel in enumerate(range(power_prolonged_close_to_open.shape[0])):
            ax[condition_idx,channel_idx].set_title(titles[condition_idx] + '__channel:' + str(channel_idx))
            ax[condition_idx,channel_idx].plot(power_prolonged_close_to_open[channel,:,:].transpose())
    plt.show()


    ### mean plots for checking power during different spatial dependencies
    iter = int(len(power_mpfc[0, :]) / window_samples)
    idxs = list(range(iter))
    mean_power_vhipp = slice_from_arr(power_vhipp, idxs, channels=vhipp_channels)
    mean_power_mpfc = slice_from_arr(power_mpfc, idxs, channels=mpfc_channels)

    open_power_vhipp = slice_from_arr(power_vhipp, open_idx, channels=vhipp_channels, window=0)
    #TODO: fix nan
    closed_power_vhipp = slice_from_arr(power_vhipp, close_idx, channels=mpfc_channels, window=0)

    power_prolonged_close_to_open = slice_from_arr(power_vhipp,
                                                   prolonged_close_to_open_idx,
                                                   channels=vhipp_channels,
                                                   window=1,
                                                   mean=True)
    power_prolonged_open_to_close = slice_from_arr(power_vhipp,
                                                   prolonged_open_to_close_idx,
                                                   channels=vhipp_channels,
                                                   window=1,
                                                   mean=True)
    power_open_to_close = slice_from_arr(power_vhipp,
                                                   open_to_close_idx,
                                                   channels=vhipp_channels,
                                                   window=1,
                                                   mean=True)

    conditions = [mean_power_vhipp.flatten(),
                  open_power_vhipp.flatten(),
                  closed_power_vhipp.flatten(),
                  power_prolonged_close_to_open.flatten(),
                  power_prolonged_open_to_close.flatten()]
    titles = ['mean_power_vhipp','open_power_vhipp','closed_power_vhipp','power_prolonged_close_to_open','power_prolonged_open_to_close']

    import seaborn as sns
    sns.boxplot(x=titles, y=conditions,)

    plotting = 0
    if plotting:

        fig, ax = plt.subplots(nrows=len(mpfc_channels),
                               ncols=len(vhipp_channels), figsize=(10, 13))
        for chan_mpfc, _ in enumerate(mean_power_mpfc):
            for chan_vhipp, _ in enumerate(mean_power_vhipp):
                x = mean_power_vhipp[chan_vhipp, :]
                y = mean_power_mpfc[chan_mpfc, :]
                # TODO: zscore here? how to remove the outlier ?
                x = scipy.stats.zscore(x)
                y = scipy.stats.zscore(y)
                r2 = scipy.stats.pearsonr(x, y)
                # plt.text('R-squared = %0.2f' % r2)
                # test = scipy.stats.zscore(x)
                # TODO: double check to square the r value?!
                slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x, y)
                ax[chan_mpfc, chan_vhipp].scatter(x, y)
                ax[chan_mpfc, chan_vhipp].set_title('R-squared = %0.2f' % r_value)
                ax[chan_mpfc, chan_vhipp].set_xlabel('mpfc ' + str(mpfc_channels[chan_mpfc]))
                ax[chan_mpfc, chan_vhipp].set_ylabel('vhipp' + str(vhipp_channels[chan_vhipp]))
                ax[chan_mpfc, chan_vhipp].plot(np.unique(x), np.poly1d(np.polyfit(x, y, 1))(np.unique(x)))
        fig.tight_layout()
        fig.show()

        fig, ax = plt.subplots(nrows=len(mpfc_channels),
                               ncols=len(vhipp_channels), figsize=(10, 13))
        for chan_mpfc, _ in enumerate(mean_power_mpfc):
            for chan_vhipp, _ in enumerate(mean_power_vhipp):
                x = mean_power_vhipp[chan_vhipp, :]
                y = mean_power_mpfc[chan_mpfc, :]
                # TODO: zscore here?
                x = scipy.stats.zscore(x)
                y = scipy.stats.zscore(y)
                r2 = scipy.stats.pearsonr(x, y)
                # plt.text('R-squared = %0.2f' % r2)
                # test = scipy.stats.zscore(x)
                # TODO: double check to square the r value?!
                slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x, y)
                ax[chan_mpfc, chan_vhipp].scatter(x, y)
                ax[chan_mpfc, chan_vhipp].set_title('R-squared = %0.2f' % r_value)
                ax[chan_mpfc, chan_vhipp].set_xlabel('mpfc ' + str(mpfc_channels[chan_mpfc]))
                ax[chan_mpfc, chan_vhipp].set_ylabel('vhipp' + str(vhipp_channels[chan_vhipp]))
                ax[chan_mpfc, chan_vhipp].plot(np.unique(x), np.poly1d(np.polyfit(x, y, 1))(np.unique(x)))
        fig.tight_layout()
        # xlabel('vhipp representative channels')
        # ylabel('mpfc representative channels')
        fig.show()

        ### --- overall coherence analysis

        coherence_mpfc_to_vhipp = np.zeros((len(vhipp_channels), len(mpfc_channels)))
        coherence_bands = []
        correlation_vals = []

        new_freq = 50
        old_freq = 500

        for vhipp_id, vhipp_channel in enumerate(vhipp_channels):
            vhipp_data = lfp_vhipp[vhipp_channel, :old_freq * 100]
            for mpfc_id, mpfc_channel in enumerate(mpfc_channels):
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


def explore_clusters(dataset, area, cluster_threshold, plot=True):
    lfp = ephys.get_lfp(dataset, brain_area=area)
    power = ephys.get_power(dataset, area)
    chanl_list = ephys.get_ch(dataset, brain_area=area)
    coherence_matrix = np.zeros((len(lfp), len(lfp)))
    start = 30
    srate = 500

    frequs = [5, 25]
    idxs = [3, 13]

    for x_id, x in tqdm(enumerate(lfp[:, start * srate:(start + 100) * srate])):
        for y_id, y in enumerate(lfp[:, start * srate:(start + 100) * srate]):
            coherence_matrix[x_id, y_id] = coherence(x=x, y=y, fs=500)[1][:20].mean()

    if plot:
        plt.figure(figsize=(10, 10))
        plt.imshow(coherence_matrix, cmap='jet')
        plt.colorbar()
        plt.show()

    corr_linkage = hierarchy.ward(coherence_matrix)
    dendro = hierarchy.dendrogram(corr_linkage, leaf_rotation=90)
    dendro_idx = np.arange(0, len(dendro["ivl"]))

    dendro_idx_pad = [str(ephys.arr_to_pad(chanl_list[int(el)])) for el in list(dendro["ivl"])]

    if plot:
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        im = ax.imshow(coherence_matrix[dendro["leaves"], :][:, dendro["leaves"]], cmap="jet")
        ax.set_xticks(dendro_idx)
        ax.set_yticks(dendro_idx)
        ax.set_xticklabels(dendro_idx_pad, rotation="vertical")
        ax.set_yticklabels(dendro_idx_pad)
        fig.tight_layout()
        fig.colorbar(im, orientation="vertical")
        plt.show()

    # clusters based on thresholding
    clusters = fcluster(corr_linkage, cluster_threshold, criterion='distance')

    # sort elements by clusters and put into dictionary
    test1 = [x for _, x in sorted(zip(clusters, dendro_idx))]
    test2 = [x for x, _ in sorted(zip(clusters, dendro_idx))]
    clusts = [str(ephys.arr_to_pad(chanl_list[int(el)])) for el in test1]
    clusters_array = {}
    clusters_pad = {}
    for id, cluster in enumerate(test2):
        if cluster not in clusters_array.keys():
            clusters_array[cluster] = []
            clusters_pad[cluster] = []
        else:
            clusters_array[cluster].append(test1[id])
            clusters_pad[cluster].append(clusts[id])

    # double checking clustering single channel
    # chan1 = 16
    # chan2 = 19
    #
    # channels = coherence(x=lfp[chan1, :500 * 100], y=lfp[chan2, :500 * 100], fs=500)[1][
    #            :20].mean()
    # print(dendro["leaves"])
    # print(channels)

    for cluster in clusters_array.keys():
        means = []
        for channel_1 in clusters_array[cluster]:
            for channel_2 in clusters_array[cluster]:
                if channel_2 == channel_1:
                    continue
                means.append(
                    coherence(x=lfp[channel_1, start * srate:(start + 100) * srate],
                              y=lfp[channel_2, start * srate:(start + 100) * srate], fs=srate)[1][:20].mean())

        print('mean coherence for cluster :' + str(cluster) + '  is:' + str(np.mean(means)) + 'and std: ' + str(
            np.std(means)))
        print(clusters_array[cluster])
        print(clusters_pad[cluster])

    # TODO: check for outliers like channel 41 in vHipp
    # TODO: check smarter solution than just selecting one

    # select channel with highest power
    representative_channels = []
    for cluster in clusters_array.keys():
        cluster_power = power[clusters_array[cluster], :]
        cluster_power = np.nanmean(cluster_power, axis=1)  # cluster_power.mean(axis=1)
        channel_idx = np.where(cluster_power == np.max(cluster_power))[0][0]  ### this line caused error in some dataset
        representative_channels.append(clusters_array[cluster][channel_idx])

    return representative_channels


if __name__ == '__main__':
    main()
