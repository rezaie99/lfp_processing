import pickle
import numpy as np
import pandas as pd
from scipy.signal import coherence
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.cluster import hierarchy
from scipy.cluster.hierarchy import fcluster
from sklearn.cluster import KMeans
import os
import sys


# sys.path.append('D:\ephys')


def load_data(session):
    print(session)
    file = 'D:\\ephys\\' + session + '\\ephys_processed\\' + session + '_dataset.pkl'
    # with open(session + '/ephys_processed/' + session + '_dataset.pkl', "rb") as f:
    with open(file, 'rb') as f:
        data = pickle.load(f)
        f.close()
    return data


def get_ch(data, brain_area):
    if brain_area == 'all':
        return sum(data['info']['ch_names'].values(), [])
    else:
        return data['info']['ch_names']['ch_' + brain_area]


def get_lfp(dataset, brain_area, f_ephys=500):
    print(dataset['lfp']['amplifier_data'].shape)
    ephys_trigger = dataset['info']['ephys_trigger']
    crop_from = int(f_ephys * ephys_trigger)

    if brain_area == 'all':
        lfp = dataset['lfp']['amplifier_data'][:, crop_from:]
        ch_list = get_ch(dataset, 'all')
        lfp = pd.DataFrame(data=lfp.T, columns=ch_list)
    if brain_area == 'mpfc':
        lfp = dataset['lfp']['amplifier_data'][:len(get_ch(dataset, 'mpfc')), crop_from:]
        lfp = pd.DataFrame(data=lfp.T, columns=get_ch(dataset, 'mpfc'))
    if brain_area == 'vhipp':
        lfp = dataset['lfp']['amplifier_data'][len(get_ch(dataset, 'mpfc')):, crop_from:]
        lfp = pd.DataFrame(data=lfp.T, columns=get_ch(dataset, 'vhipp'))
    print(lfp.shape)
    return lfp


def get_power(dataset, brain_area, band='theta', f_ephys=500):
    print(dataset['lfp']['amplifier_data'].shape)
    ephys_trigger = dataset['info']['ephys_trigger']
    crop_from = int(f_ephys * ephys_trigger)
    if brain_area == 'all':
        ch_list = get_ch(dataset, 'all')
        dat = dataset['bands'][band]['power'][:, crop_from:]
        power = pd.DataFrame(data=dat.T, columns=ch_list)

    if brain_area == 'mpfc':
        ch_list = get_ch(dataset, 'mpfc')
        dat = dataset['bands'][band]['power'][:len(ch_list), crop_from:]
        power = pd.DataFrame(data=dat.T, columns=ch_list)

    if brain_area == 'vhipp':
        ch_list = get_ch(dataset, 'vhipp')
        dat = dataset['bands'][band]['power'][len(get_ch(dataset, 'mpfc')):, crop_from:]
        power = pd.DataFrame(data=dat.T, columns=ch_list)

    print(power.shape)
    return power


def get_phase(dataset, brain_area, band='theta', f_ephys=500):
    # print(dataset['lfp']['amplifier_data'].shape)
    ephys_trigger = dataset['info']['ephys_trigger']
    crop_from = int(f_ephys * ephys_trigger)

    if brain_area == 'all':
        ch_list = get_ch(dataset, 'all')
        dat = dataset['bands'][band]['phase'][:, crop_from:]
        phase = pd.DataFrame(data=dat.T, columns=ch_list)

    if brain_area == 'mpfc':
        ch_list = get_ch(dataset, 'mpfc')
        dat = dataset['bands'][band]['phase'][:len(ch_list), crop_from:]
        phase = pd.DataFrame(data=dat.T, columns=ch_list)

    if brain_area == 'vhipp':
        ch_list = get_ch(dataset, 'vhipp')
        dat = dataset['bands'][band]['phase'][len(get_ch(dataset, 'mpfc')):, crop_from:]
        phase = pd.DataFrame(data=dat.T, columns=ch_list)

    # print(phase.shape)
    return phase


def get_speed(data):
    return data['instant_speed']


def get_distance(data):
    return data['accumulative_distance']


def sixtyfour_ch_solder_pad_to_zif(zif):  ###
    channel_map = {'T1': 9, 'T2': 10, 'T3': 11, 'T4': 12, 'T5': 13, 'T6': 14, 'T7': 15, 'T8': 16,
                   'T9': 'GND',
                   'T10': 49, 'T11': 50, 'T12': 51, 'T13': 52, 'T14': 53, 'T15': 54, 'T16': 55, 'T17': 56,
                   'T18': 48, 'T19': 47, 'T20': 46, 'T21': 45, 'T22': 44, 'T23': 43, 'T24': 42, 'T25': 41,
                   'T26': 'REF',
                   'T27': 24, 'T28': 23, 'T29': 22, 'T30': 21, 'T31': 20, 'T32': 19, 'T33': 18, 'T34': 17,
                   'B1': 57, 'B2': 58, 'B3': 59, 'B4': 60, 'B5': 61, 'B6': 62, 'B7': 63, 'B8': 64,
                   'B9': 'GND',
                   'B10': 1, 'B11': 2, 'B12': 3, 'B13': 4, 'B14': 5, 'B15': 6, 'B16': 7, 'B17': 8,
                   'B18': 32, 'B19': 31, 'B20': 30, 'B21': 29, 'B22': 28, 'B23': 27, 'B24': 26, 'B25': 25,
                   'B26': 'REF',
                   'B27': 40, 'B28': 39, 'B29': 38, 'B30': 37, 'B31': 36, 'B32': 35, 'B33': 34, 'B34': 33
                   }
    solder_pads = np.zeros(64)
    for ch in range(len(zif)):
        solder_pads[ch] = channel_map[zif[ch]]
    solder_pads = solder_pads.astype('int')
    return solder_pads


zif_connector_to_channel_id = {'T1': 9, 'T2': 10, 'T3': 11, 'T4': 12, 'T5': 13, 'T6': 14, 'T7': 15, 'T8': 16,
                               'T9': 'GND',
                               'T10': 49, 'T11': 50, 'T12': 51, 'T13': 52, 'T14': 53, 'T15': 54, 'T16': 55, 'T17': 56,
                               'T18': 48, 'T19': 47, 'T20': 46, 'T21': 45, 'T22': 44, 'T23': 43, 'T24': 42, 'T25': 41,
                               'T26': 'REF',
                               'T27': 24, 'T28': 23, 'T29': 22, 'T30': 21, 'T31': 20, 'T32': 19, 'T33': 18, 'T34': 17,
                               'B1': 57, 'B2': 58, 'B3': 59, 'B4': 60, 'B5': 61, 'B6': 62, 'B7': 63, 'B8': 64,
                               'B9': 'GND',
                               'B10': 1, 'B11': 2, 'B12': 3, 'B13': 4, 'B14': 5, 'B15': 6, 'B16': 7, 'B17': 8,
                               'B18': 32, 'B19': 31, 'B20': 30, 'B21': 29, 'B22': 28, 'B23': 27, 'B24': 26, 'B25': 25,
                               'B26': 'REF',
                               'B27': 40, 'B28': 39, 'B29': 38, 'B30': 37, 'B31': 36, 'B32': 35, 'B33': 34, 'B34': 33
                               }

intan_array_to_zif_connector = ['B23', 'T1', 'T34', 'B18', 'B19', 'B20', 'T33', 'B17',
                                'T2', 'T32', 'T3', 'B16', 'B13', 'B22', 'T31', 'T4',
                                'B21', 'T30', 'T5', 'B25', 'B14', 'B15', 'T29', 'T6',
                                'B24', 'T28', 'T7', 'B12', 'B10', 'B11', 'T27', 'T8',
                                'B34', 'T25', 'T10', 'B30', 'B2', 'B5', 'T24', 'T11',
                                'B29', 'T23', 'T12', 'B32', 'B1', 'B33', 'T22', 'T13',
                                'B6', 'T21', 'T14', 'B8', 'B27', 'B7', 'T20', 'T15',
                                'B31', 'T19', 'T16', 'B3', 'B28', 'B4', 'T18', 'T17']


### ch_000 ==> B23 ==> pad_27, ch_002 => T1 ...

### pad_1 ==> B10 ==> np.where(intan_array_to_zif_connector['B10'])


def arr_to_pad(el):
    array_to_pad = {}  ### how ephys data array register to the electrode pad. pad1-32 are in the mPFC, with pad1 is the deepest.

    cmap = sixtyfour_ch_solder_pad_to_zif(intan_array_to_zif_connector)
    for i in range(1, 65):
        chan_name = np.where(cmap == i)[0][0]
        if chan_name < 10:
            array_to_pad['A-00' + str(chan_name)] = str(i)
        else:
            array_to_pad['A-0' + str(chan_name)] = str(i)
    return array_to_pad[el]


def pad_to_array():
    pad_to_array = []
    pad_to_array_text = []
    cmap = sixtyfour_ch_solder_pad_to_zif(intan_array_to_zif_connector)
    for i in range(1, 65):
        pad_to_array.append(np.where(cmap == i)[0][0])
        pad_to_array_text.append('pad' + str(i) + '== A-0' + str(np.where(cmap == i)[0][0]))

    return pad_to_array, pad_to_array_text


def get_pad_name(df):
    pad_arr_idx = np.array(pad_to_array()[0])
    pad_list = []
    for col in df.columns:
        col_idx = int(col.split('-')[1])
        pad_list.append(np.where(pad_arr_idx == col_idx)[0][0])

    return pad_list

### rearrange the columns in the order of the electrode depth
def column_by_pad(df):
    col_name = get_pad_name(df)
    df.columns = col_name
    df = df.reindex(sorted(df.columns), axis=1)
    return df


def explore_clusters(dataset, area, cluster_threshold, plot=True, n_clusters=4):
    lfp = get_lfp(dataset, brain_area=area)
    power = get_power(dataset, area)
    chanl_list = get_ch(dataset, brain_area=area)
    coherence_matrix = np.zeros((len(lfp), len(lfp)))
    start = 30
    fps = 500

    # frequs = [5, 25]
    # idxs = [3, 13]

    for x_id, x in tqdm(enumerate(lfp[:, start * fps:(start + 20) * fps])):
        for y_id, y in enumerate(lfp[:, start * fps:(start + 20) * fps]):
            coherence_matrix[x_id, y_id] = coherence(x=x, y=y, fs=500)[1][:20].mean()

    if plot:
        plt.figure(figsize=(10, 10))
        plt.imshow(coherence_matrix, cmap='jet')
        plt.colorbar()
        plt.show()

    corr_linkage = hierarchy.ward(coherence_matrix)
    dendro = hierarchy.dendrogram(corr_linkage, leaf_rotation=90)
    dendro_idx = np.arange(0, len(dendro["ivl"]))
    dendro_idx_pad = [str(arr_to_pad(chanl_list[int(el)])) for el in list(dendro["ivl"])]

    if plot:
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        im = ax.imshow(coherence_matrix[dendro["leaves"], :][:, dendro["leaves"]], cmap="jet")

        ax.set_xticks(dendro_idx)
        ax.set_yticks(dendro_idx)
        ax.set_xticklabels(dendro_idx_pad, rotation="vertical")
        ax.set_yticklabels(dendro_idx_pad)
        fig.tight_layout()
        fig.colorbar(im, orientation="vertical")
        plt.title('Coherence_ch_in_' + area)
        plt.show()

    # clusters based on thresholding
    clusters = fcluster(corr_linkage, cluster_threshold, criterion='distance')
    # sort elements by clusters and put into dictionary
    test1 = [x for _, x in sorted(zip(clusters, dendro_idx))]
    test2 = [x for x, _ in sorted(zip(clusters, dendro_idx))]
    clusts = [str(arr_to_pad(chanl_list[int(el)])) for el in test1]
    clusters_array = {}
    clusters_pad = {}
    for id, cluster in enumerate(test2):
        if cluster not in clusters_array.keys():
            clusters_array[cluster] = []
            clusters_pad[cluster] = []
        else:
            clusters_array[cluster].append(test1[id])
            clusters_pad[cluster].append(clusts[id])

    for cluster in clusters_array.keys():
        means = []
        for channel_1 in clusters_array[cluster]:
            for channel_2 in clusters_array[cluster]:
                if channel_2 == channel_1:
                    continue
                means.append(
                    coherence(x=lfp[channel_1, start * fps:(start + 20) * fps],
                              y=lfp[channel_2, start * fps:(start + 20) * fps], fs=fps)[1][:20].mean())

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


def explore_clusters2(dataset, area, plot=True, n_clusters=4):
    lfp = get_lfp(dataset, brain_area=area)
    power = get_power(dataset, area)
    channel_list = get_ch(dataset, brain_area=area)
    coherence_matrix = np.zeros((len(lfp), len(lfp)))
    start = 30
    srate = 500

    for x_id, x in tqdm(enumerate(lfp[:, start * srate:(start + 20) * srate])):
        for y_id, y in enumerate(lfp[:, start * srate:(start + 20) * srate]):
            coherence_matrix[x_id, y_id] = coherence(x=x, y=y, fs=500)[1][:20].mean()  # ? take mean or theta band?

    if plot:
        plt.figure(figsize=(10, 10))
        plt.imshow(coherence_matrix, cmap='jet')
        plt.colorbar()
        plt.show()

    kmeans = KMeans(n_clusters=n_clusters, n_jobs=-1)
    kmeans.fit(coherence_matrix)
    kmclust = kmeans.labels_
    # corr_linkage = hierarchy.ward(coherence_matrix)
    # dendro = hierarchy.dendrogram(corr_linkage, leaf_rotation=90)
    # dendro_idx = np.arange(0, len(dendro["ivl"]))
    dendro_idx = np.arange(0, len(kmclust))
    numbering = np.where(kmclust == 0)[0]
    for i in range(1, kmclust.size):
        numbering = np.append(numbering, np.where(kmclust == i)[0])

    # dendro_idx_pad = [str(arr_to_pad(chanl_list[int(el)])) for el in list(dendro["ivl"])]
    dendro_idx_pad = [str(arr_to_pad(channel_list[int(el)])) for el in list(kmclust)]

    if plot:
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        # im = ax.imshow(coherence_matrix[dendro["leaves"], :][:, dendro["leaves"]], cmap="jet")
        im = ax.imshow(coherence_matrix[numbering, :][:, numbering], cmap="jet")

        ax.set_xticks(dendro_idx)
        ax.set_yticks(dendro_idx)
        ax.set_xticklabels(dendro_idx_pad, rotation="vertical")
        ax.set_yticklabels(dendro_idx_pad)
        fig.tight_layout()
        fig.colorbar(im, orientation="vertical")
        plt.title('plot1 ' + area)
        plt.show()

    clusters = kmclust
    # sort elements by clusters and put into dictionary
    test1 = [x for _, x in sorted(zip(clusters, dendro_idx))]
    test2 = [x for x, _ in sorted(zip(clusters, dendro_idx))]
    clusts = [str(arr_to_pad(channel_list[int(el)])) for el in test1]
    clusters_array = {}
    clusters_pad = {}
    for id, cluster in enumerate(test2):
        if cluster not in clusters_array.keys():
            clusters_array[cluster] = []
            clusters_pad[cluster] = []
        else:
            clusters_array[cluster].append(test1[id])
            clusters_pad[cluster].append(clusts[id])

    for cluster in clusters_array.keys():
        means = []
        for channel_1 in clusters_array[cluster]:
            for channel_2 in clusters_array[cluster]:
                if channel_2 == channel_1:
                    continue
                means.append(
                    coherence(x=lfp[channel_1, start * srate:(start + 20) * srate],
                              y=lfp[channel_2, start * srate:(start + 20) * srate], fs=srate)[1][:20].mean())

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
        try:
            channel_idx = np.where(cluster_power == np.max(cluster_power))[0][
                0]  ### this line caused error in some dataset
        except ValueError:
            continue
        # representative_channels.append(clusters_array[cluster][channel_idx])
        representative_channels.append(clusters_pad[cluster][channel_idx])

    return representative_channels


def slice_from_arr(arr, events, channels=None, window=1, fps_ephys=500, fps_behavior=50, mean=True):
    window_samples = window * fps_ephys
    f_ratio = fps_ephys / fps_behavior
    ret = []
    if not channels:
        channels = arr.shape[0]
        for channel in range(channels):
            power_per_channel = []
            for idx in events:
                idx_in_ephys = f_ratio * idx
                window_from = np.max([int(idx_in_ephys - window_samples), 0])
                window_to = int(idx_in_ephys + window_samples)
                if window_to > arr.shape[-1] - 1:
                    continue
                # if window_to == 0:
                #     window_to = 1
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

    ret = []
    for channel in channels:
        power_per_channel = []
        for idx in events:
            idx_in_ephys = f_ratio * idx
            window_from = np.max([int(idx_in_ephys - window_samples), 0])
            window_to = int(idx_in_ephys + window_samples)
            if window_to > arr.shape[-1] - 1:
                continue
            # window_to = np.min([int(idx_in_ephys + window_samples), arr.shape[-1] - 1])  # make sure
            # if window_to == 0:
            #     window_to = 1
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


def epoch_power(power, events, channels=None, window=None):
    idx_in_ephys = [i * 10 for i in events]
    ret = []
    print(events[-1], power.shape[-1])
    for channel in channels:
        power_epoch_per_channel = [power[channel, i] for i in idx_in_ephys]
        ret.append(power_epoch_per_channel)
    return ret


def epoch_data(data_df, channels, events, window=3, f_ephys=500):
    ret = []
    idx_ephys = events * 10
    for i in idx_ephys:
        crop_from = i - window * f_ephys
        crop_to = i + window * f_ephys
        if channels:
            epoch = data_df[channels].iloc[crop_from:crop_to].to_numpy(copy=True).T

        elif channels == None:
            epoch = data_df.iloc[crop_from:crop_to].to_numpy(copy=True).T

        ret.append(epoch)
    return np.array(ret)


def plot_epochs(epochs, average=False):
    t = np.arange(-epochs.shape[-1] / 2, epochs.shape[-1] / 2)
    epoch_mean = epochs.mean(axis=0)
    if average == False:
        for i in range(epoch_mean.shape[0]):
            plt.plot(t, epoch_mean[i, :], alpha=0.4)
    if average == True:
        plt.plot(t, epoch_mean.mean(axis=0))
    plt.show()


def multiple_formatter(denominator=2, number=np.pi, latex='\pi'):
    def gcd(a, b):
        while b:
            a, b = b, a%b
        return a
    def _multiple_formatter(x, pos):
        den = denominator
        num = np.int(np.rint(den*x/number))
        com = gcd(num,den)
        (num,den) = (int(num/com),int(den/com))
        if den==1:
            if num==0:
                return r'$0$'
            if num==1:
                return r'$%s$'%latex
            elif num==-1:
                return r'$-%s$'%latex
            else:
                return r'$%s%s$'%(num,latex)
        else:
            if num==1:
                return r'$\frac{%s}{%s}$'%(latex,den)
            elif num==-1:
                return r'$\frac{-%s}{%s}$'%(latex,den)
            else:
                return r'$\frac{%s%s}{%s}$'%(num,latex,den)
    return _multiple_formatter


def plot_phase_coh(data, fname, band='theta', mpfc_index=0, srate=500, tstart=30, twin=600, nbins=60, axs=None, showfig=True):
    '''
    plot representative histogram of theta phase differences in a specific period
    time period defined by tstart (starting time, in seconds) and twin (length of time window, in seconds)
    one mPFC channel against all vHPC channels (30)
    returns an array of all plots' full width half maximum (FWHM) (unit: radians)
    number of bins is selected as 
    '''
    start = int(tstart * srate)
    end = int((tstart + twin) * srate)

    phase_mpfc = column_by_pad(get_phase(data, 'mpfc', 'theta'))
    power_mpfc = column_by_pad(get_power(data, 'mpfc', 'theta'))
    phase_vhipp = column_by_pad(get_phase(data, 'vhipp', 'theta'))
    power_vhipp = column_by_pad(get_power(data, 'vhipp', 'theta'))
    
    mpfc_pads = np.array(phase_mpfc.columns)
    vhipp_pads = np.array(phase_vhipp.columns)

    if axs == None:
        fig, axs = plt.subplots(5, len(phase_vhipp.columns)//5, 
            figsize=(6*6, 12*5), tight_layout=True, sharey=True)

    FWHM = []
    mpfc_padid = mpfc_pads[mpfc_index]

    for i in range(len(vhipp_pads)):
        vhipp_padid = vhipp_pads[i]
        power_vhipp_curr = np.array(power_vhipp[vhipp_padid])
        power_vhipp_mean = np.mean(power_vhipp_curr)
        exceedmean = np.where(power_vhipp_curr > power_vhipp_mean)
        positions = exceedmean[0]
        filtered = (positions[np.logical_and(positions>=start, positions<end)],)
        # print("%d out of %d selected for plotting" % (len(power_vhipp_curr[exceedmean]), len(power_vhipp_curr)))

        plti = i//6
        pltj = i-plti*6

        phase_diff = np.array(phase_mpfc[mpfc_padid]) - np.array(phase_vhipp[vhipp_padid])
        phase_diff_filtered = phase_diff[filtered]
        # print(len(phase_diff_filtered),' ', len(phase_diff))

        add_pos = np.where(np.logical_and(-2*np.pi <= phase_diff, phase_diff < -np.pi))
        phase_diff[add_pos] += 2*np.pi
        sub_pos = np.where(np.logical_and(np.pi <= phase_diff, phase_diff < 2*np.pi))
        phase_diff[sub_pos] -= 2*np.pi

        n,bins,patches = axs[plti, pltj].hist(phase_diff[start:end], bins=nbins, alpha=1)
        axs[plti, pltj].set_title('mPFC_pad'+str(mpfc_padid)+'-vHPC_pad'+str(vhipp_padid), fontsize=16)
        axs[plti, pltj].xaxis.set_major_locator(plt.MultipleLocator(np.pi))
        axs[plti, pltj].xaxis.set_major_formatter(plt.FuncFormatter(multiple_formatter()))
        axs[plti, pltj].grid(True)
        axs[plti, pltj].tick_params(axis='both', which='major', labelsize=10)

        peak_value = np.max(n)
        half_range = np.where(n>peak_value/2)
        FWHM.append(bins[np.max(half_range)]-bins[np.min(half_range)])

    # Create a big subplot
    ax = fig.add_subplot(111, frameon=False)
    # hide tick and tick label of the big axes
    plt.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')

    ax.set_xlabel('Phase lag (rad)', labelpad=18, fontsize=14) # Use argument `labelpad` to move label downwards.
    ax.set_ylabel('Counts', labelpad=18, fontsize=14)
    plt.savefig(fname)
    plt.show()

    return FWHM