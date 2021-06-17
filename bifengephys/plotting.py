import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
import statsmodels.stats.api as sms
import numpy as np
from scipy import signal
import ephys
import os


def plot_psd(data,
             brain_area='mpfc',
             tstart=30,
             srate=500,
             twin=600,
             fmax=40,
             title='PSD',
             show_plot=True):
    srate = srate
    start = tstart * srate
    end = (tstart + twin) * srate
    fmax = fmax
    nperseg = 1024
    noverlap = int(nperseg * 0.8)

    pxx_den = []

    if brain_area == 'mpfc':
        lfp = get_lfp(data)[:len(get_ch(data, 'mpfc')), :]
        for i in range(lfp.shape[0]):
            f = signal.welch(lfp[i, start:end] * 10e-3, srate, nperseg=nperseg, noverlap=noverlap, nfft=None)[0]
            pxx_den.append(
                signal.welch(lfp[i, start:end] * 10e-3, srate, nperseg=nperseg, noverlap=noverlap, nfft=None)[1])

    if brain_area == 'vhipp':
        lfp = get_lfp(data)[len(get_ch(data, 'mpfc')):, :]
        for i in range(lfp.shape[0]):
            f = signal.welch(lfp[i, start:end] * 10e-3, srate, nperseg=nperseg, noverlap=noverlap, nfft=None)[0]
            pxx_den.append(
                signal.welch(lfp[i, start:end] * 10e-3, srate, nperseg=nperseg, noverlap=noverlap, nfft=None)[1])

    plot_freq = [freq for freq in f if freq <= fmax]

    pxx_den_mean = np.array(pxx_den).mean(axis=0)

    ci95 = sms.DescrStatsW(pxx_den).tconfint_mean()

    plot_mean = plt.plot(plot_freq, pxx_den_mean[:len(plot_freq)], 'k', label=brain_area, alpha=1)

    plot_ci = plt.fill_between(plot_freq,
                               ci95[0][:len(plot_freq)],
                               ci95[1][:len(plot_freq)],
                               alpha=0.3)

    if show_plot:
        plt.legend()
        plt.xlim(0, fmax)
        plt.title(title)
        plt.xlabel('frequency [Hz]')
        plt.ylabel('PSD [mV**2/Hz]')
        plt.show()


def plot_psd_multi(data=[],
                   brain_area=[],
                   tstart=30,
                   srate=500,
                   twin=600,
                   fmax=40,
                   title='title'):
    for i in range(len(data)):
        plot_psd(data[i],
                 brain_area[i],
                 tstart=tstart,
                 srate=srate,
                 twin=twin,
                 fmax=fmax,
                 title='PSD',
                 show_plot=False)

    plt.legend()
    plt.xlim(0, fmax)
    plt.title(title)
    plt.xlabel('frequency [Hz]')
    plt.ylabel('PSD [mV**2/Hz]')
    plt.show()


def plot_mean_ci(data, duration=300):
    plt.figure(figsize=(10, 6))
    srate = 500
    t = np.arange(0, data.shape[1] / srate, 1 / srate)
    plt.plot(t[:duration * srate], data.mean(axis=0)[:duration * srate])
    ci95 = sms.DescrStatsW(data).tconfint_mean()
    plt.fill_between(t[:duration * srate], ci95[0][:duration * srate], ci95[1][:duration * srate], alpha=0.3)
    plt.show()


def pair_power_phase(data, animal, session, mpfc_ch, vhipp_ch, tstart=0, twin=20, srate=500, band='theta'):
    start = int(tstart * srate)
    end = int((tstart + twin) * srate)

    power_mpfc = np.array(ephys.column_by_pad(ephys.get_power(data, 'mpfc', band))[mpfc_ch])[start:end]
    power_vhipp = np.array(ephys.column_by_pad(ephys.get_power(data, 'vhipp', band))[vhipp_ch])[start:end]

    phase_mpfc = np.array(ephys.column_by_pad(ephys.get_phase(data, 'mpfc', band))[mpfc_ch])[start:end]
    phase_vhipp = np.array(ephys.column_by_pad(ephys.get_phase(data, 'vhipp', band))[vhipp_ch])[start:end]

    fig, axs = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
    time = np.linspace(tstart, tstart+twin, num=end-start)
    axs[0].set_title('Instantaneous theta amplitude', fontsize=16)
    axs[0].plot(time, power_mpfc, 'b', label='mPFC')
    axs[0].plot(time, power_vhipp, 'r', label='vHPC')
    axs[0].set_ylabel('mV')
    axs[0].tick_params(axis='both', which='major', labelsize=10)
    axs[0].legend(fontsize=10)

    axs[1].set_title('Instantaneous theta phase', fontsize=16)
    axs[1].plot(time, phase_mpfc, 'b', label='mPFC')
    axs[1].plot(time, phase_vhipp, 'r', label='vHPC')
    axs[1].set_ylabel('rad')
    axs[1].set_xlabel('Time from start (s)')
    axs[1].set_yticks([-np.pi, 0, np.pi])
    axs[1].set_yticklabels([r'$-\pi$', r'$0$', r'$\pi$'])
    axs[1].tick_params(axis='both', which='major', labelsize=10)
    axs[1].legend(fontsize=10)

    fig.suptitle('mPFC pad'+str(mpfc_ch)+' - vHPC pad'+str(vhipp_ch)+' example plot', fontsize=18)

    plt.show()


def pair_power_lag(data, animal, session, mpfc_ch, vhipp_ch, tstart=0, twin=20, srate=500, band='theta'):
    start = int(tstart * srate)
    end = int((tstart + twin) * srate)

    power_mpfc = np.array(ephys.column_by_pad(ephys.get_power(data, 'mpfc', band))[mpfc_ch])[start:end]
    power_vhipp = np.array(ephys.column_by_pad(ephys.get_power(data, 'vhipp', band))[vhipp_ch])[start:end]

    lag, lags, corr = ephys.get_lag(power_mpfc, power_vhipp)

    fig, axs = plt.subplots(2, 1, figsize=(10, 10))
    time = np.linspace(tstart, tstart+twin, num=end-start)
    axs[0].set_title('Instantaneous theta amplitude', fontsize=16)
    axs[0].plot(time, power_mpfc, 'b', label='mPFC')
    axs[0].plot(time, power_vhipp, 'r', label='vHPC')
    axs[0].set_ylabel('mV')
    axs[0].set_xlabel('Time from start (s)')
    axs[0].tick_params(axis='both', which='major', labelsize=10)
    axs[0].legend(fontsize=10)

    axs[1].plot(lags, corr)
    axs[1].scatter([lag], corr[np.argmax(corr)], color='red', s=50)
    axs[1].axvline(x=0, color='k', linestyle='--')
    axs[1].set_ylabel('vHPC-mPFC theta power crosscorr')
    axs[1].set_xlim(-100,100)
    axs[1].set_xlabel('Lag (ms)')
    axs[1].tick_params(axis='both', which='major', labelsize=10)

    fig.suptitle('mPFC pad'+str(mpfc_ch)+' - vHPC pad'+str(vhipp_ch)+' example plot', fontsize=18)

    plt.show()


def plot_pair_phase_diff(pair_result, fname, nbins=64):
    plt.figure(figsize=(8, 16))
    bin_edges = np.linspace(-np.pi, np.pi, num=nbins+1)
    n, bins, patches = plt.hist(pair_result['phase_diff_filtered'], bin_edges)
    plt.axvline(x=pair_result['HM_left'], ymin=0, ymax=1000, color='r', linestyle='--')
    plt.axvline(x=pair_result['HM_right'], ymin=0, ymax=1000, color='r', linestyle='--')
    plt.axvline(x=pair_result['peak_position'], ymin=0, ymax=1000, color='k', linestyle='-')
    xticks = [-np.pi, 0, np.pi]
    plt.xticks(xticks, [r'$-\pi$', r'$0$', r'$\pi$'], fontsize=10)
    plt.xlabel('Phase lag (rad)', fontsize=12)
    plt.yticks(fontsize=10)
    plt.ylabel('Counts', fontsize=12)
    mpfc_ch = pair_result['mpfc_channel']
    vhipp_ch = pair_result['vhipp_channel']
    plt.title('mPFC_pad'+str(mpfc_ch)+'-vHPC_pad'+str(vhipp_ch), fontsize=16)
    plt.savefig(fname)
    plt.show()


def plot_phase_diffs(results, savedir):
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    peak_positions = []
    for pairid in results:
        pair_result = results[pairid]
        mpfc_ch = pair_result['mpfc_channel']
        vhipp_ch = pair_result['vhipp_channel']
        plot_pair_phase_diff(pair_result, fname=savedir+'mPFC_pad'+str(mpfc_ch)+'_vHPC_pad'+str(vhipp_ch)+'_phasediff.jpg')
        peak_positions.append(pair_result['peak_position'])

    peak_positions_all = np.array(peak_positions).flatten()
    plt.figure(figsize=(10,10))
    fig, ax = plt.subplots()
    bin_edges = np.linspace(-np.pi, np.pi, num=64)
    n, bins, patches = ax.hist(peak_positions_all, bin_edges, histtype='stepfilled')
    xticks = np.arange(-np.pi/4, np.pi/4+np.pi/16, np.pi/16)
    ax.set_xticks(xticks)
    ax.set_xticklabels([r'$-\pi/4$', r'$-3\pi/16$', r'$-\pi/8$', r'$-\pi/16$', r'$0$', r'$\pi/16$', r'$\pi/8$', r'$3\pi/16$', r'$\pi/4$'])
    ax.axvline(x=0, ymin=0, ymax=100, color='k', linestyle='--')
    ax.tick_params(axis='both', which='major', labelsize=10)
    ax.set_xlim(-np.pi/3, np.pi/3)
    ax.set_xlabel('Phase difference (rad)', labelpad=18, fontsize=14)
    ax.set_ylabel('Counts of channel pairs', labelpad=18, fontsize=14)
    ax.set_title('Peak phase lags across all vHPC-mPFC pairs', fontsize=18)
    plt.savefig(savedir+'phase_diff_all.jpg', bbox_inches = 'tight')
    plt.show()


def plot_FWHM(FWHM_result, savedir, plottype):
    FWHMs = FWHM_result['FWHMs']
    mpfc_ch_labels = ['pad'+str(i) for i in FWHM_result['mpfc_channels']]
    vhipp_ch_labels = ['pad'+str(i) for i in FWHM_result['vhipp_channels']]

    fig, ax = plt.subplots(figsize=(10,10))

    if plottype == 'diff':
        # define the colormap
        cmap = plt.get_cmap('PuOr')
        # extract all colors from the .jet map
        cmaplist = [cmap(i) for i in range(cmap.N)]
        # create the new map
        cmap = cmap.from_list('Custom cmap', cmaplist, cmap.N)
        # define the bins and normalize and forcing 0 to be part of the colorbar!
        bounds = np.arange(-1, 1,.1)
        idx=np.searchsorted(bounds,0)
        bounds=np.insert(bounds,idx,0)
        norm = BoundaryNorm(bounds, cmap.N)
        im = ax.imshow(FWHMs, interpolation='none',norm=norm,cmap=cmap)
    else:
        im = ax.imshow(FWHMs, vmin=0.0, vmax=np.pi)

    ax.set_xticks(np.arange(len(vhipp_ch_labels)))
    ax.set_yticks(np.arange(len(mpfc_ch_labels)))

    ax.set_xticklabels(vhipp_ch_labels)
    ax.set_yticklabels(mpfc_ch_labels)

    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(10)
        tick.label.set_rotation('vertical')

    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(10)

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    fig.tight_layout()
    fig.colorbar(im, shrink=0.5)
    plt.xlabel('vHPC')
    plt.ylabel('mPFC')
    plt.savefig(savedir + plottype + '_fwhms.jpg', bbox_inches = 'tight')
    plt.show()


def plot_lagstats(lagstat, savedir, plottype):
    plt.figure(figsize=(10,16))
    fig, ax = plt.subplots()
    bin_edges = np.linspace(-50, 50, num=50)
    n, bins, patches = ax.hist(lagstat, bin_edges, histtype='stepfilled')

    ax.tick_params(axis='both', which='major', labelsize=8)
    ax.set_xlabel('Time lag (ms)', labelpad=18, fontsize=12)
    ax.set_ylabel('Counts of mPFC-vHPC channel pairs', labelpad=18, fontsize=12)
    ax.set_title(plottype + ' vHPC-mPFC time lag occurrance distribution', fontsize=14)
    ax.set_ylim(top=800)
    plt.savefig(savedir + plottype + '_lags_allpairs.jpg', bbox_inches = 'tight')


def plot_seg_lags(results, savedir, seglen=0.5, srate=500):
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    
    allpeak = []
    allmedian = []
    allmean = []

    for pairid in results:
        if pairid=='segment_starts':
            continue
        pair_result = results[pairid]
        mpfc_ch = pair_result['mpfc_channel']
        vhipp_ch = pair_result['vhipp_channel']
        pair_lags = pair_result['allseg_lags']
        plt.figure(figsize=(10,16))
        fig, ax = plt.subplots()
        bin_edges = np.linspace(-int(seglen*srate*2), int(seglen*srate*2), num=int(seglen*srate*2))
        pair_lags_all = np.array(pair_lags).flatten()
        n, bins, patches = ax.hist(pair_lags_all, bin_edges, histtype='stepfilled')
        ax.tick_params(axis='both', which='major', labelsize=8)
        ax.set_xlabel('Time lag (ms)', labelpad=18, fontsize=12)
        ax.set_ylabel('Counts of time segments', labelpad=18, fontsize=12)
        ax.set_title('Time lags of channel pair ' + str(mpfc_ch) + '-' + str(vhipp_ch) + ' across time intervals', fontsize=14)
        plt.savefig(savedir + 'mPFC_pad' + str(mpfc_ch) + '_vHPC_pad' + str(vhipp_ch)+ '_allseglags.jpg', bbox_inches = 'tight')
        plt.show()

        allpeak.append(pair_result['peak_lag'])
        allmedian.append(pair_result['median_lag'])
        allmean.append(pair_result['mean_lag'])
    
    plot_lagstats(allpeak, savedir, plottype='peak')
    plot_lagstats(allmedian, savedir, plottype='median')
    plot_lagstats(allmean, savedir, plottype='mean')
