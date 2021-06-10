import matplotlib.pyplot as plt
import statsmodels.stats.api as sms
import numpy as np
from scipy import signal
import ephys


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
    

def plot_phase_coh_pairs(data, animal, session, savedir, band='theta', exclude=[], srate=500, beh_srate=50, tstart=0, twin=600, nbins=64, axs=None, showfig=True, select_idx=None):
    phase_mpfc = ephys.column_by_pad(ephys.get_phase(data, 'mpfc', band))
    phase_vhipp = ephys.column_by_pad(ephys.get_phase(data, 'vhipp', band))
    mpfc_pads = np.array(phase_mpfc.columns)
    vhipp_pads = np.array(phase_vhipp.columns)
    
    start = int(tstart * srate)
    end = int((tstart + twin) * srate)
    points = []
    for i in range(start, end):
        beh_time_to_start = int(i / srate * beh_srate)
        if select_idx is not None:
            if beh_time_to_start in select_idx:
                points.append(i)
        else:
            points.append(i)
    print(len(points))
    print(end-start)

    FWHMs = []
    peak_positions = []
    for i in range(len(mpfc_pads)):
        if not mpfc_pads[i] in exclude:
            FWHM, phase_diff_peakpos = ephys.plot_phase_coh(data, 
                                        fname=savedir+animal[session]+'_mPFC_pad'+str(mpfc_pads[i])+'_phasecoh.jpg', pointselect=points,
                                        band='theta', exclude=exclude, mpfc_index=i, nbins=nbins)
            FWHMs.append(FWHM)
            peak_positions.append(phase_diff_peakpos)

    FWHMs_np = np.array(FWHMs)
    mpfc_ch_labels = ['pad'+str(i) for i in mpfc_pads if i not in exclude]
    vhipp_ch_labels = ['pad'+str(i) for i in vhipp_pads if i not in exclude]

    fig, ax = plt.subplots(figsize=(10,10))
    im = ax.imshow(FWHMs_np, vmin=0.0, vmax=np.pi)

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
    plt.title('Heatmap of FWHM values of all mPFC and vHPC channel pairs', fontsize=16)
    plt.xlabel('vHPC')
    plt.ylabel('mPFC')
    plt.savefig(savedir+animal[session]+'_phasecoh_fwhms.jpg', bbox_inches = 'tight')
    plt.show()

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
    plt.savefig(savedir+animal[session]+'_phase_diff_all.jpg', bbox_inches = 'tight')
    plt.show()

    return mpfc_ch_labels, vhipp_ch_labels, FWHMs_np


def plot_seg_lags_pairs(data, animal, session, savedir, band='theta', exclude=[], srate=500, beh_srate=50, tstart=0, twin=600, select_idx=None):
    segs = []
    maxtime = int((tstart + twin) * beh_srate)

    pos = 0
    segstart = pos

    while pos < maxtime:
        while (not pos in select_idx) and (pos < maxtime):
            pos += 1
        start = pos
        count = 0
        if pos in select_idx:
            while count < int(beh_srate * 2.5) and pos in select_idx:
                count += 1
                pos += 1
        if count >= int(beh_srate * 2.5):
            segs.append(start)
        else:
            pos += 1
    print('Number of time segments extracted: ', len(segs))
    
    power_mpfc = ephys.column_by_pad(ephys.get_power(data, 'mpfc', 'theta'))
    power_vhipp = ephys.column_by_pad(ephys.get_power(data, 'vhipp', 'theta'))

    mpfc_pads = np.array(power_mpfc.columns)
    vhipp_pads = np.array(power_vhipp.columns)
    
    timelag_peakpos = []
    timelag_medianpos = []
    
    for mpfc_chid in range(len(mpfc_pads)):
        for vhipp_chid in range(len(vhipp_pads)):
            if (not vhipp_pads[vhipp_chid] in exclude) and (not mpfc_pads[mpfc_chid] in exclude):
                peakpos, medianpos = ephys.plot_seg_lags(animal, session, power_mpfc, power_vhipp, mpfc_pads, vhipp_pads, mpfc_chid, vhipp_chid, segs, savedir, beh_srate=50, srate=500)
                timelag_peakpos.append(peakpos)
                timelag_medianpos.append(medianpos)
    
    peaks_all = np.array(timelag_peakpos).flatten()
    plt.figure(figsize=(10,16))
    fig, ax = plt.subplots()
    bin_edges = np.linspace(-50, 50, num=50)
    n, bins, patches = ax.hist(peaks_all, bin_edges, histtype='stepfilled')

    ax.tick_params(axis='both', which='major', labelsize=8)
    ax.set_xlabel('Time lag (ms)', labelpad=18, fontsize=12)
    ax.set_ylabel('Counts of mPFC-vHPC channel pairs', labelpad=18, fontsize=12)
    ax.set_title('Peak vHPC-mPFC time lag occurrance distribution', fontsize=14)
    ax.set_ylim(top=800)
    plt.savefig(savedir+animal[session]+'_peak_lags_allpairs.jpg', bbox_inches = 'tight')
    
    medians_all = np.array(timelag_medianpos).flatten()
    plt.figure(figsize=(10,16))
    fig, ax = plt.subplots()
    bin_edges = np.linspace(-50, 50, num=50)
    n, bins, patches = ax.hist(medians_all, bin_edges, histtype='stepfilled')

    ax.tick_params(axis='both', which='major', labelsize=8)
    ax.set_xlabel('Time lag (ms)', labelpad=18, fontsize=12)
    ax.set_ylabel('Counts of mPFC-vHPC channel pairs', labelpad=18, fontsize=12)
    ax.set_title('Median vHPC-mPFC time lags in all time segments', fontsize=14)
    ax.set_ylim(top=800)
    plt.savefig(savedir+animal[session]+'_median_lags_allpairs.jpg', bbox_inches = 'tight')


def plot_crosscorr_pairs(data, animal, session, savedir, band='theta', exclude=[], srate=500, beh_srate=50, tstart=30, twin=600, axs=None, showfig=True, select_idx=None):
    power_mpfc = ephys.column_by_pad(ephys.get_power(data, 'mpfc', band))
    power_vhipp = ephys.column_by_pad(ephys.get_power(data, 'vhipp', band))
    mpfc_pads = np.array(power_mpfc.columns)
    vhipp_pads = np.array(power_vhipp.columns)
    
    start = int(tstart * srate)
    end = int((tstart + twin) * srate)
    points = []
    for i in range(start, end):
        beh_time_to_start = int(i / srate * beh_srate)
        if select_idx is not None:
            if beh_time_to_start in select_idx:
                points.append(i)
        else:
            points.append(i)
    print(len(points))
    print(end-start)

    mpfc_lags = []
    for i in range(len(mpfc_pads)):
        if not mpfc_pads[i] in exclude:
            mpfc_lags_curr = ephys.plot_crosscorr(data, 
                                        fname=savedir+animal[session]+'_mPFC_pad'+str(mpfc_pads[i])+'_power_crosscorr.jpg', pointselect=points,
                                        band=band, exclude=exclude, mpfc_index=i, srate=srate, 
                                        tstart=tstart, twin=twin)
            mpfc_lags.append(mpfc_lags_curr)
            plt.figure(figsize=(6,12))
            bin_edges = np.linspace(-50, 50, num=50)
            n, bins, patches = plt.hist(mpfc_lags_curr, bin_edges, histtype='stepfilled')
            plt.xlim(-50, 50)
            plt.xticks(fontsize=14)
            plt.yticks(fontsize=14)
            plt.vlines(0,0,8, colors='r', linestyles='dashed')
            plt.xlabel('Lag (ms)', fontsize=18)
            plt.ylabel('counts', fontsize=18)
            plt.title('vHPC channels-mPFC_pad'+str(mpfc_pads[i])+' lag distribution', fontsize=20)
            plt.savefig(savedir+animal[session]+'_mPFC_pad'+str(mpfc_pads[i])+'_lag_distrib.jpg')
            plt.show()
    
    mpfc_lags_all = np.array(mpfc_lags).flatten()
    plt.figure(figsize=(6,12))
    bin_edges = np.linspace(-50, 50, num=50)
    n, bins, patches = plt.hist(mpfc_lags_all, bin_edges, histtype='stepfilled')
    plt.xlim(-50, 50)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.vlines(0,0,50, colors='r', linestyles='dashed')
    plt.xlabel('Lag (ms)', fontsize=18)
    plt.ylabel('counts of mPFC-vHPC pairs', fontsize=18)
    plt.title('Time lags all vHPC-mPFC channel pairs', fontsize=20)
    plt.savefig(savedir+animal[session]+'_all_lag_distrib.jpg', bbox_inches = 'tight')
    plt.show()
