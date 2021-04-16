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
    end = (start + twin) * srate
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
