import pickle
import numpy as np


def load_data(session):
    print(session)
    with open(session + '/ephys_processed/' + session + '_dataset.pkl', "rb") as f:
        data = pickle.load(f)
        f.close()
    return data


def load_traj(session):
    print(session)
    return pickle.load(open(session + '/ephys_processed/' + session + '_movement.pkl', "rb"))


def load_rois(session):
    print(session)
    return pickle.load(open(session + '/ephys_processed/' + session + '_rois.pkl', "rb"))


def load_event(session):
    print(session)
    return pickle.load(open(session + '/ephys_processed/' + session + '_transition.pkl', "rb"))


def get_ch(data, brain_area):
    return data['info']['ch_names']['ch_' + brain_area]


def get_lfp(dataset, brain_area):
    print(dataset['lfp']['amplifier_data'].shape)
    if brain_area == 'mpfc':
        lfp = dataset['lfp']['amplifier_data'][:len(get_ch(dataset, 'mpfc')), :]
    if brain_area == 'vhipp':
        lfp = dataset['lfp']['amplifier_data'][len(get_ch(dataset, 'mpfc')):, :]
    print(lfp.shape)
    return lfp


def get_power(dataset, brain_area, band='theta'):
    print(dataset['bands'][band]['power'].shape)
    if brain_area == 'mpfc':
        power = dataset['bands'][band]['power'][:len(get_ch(dataset, 'mpfc')), :]
    if brain_area == 'vhipp':
        power = dataset['bands'][band]['power'][len(get_ch(dataset, 'mpfc')):, :]
    print(power.shape)
    return power


def get_phase(dataset, brain_area, band='theta'):
    print(dataset['bands'][band]['phase'].shape)
    if brain_area == 'mpfc':
        phase = dataset['bands'][band]['phase'][:len(get_ch(dataset, 'mpfc')), :]
    if brain_area == 'vhipp':
        phase = dataset['bands'][band]['phase'][len(get_ch(dataset, 'mpfc')):, :]
    print(phase.shape)
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
    array_to_pad = {} ### how ephys data array register to the electrode pad. pad1-32 are in the mPFC, with pad1 is the deepest.

    cmap = sixtyfour_ch_solder_pad_to_zif(intan_array_to_zif_connector)
    for i in range(1,65):
        chan_name = np.where(cmap==i)[0][0]
        if chan_name < 10:
            array_to_pad['A-00'+ str(chan_name)]=str(i)
        else:
            array_to_pad['A-0'+ str(chan_name)]=str(i)
    return array_to_pad[el]


def pad_to_array():
    pad_to_array = []
    pad_to_array_text = []
    cmap = sixtyfour_ch_solder_pad_to_zif(intan_array_to_zif_connector)
    for i in range(1, 65):
        pad_to_array.append(np.where(cmap == i)[0][0])
        pad_to_array_text.append('pad' + str(i) + '== A-0' + str(np.where(cmap == i)[0][0]))
    return pad_to_array, pad_to_array_text


