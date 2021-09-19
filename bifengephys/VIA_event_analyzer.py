import pandas as pd
import numpy as np
import os
import glob
from matplotlib import pyplot as plt
import scipy.stats as stats

path = 'F:\SAPAP3_Bifeng'
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
fnames = glob.glob('*/*EZM_behav.csv')
for fname in fnames:
    print(fname, '\n')

#%%
for fname in fnames:
    ret = {}
    events = pd.read_csv(fname, header=1)
    close = events[events['metadata'] == '{"EZM":"Close"}'][['temporal_segment_start', 'temporal_segment_end']]
    open = events[events['metadata'] == '{"EZM":"Open"}'][['temporal_segment_start', 'temporal_segment_end']]
    LED_off = events[events['metadata'] == '{"EZM":"LED_off"}']['temporal_segment_start'].to_numpy()

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
        if open_end[i] <= five_min and open_start[i] <= five_min:
            dtime = open_end[i] - open_start[i]
            open_time_5min += dtime
        elif open_end[i] > five_min and open_start[i] <= five_min:
            dtime = five_min - open_start[i]
            open_time_5min += dtime
        else:
            break

    open_time_10min = 0
    for i in range(len(open_end)):
        if open_end[i] <= ten_min and open_start[i] <= ten_min:
            dtime = open_end[i] - open_start[i]
            open_time_10min += dtime
        elif open_end[i] > ten_min and open_start[i] <= ten_min:
            dtime = ten_min - open_start[i]
            open_time_10min += dtime
        else:
            break

    close_time_5min = 0

    for i in range(len(close_end)):
        if close_end[i] <= five_min and close_start[i] <= five_min:
            dtime = close_end[i] - close_start[i]
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
        if close_end[i] <= ten_min and close_start[i] <= ten_min:
            dtime = close_end[i] - close_start[i]
            close_time_10min += dtime
            trans_10min.append(i)

        elif close_end[i] > ten_min and close_start[i] <= ten_min:
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
          'transition in 10 min ', trans_10min[-1], )

    ret['open_time_5min'] = open_time_5min
    ret['close_time_5min'] = close_time_5min
    ret['transition_5min'] = trans_5min[-1]

    ret['open_time_10min'] = open_time_10min
    ret['close_time_10min'] = close_time_10min
    ret['transition_10min'] = trans_10min[-1]

    result = pd.DataFrame(ret, index=[0])
    result.to_csv(fname.split('.')[0] + '_result.csv')

# %%

data = pd.read_csv('EZM_behavior_result.csv', header=1)
data_1 = pd.read_csv('EZM_ephys_result.csv', header=0)
ko_test1 = data['open_time_10min'].iloc[:6]
ko_test2 = data['open_time_10min'].iloc[7:13]
wt_test1 = data['open_time_10min'].iloc[14:22]

# %%
import seaborn as sns

ax = sns.barplot(x='group', y='open_time_10min', data=data, ci='sd')
plt.show()

# %%
ax = sns.barplot(x='Injection', y='time_in_open_10min', data=data_1, ci='sd')
plt.show()

# %%
drug = data_1[data_1['Injection'] == 'Drug']['time_in_open_10min']
vehicle = data_1[data_1['Injection'] == 'Vehicle']['time_in_open_10min']
stats.wilcoxon(drug, vehicle)

# %%
stats.describe(ko_test1)
stats.describe(ko_test2)
stats.describe(wt_test1)
stats.wilcoxon(ko_test1, ko_test2)
stats.mannwhitneyu(ko_test2, wt_test1)


