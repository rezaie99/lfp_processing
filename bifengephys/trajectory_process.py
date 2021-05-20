import glob
import numpy as np
import pandas as pd
from collections import namedtuple
import pickle

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math
from scipy import interpolate
from shapely.geometry import Point, Polygon


def load_traj(session):
    file_dir = 'D:\\ephys\\' + session + '\\ephys_processed\\'
    try:
        fname = glob.glob(file_dir + '*.h5')[0]
        print(file_dir)
    except IndexError:
        print('Key coordinates for EPM dictionary file not found!')
        exit()
    traj_df = pd.read_hdf(fname)
    scorer = traj_df.columns[0][0]
    return traj_df, scorer


def load_points(session, behavior):
    file_dir = 'D:\\ephys\\' + session + '\\ephys_processed\\'
    if behavior == 'epm':
        fname = glob.glob(file_dir + 'EPM_points.pkl')[0]
    #TODO: OFT
    print(fname + ' loaded')
    points_file = open(fname, 'rb')
    coors = pickle.load(points_file)
    return coors


def calib_traj(traj_df, start_time, duration, fps=50, bp='head', XYMAX=400, THRL=0.95, THRD=5):
    scorer = traj_df.columns[0][0]
    bd_x = traj_df[scorer][bp]['x']
    bd_y = traj_df[scorer][bp]['y']
    bd_confidence = traj_df[scorer][bp]['likelihood']

    bd_x_np = np.array(bd_x)
    bd_y_np = np.array(bd_y)
    bd_x_new = bd_x.copy(deep=True)
    bd_y_new = bd_y.copy(deep=True)

    length = len(bd_x)
    bd_x_diff = np.concatenate(([0], bd_x_np[1:] - bd_x_np[:length - 1]))
    bd_y_diff = np.concatenate(([0], bd_y_np[1:] - bd_y_np[:length - 1]))
    step_dis = np.sqrt(bd_x_diff ** 2 + bd_y_diff ** 2)

    print('total number of frames:', length)
    # start_time unit: seconds
    # index of the first frame for evaluation
    start_frame = int(start_time * fps)
    # index of the last frame for evaluation
    # duration is time in minutes
    end_frame = min(int(start_frame + duration * 60 * fps), length)
    print('frames to process:', end_frame - start_frame)

    # process data point: (A) if the distance of current point to the previous one exceeds a certain threshold;
    # (B) if the likelihood value is below a certain threshold
    # or (C) the point is currently out of frame
    # the video frame is 330 by 330 pixels in size in OFT sessions, but approx. 400 by 400 in EZM sessions
    XMIN = 0
    XMAX = XYMAX
    YMIN = 0
    YMAX = XYMAX

    num_processed = 0
    # for i in range(start_frame, end_frame):
    for i in range(length):
        process = False
        if step_dis[i] > THRD:
            process = True
        if bd_confidence[i] < THRL:
            process = True
        if bd_x_new[i] < XMIN or bd_x_new[i] > XMAX:
            process = True
        if bd_y_new[i] < YMIN or bd_y_new[i] > YMAX:
            process = True
        if process:
            num_processed += 1
            bd_x_new.loc[i] = np.nan
            bd_y_new.loc[i] = np.nan
    # set limit_direction so that consecutive NaNs are filled with interpolation
    bd_x_new = bd_x_new.interpolate(method='linear', limit_direction='both')
    bd_y_new = bd_y_new.interpolate(method='linear', limit_direction='both')
    print("edited " + str(num_processed) + " data points")

    plt.figure(figsize=(5, 5))
    plt.plot(bd_x[start_frame:end_frame], bd_y[start_frame:end_frame], label='unprocessed trajectory')
    plt.plot(bd_x_new[start_frame:end_frame], bd_y_new[start_frame:end_frame], alpha=0.4, label='processed trajectory')
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize=12)
    plt.show()

    # two new pandas Series are returned, same in length as the original ones
    # ONLY the frames defined by Start_time and Duration are processed !!
    return bd_x_new, bd_y_new


def calculate_speed_acc(traj_x, traj_y, start_time, duration, fps, move_cutoff=5, period=5):
    # "period" is the length (in frames) of sliding window within which average speed is calculated
    # "move cutoff" is the threshold (in pixels per second, same as speed) below which a frame is labeled as "not moving"

    traj_x_np = np.array(traj_x)
    traj_y_np = np.array(traj_y)
    length = len(traj_x)
    # start_time unit: seconds
    # index of the first frame for evaluation
    start_frame = int(start_time * fps)
    # index of the last frame for evaluation
    # duration is time in minutes
    end_frame = min(int(start_frame + duration * 60 * fps), length)

    accumulated_distance = []
    inst_distance = []
    distsum = 0.0
    for i in range(length):
        if start_frame <= i <= end_frame:
            step = 0.0
            if i >= 1:
                step = np.sqrt((traj_x[i] - traj_x[i - 1]) ** 2 + (traj_y[i] - traj_y[i - 1]) ** 2)
                if not np.isnan(step):
                    distsum += step
            accumulated_distance.append(distsum)
            inst_distance.append(step)
        else:
            accumulated_distance.append(0)
            inst_distance.append(0)

    accumulated_distance = np.array(accumulated_distance)
    inst_distance = np.array(inst_distance)

    speed = inst_distance * fps  # unit of speed: pixels per second
    avgspd = np.convolve(speed, np.ones(period) / float(period), mode='valid')
    ismoving = avgspd > move_cutoff
    # acc unit: pixels per timestep squared
    acc = np.concatenate(([0], speed[1:] - speed[:length - 1])) * fps

    return accumulated_distance, speed, acc, ismoving


def create_rois_oft():
    # two points defining each roi: topleft(X,Y) and bottomright(X,Y).
    position = namedtuple('position', ['topleft', 'bottomright'])
    rois = {'center': position((87.5, 87.5), (252.5, 252.5)), 'cornertl': position((5, 252.5), (87.5, 335)),
            'cornertr': position((252.5, 252.5), (335, 335)),
            'cornerbl': position((5, 5), (87.5, 87.5)), 'cornerbr': position((252.5, 5), (335, 87.5)),
            'peritop': position((87.5, 252.5), (252.5, 335)),
            'perileft': position((5, 87.5), (87.5, 252.5)), 'peribottom': position((87.5, 5), (252.5, 87.5)),
            'periright': position((252.5, 87.5), (335, 252.5))}
    return rois


def create_rois_epm(epm_coors):
    rois = {
    'closedtop': Polygon([epm_coors['tr'], epm_coors['tl'], epm_coors['ctcl'], epm_coors['ctcr']]),
    'center': Polygon([epm_coors['ctcr'], epm_coors['ctcl'], epm_coors['cbcl'], epm_coors['cbcr']]),
    'closedbottom': Polygon([epm_coors['cbcr'], epm_coors['cbcl'], epm_coors['bl'], epm_coors['br']]),
    'openleft': Polygon([epm_coors['ctcl'], epm_coors['ctl'], epm_coors['cbl'], epm_coors['cbcl']]),
    'openright': Polygon([epm_coors['ctr'], epm_coors['ctcr'], epm_coors['cbcr'], epm_coors['cbr']]),
    }
    return rois


def plot_traj_roi(traj_x, traj_y, rois):
    fig, ax = plt.subplots(1, figsize=(8, 8))

    # plot trajectory + bounding boxes for rois
    plt.plot(traj_x, traj_y, '-', linewidth=.2, alpha=.4)

    rect = patches.Rectangle(rois['center'].topleft, rois['center'].bottomright[0] - rois['center'].topleft[0],
                             rois['center'].bottomright[1] - rois['center'].topleft[1], linewidth=2, edgecolor='purple',
                             facecolor='none')
    ax.add_patch(rect)
    rect = patches.Rectangle(rois['cornertl'].topleft, rois['cornertl'].bottomright[0] - rois['cornertl'].topleft[0],
                             rois['cornertl'].bottomright[1] - rois['cornertl'].topleft[1], linewidth=2,
                             edgecolor='orange', facecolor='none')
    ax.add_patch(rect)
    rect = patches.Rectangle(rois['cornertr'].topleft, rois['cornertr'].bottomright[0] - rois['cornertr'].topleft[0],
                             rois['cornertr'].bottomright[1] - rois['cornertr'].topleft[1], linewidth=2,
                             edgecolor='orange', facecolor='none')
    ax.add_patch(rect)
    rect = patches.Rectangle(rois['cornerbl'].topleft, rois['cornerbl'].bottomright[0] - rois['cornerbl'].topleft[0],
                             rois['cornerbl'].bottomright[1] - rois['cornerbl'].topleft[1], linewidth=2,
                             edgecolor='orange', facecolor='none')
    ax.add_patch(rect)
    rect = patches.Rectangle(rois['cornerbr'].topleft, rois['cornerbr'].bottomright[0] - rois['cornerbr'].topleft[0],
                             rois['cornerbr'].bottomright[1] - rois['cornerbr'].topleft[1], linewidth=2,
                             edgecolor='orange', facecolor='none')
    ax.add_patch(rect)
    plt.ylim(0, 350)
    plt.show()


def sort_roi_points(roi):
    return np.sort([roi.topleft[0], roi.bottomright[0]]), np.sort([roi.topleft[1], roi.bottomright[1]])


def get_indexes(lst, match):
    return np.asarray([i for i, x in enumerate(lst) if x == match])


def get_roi_center(rois):
    centers_roi = []
    for points in rois.values():
        center_x = (points.topleft[0] + points.bottomright[0]) / 2
        center_y = (points.topleft[1] + points.bottomright[1]) / 2
        center = np.asarray([center_x, center_y])
        centers_roi.append(center)
    return centers_roi


def assign_rois(traj_x, traj_y, rois):
    # calculate distance to each roi for each frame's point on trajectory
    data_length = len(traj_x)
    centers_roi = get_roi_center(rois)
    distances = np.zeros((data_length, len(centers_roi)))
    for idx, center in enumerate(centers_roi):
        cnt = np.tile(center, data_length).reshape((data_length, 2))
        dist = np.hypot(np.subtract(
            cnt[:, 0], traj_x), np.subtract(cnt[:, 1], traj_y))
        distances[:, idx] = dist

    # get which roi is closest at each frame
    # this is based on the distances between each ROI's center and the point in the frame
    # if a point is not bounded by any ROI, it will still be labeled as belonging to the nearest ROI
    roi_names = list(rois.keys())
    sel_rois = np.argmin(distances, 1)
    roi_nearest = tuple([roi_names[x] for x in sel_rois])
    roi_at_each_frame = []
    for i in range(data_length):
        x, y = traj_x[i], traj_y[i]
        bounded = False
        for j, curr_roi in enumerate(rois):
            X, Y = sort_roi_points(rois[curr_roi])
            if X[0] <= x <= X[1] and Y[0] <= y <= Y[1]:
                bounded = True
                roi_at_each_frame.append(curr_roi)
                break
        if not bounded:
            roi_at_each_frame.append(roi_nearest[i])

    data_time_inrois = {name: roi_at_each_frame.count(name) for name in set(
        roi_at_each_frame)}  # total time (frames) in each roi
    return roi_at_each_frame, data_time_inrois


def assign_rois_epm(traj_x, traj_y, rois, start_frame, end_frame):
    data_length = len(traj_x)

    roi_at_each_frame = []
    in_maze = []
    for i in range(data_length):
        if start_frame <= i <= end_frame:
            x, y = traj_x[i], traj_y[i]
            curr_point = Point(x, y)
            bounded = False
            for j, curr_roi in enumerate(rois):
                if curr_point.within(rois[curr_roi]):
                    bounded = True
                    roi_at_each_frame.append(curr_roi)
            if not bounded:
                in_maze.append('out')
                if x < 225:
                    roi_at_each_frame.append('openleft')
                else:
                    roi_at_each_frame.append('openright')
            else:
                in_maze.append('in')
        else:
            roi_at_each_frame.append('nan')
            in_maze.append('nan')

    data_time_inrois = {name: roi_at_each_frame.count(name) for name in set(
        roi_at_each_frame)}  # total time (frames) in each roi

    return roi_at_each_frame, data_time_inrois, in_maze


def get_transitions(roi_at_each_frame):
    transitions = []
    prev = []
    frame_trans = []

    for i, n in enumerate(list(roi_at_each_frame)):
        if i == 0 or n != list(roi_at_each_frame)[i - 1]:
            transitions.append(n)
            if i == 0:
                prev.append('nan')
            else:
                prev.append(list(roi_at_each_frame)[i - 1])
            frame_trans.append(i)
    return prev, transitions, frame_trans


def get_corner_to_center_events(prev, dest, frame_trans):
    # detect corner-to-center entry events
    corner_to_center_entrytime = []
    corner_to_center_starttime = []

    source = 1
    end = source
    entry_count = 0
    while source < len(prev) and end < len(dest):
        end = source
        if not source < len(prev):
            break
        while (source < len(prev)) and (not prev[source].startswith('corner')):
            source += 1
            end = source
        if source < len(frame_trans):
            start_frame = frame_trans[source]
            entry_frame = frame_trans[end]
            while (end < len(dest)) and (not dest[end].startswith('center')):
                end += 1
                if end < len(dest):
                    entry_frame = frame_trans[end]
            if (end < len(dest)) and (dest[end].startswith('center')) and (entry_frame - start_frame < 100): # 2 second
                entry_frame = frame_trans[end]
                entry_count += 1
                corner_to_center_entrytime.append(entry_frame)
                corner_to_center_starttime.append(start_frame)
        source = end + 1

    return corner_to_center_entrytime, corner_to_center_starttime


def get_corner_to_corner_events(prev, dest, frame_trans):
    corner_to_corner_entrytime = []
    corner_to_corner_starttime = []

    source = 1
    end = source
    entry_count = 0
    while source < len(prev) and end < len(dest):
        if not source < len(prev):
            break
        end = source
        while not prev[source].startswith('corner'):
            source += 1
            end = source
        if source < len(frame_trans):
            start_frame = frame_trans[source]
            entry_frame = frame_trans[end]
            while (end < len(dest)) and (not dest[end].startswith('corner')):
                end += 1
                if end < len(dest):
                    entry_frame = frame_trans[end]
            if (end < len(dest)) and (dest[end].startswith('corner')) and (entry_frame - start_frame < 100) and (not prev[source] == dest[end]):
                entry_frame = frame_trans[end]
                entry_count += 1
                corner_to_corner_entrytime.append(entry_frame)
                corner_to_corner_starttime.append(start_frame)
        source = end+1
    return corner_to_corner_entrytime, corner_to_corner_starttime


def get_center_exit_events(prev, dest, frame_trans):
    center_exit_time = []

    source = 1
    end = source
    exit_count = 0

    while source < len(prev) and end < len(dest):
        if not source < len(prev):
            break
        end = source
        while not prev[source].startswith('center'):
            source += 1
            end = source
        if source < len(frame_trans):
            start_frame = frame_trans[source]
            entry_frame = frame_trans[end]
            while (end < len(dest)) and (not dest[end].startswith('center')):
                end += 1
                if end < len(dest):
                    entry_frame = frame_trans[end]
            if (end == len(dest) or not dest[end].startswith('center')) or (entry_frame - start_frame > 100):
                if end < len(dest):
                    entry_frame = frame_trans[end]
                    exit_count += 1
                    center_exit_time.append(start_frame)
        source = end + 1

    return center_exit_time


def get_rois_dists(traj_x, traj_y, start_frame, end_frame):
    roi_at_each_frame = []
    rois = []
    dists = []
    ## center of circle: x=202, y=204
    ## inner radius = 144
    ## outer radius = 176
    ## l1: (44,20) to (360,388)
    ## l2: (400,40) to (4,368)

    ## ROIs: open, closed, inner, outer

    center_x = 202
    center_y = 204

    in_rad = 144
    out_rad = 176

    k1 = (388 - 20) / (360 - 44)
    k2 = (368 - 40) / (4 - 400)
    # print(k1, k2)

    # l1: y-20=k_1(x-44)
    # l2: y-40=k_2(x-400)

    k3 = (k1 - math.tan(math.radians(20))) / (1 + k1 * math.tan(math.radians(20)))
    k4 = (k2 + math.tan(math.radians(20))) / (1 - k2 * math.tan(math.radians(20)))

    # l3: y-center_y = k_3(x-center_x)
    # l4: y-center_y = k_4(x-center_x)
    # below l3 and above l4; or; above l3 and below l4 --> inner closed

    k5 = (k1 + math.tan(math.radians(10))) / (1 - k1 * math.tan(math.radians(10)))
    k6 = (k2 - math.tan(math.radians(10))) / (1 + k2 * math.tan(math.radians(10)))

    # l5: y-center_y = k_5(x-center_x)
    # l6: y-center_y = k_6(x-center_x)
    # below l5 & l6; or; above l5 & l6
    # print(k3, k4)
    # print(k5, k6)

    for i in range(len(traj_x)):
        if start_frame <= i <= end_frame:
            dist_center = np.sqrt(
                (center_x-traj_x[i])**2 + (center_y-traj_y[i])**2)
            dists.append(dist_center)
            y1 = 20 + k1 * (traj_x[i] - 44)
            y2 = 40 + k2 * (traj_x[i] - 400)
            if (traj_y[i] > y1 and traj_y[i] > y2) or (traj_y[i] < y1 and traj_y[i] < y2):
                roi_at_each_frame.append('open')
            else:
                roi_at_each_frame.append('closed')

            y3 = center_y + k3 * (traj_x[i] - center_x)
            y4 = center_y + k4 * (traj_x[i] - center_x)
            y5 = center_y + k5 * (traj_x[i] - center_x)
            y6 = center_y + k6 * (traj_x[i] - center_x)

            if (traj_y[i] < y3 and traj_y[i] > y4) or (traj_y[i] > y3 and traj_y[i] < y4):
                rois.append('closed')
            elif (traj_y[i] < y5 and traj_y[i] < y6) or (traj_y[i] > y5 and traj_y[i] > y6):
                rois.append('open')
            else:
                rois.append('transition')
        else:
            dists.append(-1)
            rois.append('nan')
            roi_at_each_frame.append('nan')

    data_time_inrois = {name: roi_at_each_frame.count(name) for name in set(
        roi_at_each_frame)}  # total time (frames) in each roi
    # these returned arrays (ROIs & dists) are in length of the whole video, unprocessed frames marked as 'nan'
    return rois, roi_at_each_frame, data_time_inrois, dists


def get_open_closed_events(transitions, prev, frame_trans, start_frame, end_frame, fps=50):
    open_closed_entrytime = []
    open_closed_exittime = []

    closed_open_entrytime = []
    closed_open_exittime = []

    source = 0
    dest = source
    cross_count = 0

    while source < len(transitions) and dest < len(prev):
        dest = source
        if not source < len(transitions):
            break
        while (source < len(transitions)) and (not transitions[source].startswith('transition')):
            source += 1
            dest = source

        if source < len(frame_trans):
            entrytime = frame_trans[source]
            exittime = frame_trans[dest]
            while (dest < len(prev)) and (not prev[dest].startswith('transition')):
                dest += 1
                if dest < len(prev):
                    exittime = frame_trans[dest]
            if (dest < len(prev)) and (prev[dest].startswith('transition')) and (exittime - entrytime < fps*2) and (
                    prev[source] != transitions[dest]):
                # print(dest, transitions[dest])
                exittime = frame_trans[dest]
                if start_frame <= entrytime <=end_frame and start_frame <= exittime <= end_frame:
                    cross_count += 1
                    if prev[source] == 'open':
                        open_closed_entrytime.append(entrytime)
                        open_closed_exittime.append(exittime)
                    elif prev[source]=='closed':
                        closed_open_entrytime.append(entrytime)
                        closed_open_exittime.append(exittime)

        source = dest + 1

    return open_closed_entrytime, open_closed_exittime, closed_open_entrytime, closed_open_exittime


def get_open_closed_events_epm(transitions, prev, frame_trans, start_frame, end_frame, fps=50):
    open_closed_entrytime = []
    open_closed_exittime = []

    source = 0
    dest = source
    cross_count = 0

    while source < len(prev) and dest < len(prev):
        dest = source
        if not source < len(prev):
            break
        while (source < len(prev)) and (not prev[source].startswith('open')):
            source += 1
            dest = source

        if source < len(frame_trans):
            exittime = frame_trans[source]
            entrytime = frame_trans[dest]
            while (dest < len(transitions)) and (not transitions[dest].startswith('closed')):
                dest += 1
                if dest < len(transitions):
                    entrytime = frame_trans[dest]
            if (dest < len(transitions)) and (transitions[dest].startswith('closed')) and (
                    prev[source] != transitions[dest]):
                # print(dest, transitions[dest])
                entrytime = frame_trans[dest]
                if start_frame <= entrytime <=end_frame and start_frame <= exittime <= end_frame:
                    cross_count += 1
                    open_closed_entrytime.append(entrytime)
                    open_closed_exittime.append(exittime)

        source = dest + 1

    return open_closed_entrytime, open_closed_exittime


def get_closed_open_events_epm(transitions, prev, frame_trans, start_frame, end_frame, fps=50):
    closed_open_entrytime = []
    closed_open_exittime = []

    source = 0
    dest = source
    cross_count = 0

    while source < len(prev) and dest < len(prev):
        dest = source
        if not source < len(prev):
            break
        while (source < len(prev)) and (not prev[source].startswith('closed')):
            source += 1
            dest = source

        if source < len(frame_trans):
            exittime = frame_trans[source]
            entrytime = frame_trans[dest]
            while (dest < len(transitions)) and (not transitions[dest].startswith('open')):
                dest += 1
                if dest < len(transitions):
                    entrytime = frame_trans[dest]
            if (dest < len(transitions)) and (transitions[dest].startswith('open')) and (
                    prev[source] != transitions[dest]):
                # print(dest, transitions[dest])
                entrytime = frame_trans[dest]
                if start_frame <= entrytime <=end_frame and start_frame <= exittime <= end_frame:
                    cross_count += 1
                    closed_open_entrytime.append(entrytime)
                    closed_open_exittime.append(exittime)

        source = dest + 1

    return closed_open_entrytime, closed_open_exittime


def get_lingerings(transitions, prev, frame_trans, start_frame, end_frame, fps=50):
    # detect lingering in intermediate region
    lingering_entrytime = []
    lingering_exittime = []
    prolonged_open_closed_entrytime = []
    prolonged_open_closed_exittime = []
    prolonged_closed_open_entrytime = []
    prolonged_closed_open_exittime = []
    withdraw_entrytime = []
    withdraw_exittime = []

    source = 0
    dest = source
    cross_count = 0

    while source < len(transitions) and dest < len(prev):
        dest = source
        if not source < len(transitions):
            break
        while (source < len(transitions)) and (not transitions[source].startswith('transition')):
            source += 1
            dest = source

        if source < len(frame_trans):
            entrytime = frame_trans[source]
            exittime = frame_trans[dest]
            while (dest < len(prev)) and (not prev[dest].startswith('transition')):
                dest += 1
                if dest < len(prev):
                    exittime = frame_trans[dest]
            if (dest < len(prev)) and (prev[dest].startswith('transition')) and (exittime - entrytime > fps*2):
                # print(dest, transitions[dest])
                exittime = frame_trans[dest]
                if start_frame <= entrytime <= end_frame and start_frame <= exittime <= end_frame:
                    cross_count += 1
                    lingering_entrytime.append(entrytime)
                    lingering_exittime.append(exittime)
                    if prev[source] == transitions[dest]:
                        withdraw_entrytime.append(entrytime)
                        withdraw_exittime.append(exittime)
                    elif prev[source].startswith('open'):
                        prolonged_open_closed_entrytime.append(entrytime)
                        prolonged_open_closed_exittime.append(exittime)
                    else:
                        prolonged_closed_open_entrytime.append(entrytime)
                        prolonged_closed_open_exittime.append(exittime)

        source = dest + 1

    return lingering_entrytime, lingering_exittime, prolonged_open_closed_entrytime, prolonged_open_closed_exittime, prolonged_closed_open_entrytime, prolonged_closed_open_exittime, withdraw_entrytime, withdraw_exittime


def get_nose_dips(dists, start_frame, end_frame, fps=50):
    dip_starttime = []
    dip_stoptime = []

    begin = 0
    end = begin
    dip_count = 0

    in_rad = 144
    out_rad = 176

    while begin < len(dists) and end < len(dists):
        while begin < len(dists) and in_rad <= dists[begin] <= out_rad:
            begin += 1
        end = begin + 1

        if begin < len(dists) and (not (in_rad <= dists[begin] <= out_rad)):
            while end < len(dists) and (not (in_rad <= dists[end] <= out_rad)):
                end += 1

            if end < len(dists) and (in_rad <= dists[end] <= out_rad):
                if end - begin > fps // 2:
                    if start_frame <= begin <= end_frame and start_frame <= end <= end_frame:
                        dip_starttime.append(begin)
                        dip_stoptime.append(end)
        begin = end

    return dip_starttime, dip_stoptime


def get_nose_dips_epm(in_maze, start_frame, end_frame, fps=50):
    dip_starttime = []
    dip_stoptime = []

    begin = 0
    end = begin
    dip_count = 0

    while begin < len(in_maze) and end < len(in_maze):
        while begin < len(in_maze) and in_maze[begin]=='in':
            begin += 1
        end = begin + 1

        if begin < len(in_maze) and in_maze[begin]=='out':
            while end < len(in_maze) and (in_maze[end]=='out'):
                end += 1

            if end < len(in_maze):
                if end - begin > fps // 2:
                    if start_frame <= begin <= end_frame and start_frame <= end <= end_frame:
                        dip_starttime.append(begin)
                        dip_stoptime.append(end)
        begin = end

    return dip_starttime, dip_stoptime    


def find_transition_oft(traj_x, traj_y, fps=50):
    rois = create_rois_oft()
    roi_at_each_frame, data_time_inrois = assign_rois(traj_x, traj_y, rois)
    prev, transitions, frame_trans = get_transitions(roi_at_each_frame)
    transitions_count = {name: transitions.count(name) for name in transitions}
    # average time in each roi (frames)
    avg_time_in_roi = {dest[0]: time / dest[1]
                       for dest, time in zip(transitions_count.items(), data_time_inrois.values())}
    data_time_inrois_sec = {name: t / fps for name,
                                              t in data_time_inrois.items()}
    avg_time_in_roi_sec = {name: t / fps for name,
                                             t in avg_time_in_roi.items()}

    # avg_vel_per_roi = {}
    # for name in set(roi_at_each_frame):
    #     indexes = get_indexes(roi_at_each_frame, name)
    #     vels = spd[indexes]
    #     avg_vel_per_roi[name] = np.average(np.asarray(vels))

    # Save summary
    # transitions_count['tot'] = np.sum(list(transitions_count.values()))
    # data_time_inrois['tot'] = np.sum(list(data_time_inrois.values()))
    # data_time_inrois_sec['tot'] = np.sum(list(data_time_inrois_sec.values()))
    # avg_time_in_roi['tot'] = np.nan
    # avg_time_in_roi_sec['tot'] = np.nan
    # avg_vel_per_roi['tot'] = np.nan

    roinames = sorted(list(data_time_inrois.keys()))
    rois_stats = {
    #     "ROI_name": roinames,
        "roi_at_each_frame": roi_at_each_frame,
    #     "transitions_per_roi": [transitions_count[r] for r in roinames],
    #     "cumulative_time_in_roi": [data_time_inrois[r] for r in roinames],
    #     "cumulative_time_in_roi_sec": [data_time_inrois_sec[r] for r in roinames],
    #     "avg_time_in_roi": [avg_time_in_roi[r] for r in roinames],
    #     "avg_time_in_roi_sec": [avg_time_in_roi_sec[r] for r in roinames],
    #     "avg_vel_in_roi": [avg_vel_per_roi[r] for r in roinames],
    }

    # detect corner to center events
    corner_to_center_entrytime, corner_to_center_starttime = get_corner_to_center_events(prev, transitions, frame_trans)
    print("number of corner to center detected: %d" % len(corner_to_center_entrytime))

    # detect corner to corner entry time
    corner_to_corner_entrytime, corner_to_corner_starttime = get_corner_to_corner_events(prev, transitions, frame_trans)
    print("number of corner to corner detected: %d" % len(corner_to_corner_entrytime))

    # detect center exit time
    center_exit_time = get_center_exit_events(prev, transitions, frame_trans)
    print("number of center exit detected: %d" % len(center_exit_time))

    transitions = {
        'corner_to_center_entrytime': corner_to_center_entrytime,
        'corner_to_center_starttime': corner_to_center_starttime,
        'corner_to_corner_entrytime': corner_to_corner_entrytime,
        'corner_to_corner_starttime': corner_to_corner_starttime,
        'center_exit': center_exit_time
    }
    return rois_stats, transitions


def find_transition_epm(traj_x, traj_y, epm_coors, start_time, duration, spd, fps=50):
    # ROI Analysis (EZM)
    length = len(np.array(traj_x))
    # start_time unit: seconds
    # index of the first frame for evaluation
    start_frame = int(start_time * fps)
    # index of the last frame for evaluation
    # duration is time in minutes
    end_frame = min(int(start_frame + duration * 60 * fps), length)

    rois = create_rois_epm(epm_coors)
    roi_at_each_frame, data_time_inrois, in_maze = assign_rois_epm(traj_x, traj_y, rois, start_frame, end_frame)
    prev, transitions, frame_trans = get_transitions(roi_at_each_frame)
    transitions_count = {name: transitions.count(name) for name in transitions}
    # average time in each roi (frames)
    avg_time_in_roi = {dest[0]: time / dest[1]
                       for dest, time in zip(transitions_count.items(), data_time_inrois.values())}
    data_time_inrois_sec = {name: t / fps for name,
                                              t in data_time_inrois.items()}
    avg_time_in_roi_sec = {name: t / fps for name,
                                             t in avg_time_in_roi.items()}

    avg_vel_per_roi = {}
    for name in set(roi_at_each_frame):
        indexes = get_indexes(roi_at_each_frame, name)
        vels = spd[indexes]
        avg_vel_per_roi[name] = np.average(np.asarray(vels))

    # Save summary
    transitions_count['tot'] = np.sum(list(transitions_count.values()))
    data_time_inrois['tot'] = np.sum(list(data_time_inrois.values()))
    data_time_inrois_sec['tot'] = np.sum(list(data_time_inrois_sec.values()))
    avg_time_in_roi['tot'] = np.nan
    avg_time_in_roi_sec['tot'] = np.nan
    avg_vel_per_roi['tot'] = np.nan

    roinames = sorted(list(data_time_inrois.keys()))
    rois_stats = {
        "ROI_name": roinames,
        "roi_at_each_frame": roi_at_each_frame,
        "transitions_per_roi": [transitions_count[r] for r in roinames],
        "cumulative_time_in_roi": [data_time_inrois[r] for r in roinames],
        "cumulative_time_in_roi_sec": [data_time_inrois_sec[r] for r in roinames],
        "avg_time_in_roi": [avg_time_in_roi[r] for r in roinames],
        "avg_time_in_roi_sec": [avg_time_in_roi_sec[r] for r in roinames],
        "avg_vel_in_roi": [avg_vel_per_roi[r] for r in roinames],
    }

    open_closed_entrytime, open_closed_exittime = get_open_closed_events_epm(transitions, prev, frame_trans, start_frame, end_frame)
    print("Number of open-to-closed crossings detected: %d" % len(open_closed_entrytime))
    closed_open_entrytime, closed_open_exittime = get_closed_open_events_epm(transitions, prev, frame_trans, start_frame, end_frame)
    print("Number of closed-to-open crossings detected: %d" % len(closed_open_entrytime))
    dip_starttime, dip_stoptime = get_nose_dips_epm(in_maze, start_frame, end_frame)
    print("Number of nosedips detected: %d" % len(dip_starttime))

    transitions = {
        'open_closed_entrytime': open_closed_entrytime,
        'open_closed_exittime': open_closed_exittime,
        'closed_open_entrytime': closed_open_entrytime,
        'closed_open_exittime': closed_open_exittime,
        'dip_starttime': dip_starttime,
        'dip_stoptime': dip_stoptime
    }

    return rois_stats, transitions


def find_transition_ezm(traj_x, traj_y, start_time, duration, spd, fps=50):
    # ROI Analysis (EZM)
    length = len(traj_x)
    # start_time unit: seconds
    # index of the first frame for evaluation
    start_frame = int(start_time * fps)
    # index of the last frame for evaluation
    # duration is time in minutes
    end_frame = min(int(start_frame + duration * 60 * fps), length)

    rois, roi_at_each_frame, data_time_inrois, dists = get_rois_dists(traj_x, traj_y, start_frame, end_frame)
    transitions, prev, frame_trans = get_transitions(roi_at_each_frame)

    transitions_count = {name: transitions.count(name) for name in transitions}
    # average time in each roi (frames)
    avg_time_in_roi = {transitions[0]: time / transitions[1]
                       for transitions, time in zip(transitions_count.items(), data_time_inrois.values())}
    data_time_inrois_sec = {name: t / fps for name,
                                              t in data_time_inrois.items()}
    avg_time_in_roi_sec = {name: t / fps for name,
                                             t in avg_time_in_roi.items()}
    
    avg_vel_per_roi = {}
    for name in set(roi_at_each_frame):
        indexes = get_indexes(roi_at_each_frame, name)
        vels = spd[indexes]
        avg_vel_per_roi[name] = np.average(np.asarray(vels))

    # Save summary
    transitions_count['tot'] = np.sum(list(transitions_count.values()))
    data_time_inrois['tot'] = np.sum(list(data_time_inrois.values()))
    data_time_inrois_sec['tot'] = np.sum(list(data_time_inrois_sec.values()))
    avg_time_in_roi['tot'] = np.nan
    avg_time_in_roi_sec['tot'] = np.nan
    avg_vel_per_roi['tot'] = np.nan

    roinames = sorted(list(data_time_inrois.keys()))
    rois_stats = {
        "ROI_name": roinames,
        "roi_at_each_frame": roi_at_each_frame,
        "transitions_per_roi": [transitions_count[r] for r in roinames],
        "cumulative_time_in_roi": [data_time_inrois[r] for r in roinames],
        "cumulative_time_in_roi_sec": [data_time_inrois_sec[r] for r in roinames],
        "avg_time_in_roi": [avg_time_in_roi[r] for r in roinames],
        "avg_time_in_roi_sec": [avg_time_in_roi_sec[r] for r in roinames],
        "avg_vel_in_roi": [avg_vel_per_roi[r] for r in roinames],
    }

    # Detect events open-closed & closed-open
    prev, transitions, frame_trans = get_transitions(rois)
    
    open_closed_entrytime, open_closed_exittime, closed_open_entrytime, closed_open_exittime = get_open_closed_events(transitions, prev, frame_trans, start_frame, end_frame)
    print("Number of open-to-closed crossings detected: %d" % len(open_closed_entrytime))
    print("Number of closed-to-open crossings detected: %d" % len(closed_open_entrytime))

    lingering_entrytime, lingering_exittime, prolonged_open_closed_entrytime, prolonged_open_closed_exittime, prolonged_closed_open_entrytime, prolonged_closed_open_exittime, withdraw_entrytime, withdraw_exittime = get_lingerings(transitions, prev, frame_trans, start_frame, end_frame)
    print("Number of lingerings in transition region detected: %d" % len(lingering_entrytime))
    print("Number of prolonged open to closed crossings detected: %d" % len(prolonged_open_closed_entrytime))
    print("Number of prolonged closed to open crossings detected: %d" % len(prolonged_closed_open_entrytime))
    print("Number of withdraws detected: %d" % len(withdraw_entrytime))

    dip_starttime, dip_stoptime = get_nose_dips(dists, start_frame, end_frame)
    print("Number of nosedips detected: %d" % len(dip_starttime))

    transitions = {
        'open_closed_entrytime': open_closed_entrytime,
        'open_closed_exittime': open_closed_exittime,
        'closed_open_entrytime': closed_open_entrytime,
        'closed_open_exittime': closed_open_exittime,
        'lingering_entrytime': lingering_entrytime,
        'lingering_exittime': lingering_exittime,
        'prolonged_open_closed_entrytime': prolonged_open_closed_entrytime,
        'prolonged_open_closed_exittime': prolonged_open_closed_exittime,
        'prolonged_closed_open_entrytime': prolonged_closed_open_entrytime,
        'prolonged_closed_open_exittime': prolonged_closed_open_exittime,
        'withdraw_entrytime': withdraw_entrytime,
        'withdraw_exittime': withdraw_exittime,
        'nosedip_starttime': dip_starttime,
        'nosedip_stoptime': dip_stoptime
    }
    return rois_stats, transitions


def traj_process(session, start_time, duration, behavior=None, bp='head', fps=50):
    fps = fps
    traj_df, scorer = load_traj(session)
    if behavior == 'epm':
        traj_x, traj_y = calib_traj(traj_df, start_time, duration, fps, bp=bp, XYMAX=450)
    elif behavior == 'ezm':
        traj_x, traj_y = calib_traj(traj_df, start_time, duration, fps, bp=bp, XYMAX=400)

    accumulated_distance, spd, acc, ismoving = calculate_speed_acc(traj_x, traj_y, start_time, duration, fps, move_cutoff=5)

    results = {}
    movement = {
        'start_time': start_time,
        'duration': duration,
        'calib_traj_x': traj_x,
        'calib_traj_y': traj_y,
        'instant_speed': spd,
        'accumulative_distance': accumulated_distance,
        'is_moving': ismoving
    }
    results.update({'movement': movement})

    # ROI analysis
    # Given position data for a bodypart and the position of a list of rois, this function calculates which roi is the closest to the bodypart at each frame

    # if behavior == 'oft':
    #     rois_stats, transitions = find_transition_oft(traj_x, traj_y)
    #     results.update({
    #         'rois_stats': rois_stats,
    #         'transitions': transitions})

    if behavior == 'epm':
        EPM_points = load_points(session, behavior)
        rois_stats, transitions = find_transition_epm(traj_x, traj_y, EPM_points, start_time, duration, spd)
        results.update({'rois_stats': rois_stats, 'transitions': transitions})

    if behavior == 'ezm':
        rois_stats, transitions = find_transition_ezm(traj_x, traj_y, start_time, duration, spd, fps)
        results.update({'rois_stats': rois_stats, 'transitions': transitions})

    return results


def get_events(events, video_trigger, video_duration, f_video=50):
    crop_from = int(f_video * video_trigger)
    crop_to = int((video_trigger + video_duration) * f_video)

    open_idx = [i for i, el in enumerate(
        events['rois_stats']['roi_at_each_frame'][crop_from:crop_to])  ## cropped the frame before trigger
                if
                el == 'open']

    close_idx = [i for i, el in enumerate(
        events['rois_stats']['roi_at_each_frame'][crop_from:crop_to])
                 if
                 el == 'closed']
    OTC_idx = np.array(events['transitions']['open_closed_exittime']) - crop_from  ## crop the frame before trigger

    prOTC_idx = np.array(events['transitions']['prolonged_open_closed_exittime']) - crop_from
    prCTO_idx = np.array(events['transitions']['prolonged_closed_open_exittime']) - crop_from
    nosedip_idx = np.array(events['transitions']['nosedip_starttime']) - crop_from

    return open_idx, close_idx, OTC_idx, prOTC_idx, prCTO_idx, nosedip_idx