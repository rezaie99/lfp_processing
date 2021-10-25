import glob
import numpy as np
import pandas as pd
from collections import namedtuple
import pickle
from sys import exit

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math
from scipy import interpolate
from shapely.geometry import Point, Polygon


def load_location(session):
    file_dir = session + '/video_processed/'
    fname = glob.glob(file_dir + '*filtered.h5')[0]
    print(fname)
    loc = pd.read_hdf(fname) # read the H5 file from Deeplabcut as a dataframe
    scorer = loc.columns[0][0]  # scorer is the trained network by Deeplabcut for estimate the location of a animal
    return loc, scorer


def calib_location(loc_df, sdir, xymax, THRL=0.95, THRD=5):
    scorer = loc_df.columns[0][0]
    bodyparts = loc_df.columns.levels[1].to_list()
    length = len(loc_df)
    print('total number of frames: ', length)

    for bd in bodyparts:

        bd_x = loc_df[scorer, bd, 'x']
        bd_y = loc_df[scorer, bd, 'y']
        bd_confidence = loc_df[scorer, bd, 'likelihood']

        bd_x_diff = bd_x.diff()
        bd_x_diff.iloc[0] = 1
        bd_y_diff = bd_y.diff()
        bd_y_diff.iloc[0] = 1
        step_distance = np.sqrt(bd_x_diff ** 2 + bd_y_diff ** 2)

        bd_x_new = bd_x.copy(deep=True)
        bd_y_new = bd_y.copy(deep=True)

        # process data point:
        # (A) if the distance of current point to the previous one exceeds a certain threshold;
        # (B) if the likelihood value is below a certain threshold
        # or (C) the point is currently out of frame
        # the video frame is 330 by 330 pixels in size in OFT sessions, but approx. 400 by 400 in EZM sessions

        XMIN = 0
        XMAX = xymax
        YMIN = 0
        YMAX = xymax

        num_processed = 0
        process = False
        ## process the entire recording
        for i in range(length):
            if step_distance[i] > THRD or \
                    bd_confidence[i] < THRL or \
                    bd_x_new[i] < XMIN or \
                    bd_x_new[i] > XMAX or \
                    bd_y_new[i] < YMIN or \
                    bd_y_new[i] > YMAX:
                num_processed += 1
                bd_x_new.loc[i] = np.nan # replace the point with nan
                bd_y_new.loc[i] = np.nan
        # set limit_direction so that consecutive NaNs are filled with interpolation
        bd_x_new = bd_x_new.interpolate(method='linear', limit_direction='both')
        bd_y_new = bd_y_new.interpolate(method='linear', limit_direction='both')
        print("edited " + str(num_processed) + " data points")
        # two new pandas Series are returned, same in length as the original ones
        # ONLY the frames defined by Start_time and Duration are processed !!

        loc_df[scorer, bd, 'x'] = bd_x_new
        loc_df[scorer, bd, 'y'] = bd_y_new

    fig, ax = plt.subplots(figsize=(8, 8))
    for bd in bodyparts:
        ax.plot(loc_df[scorer, bd, 'x'], loc_df[scorer, bd, 'y'],
                 label=bd, alpha=0.3)

    title = 'Locomotion trajectory'
    ax.set_title(title)
    plt.tight_layout()
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize=12)
    plt.savefig(sdir + 'locomotion trajectory.png')
    plt.show()

    return loc_df


def get_locomotion(loc_df, fps=50, move_cutoff=5, avgspd_win=9):
    # "win" is the length (in frames) of sliding window within which average speed is calculated
    # "move cutoff" is the threshold (in pixels per second, same as speed) below which a frame is labeled as "not moving"

    scorer = loc_df.columns[0][0]
    bodyparts = loc_df.columns.levels[1].to_list()
    length = len(loc_df)
    print('total number of frames: ', length)

    for bp in bodyparts:
        x = loc_df[scorer, bp, 'x']
        y = loc_df[scorer, bp, 'y']

        x_diff = x.diff().fillna(0)  # x change between neighboring two frames
        y_diff = y.diff().fillna(0)  # y change between neighboring two frames
        step = np.sqrt(x_diff ** 2 + y_diff ** 2)  # distance between two video frames
        accumulated_distance = step.cumsum()

        speed = step * fps  # unit of speed: pixels per second
        acceleration = speed.diff().fillna(0)
        spd_filtered = speed.rolling(window=avgspd_win).mean().fillna(0)  # compute the mean of a rolling window, fill na with 0
        ismoving = spd_filtered > move_cutoff
        # acc unit: pixels per timestep squared

        loc_df[scorer, bp, 'speed'] = speed
        loc_df[scorer, bp, 'speed_filtered'] = spd_filtered
        loc_df[scorer, bp, 'acceleration'] = acceleration
        loc_df[scorer, bp, 'accumlated_dist'] = accumulated_distance
        loc_df[scorer, bp, 'ismoving'] = ismoving

    return loc_df


def cart2pol_point(x, y):
    rho = np.sqrt(x ** 2 + y ** 2)
    phi = np.arctan2(y, x)
    return (rho, phi)


def cart2pol(loc_df): ## Cartesian coordinate system to Polar coordinate system
    scorer = loc_df.columns[0][0]
    bodyparts = loc_df.columns.levels[1].to_list()

    center = (200, 200)

    for bd in bodyparts:
        x = loc_df[scorer, bd, 'x'] - center[0]
        y = loc_df[scorer, bd, 'y'] - center[1]
        rho, phi = cart2pol_point(x, y)

        loc_df[scorer, bd, 'rho'] = rho
        loc_df[scorer, bd, 'phi'] = phi

    return loc_df


def load_rois(session):
    file_dir = session + '/ephys_processed/'
    print(file_dir)
    try:
        with open(file_dir + session + '_rois.pkl', 'rb') as f:
            coors = pickle.load(f)
            f.close()
    except IndexError:
        print('Key coordinates for EPM dictionary file not found!')
        exit()
    return coors

def save_rois(session, data):
    dir_save = session + '/ephys_processed/'
    print(dir_save)
    try:
        with open(dir_save + session + '_rois.pkl', 'wb') as f:
            pickle.dump(data, f)
            f.close()
    except IndexError:
        print('Failed to save the file')
        exit()


def create_rois(session, behav, save=False):
    if behav == 'oft':
        # two points defining each roi: topleft(X,Y) and bottomright(X,Y).
        position = namedtuple('position', ['topleft', 'bottomright'])
        rois = {'center': position((87.5, 87.5), (252.5, 252.5)),
                'cornertl': position((5, 252.5), (87.5, 335)),
                'cornertr': position((252.5, 252.5), (335, 335)),
                'cornerbl': position((5, 5), (87.5, 87.5)),
                'cornerbr': position((252.5, 5), (335, 87.5)),
                'peritop': position((87.5, 252.5), (252.5, 335)),
                'perileft': position((5, 87.5), (87.5, 252.5)),
                'peribottom': position((87.5, 5), (252.5, 87.5)),
                'periright': position((252.5, 87.5), (335, 252.5))}

    if behav == 'epm':
        rois = {
            'closedtop': Polygon(
                [epm_coors['tclosedtr'], epm_coors['tclosedtl'], epm_coors['tclosedbl'], epm_coors['tclosedbr']]),
            'center': Polygon(
                [epm_coors['centertl'], epm_coors['centertr'], epm_coors['centerbr'], epm_coors['centerbl']]),
            'closedbottom': Polygon(
                [epm_coors['bclosedbl'], epm_coors['bclosedbr'], epm_coors['bclosedtr'], epm_coors['bclosedtl']]),
            'openleft': Polygon(
                [epm_coors['lopentl'], epm_coors['lopenbl'], epm_coors['lopenbr'], epm_coors['lopentr']]),
            'openright': Polygon(
                [epm_coors['ropentl'], epm_coors['ropenbl'], epm_coors['ropenbr'], epm_coors['ropentr']]),
        }
        
    if behav == 'ezm':
        pass

    if save:
        save_rois(session, rois)

    return rois


def plot_loc_with_rois(loc_df, bp, rois, start, duration, fps=50):
    fig, ax = plt.subplots(1, figsize=(6, 6))

    # plot trajectory + bounding boxes for rois
    scorer = loc_df.columns[0][0]
    # start_time unit: seconds
    # index of the first frame for evaluation
    start_frame = int(start * fps)
    # index of the last frame for evaluation
    # duration is time in seconds
    end_frame = int(start + duration * fps)

    x = loc_df[scorer, bp, 'x']
    y = loc_df[scorer, bp, 'y']

    plt.plot(x[start_frame:end_frame], y[start_frame:end_frame], '-', linewidth=.2, alpha=.6)

    coords = np.array(rois['center'].exterior.coords.xy).T
    polygon = patches.Polygon(coords, linewidth=2, edgecolor='purple',
                              facecolor='none')
    ax.add_patch(polygon)

    coords = np.array(rois['openleft'].exterior.coords.xy).T
    polygon = patches.Polygon(coords, linewidth=2, edgecolor='orange',
                              facecolor='none')
    ax.add_patch(polygon)

    coords = np.array(rois['openright'].exterior.coords.xy).T
    polygon = patches.Polygon(coords, linewidth=2, edgecolor='orange',
                              facecolor='none')
    ax.add_patch(polygon)

    coords = np.array(rois['closedtop'].exterior.coords.xy).T
    polygon = patches.Polygon(coords, linewidth=2, edgecolor='green',
                              facecolor='none')
    ax.add_patch(polygon)

    coords = np.array(rois['closedbottom'].exterior.coords.xy).T
    polygon = patches.Polygon(coords, linewidth=2, edgecolor='green',
                              facecolor='none')
    ax.add_patch(polygon)

    plt.title('Calibrated trajectory of ' + bp + ' with ROIs', fontsize=18)
    plt.xlim(0, 450)
    plt.ylim(0, 450)
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


def assign_rois(x, y, rois):
    # calculate distance between the location to each roi
    data_length = len(x)
    centers_roi = get_roi_center(rois)
    distances = np.zeros((data_length, len(centers_roi)))
    for idx, center in enumerate(centers_roi):
        cnt = np.tile(center, data_length).reshape((data_length, 2))
        dist = np.hypot(np.subtract(
            cnt[:, 0], x), np.subtract(cnt[:, 1], y))
        distances[:, idx] = dist

    # get which roi is closest at each frame
    # this is based on the distances between each ROI's center and the point in the frame
    # if a point is not bounded by any ROI, it will still be labeled as belonging to the nearest ROI
    roi_names = list(rois.keys())
    sel_rois = np.argmin(distances, 1)
    roi_nearest = tuple([roi_names[x] for x in sel_rois])
    roi_at_each_frame = []
    for i in range(data_length):
        xi, yi = x[i], y[i]
        bounded = False
        for j, curr_roi in enumerate(rois):
            X, Y = sort_roi_points(rois[curr_roi])
            if X[0] <= xi <= X[1] and Y[0] <= yi <= Y[1]:
                bounded = True
                roi_at_each_frame.append(curr_roi)
                break
        if not bounded:
            roi_at_each_frame.append(roi_nearest[i])

    data_time_inrois = {name: roi_at_each_frame.count(name) for name in set(
        roi_at_each_frame)}  # total time (frames) in each roi
    return roi_at_each_frame, data_time_inrois


def assign_rois_epm(traj_x, traj_y, traj_x_head, traj_y_head, rois, start_frame, end_frame):
    data_length = len(traj_x)
    roi_at_each_frame = []
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
                if x < 225:
                    roi_at_each_frame.append('openleft')
                else:
                    roi_at_each_frame.append('openright')
        else:
            roi_at_each_frame.append('nan')

    in_maze = []
    for i in range(data_length):
        if start_frame <= i <= end_frame:
            x, y = traj_x_head[i], traj_y_head[i]
            curr_point = Point(x, y)
            bounded = False
            for j, curr_roi in enumerate(rois):
                if curr_point.within(rois[curr_roi]):
                    bounded = True
            if bounded:
                in_maze.append('in')
            else:
                in_maze.append('out')
        else:
            in_maze.append('nan')

    data_time_inrois = {name: roi_at_each_frame.count(name) for name in set(
        roi_at_each_frame)}  # total time (frames) in each roi

    return roi_at_each_frame, data_time_inrois, in_maze


def get_transitions(roi_at_each_frame):  ### detect the frames when animal transit from one zone to the others
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


def get_corner_to_center_events(prev, dest, frame_trans, thresh=2, fps=50):
    # detect corner-to-center entry events
    corner_to_center_starttime = []  ## animal leave one corner
    corner_to_center_entrytime = []  ## animal enter another corner

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
            if (end < len(dest)) and (dest[end].startswith('center')) and (
                    entry_frame - start_frame < thresh * fps):  # 2 second
                entry_frame = frame_trans[end]
                entry_count += 1
                corner_to_center_starttime.append(start_frame)
                corner_to_center_entrytime.append(entry_frame)

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
            if (end < len(dest)) and (dest[end].startswith('corner')) and (entry_frame - start_frame < 100) and (
                    not prev[source] == dest[end]):
                entry_frame = frame_trans[end]
                entry_count += 1
                corner_to_corner_entrytime.append(entry_frame)
                corner_to_corner_starttime.append(start_frame)
        source = end + 1
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

## for
def get_rois_dists(loc_x, loc_y, start_frame, end_frame):
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

    # scorer = loc_df.columns[0][0]
    x = loc_x
    y = loc_y

    for i in range(len(x)):
        if start_frame <= i <= end_frame:
            dist_center = np.sqrt(
                (center_x - x[i]) ** 2 + (center_y - y[i]) ** 2)
            dists.append(dist_center)
            y1 = 20 + k1 * (x[i] - 44)
            y2 = 40 + k2 * (x[i] - 400)
            if (y[i] > y1 and y[i] > y2) or (y[i] < y1 and y[i] < y2):
                roi_at_each_frame.append('open')
            else:
                roi_at_each_frame.append('closed')

            y3 = center_y + k3 * (x[i] - center_x)
            y4 = center_y + k4 * (x[i] - center_x)
            y5 = center_y + k5 * (x[i] - center_x)
            y6 = center_y + k6 * (x[i] - center_x)

            if (y[i] < y3 and y[i] > y4) or (y[i] > y3 and y[i] < y4):
                rois.append('closed')
            elif (y[i] < y5 and y[i] < y6) or (y[i] > y5 and y[i] > y6):
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
            if (dest < len(prev)) and (prev[dest].startswith('transition')) and (exittime - entrytime < fps * 2) and (
                    prev[source] != transitions[dest]):
                # print(dest, transitions[dest])
                exittime = frame_trans[dest]
                if start_frame <= entrytime <= end_frame and start_frame <= exittime <= end_frame:
                    cross_count += 1
                    if prev[source] == 'open':
                        open_closed_entrytime.append(entrytime)
                        open_closed_exittime.append(exittime)
                    elif prev[source] == 'closed':
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
                if start_frame <= entrytime <= end_frame and start_frame <= exittime <= end_frame:
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
                if start_frame <= entrytime <= end_frame and start_frame <= exittime <= end_frame:
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
            if (dest < len(prev)) and (prev[dest].startswith('transition')) and (exittime - entrytime > fps * 2):
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
        while begin < len(in_maze) and in_maze[begin] == 'in':
            begin += 1
        end = begin + 1

        if begin < len(in_maze) and in_maze[begin] == 'out':
            while end < len(in_maze) and (in_maze[end] == 'out'):
                end += 1

            if end < len(in_maze):
                if end - begin > fps // 4:
                    if start_frame <= begin <= end_frame and start_frame <= end <= end_frame:
                        dip_starttime.append(begin)
                        dip_stoptime.append(end)
        begin = end

    return dip_starttime, dip_stoptime


def find_transition_oft(loc_x, loc_y, fps=50):
    rois = create_rois(behav='oft')
    roi_at_each_frame, data_time_inrois = assign_rois(loc_x, loc_y, rois)
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


def analyze_trajectory_epm(traj_x, traj_y, traj_x_head, traj_y_head, epm_coors, start_time, duration, spd, fps=50):
    # ROI Analysis (EZM)
    length = len(np.array(traj_x))
    # start_time unit: seconds
    # index of the first frame for evaluation
    start_frame = int(start_time * fps)
    # index of the last frame for evaluation
    # duration is time in seconds
    end_frame = min(int(start_frame + duration * fps), length)

    rois = create_rois_epm(epm_coors)
    print("ROIs defined")
    plot_traj_roi_epm(traj_x_head, traj_y_head, rois, start_frame, end_frame, bp='head')
    plot_traj_roi_epm(traj_x, traj_y, rois, start_frame, end_frame, bp='body')

    roi_at_each_frame, data_time_inrois, in_maze = assign_rois_epm(traj_x, traj_y, traj_x_head, traj_y_head, rois,
                                                                   start_frame, end_frame)
    print("ROIs assigned")
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

    open_closed_entrytime, open_closed_exittime = get_open_closed_events_epm(transitions, prev, frame_trans,
                                                                             start_frame, end_frame)
    print("Number of open-to-closed crossings detected: %d" % len(open_closed_entrytime))
    closed_open_entrytime, closed_open_exittime = get_closed_open_events_epm(transitions, prev, frame_trans,
                                                                             start_frame, end_frame)
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


def ezm_analyzer(loc_df, bp, start_time, duration, fps=25):
    ''''
    find the transition event in the elevated zero moze
    loc_df: coordinates of the mouse
    bd: the body part to be used (head, shoulder, tail etc.)
    start_time: alwoys set to zero, in second
    duration: the duration of data to be analyzed, in seconds
    fps: frame rate of the video
    '''

    length = len(loc_df)
    start = int(start_time * fps)
    end = min(int(start + duration * fps), length)

    scorer = loc_df.columns[0][0]
    loc_x = loc_df[scorer, bp, 'x']
    loc_y = loc_df[scorer, bp, 'y']

    rois, roi_at_each_frame, data_time_inrois, dists = get_rois_dists(loc_x, loc_y, start, end)

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
        vels = loc_df[scorer, bp, 'avgspd'][indexes] # if not working , use .loc
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

    open_closed_entrytime, open_closed_exittime, closed_open_entrytime, closed_open_exittime = get_open_closed_events(
        transitions, prev, frame_trans, start, end)
    print("Number of open-to-closed crossings detected: %d" % len(open_closed_entrytime))
    print("Number of closed-to-open crossings detected: %d" % len(closed_open_entrytime))

    lingering_entrytime, lingering_exittime, prolonged_open_closed_entrytime, prolonged_open_closed_exittime, prolonged_closed_open_entrytime, prolonged_closed_open_exittime, withdraw_entrytime, withdraw_exittime = get_lingerings(
        transitions, prev, frame_trans, start, end)
    print("Number of lingerings in transition region detected: %d" % len(lingering_entrytime))
    print("Number of prolonged open to closed crossings detected: %d" % len(prolonged_open_closed_entrytime))
    print("Number of prolonged closed to open crossings detected: %d" % len(prolonged_closed_open_entrytime))
    print("Number of withdraws detected: %d" % len(withdraw_entrytime))

    dip_starttime, dip_stoptime = get_nose_dips(dists, start, end)
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


def loc_analyzer(session, start_time, duration, task='ezm', bp='head', fps=25):
    fps = fps
    loc, scorer = load_location(session)
    loc = calib_location(loc, task='ezm', fps=fps)
    loc = get_locomotion(loc, fps)
    loc_x = loc[scorer, bp, 'x']
    loc_y = loc[scorer, bp, 'y']

    # ROI analysis
    # Given position data for a bodypart and the position of a list of rois, this function calculates which roi is the closest to the bodypart at each frame
    results = {}
    if task == 'oft':
        rois_stats, transitions = find_transition_oft(loc_x, loc_y)
        results.update({
            'rois_stats': rois_stats,
            'transitions': transitions})

    # if behav == 'epm':
    #     EPM_points = load_rois(session, behav)
    #     rois_stats, transitions = analyze_trajectory_epm(loc_x, loc_y, loc_x_head, loc_y_head, EPM_points,
    #                                                      start_time, duration, spd)
    #     results.update({'rois_stats': rois_stats, 'transitions': transitions})

    if task == 'ezm':
        rois_stats, transitions = ezm_analyzer(loc, bp,  start_time, duration, fps)
        results.update({'rois_stats': rois_stats, 'transitions': transitions})

    return loc, results


def get_events(events: object, video_trigger: object, video_duration: object, f_video: object = 50) -> object:
    crop_start = int(f_video * video_trigger)
    crop_end = int((video_trigger + video_duration) * f_video)

    open_idx = [i for i, el in enumerate(
        events['rois_stats']['roi_at_each_frame'][crop_start:crop_end])  ## crop the frame before trigger
                if
                el == 'open']

    close_idx = [i for i, el in enumerate(
        events['rois_stats']['roi_at_each_frame'][crop_start:crop_end])
                 if
                 el == 'closed']
    OTC_idx = np.array(events['transitions']['open_closed_exittime']) - crop_start  ## crop the frame before trigger

    prOTC_idx = np.array(events['transitions']['prolonged_open_closed_exittime']) - crop_start
    prCTO_idx = np.array(events['transitions']['prolonged_closed_open_exittime']) - crop_start
    nosedip_idx = np.array(events['transitions']['nosedip_starttime']) - crop_start

    return open_idx, close_idx, OTC_idx, prOTC_idx, prCTO_idx, nosedip_idx


def create_mne_events(idx, event_id):
    mne_events = np.zeros((len(idx), 3)).astype(int)
    mne_events[:, 0] = idx
    mne_events[:, 2] = [event_id] * len(idx)

    return mne_events


def merge_events(OTC, prOTC, prCTO, nosedip):
    events = [OTC, prOTC, prCTO, nosedip]
    events = np.concatenate(events, axis=0)
    events = events[events[:, 0].argsort()]
    return events


def evaluate_OF(traj_x, traj_y, srate, center_area, corners):
    ### create trajectory corordinates of the animal (body parts)
    traj_xy = list(zip(traj_x, traj_y))

    center_area_count = 0
    peri_area_count = 0
    corners_count = 0

    # Create Point objects
    head_cord = Point(traj_xy[0])

    # initiate the position of the animal
    mouse_in_center = head_cord.within(center_area)
    center_idx = []
    peri_idx = []
    corners_idx=[]

    ### find out where is the animal in a frame
    for i in range(len(traj_x)):
        # Create Point objects
        head_cord = Point(traj_xy[i])

        if mouse_in_center == True:
            if head_cord.within(center_area):
                center_area_count += 1
                center_idx.append(i)
                # print('mouse_in_center', center_area_count)

            if head_cord.within(center_area) == False:
                peri_area_count += 1
                mouse_in_center = False
                # print('mouse_in_peri', peri_area_count)

        elif mouse_in_center == False:
            if head_cord.within(center_area):
                center_area_count += 1
                mouse_in_center = True
                center_idx.append(i)
                # print('mouse_in_center', center_area_count)

            elif head_cord.within(center_area) == False:
                peri_area_count += 1
                peri_idx.append(i)

                for corner in corners:
                    if head_cord.within(corner):
                        corners_count += 1
                        corners_idx.append(i)
                # print('mouse_in_peri', peri_area_count)

    time_in_center = center_area_count * 1/srate  ## 25 ms per frame
    time_in_peri = peri_area_count * 1/srate  ## 25 ms per frame

    percentage_in_center_area = time_in_center / (time_in_center + time_in_peri)
    ### Validate the result, two values should be the same

    print(peri_area_count + center_area_count)

    print('time_in_center = ', time_in_center, '\n')
    print('time_in_peri = ', time_in_peri, '\n', '\n',
          'percentage_in_center_area = ', percentage_in_center_area, '\n', '\n')

    return time_in_center, time_in_peri, percentage_in_center_area, center_idx, peri_idx, corners_idx