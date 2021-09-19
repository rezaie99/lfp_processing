import tqdm
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn import mixture
import pandas as pd
import argparse

ref_file = 'ref_image.npz'

def get_avg(file):
    input_video_path = file

    cap = cv2.VideoCapture(input_video_path)
    total_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print('Loading: ' ,file ,'...')
    all_frames = []
    for i in tqdm.tqdm(range(total_count)) :
        ret, frame = cap.read()
        # print(np.shape(frame))
        if ret:
            # plt.imshow(frame)

            if len(all_frames) == 0:
                all_frames = np.array(frame).astype('float64')
            else:
                all_frames += np.array(frame).astype('float64')
        else:
            break

    ref_data = np.load(ref_file)

    plt.figure()
    plt.imshow((all_frames /total_count).astype('uint8'))

    cap.release()
    cv2.destroyAllWindows()

    mx = ref_data['position'][0]
    my = ref_data['position'][1]
    r1 = ref_data['position'][2]
    r2 = ref_data['position'][3]

    mask   = np.fft.fft2(ref_data['ref'])
    image  = np.fft.fft2(all_frames[: ,: ,1])
    spect  = np.fft.ifft2(mask * np.conj(image))
    mx_off = np.argwhere(spect==spect.max())[0][0]
    my_off = np.argwhere(spect==spect.max())[0][1]

    mx_off = np.shape(spect)[0] - mx_off
    my_off = np.shape(spect)[1] - my_off
    if mx_off > np.shape(spect)[0 ] /2:
        mx_off = mx_off - np.shape(spect)[0]

    if my_off > np.shape(spect)[1 ] /2:
        my_off = my_off - np.shape(spect)[1]

    mx = mx + mx_off
    my = my + my_off
    print(mx_off, my_off)
    return (mx, my, r1, r2, all_frames, total_count)


def process_mice_loc(mx, my, r1, r2, all_frames, file):
    input_video_path = file

    cap = cv2.VideoCapture(input_video_path)
    ny, nx ,_ = np.shape(all_frames)
    x = np.arange(nx)
    y = np.arange(ny)
    xv, yv = np.meshgrid(x, y)
    zv = ((np.square(xv - mx ) +np.square(yv - my)) > (r1**2)) \
         & ((np.square(xv - mx ) +np.square(yv - my)) < (r2**2))

    zv = zv.astype('float64')


    total_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print('Process ...')
    frame_seq = []
    pos = []
    for i in tqdm.tqdm(range(total_count)) :
        ret, frame = cap.read()
        if ret:
            frame_seq.append(np.array(frame).astype('float64'))
            if (1):
                diff_frame = np.abs(np.array(frame).astype('float64') - (all_frames /total_count))
                df = (diff_frame[: ,: ,0 ] + diff_frame[: ,: ,1 ] + diff_frame[: ,: ,2] ) *zv
                diff_max = (np.sort(df.flatten())[-900]) + 5

                data = np.where(df > diff_max)
                data = [[a ,b] for a ,b in zip(data[0], data[1])]

                gmm = mixture.GaussianMixture(n_components=3).fit(data)

                if zv[int(gmm.means_[0][0]), int(gmm.means_[0][1])] > 0:
                    pos.append((gmm.means_[0][1], gmm.means_[0][0], diff_max))
                elif zv[int(gmm.means_[1][0]), int(gmm.means_[1][1])] > 0:
                    pos.append((gmm.means_[1][1], gmm.means_[1][0], diff_max))
                elif zv[int(gmm.means_[2][0]), int(gmm.means_[2][1])] > 0:
                    pos.append((gmm.means_[2][1], gmm.means_[2][0], diff_max))
                else:
                    pos.append(pos[-1])
        else:
            break

    cap.release()
    cv2.destroyAllWindows()
    return pos

def imshow_pos(pos, total_count, all_frames):

    y = [y if t> 100 else 0 for y, _, t in pos]
    x = [x if t > 100 else 0 for _, x, t in pos]
    t = [t for _, _, t in pos]
    plt.figure()
    plt.imshow((all_frames / total_count).astype('uint8'))
    plt.figure()


def save_to(file, pos):
    pd.DataFrame.from_dict({x: [x if t > 100 else 0 for y, x, t in pos],
                            y: [x if t > 100 else 0 for y, x, t in pos],
                            t: [x if t > 100 else 0 for _, x, t in pos]
                            }).to_csv(file.split('.')[0] + 'pos.csv')

