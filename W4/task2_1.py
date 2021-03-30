import os, sys, cv2, pickle, argparse
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
from scipy.signal import savgol_filter
from block_matching import estimate_flow
from flow_utils import plot_flow

sys.path.append("../W1")
from utils import save_gif

def parse_args(args=sys.argv[1:]):
    parser = argparse.ArgumentParser(description='Video stabilization using optical flow (block matching)')

    parser.add_argument('--video_path', type=str, default='../data/video_stabilization/oscar_pc4.mp4')

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()

    cap = cv2.VideoCapture(args.video_path)
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    compute_acc_motions = False  # caution: this is slow (max 5-6 sec video!!!)
    visualize_flow = False
    stabilize_video = True
    smooth_trajectories = False
    visualize_stabilization = True
    plot_corrections = True
    create_gifs = False

    pkl_path = './pkls'

    if compute_acc_motions:
        acc_mean_x_motion = [0]
        acc_mean_y_motion = [0]

        acc_median_x_motion = [0]
        acc_median_y_motion = [0]

        for frame_id in range(num_frames-1):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
            _, frame_prev = cap.read()
            _, frame_next = cap.read()

            flow = estimate_flow(motion_type='forward', N=16, P=32, distance_metric='ncc',
                                 img_reference=frame_prev, img_current=frame_next)

            if visualize_flow:
                plot_flow(frame_prev, flow)
                # sleep or something

            acc_mean_x_motion.append(acc_mean_x_motion[frame_id] + np.mean(flow[:, :, 0]))
            acc_mean_y_motion.append(acc_mean_y_motion[frame_id] + np.mean(flow[:, :, 1]))

            acc_median_x_motion.append(acc_median_x_motion[frame_id] + np.median(flow[:, :, 0]))
            acc_median_y_motion.append(acc_median_y_motion[frame_id] + np.median(flow[:, :, 1]))

            print(frame_id)

        with open(os.path.join(pkl_path, 'acc_mean_y_motion.pkl'), 'wb') as f:
            pickle.dump(acc_mean_y_motion, f)

        with open(os.path.join(pkl_path, 'acc_mean_x_motion.pkl'), 'wb') as f:
            pickle.dump(acc_mean_x_motion, f)

        with open(os.path.join(pkl_path, 'acc_median_y_motion.pkl'), 'wb') as f:
            pickle.dump(acc_median_y_motion, f)

        with open(os.path.join(pkl_path, 'acc_median_x_motion.pkl'), 'wb') as f:
            pickle.dump(acc_median_x_motion, f)


    if stabilize_video:
        trajectory_type = 'median'  # median or mean
        with open(os.path.join(pkl_path, 'acc_'+trajectory_type+'_y_motion.pkl'), 'rb') as f:
            acc_y_motions = pickle.load(f)

        with open(os.path.join(pkl_path, 'acc_'+trajectory_type+'_x_motion.pkl'), 'rb') as f:
            acc_x_motions = pickle.load(f)

        if smooth_trajectories:
            acc_y_motions = savgol_filter(acc_y_motions, 51, 3)
            acc_x_motions = savgol_filter(acc_x_motions, 51, 3)

        if plot_corrections:
            plt.figure(figsize=(20,8))
            plt.title('Smoothed (SavGol filter) horizontal and vertical corrections due to accumulated ' + trajectory_type + ' motions')
            plt.plot(-np.array(acc_x_motions), label='horizontal motion')
            plt.plot(-np.array(acc_y_motions), label='vertical motion')
            plt.ylabel('Correction (pxs)')
            plt.xlabel('Frame')
            plt.legend()
            plt.show()

        start_flag = True

        first = True

        for frame_id in tqdm(range(num_frames), desc='Stabilizing video frames'):
            _, frame = cap.read()

            transl_matrix = np.array([[1, 0, -acc_x_motions[frame_id]],
                                      [0, 1, -acc_y_motions[frame_id]]],
                                     dtype=np.float32)

            frame_stab = cv2.warpAffine(frame, transl_matrix, (frame.shape[1], frame.shape[0]))

            s = frame_stab.shape
            T = cv2.getRotationMatrix2D((s[1]/2, s[0]/2), 0, 1.3)
            frame = cv2.warpAffine(frame, T, (s[1], s[0]))
            frame_stab = cv2.warpAffine(frame_stab, T, (s[1], s[0]))

            # mask = np.where(frame_stab==[0,0,0], 1, 0).astype(np.uint8)
            #
            # if first:
            #     frame_stab = frame_stab + frame*mask
            #     first = False
            # else:
            #     frame_stab = frame_stab + prev_frame*mask
            #
            # prev_frame = frame_stab

            if visualize_stabilization:
                cv2.namedWindow('prev', cv2.WINDOW_NORMAL)
                cv2.resizeWindow('prev', 1000, 800)
                cv2.imshow('prev', frame)

                cv2.namedWindow('stabilized', cv2.WINDOW_NORMAL)
                cv2.resizeWindow('stabilized', 1000, 800)
                cv2.imshow('stabilized', frame_stab)

                if start_flag:
                    cv2.waitKey()
                    start_flag = False
                else:
                    cv2.waitKey(10)

            if create_gifs:
                scale = 0.3
                w = int(frame.shape[1] * scale)
                h = int(frame.shape[0] * scale)
                dim = (w, h)

                cv2.imwrite('./gif_original/'+str(frame_id+100)+'.png', cv2.resize(frame, dim, interpolation=cv2.INTER_AREA))
                cv2.imwrite('./gif_stab/' + str(frame_id + 100) + '.png', cv2.resize(frame_stab, dim, interpolation=cv2.INTER_AREA))

        if create_gifs:
            print('Creating GIFs')
            save_gif('./gif_original/', './gif_original.gif', fps=30)
            save_gif('./gif_stab/', './gif_stab.gif', fps=30)