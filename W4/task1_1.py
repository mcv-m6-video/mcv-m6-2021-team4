import sys, os, cv2, argparse, time
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from tqdm import tqdm
import pandas as pd
from itertools import product

sys.path.append('../W1')
from flow_reader import read_flow
from flow_evaluation import compute_msen, compute_pepn


def plot_flow(img, flow, step=16):
    h, w = img.shape[:2]
    X, Y = np.meshgrid(np.arange(0, w, 1), np.arange(0, h, 1))
    M = np.hypot(flow[:, :, 0], flow[:, :, 1])
    plt.figure(figsize=(20,8))
    plt.quiver(X[::step, ::step], Y[::step, ::step], flow[::step, ::step, 0], flow[::step, ::step, 1],
               M[::step, ::step], width=0.001)
    plt.colorbar()
    plt.imshow(img, cmap='gray')
    plt.show()


def distance(blockA, blockB, distance_metric='ncc'):
    if distance_metric == 'sad':
        return np.sum(np.abs(blockA-blockB))

    elif distance_metric == 'ssd':
        return np.sum((blockA-blockB)**2)

    elif distance_metric == 'ncc':
        return -cv2.matchTemplate(blockA, blockB, method=cv2.TM_CCORR_NORMED).squeeze()

    else:
        raise ValueError('Unknown distance metric')

def estimate_flow_block(N, distance_metric, blocks_positions, img_reference, img_current):
    tlx_ref = blocks_positions['tlx_ref']
    tly_ref = blocks_positions['tly_ref']
    init_tlx_curr = blocks_positions['init_tlx_curr']
    init_tly_curr = blocks_positions['init_tly_curr']
    end_tlx_curr = blocks_positions['end_tlx_curr']
    end_tly_curr = blocks_positions['end_tly_curr']

    if distance_metric == 'ncc':
        corr = cv2.matchTemplate(img_current[init_tly_curr:end_tly_curr + N, init_tlx_curr:end_tlx_curr + N],
                                 img_reference[tly_ref:tly_ref + N, tlx_ref:tlx_ref + N],
                                 method=cv2.TM_CCORR_NORMED)

        motion = list(np.unravel_index(np.argmax(corr), corr.shape))
        motion[0] = motion[0] + init_tly_curr - tly_ref
        motion[1] = motion[1] + init_tlx_curr - tlx_ref

        return [motion[1], motion[0]]

    else:
        min_dist = np.inf

        for tly_curr in np.arange(init_tly_curr, end_tly_curr):
            for tlx_curr in np.arange(init_tlx_curr, end_tlx_curr):

                dist = distance(img_reference[tly_ref:tly_ref + N, tlx_ref:tlx_ref + N],
                                img_current[tly_curr:tly_curr + N, tlx_curr:tlx_curr + N],
                                distance_metric)

                if dist < min_dist:
                    min_dist = dist
                    motion = [tlx_curr - tlx_ref, tly_curr - tly_ref]

        return motion


def estimate_flow(motion_type, N, P, distance_metric, img_reference, img_current):
    h, w = img_reference.shape[:2]
    flow = np.zeros(shape=(h, w, 2))

    for tly_ref in tqdm(range(0, h - N, N), desc='Motion estimation'):
        for tlx_ref in range(0, w - N, N):

            blocks_positions = {
                'tlx_ref': tlx_ref,
                'tly_ref': tly_ref,
                'init_tlx_curr': max(tlx_ref - P, 0),
                'init_tly_curr': max(tly_ref - P, 0),
                'end_tlx_curr': min(tlx_ref + P, w - N),
                'end_tly_curr': min(tly_ref + P, h - N)
            }

            flow[tly_ref:tly_ref + N, tlx_ref:tlx_ref + N, :] = estimate_flow_block(N, distance_metric, blocks_positions, img_reference, img_current)

    if motion_type == 'backward':
        flow = -flow

    return flow


def evaluate_flow(flow, gt_path, plot_error):
    gt_flow = read_flow(gt_path)

    msen, sen = compute_msen(gt_flow, flow, debug=False, visualize=plot_error)
    pepn = compute_pepn(gt_flow, flow, sen)

    return msen, pepn


def parse_args(args=sys.argv[1:]):
    parser = argparse.ArgumentParser(description='Block matching algorithm to estimate optical flow')

    parser.add_argument('--motion_type', type=lambda s: [i for i in s.split(',')], default=['forward'],
                        help='motion type to use when estimating the optical flow')

    parser.add_argument('--block_size', type=lambda s: [int(i) for i in s.split(',')], default=[16],
                        help='size/s of the square blocks in which the image is divided (N)')

    parser.add_argument('--search_area', type=lambda s: [int(i) for i in s.split(',')], default=[16],
                        help='related to the range of expected movement, number/s of pixels in every direction (P)')

    parser.add_argument('--distance_metric', type=lambda s: [i for i in s.split(',')], default=['ncc'],
                        help='distance metric used to match blocks (sad, ssd, ncc)')

    parser.add_argument('--gt_path', type=str, default='../data/of_ground_truth/000045_10.png',
                        help='path to ground truth optical flow')

    parser.add_argument('--data_path', type=str, default='../data/OF',
                        help='path to folder containing images of interest')

    parser.add_argument('--plot_flow', action='store_true',
                        help='plot optical flow and error')

    parser.add_argument('--save_results', action='store_true',
                        help='save results into csv')

    parser.add_argument('--results_path', type=str, default='./csv/task1_1_grid_search.csv',
                        help='path to save csv file with results')

    return parser.parse_args(args)


if __name__ == '__main__':
    args = parse_args()

    img_prev = np.array(Image.open(os.path.join(args.data_path, '000045_10.png')))
    img_next = np.array(Image.open(os.path.join(args.data_path, '000045_11.png')))

    log = []

    for motion_type, N, P, distance_metric in product(args.motion_type, args.block_size, args.search_area, args.distance_metric):

        if motion_type == 'forward':
            img_reference = img_prev
            img_current = img_next

        elif motion_type == 'backward':
            img_reference = img_next
            img_current = img_prev

        else:
            raise ValueError('Unkown motion type')

        start = time.time()
        flow = estimate_flow(motion_type, N, P, distance_metric, img_reference, img_current)
        end = time.time()

        if args.plot_flow:
            plot_flow(img_reference, flow)

        msen, pepn = evaluate_flow(flow, args.gt_path, args.plot_flow)

        log.append([motion_type, N, P, distance_metric, msen, pepn, end-start])

    df = pd.DataFrame(log, columns=['motion_type', 'block_size', 'search_area', 'distance_metric', 'msen', 'pepn', 'runtime'])
    print(df)
    if args.save_results:
        df.to_csv(args.results_path, index=False)