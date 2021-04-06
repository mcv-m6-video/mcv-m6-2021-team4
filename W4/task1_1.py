import sys, os,  argparse, time
import numpy as np
from PIL import Image
import pandas as pd
from itertools import product

from block_matching import estimate_flow
from flow_utils import plot_flow
sys.path.append('../W1')
from flow_evaluation import evaluate_flow

def parse_args(args=sys.argv[1:]):
    parser = argparse.ArgumentParser(description='Block matching algorithm to estimate optical flow')

    parser.add_argument('--motion_type', type=lambda s: [i for i in s.split(',')], default=['forward'],
                        help='motion type to use when estimating the optical flow')

    parser.add_argument('--block_size', type=lambda s: [int(i) for i in s.split(',')], default=[16],
                        help='size/s of the square blocks in which the image is divided (N)')

    parser.add_argument('--search_area', type=lambda s: [int(i) for i in s.split(',')], default=[32],
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
            title='Motion type: ' + motion_type + ', Block size: ' + str(N) + ', Search area: ' + str(P) + ', Distance metric: ' + distance_metric
            plot_flow(img_reference, flow, title)

        msen, pepn = evaluate_flow(flow, args.gt_path, args.plot_flow)

        log.append([motion_type, N, P, distance_metric, msen, pepn, end-start])

    df = pd.DataFrame(log, columns=['motion_type', 'block_size', 'search_area', 'distance_metric', 'msen', 'pepn', 'runtime'])
    print(df)
    if args.save_results:
        df.to_csv(args.results_path, index=False)