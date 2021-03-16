import sys
import numpy as np
import argparse
from task1 import run as t1_run
from task2 import run as t2_run
from task3 import run as t3_run
from task4 import run as t4_run


def parse_args(args=sys.argv[1:]):
    parser = argparse.ArgumentParser(description='Video Surveillance for Road Traffic Monitoring. MCV-M6-Project, Team 4')

    parser.add_argument('--t1', action='store_true',
                        help='execute task 1: static method to estimate backround')

    parser.add_argument('--t2', action='store_true',
                        help='execute task 2: adaptive method to estimate backround')

    parser.add_argument('--t3', action='store_true',
                        help='execute task 3: SOTA methods')

    parser.add_argument('--t4', action='store_true',
                        help='execute task 4: color method to estimate background')

    parser.add_argument('--video_path', type=str, default="./data/vdo.avi",
                        help='path to video')

    parser.add_argument('--roi_path', type=str, default="./data/AICity_data/train/S03/c010/roi.jpg",
                        help='path to roi')

    parser.add_argument('--gt_path', type=str, default="./data/AICity_data/train/S03/c010/ai_challenge_s03_c010-full_annotation.xml",
                        help='path to annotations')

    parser.add_argument('--results_path', type=str, default="./W2/output/",
                        help='path to save results')

    parser.add_argument('--num_frames_eval', type=int, default=1606,
                        help='number of frames to evaluate')

    parser.add_argument('--show_boxes', action='store_true',
                        help='show bounding boxes')

    parser.add_argument('--save_results', action='store_true',
                        help='save detections')

    return parser.parse_args(args)

if __name__ == '__main__':
    args = parse_args()

    print(args)

    if args.t1:
        print('Executing task 1')

        alphas = np.linspace(0, 10, 21)
        for alpha in alphas:
            rec, prec, ap = t1_run(args, alpha=alpha)
            print(f'alpha: {alpha}, ap: {ap}')

    if args.t2:
        print('Executing task 2')
        t2_run(args, alpha=3, rho=0.05)

    if args.t3:
        print('Executing task 3')
        t3_run(args)

    if args.t4:
        print('Executing task 4')
        t4_run(args, bg_est='adaptive', alpha=3, rho=0.05, color_space='RGB', voting='simple')
