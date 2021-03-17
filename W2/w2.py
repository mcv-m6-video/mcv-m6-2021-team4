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

    parser.add_argument('--alpha', type=float, default=3,
                        help='alpha parameter')

    parser.add_argument('--rho', type=float, default=0.03,
                        help='rho parameter')

    parser.add_argument('--show_boxes', action='store_true',
                        help='show bounding boxes')

    parser.add_argument('--color_space', type=str, default='grayscale',
                        help='color space')

    parser.add_argument('--voting', type=str, default='unanimous',
                        help='voting method')

    parser.add_argument('--save_results', action='store_true',
                        help='save detections')

    parser.add_argument('--sota_method', type=str, default="MOG2",
                        choices=['MOG', 'MOG2', 'LSBP', 'KNN', 'GSOC'],
                        help='State of the art method for Background Substraction task')

    return parser.parse_args(args)

if __name__ == '__main__':
    args = parse_args()

    print(args)

    if args.t1:
        print('Executing task 1')
        ap = t1_run(args, alpha=args.alpha)
        print(f'alpha: {args.alpha}, ap: {ap}')

    if args.t2:
        print('Executing task 2')
        ap = t2_run(args, alpha=args.alpha, rho=args.rho)
        print(f'alpha: {args.alpha}, rho: {args.rho}, ap: {ap}')

    if args.t3:
        print('Executing task 3')
        ap = t3_run(args)
        print(f'sota_method: {args.sota_method}, ap: {ap}')

    if args.t4:
        print('Executing task 4')
        bg_est = 'adaptive'
        ap = t4_run(args, bg_est=bg_est, alpha=args.alpha, rho=args.rho,
                    color_space=args.color_space, voting=args.voting)
        print(f'bg_est: {bg_est}, alpha: {args.alpha}, rho: {args.rho}, color_space: {args.color_space}, voting: {args.voting}, ap: {ap}')
