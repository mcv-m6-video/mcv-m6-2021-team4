import os
import sys
import argparse
from task1 import run as t1_run
from task2 import run as t2_run
from task3 import run as t3_run
from task4 import run as t4_run


def parse_args(args=sys.argv[1:]):
    parser = argparse.ArgumentParser(description='Video Surveillance for Road Traffic Monitoring. MCV-M6-Project, Team 4')

    parser.add_argument('--t1', action='store_true',
                        help='execute task 1: generate noisy boxes from annotations and compute AP/mIoU')

    parser.add_argument('--t2', action='store_true',
                        help='execute task 2: compute AP/mIoU vs frame (temporal) for a specific detector')

    parser.add_argument('--t3', action='store_true',
                        help='execute task 3: compute MSEN, PEPN, and visualize the errors')

    parser.add_argument('--t4', action='store_true',
                        help='execute task 4: visualize optial flow (two methods)')

    return parser.parse_args(args)

if __name__ == '__main__':
    args = parse_args()

    print(args)

    if args.t1:
        print('Executing task 1')
        t1_run()

    if args.t2:
        print('Executing task 2')
        t2_run()

    if args.t3:
        print('Executing task 3')
        t3_run()

    if args.t4:
        print('Executing task 4')
        t4_run()
