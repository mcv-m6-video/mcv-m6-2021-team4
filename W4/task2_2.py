import sys, argparse
from vidstab import VidStab
import matplotlib.pyplot as plt


def vidstab(input_path, output_path, kp_method='GFTT', smoothing_window=30, border_type='black', border_size=0):
    # stabilizer = VidStab(kp_method='FAST', threshold=42, nonmaxSuppression=False)  # Using different parameters
    stabilizer = VidStab(kp_method=kp_method)  # Default
    stabilizer.stabilize(input_path=input_path,
                         output_path=output_path,
                         smoothing_window=smoothing_window,
                         border_type=border_type,
                         border_size=border_size)

    stabilizer.plot_trajectory()
    plt.show()

    stabilizer.plot_transforms()
    plt.show()
    return


def parse_args(args=sys.argv[1:]):
    parser = argparse.ArgumentParser(description='Video stabilization')
    parser.add_argument('--method', type=str, default='vidstab',
                        choices=['vidstab'],
                        help='method used to stabilize video')

    parser.add_argument('--input_path', type=str, default='../data/video_stabilization/oscar_pc4.mp4',
                        help='path to unstable video')

    parser.add_argument('--output_path', type=str, default='./results/vidstab/open_pc4_stabilized.mp4',
                            help='path to save stabilized video')

    return parser.parse_args(args)


video_stabilizer = {
    'vidstab': vidstab
}


if __name__ == '__main__':
    args=parse_args()
    video_stabilizer[args.method](args.input_path, args.output_path)



