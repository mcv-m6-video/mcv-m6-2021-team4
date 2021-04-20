import sys, argparse
import pandas as pd
from siamese_network import train, get_data_loader


def parse_args(args=sys.argv[1:]):
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt_csv', type=str, default='./W5/gt/gt_car_patches_annotations.csv',
                        help='path to gt csv containing annotations')
    parser.add_argument('--gt_patches', type=str, default='./gt/gt_car_patches/',
                        help='path to gt folder containing car ptches')
    parser.add_argument('--save_model', type=str, default=None,
                        help='path to save trained model')
    return parser.parse_args(args)


if __name__ == '__main__':
    args = parse_args()

    labels = pd.read_csv(args.gt_csv)

    train_data, train_loader = get_data_loader('train', labels, args.gt_patches)
    # test_data, test_loader = get_data_loader('test', labels, args.gt_patches)

    train(train_data, args.save_model)

