import os, sys, time, cv2, argparse
from PIL import Image
import numpy as np
sys.path.append("../W1")
import pyflow.pyflow as pyflow
from flow_evaluation import evaluate_flow
from flow_utils import plot_flow

sys.path.append("MaskFlownet")
from predict_new_data import flow_maskflownet

sys.path.append("RAFT")
from demo import flow_raft


def flow_LK(img_prev, img_next):

    img_prev = cv2.cvtColor(img_prev, cv2.COLOR_BGR2GRAY)
    img_next = cv2.cvtColor(img_next, cv2.COLOR_BGR2GRAY)

    # Parameters for lucas kanade optical flow
    lk_params = dict(winSize=(15, 15),
                     maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    # Take all pixels
    height, width = img_prev.shape[:2]
    p0 = np.array([[x, y] for y in range(height) for x in range(width)], dtype=np.float32).reshape((-1, 1, 2))

    start = time.time()
    p1, st, err = cv2.calcOpticalFlowPyrLK(img_prev, img_next, p0, None, **lk_params)
    end = time.time()
    p0 = p0.reshape((height, width, 2))
    p1 = p1.reshape((height, width, 2))
    st = st.reshape((height, width))

    flow = p1 - p0
    flow[st == 0] = 0

    return flow, end-start


def flow_pyflow(img_prev, img_next):
    img_prev = img_prev.astype(float) / 255.
    img_next = img_next.astype(float) / 255.

    # Flow Options:
    alpha = 0.012
    ratio = 0.75
    minWidth = 20
    nOuterFPIterations = 7
    nInnerFPIterations = 1
    nSORIterations = 30
    colType = 0  # 0 or default:RGB, 1:GRAY (but pass gray image with shape (h,w,1))

    start = time.time()
    u, v, _ = pyflow.coarse2fine_flow(
        img_prev, img_next, alpha, ratio, minWidth, nOuterFPIterations, nInnerFPIterations,
        nSORIterations, colType)
    end = time.time()
    flow = np.concatenate((u[..., None], v[..., None]), axis=2)

    return flow, end-start


estimate_flow = {
    'pyflow': flow_pyflow,
    'LK': flow_LK,
    'maskflownet': flow_maskflownet,
    'raft': flow_raft
}


def parse_args(args=sys.argv[1:]):
    parser = argparse.ArgumentParser(description='Block matching algorithm to estimate optical flow')

    parser.add_argument('--gt_path', type=str, default='../data/of_ground_truth/000045_10.png',
                        help='path to ground truth optical flow')

    parser.add_argument('--data_path', type=str, default='../data/OF_color',
                        help='path to folder containing images of interest')

    parser.add_argument('--plot_flow', action='store_true',
                        help='plot optical flow and error')

    parser.add_argument('--method', type=str, default='LK',
                        choices=['pyflow', 'LK', 'maskflownet', 'raft'],
                        help='')

    return parser.parse_args(args)

from skimage import color

if __name__ == '__main__':
    args = parse_args()

    img_prev = np.array(Image.open(os.path.join(args.data_path, '000045_10.png')))
    img_next = np.array(Image.open(os.path.join(args.data_path, '000045_11.png')))

    flow, runtime = estimate_flow[args.method](img_prev, img_next)

    if args.plot_flow:
        title = args.method + ' optical flow estimation'
        plot_flow(color.rgb2gray(img_prev), flow, title)

    msen, pepn = evaluate_flow(flow, args.gt_path, args.plot_flow)

    print('MSEN: ', msen, ', PEPN(%): ', pepn, ', runtime(s): ', runtime)