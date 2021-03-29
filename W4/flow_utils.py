import sys
import numpy as np
from matplotlib import pyplot as plt

sys.path.append('../W1')
from flow_reader import read_flow
from flow_evaluation import compute_msen, compute_pepn


def plot_flow(img, flow, title, step=16):
    h, w = img.shape[:2]
    X, Y = np.meshgrid(np.arange(0, w, 1), np.arange(0, h, 1))
    M = np.hypot(flow[:, :, 0], flow[:, :, 1])
    plt.figure(figsize=(20,8))
    plt.quiver(X[::step, ::step], Y[::step, ::step], flow[::step, ::step, 0], flow[::step, ::step, 1],
               M[::step, ::step], width=0.001)
    plt.colorbar()
    plt.title(title)
    plt.imshow(img, cmap='gray')
    plt.show()


def evaluate_flow(flow, gt_path, plot_error):
    gt_flow = read_flow(gt_path)

    msen, sen = compute_msen(gt_flow, flow, debug=False, visualize=plot_error)
    pepn = compute_pepn(gt_flow, flow, sen)

    return msen, pepn