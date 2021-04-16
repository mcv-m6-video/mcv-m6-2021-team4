import sys
import numpy as np
from matplotlib import pyplot as plt

def plot_flow(img, flow, title='', step=4, fname=''):
    h, w = img.shape[:2]
    X, Y = np.meshgrid(np.arange(0, w, 1), np.arange(0, h, 1))
    M = np.hypot(flow[:, :, 0], flow[:, :, 1])
    plt.figure(figsize=(20,8))
    plt.quiver(X[::step, ::step], Y[::step, ::step], flow[::step, ::step, 0], flow[::step, ::step, 1],
               M[::step, ::step], width=0.002, scale=400.0)
    plt.colorbar()
    plt.title(title)
    plt.imshow(img, cmap='gray')
    plt.savefig(fname)
    plt.show()