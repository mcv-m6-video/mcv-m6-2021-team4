from os import path
import cv2
import numpy as np
import matplotlib.pyplot as plt
from flow_reader import read_flow

if __name__ == '__main__':

    data_path = "../data/"
    gt_path = "../data/of_ground_truth"
    frames = ["000045_10.png", "000157_10.png"]

    for frame in frames:

        gt = read_flow(path.join(gt_path, frame))

        X, Y = np.meshgrid(np.arange(0, gt.shape[1], 1),
                           np.arange(0, gt.shape[0], 1))

        originalImage = cv2.imread(path.join(data_path, frame))
        grayImage = cv2.cvtColor(originalImage, cv2.COLOR_BGR2GRAY)

        U = gt[:, :, 0]
        V = gt[:, :, 1]

        plt.figure()
        plt.title("pivot='tip'; scales with x view")
        M = np.hypot(U, V)
        Q = plt.quiver(X[::5, ::5], Y[::5, ::5], U[::5, ::5], V[::5, ::5], M[::5, ::5],
                       units='x', pivot='tail', width=0.9, scale=0.5)
        qk = plt.quiverkey(Q, 0.9, 0.9, 1, r'$1 \frac{m}{s}$',
                           labelpos='E', coordinates='data')

        plt.imshow(grayImage, cmap='gray')

        plt.show()
        # plt.savefig('task4'+frame)