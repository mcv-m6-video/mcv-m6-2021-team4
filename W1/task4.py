from os import path
import cv2
import numpy as np
import matplotlib.pyplot as plt
from flow_reader import read_flow


def option1():

    data_path = "../data/"
    gt_path = "../data/of_ground_truth"
    frames = ["000045_10.png", "000157_10.png"]

    gt_path = "../data/of_estimations_LK"
    frames = ["LKflow_000045_10.png", "LKflow_000157_10.png"]

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
                       units='x', pivot='tail', width=0.9, scale=0.05)
        qk = plt.quiverkey(Q, 0.9, 0.9, 1, r'$1 \frac{m}{s}$',
                           labelpos='E', coordinates='data')

        plt.colorbar()
        plt.imshow(grayImage, cmap='gray')

        plt.show()
        # plt.savefig('task4'+frame)

def option2():
    data_path = "../data/"
    # gt_path = "../data/of_ground_truth"
    # frames = ["000045_10.png", "000157_10.png"]
    gt_path = "../data/of_estimations_LK"
    frames = ["LKflow_000045_10.png", "LKflow_000157_10.png"]

    for frame in frames:

        gt = read_flow(path.join(gt_path, frame))
        X, Y = np.meshgrid(np.arange(0, gt.shape[1], 1), np.arange(0, gt.shape[0], 1))

        originalImage = cv2.imread(path.join(data_path, frame))
        grayImage = cv2.cvtColor(originalImage, cv2.COLOR_BGR2GRAY)


        U = gt[:, :, 0]
        V = gt[:, :, 1]

        print(originalImage.shape)
        windowSize = 50

        u_new = np.zeros([originalImage.shape[0], originalImage.shape[1]])
        v_new = np.zeros([originalImage.shape[0], originalImage.shape[1]])

        print(u_new.shape)
        print(v_new.shape)

        for i in range(0, originalImage.shape[0]-windowSize, windowSize):
            for j in range(0, originalImage.shape[1]-windowSize, windowSize):
                print(i, j)

                u_mean = np.mean(U[i:i+windowSize, j:j+windowSize])
                v_mean = np.mean(V[i:i+windowSize, j:j+windowSize])
                
                u_new[i+int(windowSize/2), int(j+windowSize/2)] = u_mean
                v_new[i+int(windowSize/2), int(j+windowSize/2)] = v_mean
                print("MEAN: ", u_mean, v_mean)

        plt.figure()
        plt.title("pivot='tip'; scales with x view")
        M = np.hypot(u_new, v_new)
        Q = plt.quiver(X[::5, ::5], Y[::5, ::5], u_new[::5, ::5], v_new[::5, ::5], M[::5, ::5],
                        units='x', pivot='tail', width=3, scale=0.009)
        qk = plt.quiverkey(Q, 0.9, 0.9, 1, r'$1 \frac{m}{s}$',
                            labelpos='E', coordinates='data')
        # plt.colorbar()
        plt.imshow(grayImage, cmap='gray')
        # plt.savefig('task4_opt2_'+frame)    
        plt.show()
    




if __name__ == '__main__':
    option1()
    # option2()
