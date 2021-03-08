import matplotlib
import matplotlib.pyplot as plt

import numpy as np
from os import path
import cv2
import task3
from numpy import ma

data_path = "../data/"
ground_truth_path = "../data/of_ground_truth"
frames = ["000045_10.png","000157_10.png"]

for frame in frames:

    ground_truth = task3.read_flow(path.join(ground_truth_path, frame))

    X, Y = np.meshgrid(np.arange(0,ground_truth.shape[1],1), np.arange(0,ground_truth.shape[0],1))

    originalImage = cv2.imread(path.join(data_path, frame))
    grayImage = cv2.cvtColor(originalImage, cv2.COLOR_BGR2GRAY)

    U = ground_truth[:,:,0] 
    V = ground_truth[:,:,1]


    plt.figure()
    plt.title("pivot='tip'; scales with x view")
    M = np.hypot(U, V)
    Q = plt.quiver(X[::5, ::5], Y[::5, ::5], U[::5, ::5], V[::5, ::5], M[::5, ::5], units='x', pivot='tail', width=0.9,
                scale=0.5)
    qk = plt.quiverkey(Q, 0.9, 0.9, 1, r'$1 \frac{m}{s}$', labelpos='E',
                    coordinates='data')
    
    plt.imshow(grayImage,cmap='gray')                  

    plt.show()
    # plt.savefig('task4'+frame)