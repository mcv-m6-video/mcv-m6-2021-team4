import cv2
from os import path
import numpy as np
import matplotlib.pyplot as plt


def read_flow (of_file):
    im = cv2.imread(of_file, cv2.IMREAD_UNCHANGED).astype(np.double) #the decoded images will have the channels stored in B G R order.

    im_u = (im[:, :, 2] - 2 ** 15) / 64
    im_v = (im[:, :, 1] - 2 ** 15) / 64

    im_exists = im[:, :, 0]
    im_exists[im_exists > 1] = 1

    im_u[im_exists == 0] = 0
    im_v[im_exists == 0] = 0

    optical_flow = np.dstack((im_u, im_v, im_exists))

    return optical_flow

def compute_msen (ground_truth, predicted):
    u_diff = ground_truth[:,:,0] - predicted[:,:,0]
    v_diff = ground_truth[:,:,1] - predicted[:,:,1]
    
    se = np.sqrt(u_diff ** 2 + v_diff ** 2)
    sen = se[ground_truth[:,:,2]==1]

    se[ground_truth[:,:,2]==0] = 0 #To exclude the non valid
    img_plot = plt.imshow(se)
    img_plot.set_cmap("nipy_spectral")
    plt.colorbar()
    plt.show()

    # img_plot = plt.imshow(predicted)
    # img_plot.set_cmap("nipy_spectral")
    # plt.colorbar()
    # plt.show()

    msen = np.mean(sen)

    return msen,sen

def compute_pepn(ground_truth,predicted,sen,th=3):

    n_pixels_n= len(sen)

    error_count = np.sum(sen > th)

    pepn = (error_count/n_pixels_n) * 100

    return pepn


def task3_1_2(ground_truth_path,estimated_path,frame): 
    print("Task 3 - Quantitative evaluation of optical flow")

    ground_truth = read_flow(path.join(ground_truth_path, frame))
    estimated_flow = read_flow(path.join(estimated_path, "LKflow_" + frame))

    msen,sen = compute_msen(ground_truth,estimated_flow)

    pepn = compute_pepn(ground_truth,estimated_flow,sen)
    print(msen,pepn) # put the outputs nicer

    return msen,pepn,sen



def task3_3 (sen,frame):
    print("Task 3.3 - Visualization of Optical flow error")

    plt.hist(x=sen,bins=50)
    plt.savefig(frame)
    plt.clf()

if __name__ == "__main__":


    ground_truth_path = "../data/of_ground_truth"
    estimated_path = "../data/of_estimations_LK"
    frames = ["000045_10.png","000157_10.png"]


    for frame in frames:
        msen,pepn,sen = task3_1_2(ground_truth_path,estimated_path,frame)

        asfd = task3_3(sen,frame)