import cv2
from os import path
import numpy as np

def read_flow (of_file):
    print(of_file)
    im = cv2.imread(of_file, cv2.IMREAD_UNCHANGED).astype(np.double)

    im_u = (im[:, :, 2] - 2 ** 15) / 64
    im_v = (im[:, :, 1] - 2 ** 15) / 64

    im_exists = im[:, :, 0]
    im_exists[im_exists > 1] = 1

    im_u[im_exists == 0] = 0
    im_v[im_exists == 0] = 0

    optical_flow = np.dstack((im_u, im_v, im_exists))

    return optical_flow


def task3_1(): 
    print("Task 3 - Quantitative evaluation of optical flow")

    ground_truth_path = "../data/of_ground_truth"
    estimated_path = "../data/of_estimations_LK"
    frames = ["000045_10.png","000157_10.png"]

    for frame in frames:
        gt_flow = read_flow(path.join(ground_truth_path, frame))
        estimated_flow = read_flow(path.join(estimated_path, frame))
        print(gt_flow)
        print(estimated_flow)


if __name__ == "__main__":
    task3_1()