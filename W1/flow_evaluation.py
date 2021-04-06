import numpy as np
import matplotlib.pyplot as plt

import cv2

def compute_msen(ground_truth, predicted, debug=True, visualize=True):
    """
    Mean Square Error in Non-occluded areas
    """

    u_diff = ground_truth[:, :, 0] - predicted[:, :, 0]
    v_diff = ground_truth[:, :, 1] - predicted[:, :, 1]
    se = np.sqrt(u_diff ** 2 + v_diff ** 2)
    sen = se[ground_truth[:, :, 2] == 1]
    msen = np.mean(sen)

    if debug:
        print(ground_truth[0, -1])
        print(predicted[0, -1])
        print(u_diff[0, -1])
        print(v_diff[0, -1])
        print(se[0, -1])

    if visualize:
        se[ground_truth[:, :, 2] == 0] = 0  # To exclude the non valid
        plt.figure(figsize=(20,8))
        img_plot = plt.imshow(se)
        img_plot.set_cmap("nipy_spectral")
        plt.title("Squared Error")
        plt.colorbar()
        plt.show()

        predicted, _ = cv2.cartToPolar(predicted[:, :, 0], predicted[:, :, 1])
        plt.figure(figsize=(20, 8))
        img_plot = plt.imshow(predicted)
        img_plot.set_cmap("nipy_spectral")
        plt.title("Predicted Optical flow")
        plt.colorbar()
        plt.show()

    return msen, sen


def compute_pepn(ground_truth, predicted, sen, th=3):
    """
    Percentage of Erroneous Pixels in Non-occluded areas
    """

    n_pixels_n = len(sen)

    error_count = np.sum(sen > th)

    pepn = (error_count / n_pixels_n) * 100

    return pepn