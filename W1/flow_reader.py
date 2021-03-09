import cv2
import numpy as np


def read_flow(flow_file):
    # channels: BGR
    im = cv2.imread(flow_file, cv2.IMREAD_UNCHANGED).astype(np.double)

    # Optical flow maps are saved as 3-channel uint16 PNG images: The first channel
    # contains the u-component, the second channel the v-component and the third
    # channel denotes if a valid ground truth optical flow value exists for that
    # pixel (1 if true, 0 otherwise). To convert the u-/v-flow into floating point
    # values, convert the value to float, subtract 2^15 and divide the result by 64.

    im_u = (im[:, :, 2] - 2 ** 15) / 64
    im_v = (im[:, :, 1] - 2 ** 15) / 64

    im_exists = im[:, :, 0]
    im_exists[im_exists > 1] = 1

    im_u[im_exists == 0] = 0
    im_v[im_exists == 0] = 0

    optical_flow = np.dstack((im_u, im_v, im_exists))

    return optical_flow
