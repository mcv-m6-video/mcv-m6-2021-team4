import sys
from PIL import Image
import time
import numpy as np
from os import path
sys.path.append("W1")
import pyflow.pyflow as pyflow
from flow_evaluation import compute_pepn,compute_msen
from flow_reader import read_flow

im1 = np.array(Image.open('data/000045_10.png'))
im2 = np.array(Image.open('data/000045_11.png'))
im1 = im1.astype(float) / 255.
im2 = im2.astype(float) / 255.

# Flow Options:
alpha = 0.012
ratio = 0.75
minWidth = 20
nOuterFPIterations = 7
nInnerFPIterations = 1
nSORIterations = 30
colType = 0  # 0 or default:RGB, 1:GRAY (but pass gray image with shape (h,w,1))




s = time.time()
u, v, im2W = pyflow.coarse2fine_flow(
    im1, im2, alpha, ratio, minWidth, nOuterFPIterations, nInnerFPIterations,
    nSORIterations, colType)
e = time.time()
print('Time Taken: %.2f seconds for image of size (%d, %d, %d)' % (
    e - s, im1.shape[0], im1.shape[1], im1.shape[2]))
flow = np.concatenate((u[..., None], v[..., None]), axis=2)
print(flow.shape)
np.save('W4/outFlow.npy', flow)

if True:
    import cv2
    # hsv = np.zeros(im1.shape, dtype=np.uint8)
    # hsv[:, :, 0] = 255
    # hsv[:, :, 1] = 255
    # mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    # hsv[..., 0] = ang * 180 / np.pi / 2
    # hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    # rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    # cv2.imwrite('W4/outFlow_new.png', flow[:,:,0])
    # cv2.imwrite('W4/outFlow_new1.png', flow[:,:,1])
    cv2.imwrite('W4/outFlow_new.png', u)
    cv2.imwrite('W4/outFlow_new1.png', v)


gt_path = "data/of_ground_truth"
frame = "000045_10.png"

def task3_1_2(gt_path, flow, frame):
    print("Task 3 - Quantitative evaluation of optical flow")

    gt = read_flow(path.join(gt_path, frame))
    estimated_flow = flow

    msen, sen = compute_msen(gt, estimated_flow)
    pepn = compute_pepn(gt, estimated_flow, sen)

    print(msen, pepn) # put the outputs nicer

    return msen, pepn, sen


msen, pepn, sen = task3_1_2(gt_path, flow, frame)
