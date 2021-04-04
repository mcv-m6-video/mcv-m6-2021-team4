import sys
from PIL import Image
import time
import numpy as np
import matplotlib.pyplot as plt
from os import path
sys.path.append("W1")
import pyflow.pyflow as pyflow
from flow_evaluation import compute_pepn,compute_msen
from flow_reader import read_flow
import cv2


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

gt_path = "data/of_ground_truth"
frame = "000045_10.png"

gt = read_flow(path.join(gt_path, frame))

X, Y = np.meshgrid(np.arange(0, gt.shape[1], 1), np.arange(0, gt.shape[0], 1))

originalImage = cv2.imread('data/000045_10.png')
grayImage = cv2.cvtColor(originalImage, cv2.COLOR_BGR2GRAY)



print(originalImage.shape)
windowSize = 50

u_new = np.zeros([originalImage.shape[0], originalImage.shape[1]])
v_new = np.zeros([originalImage.shape[0], originalImage.shape[1]])

print(u_new.shape)
print(v_new.shape)

for i in range(0, originalImage.shape[0]-windowSize, windowSize):
    for j in range(0, originalImage.shape[1]-windowSize, windowSize):
        print(i, j)

        u_mean = np.mean(u[i:i+windowSize, j:j+windowSize])
        v_mean = np.mean(v[i:i+windowSize, j:j+windowSize])
        
        u_new[i+int(windowSize/2), int(j+windowSize/2)] = u_mean
        v_new[i+int(windowSize/2), int(j+windowSize/2)] = v_mean
        print("MEAN: ", u_mean, v_mean)

plt.figure()
plt.title("pivot='tip'; scales with x view")
M = np.hypot(u_new, v_new)
Q = plt.quiver(X[::5, ::5], Y[::5, ::5], u_new[::5, ::5], v_new[::5, ::5], M[::5, ::5],
                units='x', pivot='tail', width=3, scale=0.5)
qk = plt.quiverkey(Q, 0.9, 0.9, 1, r'$1 \frac{m}{s}$',
                    labelpos='E', coordinates='data')
# plt.colorbar()
plt.imshow(grayImage, cmap='gray')
# plt.savefig('task4_opt2_'+frame)    
plt.show()




if True:
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




def task3_1_2(gt_path, flow, frame):
    print("Task 3 - Quantitative evaluation of optical flow")

    gt = read_flow(path.join(gt_path, frame))
    estimated_flow = flow

    msen, sen = compute_msen(gt, estimated_flow)
    pepn = compute_pepn(gt, estimated_flow, sen)

    print(msen, pepn) # put the outputs nicer

    return msen, pepn, sen


msen, pepn, sen = task3_1_2(gt_path, flow, frame)

