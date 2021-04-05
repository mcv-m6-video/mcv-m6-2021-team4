import numpy as np
import cv2
import sys
sys.path.append("W1")
from flow_evaluation import compute_msen, compute_pepn
from flow_reader import read_flow
import matplotlib.pyplot as plt


compute_LK_OF(path_image1, path_image2, path_gt)
gt = read_flow("data/of_ground_truth/000045_10.png")

im1 = cv2.imread('data/000045_10.png')
im2 = cv2.imread('data/000045_11.png')

im1 = cv2.imread('data/vlcsnap-2021-04-04-19h56m36s549.png')
im2 = cv2.imread('data/vlcsnap-2021-04-04-19h56m38s394.png')

im1_gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
im2_gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

# params for ShiTomasi corner detection
feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )

# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))


# Create some random colors
color = np.random.randint(0,255,(100,3))

# Take first frame and find corners in it
# p0 = cv2.goodFeaturesToTrack(im1_gray, mask = None, **feature_params)

# Take all pixels
height, width = im1_gray.shape[:2]
p0 = np.array([[x, y] for y in range(height) for x in range(width)], dtype=np.float32).reshape((-1, 1, 2))

# Create a mask image for drawing purposes
mask = np.zeros_like(im1_gray)

# calculate optical flow
p1, st, err = cv2.calcOpticalFlowPyrLK(im1_gray, im2_gray, p0, None, **lk_params)
p0 = p0.reshape((height, width, 2))
p1 = p1.reshape((height, width, 2))
st = st.reshape((height, width))

flow = p1 - p0
flow[st == 0] = 0
# flow = np.reshape(flow, (im1.shape[0],im1.shape[1],2))

# msen, sen = compute_msen(gt, flow, debug=False, visualize=False)
# pepn = compute_pepn(gt, flow, sen, th=3)
# print('MSEN: ', msen)
# print('PEPN: ', pepn)

U = flow[:, :, 0]
V = flow[:, :, 1]

X, Y = np.meshgrid(np.arange(0, flow.shape[1], 1),
                    np.arange(0, flow.shape[0], 1))

print(im1.shape)
windowSize = 50

u_new = np.zeros([im1.shape[0], im1.shape[1]])
v_new = np.zeros([im1.shape[0], im1.shape[1]])

print(u_new.shape)
print(v_new.shape)

for i in range(0, im1.shape[0]-windowSize, windowSize):
    for j in range(0, im1.shape[1]-windowSize, windowSize):
        # print(i, j)

        u_mean = np.mean(U[i:i+windowSize, j:j+windowSize])
        v_mean = np.mean(V[i:i+windowSize, j:j+windowSize])
        
        u_new[i+int(windowSize/2), int(j+windowSize/2)] = u_mean
        v_new[i+int(windowSize/2), int(j+windowSize/2)] = v_mean
        # print("MEAN: ", u_mean, v_mean)

plt.figure()
plt.title("pivot='tip'; scales with x view")
M = np.hypot(u_new, v_new)
Q = plt.quiver(X[::5, ::5], Y[::5, ::5], u_new[::5, ::5], v_new[::5, ::5], M[::5, ::5],
                units='x', pivot='tail', width=3, scale=0.5)
qk = plt.quiverkey(Q, 0.9, 0.9, 1, r'$1 \frac{m}{s}$',
                    labelpos='E', coordinates='data')
# plt.colorbar()
plt.imshow(im1_gray, cmap='gray')
# plt.savefig('task4_opt2_'+frame)    
plt.show()

# print(flow.shape)
# cv2.imwrite("W4/LK_result_U.png", flow[:,:,0])
# cv2.imwrite("W4/LK_result_V.png", flow[:,:,1])
# cv2.imwrite("W4/LK_result.png", flow)

# # Create some random colors
# color = np.random.randint(0,255,(100,3))

# # Take first frame and find corners in it
# ret, old_frame = cap.read()
# old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
# p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)

# # Create a mask image for drawing purposes
# mask = np.zeros_like(old_frame)

# while(1):
#     ret,frame = cap.read()
#     frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#     # calculate optical flow
#     p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

#     # Select good points
#     good_new = p1[st==1]
#     good_old = p0[st==1]

#     # draw the tracks
#     for i,(new,old) in enumerate(zip(good_new,good_old)):
#         a,b = new.ravel()
#         c,d = old.ravel()
#         mask = cv2.line(mask, (a,b),(c,d), color[i].tolist(), 2)
#         frame = cv2.circle(frame,(a,b),5,color[i].tolist(),-1)
#     img = cv2.add(frame,mask)

#     cv2.imshow('frame',img)
#     k = cv2.waitKey(30) & 0xff
#     if k == 27:
#         break

#     # Now update the previous frame and previous points
#     old_gray = frame_gray.copy()
#     p0 = good_new.reshape(-1,1,2)

# cv2.destroyAllWindows()
# cap.release()

if __name__ == "__main__":
    compute_LK_OF(path_image1, path_image2)
