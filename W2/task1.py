import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt


##Very naive approach, need to use a lib to compute on the fly with the video.

# path_to_frames_folder = r"C:\Users\Eduard\Desktop\M6\AICity_data\train\S03\c010\img\*.bmp"
path_video = "data/vdo.avi"
vidcap = cv2.VideoCapture(path_video)
frame_count = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
print(frame_count)

train_len = int(0.25*frame_count)
test_len = frame_count-train_len

print("train_len: ",train_len)
print("test_len: ",test_len)

img_list=[]

success,frame = vidcap.read()
frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
count = 0
while count < train_len:
  success,frame = vidcap.read()
  frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  img_list.append(frame) 
  count += 1

#Compute mean 
mean_train_frames= np.mean(img_list,axis=0)
print("mean_train_frames.shape", mean_train_frames.shape)

#Compute std 
std_train_frames= np.std(img_list,axis=0)
cv2.imwrite("std.png", std_train_frames)
print("std_train_frames.shape", std_train_frames.shape)

# vidcap.set(2,train_len)
# ret, frame = vidcap.read()
# while count < 20:
#     ret, frame = vidcap.read()
#     count += 1
#     print(count)


alfa = 3
kernel = np.ones((5,5),np.uint8)
img_list_processed=[]

while count < train_len + 5:
    success,frame = vidcap.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    segmentation = np.zeros((1080,1920))
    segmentation[abs(frame - mean_train_frames)>= alfa*(std_train_frames + 2)] = 255
    segmentation = cv2.erode(segmentation,kernel,iterations = 1)
    segmentation = cv2.dilate(segmentation,kernel,iterations = 1)
    img_list_processed.append(segmentation) 
    count += 1

for i,img in enumerate(img_list_processed):
    cv2.imwrite("./W2/output/seg_erode_dilate_" + str(alfa) + "_" + str(i) + ".bmp", img.astype(int))
