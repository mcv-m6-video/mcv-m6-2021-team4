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
count = 0
while count < train_len:
  success,frame = vidcap.read()
  img_list.append(frame) 
  count += 1

#Compute mean 
mean_train_frames= np.mean(img_list,axis=0)
print("mean_train_frames.shape", mean_train_frames.shape)

#Compute variance 
variance_train_frames= np.var(img_list,axis=0)
print("variance_train_frames.shape", variance_train_frames.shape)

# vidcap.set(2,train_len)
# ret, frame = vidcap.read()
# while count < 20:
#     ret, frame = vidcap.read()
#     count += 1
#     print(count)


alfa =2
img_list_processed=[]

while count < train_len + 2:
  success,image = vidcap.read()
  image[abs(image - mean_train_frames)>= alfa*(variance_train_frames + 2)] = 0
  img_list_processed.append(image) 
  count += 1

cv2.imwrite("d.bmp",img_list_processed[0].astype(int))

# list_frames = [frame for frame in glob.glob(path_to_frames_folder)]
# print(len(list_frames))
# img_list=[]

# for image in list_frames[:25]:
#     img_list.append(cv2.imread(image)) 
#     # img_list.append(cv2.imread(image,cv2.COLOR_BGR2RGB)) 

    

# print(len(img_list))
# print(img_list[0][0])

# mean_all_frames= np.mean(img_list,axis=0)
# print(mean_all_frames.shape)
# print(mean_all_frames[0])

# print(type(mean_all_frames))
# mean_all_frames = mean_all_frames.astype(int)
# mean_all_frames = cv2.convertScaleAbs(mean_all_frames)
# cv2.imwrite("d.bmp",mean_all_frames)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# # plt.imshow(mean_all_frames.astype(int),cmap='gray')
# # plt.show()
