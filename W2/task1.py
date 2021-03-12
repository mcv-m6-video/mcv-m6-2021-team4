import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt


##Very naive approach, need to use a lib to compute on the fly with the video.

path_to_frames_folder = r"C:\Users\Eduard\Desktop\M6\AICity_data\train\S03\c010\img\*.bmp"

list_frames = [frame for frame in glob.glob(path_to_frames_folder)]
print(len(list_frames))
img_list=[]

for image in list_frames[:25]:
    img_list.append(cv2.imread(image)) 
    # img_list.append(cv2.imread(image,cv2.COLOR_BGR2RGB)) 

    

print(len(img_list))
print(img_list[0][0])

mean_all_frames= np.mean(img_list,axis=0)
print(mean_all_frames.shape)
print(mean_all_frames[0])

print(type(mean_all_frames))
mean_all_frames = mean_all_frames.astype(int)
mean_all_frames = cv2.convertScaleAbs(mean_all_frames)
cv2.imwrite("d.bmp",mean_all_frames)
cv2.waitKey(0)
cv2.destroyAllWindows()
# plt.imshow(mean_all_frames.astype(int),cmap='gray')
# plt.show()
