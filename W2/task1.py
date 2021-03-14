import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


def background_estimator(image, alpha, mean_train_frames, std_train_frames):
    segmentation = np.zeros((1080,1920))
    segmentation[abs(image - mean_train_frames) >= alpha*(std_train_frames + 2)] = 255
    return segmentation

def postprocess_after_segmentation(seg):
    kernel = np.ones((5,5),np.uint8)
    seg = cv2.erode(seg,kernel,iterations = 1)
    seg = cv2.dilate(seg,kernel,iterations = 1)
    return seg

def get_bboxes(seg):
    contours, _ = cv2.findContours(seg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        rect = cv2.boundingRect(c)
        if rect[2] < 100 or rect[3] < 100: continue
        print("cv2.contourArea(c)", cv2.contourArea(c))
        x,y,w,h = rect
        cv2.rectangle(im,(x,y),(x+w,y+h),(0,255,0),2)
        cv2.putText(im,'Moth Detected',(x+w+10,y+h),0,0.3,(0,255,0))
    cv2.imshow("Show",im)
    cv2.waitKey()  
    cv2.destroyAllWindows()

    return
    
def train(vidcap, train_len, saveResults=False):
    img_list=[]
    count = 0

    for t in tqdm(range(train_len)):
        success,frame = vidcap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        img_list.append(frame) 
        count += 1

    #Compute mean 
    mean_train_frames= np.mean(img_list,axis=0)
    print("Mean computed")

    #Compute std 
    std_train_frames= np.std(img_list,axis=0) 
    print("Std computed")

    if saveResults:
        cv2.imwrite("./W2/output/mean_train.png", mean_train_frames)
        cv2.imwrite("./W2/output/std_train.png", std_train_frames)

    return mean_train_frames, std_train_frames

def eval(vidcap, mean_train_frames, std_train_frames, eval_num, saveResults=False):
    
    alpha = 4
    img_list_processed=[]

    for t in tqdm(range(eval_num)):
        success,frame = vidcap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        segmentation = background_estimator(frame, alpha, mean_train_frames, std_train_frames)
        segmentation = postprocess_after_segmentation(segmentation)
        bboxes = get_bboxes(segmentation)
        img_list_processed.append(segmentation) 
        
        if saveResults:
            cv2.imwrite(f"./W2/output/seg_{str(t)}_pp_{str(alpha)}.bmp", segmentation.astype(int))


    return img_list_processed

if __name__ == '__main__':

    path_video = "data/vdo.avi"
    vidcap = cv2.VideoCapture(path_video)
    frame_count = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    print("Total frames: ", frame_count)

    train_len = int(0.25*frame_count)
    test_len = frame_count-train_len

    print("Train frames: ", train_len)
    print("Test frames: ", test_len)

    #Train
    mean_train_frames, std_train_frames = train(vidcap, train_len, saveResults=False)

    #Evaluate
    eval(vidcap, mean_train_frames, std_train_frames, 2, saveResults=True)
