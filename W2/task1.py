import glob
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys
sys.path.append("W1")
import voc_evaluation
import aicity_reader
import bounding_box
from utils import draw_boxes
import pickle

def background_estimator(image, alpha, mean_train_frames, std_train_frames):
    segmentation = np.zeros((1080,1920))
    segmentation[abs(image - mean_train_frames) >= alpha*(std_train_frames + 2)] = 255
    return segmentation

def postprocess_after_segmentation(seg):
    kernel = np.ones((5,5),np.uint8)
    seg = cv2.erode(seg,kernel,iterations = 1)
    seg = cv2.dilate(seg,kernel,iterations = 1)
    return seg

def get_bboxes(seg,frame_id):

    # print(seg.shape)
    new_seg = cv2.cvtColor(seg.astype(np.uint8), cv2.COLOR_GRAY2RGB)
    contours, _ = cv2.findContours(seg.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bboxes = []
    output=[]
    
    for c in contours:
        rect = cv2.boundingRect(c)
        if rect[2] < 50 or rect[3] < 50: continue #Discard small contours
        # print("cv2.contourArea(c)", cv2.contourArea(c))
        x,y,w,h = rect
        bboxes.append([x,y,x+w,y+h])

    for i,bbox in enumerate(bboxes):

        # output[i]={'box': bbox,'id': i,'frame':frame_id}
        output.append(bounding_box.BoundingBox(id=i,label='car',frame=frame_id,xtl=bbox[0], ytl=bbox[1], xbr=bbox[2], ybr=bbox[3]))
        
        
        # new_seg = cv2.rectangle(new_seg,(x,y),(x+w,y+h),(0,255,0),2)
        # cv2.putText(seg,'Moth Detected',(x+w+10,y+h),0,0.3,(0,255,0))

    # print("hola", new_seg.shape)
    # cv2.imshow("Show",new_seg)
    # cv2.waitKey()  
    # cv2.destroyAllWindows()

    return output

def train(vidcap, train_len, saveResults=False, usePickle=False):

    if usePickle:
        mean_f = open('data/mean_pickle', 'rb')
        mean_train_frames = pickle.load(mean_f)

        std_f = open('data/std_pickle', 'rb')
        std_train_frames = pickle.load(std_f)

        cv2.imwrite("./W2/output/mean_train_pickle.png", mean_train_frames)
        cv2.imwrite("./W2/output/std_train_pickle.png", std_train_frames)

        print("1", type(mean_train_frames))
        print(mean_train_frames)
        return mean_train_frames, std_train_frames

    else:
        img_list=[]
        for t in tqdm(range(train_len)):
            success,frame = vidcap.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            img_list.append(frame) 

        #Compute mean 
        mean_train_frames= np.mean(img_list,axis=0)
        print("Mean computed")
        print(mean_train_frames)
        print("2",type(mean_train_frames))

        #Compute std 
        std_train_frames= np.std(img_list,axis=0) 
        print("Std computed")

        mean_pickle = open('mean_pickle','wb')
        pickle.dump(mean_train_frames, mean_pickle)

        std_pickle = open('std_pickle','wb')
        pickle.dump(std_train_frames, std_pickle)

        if saveResults:
            cv2.imwrite("./W2/output/mean_train.png", mean_train_frames)
            cv2.imwrite("./W2/output/std_train.png", std_train_frames)

        
        return mean_train_frames, std_train_frames

def eval(vidcap, mean_train_frames, std_train_frames, eval_num, saveResults=False):
    frame_num = train_len
    alpha = 3
    img_list_processed = []
    bboxes_byframe = []

    path_gt = './data/ai_challenge_s03_c010-full_annotation.xml'
    annotations = aicity_reader.read_annotations(path_gt)
    annotations_grouped = aicity_reader.group_by_frame(annotations)

    for t in tqdm(range(eval_num)):
        success,frame = vidcap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        segmentation = background_estimator(frame, alpha, mean_train_frames, std_train_frames)
        # segmentation = postprocess_after_segmentation(segmentation)
        bboxes = get_bboxes(segmentation,frame_num)
        bboxes_byframe = bboxes_byframe + bboxes
        frame_num += 1


        if saveResults:
            if not os.path.exists(f"./W2/output/{str(alpha)}"):
                os.makedirs(f"./W2/output/{str(alpha)}")
            cv2.imwrite(f"./W2/output/{str(alpha)}/seg_{str(t)}_pp.bmp", segmentation.astype(int))

        
        # if True:
        #     frame = draw_boxes(frame, frame_id, grouped_det[frame_id], color='b', det=True)
        #     frame_iou = mean_iou(grouped_det[frame_id], grouped_gt[frame_id], sort=True)



    rec, prec, ap = voc_evaluation.voc_eval(bboxes_byframe, annotations_grouped, ovthresh=0.5)
    print(rec, prec, ap)

    return

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
    mean_train_frames, std_train_frames = train(vidcap, train_len, saveResults=False, usePickle=False)

    #Evaluate
    eval(vidcap, mean_train_frames, std_train_frames, test_len, saveResults=True)
