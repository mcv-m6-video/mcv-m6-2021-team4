import glob
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
import task1



def BackgroundSubtractor(train_len, method):
        
    if method == 'MOG':
        backSub = cv2.bgsegm.createBackgroundSubtractorMOG()
    elif method == 'MOG2':
        backSub = cv2.createBackgroundSubtractorMOG2()
    elif method == 'LSBP':
        backSub = cv2.bgsegm.createBackgroundSubtractorLSBP()
    elif method == 'KNN':
        backSub = cv2.createBackgroundSubtractorKNN(history=100, detectShadows=True)

    print("Background Substractor Method: ", method)

    for t in tqdm(range(train_len)):
        success,frame = vidcap.read()
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        fgMask = backSub.apply(frame)
        det_bboxes = task1.get_bboxes(fgMask,t)
        fgMask = cv2.cvtColor(fgMask.astype(np.uint8), cv2.COLOR_GRAY2RGB)
        seg_boxes = draw_boxes(image=fgMask, boxes=det_bboxes, color='r', linewidth=3)
        # seg_boxes = draw_boxes(image=seg_boxes, boxes=gt_bboxes, color='g', linewidth=3)

        # cv2.rectangle(fgMask, (10, 2), (120,20), (255,255,255), -1)
        # cv2.putText(frame, method+" - "+str(vidcap.get(cv2.CAP_PROP_POS_FRAMES)), (15, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5 , (0,0,0))
        # cv2.imshow('Frame', frame)
        # cv2.imshow('FG Mask', fgMask)

        cv2.imshow("Segmentation mask with detected boxes and gt", seg_boxes)

        keyboard = cv2.waitKey(30)
        if keyboard == 'q' or keyboard == 27:
            break

def run():
    path_video = "data/vdo.avi"
    vidcap = cv2.VideoCapture(path_video)
    frame_count = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    print("Total frames: ", frame_count)

    train_len = int(0.25*frame_count)
    test_len = frame_count-train_len

    print("Train frames: ", train_len)
    print("Test frames: ", test_len)

    BackgroundSubtractor(train_len, 'MOG')
    BackgroundSubtractor(train_len, 'MOG2')
    BackgroundSubtractor(train_len, 'LSBP')
    BackgroundSubtractor(train_len, 'KNN')


    # #Train
    # mean_train_frames, std_train_frames = train(vidcap, train_len, saveResults=False, usePickle=True)

    # #Evaluate
    # eval(vidcap, mean_train_frames, std_train_frames, 2, saveResults=False)
