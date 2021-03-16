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
from bg_estimation import fg_bboxes
from aicity_reader import read_annotations
from bg_estimation import static_bg_est, adaptive_bg_est, postprocess_fg, fg_bboxes, temporal_filter


def train_sota(vidcap, train_len, backSub):
    print("Training SOTA")
    for t in tqdm(range(train_len)):
        #update the background model
        _ ,frame = vidcap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        segmentation = backSub.apply(frame)
        segmentation[segmentation<200] = 0
        # print(segmentation.shape)
        # print(np.unique(segmentation))
        # break
    
    return backSub #return backSub updated

def eval_sota(vidcap, test_len, backSub, showResults=True):
    print("Evaluating SOTA")  
    gt = read_annotations('./data/ai_challenge_s03_c010-full_annotation.xml', grouped=True, use_parked=False)
    frame_id = int(vidcap.get(cv2.CAP_PROP_POS_FRAMES))

    detections = []
    annotations = {}

    for t in tqdm(range(test_len)):

        _ ,frame = vidcap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        segmentation = backSub.apply(frame)
        segmentation[segmentation<200] = 0
        det_bboxes = fg_bboxes(segmentation,frame_id)
        detections += det_bboxes

        segmentation = cv2.cvtColor(segmentation.astype(np.uint8), cv2.COLOR_GRAY2RGB)

        gt_bboxes = []
        if frame_id in gt:
            gt_bboxes = gt[frame_id]
        annotations[frame_id] = gt_bboxes

        if showResults:
            segmentation = draw_boxes(image=segmentation, boxes=gt_bboxes, color='g', linewidth=3)
            cv2.rectangle(frame, (10, 2), (120,20), (255,255,255), -1)
            cv2.putText(frame, method+" - "+str(vidcap.get(cv2.CAP_PROP_POS_FRAMES)), (15, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5 , (0,0,0))
            segmentation = draw_boxes(image=segmentation, boxes=det_bboxes, color='r', linewidth=3)
            cv2.imshow("Segmentation mask with detected boxes and gt", segmentation)
            cv2.imshow('Frame', frame)

        keyboard = cv2.waitKey(30)
        if keyboard == 'q' or keyboard == 27:
            break

        frame_id += 1

    rec, prec, ap = voc_evaluation.voc_eval(detections, annotations, ovthresh=0.5, use_confidence=False)   
    print("Recall: ", rec)
    print("Precission: ", prec)
    print("AP: ", ap)


if __name__ == "__main__":

    path_video = "data/vdo.avi"
    vidcap = cv2.VideoCapture(path_video)
    frame_count = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    print("Total frames: ", frame_count)

    train_len = int(0.25*frame_count)
    test_len = frame_count-train_len

    print("Train frames: ", train_len)
    print("Test frames: ", test_len)

    method = 'MOG2'
    if method == 'MOG':
        backSub = cv2.bgsegm.createBackgroundSubtractorMOG()
    elif method == 'MOG2':
        backSub = cv2.createBackgroundSubtractorMOG2(history=500,varThreshold=16,detectShadows=True)
    elif method == 'LSBP':
        backSub = cv2.bgsegm.createBackgroundSubtractorLSBP()
    elif method == 'KNN':
        backSub = cv2.createBackgroundSubtractorKNN(history=100, detectShadows=True)

    print("Background Substractor Method: ", method)


    backSub = train_sota(vidcap, train_len, backSub)
    backSub = eval_sota(vidcap, test_len, backSub)



    # #Train
    # mean_train_frames, std_train_frames = train(vidcap, train_len, saveResults=False, usePickle=True)

    # #Evaluate
    # eval(vidcap, mean_train_frames, std_train_frames, 2, saveResults=False)
