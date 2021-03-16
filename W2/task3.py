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
from bg_estimation import static_bg_est, adaptive_bg_est, postprocess_fg, fg_bboxes, temporal_filter, train_sota, eval_sota


def run(args):
    path_video = "data/vdo.avi"
    vidcap = cv2.VideoCapture(path_video)
    frame_count = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    print("Total frames: ", frame_count)

    train_len = int(0.25*frame_count)
    test_len = frame_count-train_len

    print("Train frames: ", train_len)
    print("Test frames: ", test_len)

    params = {
        'video_path': args.video_path,
        'roi_path': args.roi_path,
        'gt_path': args.gt_path,
        'show_boxes': args.show_boxes,
        'sota_method': args.sota_method
    }


    print("Background Substractor Method: ", args.sota_method)

    if params["sota_method"] == 'MOG':
        backSub = cv2.bgsegm.createBackgroundSubtractorMOG(nmixtures = 2, history=200)
    elif params["sota_method"] == 'MOG2':
        # backSub = cv2.createBackgroundSubtractorMOG2(history=100,varThreshold=30,detectShadows=True)
        backSub = cv2.createBackgroundSubtractorMOG2()
    elif params["sota_method"] == 'LSBP':
        backSub = cv2.bgsegm.createBackgroundSubtractorLSBP()
    elif params["sota_method"] == 'KNN':
        backSub = cv2.createBackgroundSubtractorKNN(history=100, detectShadows=True)

    backSub = train_sota(vidcap, train_len, backSub)
    backSub = eval_sota(vidcap, test_len, backSub, params)



    # #Train
    # mean_train_frames, std_train_frames = train(vidcap, train_len, saveResults=False, usePickle=True)

    # #Evaluate
    # eval(vidcap, mean_train_frames, std_train_frames, 2, saveResults=False)
