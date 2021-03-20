import cv2
import numpy as np
from tqdm import tqdm
import sys

sys.path.append("W1")
sys.path.append("W2")
from bounding_box import BoundingBox
from aicity_reader import read_annotations, read_detections, group_by_frame
from utils import draw_boxes
from bg_postprocess import temporal_filter, postprocess_fg, discard_overlapping_bboxes


def eval_tracking(vidcap, test_len, params):
    print("Evaluating Tracking")  
    gt = read_annotations(params["gt_path"], grouped=True, use_parked=False)
    det =read_detections(params["det_path"],grouped=True)
    frame_id = int(vidcap.get(cv2.CAP_PROP_POS_FRAMES))

    detections = []
    annotations = {}

    for t in tqdm(range(test_len)):

        _ ,frame = vidcap.read()
        # cv2.imshow('Frame', frame)
        # keyboard = cv2.waitKey(30)

        # det_bboxes = fg_bboxes(segmentation, frame_id, params)
        # detections += det_bboxes

        gt_bboxes = []
        if frame_id in gt:
            gt_bboxes = gt[frame_id]
        annotations[frame_id] = gt_bboxes

        text_bboxes = "nbb" #no bouning boxes
        # if params['show_boxes']:
        frame = draw_boxes(image=frame, boxes=gt_bboxes, color='g', linewidth=3,boxIds=True)
        frame = draw_boxes(image=frame, boxes=det[frame_id], color='r', linewidth=3,boxIds=False,det=True)
        cv2.rectangle(frame, (10, 2), (120,20), (255,255,255), -1)
        cv2.putText(frame, str(vidcap.get(cv2.CAP_PROP_POS_FRAMES)), (15, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5 , (0,0,0))
        cv2.imshow('Frame', frame)
        text_bboxes = ""
        keyboard = cv2.waitKey(30)

        frame_id += 1


if __name__ == "__main__":

    params = {
        'video_path': "./data/vdo.avi",
        'gt_path': "./data/AICity_data/train/S03/c010/ai_challenge_s03_c010-full_annotation.xml",
        'det_path': "./data/AICity_data/train/S03/c010/det/det_mask_rcnn.txt",
        'roi_path': "./data/AICity_data/train/S03/c010/roi.jpg",
        'show_boxes': True,
        'sota_method': "MOG2",
        'save_results': False,
        'results_path': "./W3/output/"
    }

    vidcap = cv2.VideoCapture(params['video_path'])
    frame_count = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    print("Total frames: ", frame_count)

    train_len = int(0.25*frame_count)
    test_len = frame_count-train_len

    print("Train frames: ", train_len)
    print("Test frames: ", test_len)


    eval_tracking(vidcap, test_len, params)
