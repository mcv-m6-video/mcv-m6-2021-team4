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
from tracking import Tracking
import motmetrics as mm


def eval_tracking(vidcap, test_len, params):
    print("Evaluating Tracking")  
    gt = read_annotations(params["gt_path"], grouped=True, use_parked=False)
    det = read_detections(params["det_path"],grouped=True, confidenceThr=0.5)
    frame_id = int(vidcap.get(cv2.CAP_PROP_POS_FRAMES))
    print(frame_id)

    detections = []
    annotations = {}

    tracking = Tracking()
    det_bboxes_old = -1

    # Create an accumulator that will be updated during each frame
    acc = mm.MOTAccumulator(auto_id=True)

    for t in tqdm(range(test_len)):

        _ ,frame = vidcap.read()
        # cv2.imshow('Frame', frame)
        # keyboard = cv2.waitKey(30)

        det_bboxes = det[frame_id]
        det_bboxes = tracking.set_frame_ids(det_bboxes, det_bboxes_old)
        detections += det_bboxes

        gt_bboxes = []
        if frame_id in gt:
            gt_bboxes = gt[frame_id]
        annotations[frame_id] = gt_bboxes

        # acc.update(
        #     [bbox.id for bbox in det_bboxes],                     # Ground truth objects in this frame
        #     [bbox.id for bbox in det_bboxes],                  # Detector hypotheses in this frame
        #     [
        #         [0.1, np.nan, 0.3],     # Distances from object 1 to hypotheses 1, 2, 3
        #         [0.5,  0.2,   0.3]      # Distances from object 2 to hypotheses 1, 2, 3
        #     ]
        # )

        if params['show_boxes']:
            frame = draw_boxes(image=frame, boxes=gt_bboxes, color='g', linewidth=3, boxIds=True)
            # frame = draw_boxes(image=frame, boxes=det_bboxes, color='r', linewidth=3, det=False, boxIds=True)
            # frame = draw_boxes(image=frame, boxes=det_bboxes, color='g', linewidth=3, boxIds=False)
            cv2.rectangle(frame, (10, 2), (120,20), (255,255,255), -1)
            cv2.putText(frame, str(vidcap.get(cv2.CAP_PROP_POS_FRAMES)), (15, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5 , (0,0,0))
            cv2.imshow('Frame', frame)
            keyboard = cv2.waitKey(30)

        frame_id += 1
        det_bboxes_old = det_bboxes


if __name__ == "__main__":

    params = {
        'video_path': "./data/vdo.avi",
        'gt_path': "./data/AICity_data/train/S03/c010/ai_challenge_s03_c010-full_annotation.xml",
        # 'det_path': "./data/AICity_data/train/S03/c010/det/det_yolo3.txt", #YOLO
        'det_path': "./data/AICity_data/train/S03/c010/det/det_mask_rcnn.txt", #MASK RCNN
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

    vidcap.set(cv2.CAP_PROP_POS_FRAMES, train_len)

    print("Train frames: ", train_len)
    print("Test frames: ", test_len)


    eval_tracking(vidcap, test_len, params)
