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
    det = read_detections(params["det_path"],grouped=True, confidenceThr=0.4)
    frame_id = int(vidcap.get(cv2.CAP_PROP_POS_FRAMES))
    first_frame_id = frame_id
    print(frame_id)

    detections = []
    annotations = {}
    list_positions={}
    tracking = Tracking()
    det_bboxes_old = -1

    # Create an accumulator that will be updated during each frame
    accumulator = mm.MOTAccumulator(auto_id=True)

    for t in tqdm(range((train_len + test_len) - first_frame_id)):

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

        objs = [bbox.center for bbox in gt_bboxes]
        hyps = [bbox.center for bbox in det_bboxes]

        
        for object_bb in det_bboxes:
            if object_bb.id in list(list_positions.keys()):
                list_positions[object_bb.id].append([int(x) for x in object_bb.center])
            else:
                list_positions[object_bb.id] = [[int(x) for x in object_bb.center]]

        accumulator.update(
            [bbox.id for bbox in gt_bboxes],             # Ground truth objects in this frame
            [bbox.id for bbox in det_bboxes],            # Detector hypotheses in this frame
            mm.distances.norm2squared_matrix(objs, hyps) # Distances from object 1 to hypotheses 1, 2, 3 and Distances from object 2 to hypotheses 1, 2, 3
        )

        if params['show_boxes']:
            frame = draw_boxes(image=frame, boxes=gt_bboxes, color='w', linewidth=3, boxIds=False, tracker= list_positions)
            frame = draw_boxes(image=frame, boxes=det_bboxes, color='r', linewidth=3, det=False, boxIds=True, tracker = list_positions)
            # if not det_bboxes_old==-1:
            #     frame = draw_boxes(image=frame, boxes=det_bboxes_old, color='r', linewidth=3, det=False, boxIds=True,old=True)
            # frame = draw_boxes(image=frame, boxes=det_bboxes, color='g', linewidth=3, boxIds=False)
            cv2.rectangle(frame, (10, 2), (120,20), (255,255,255), -1)
            cv2.putText(frame, str(vidcap.get(cv2.CAP_PROP_POS_FRAMES)), (15, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5 , (0,0,0))
            cv2.imshow('Frame', frame)
            keyboard = cv2.waitKey(30)

        frame_id += 1
        det_bboxes_old = det_bboxes

    
    mh = mm.metrics.create()
    summary = mh.compute(accumulator, metrics=['precision','recall','idp','idr','idf1'], name='acc')
    print(summary)

if __name__ == "__main__":

    params = {
        'video_path': "./data/vdo.avi",
        'gt_path': "./data/AICity_data/train/S03/c010/ai_challenge_s03_c010-full_annotation.xml",
        'det_path': "./data/AICity_data/train/S03/c010/det/det_yolo3.txt", #YOLO
        # 'det_path': "./data/AICity_data/train/S03/c010/det/det_mask_rcnn.txt", #MASK RCNN
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
