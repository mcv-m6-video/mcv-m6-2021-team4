import cv2
import numpy as np
from tqdm import tqdm
import sys
from pprint import pprint

sys.path.append("W1")
sys.path.append("W2")
sys.path.append("W3")
sys.path.append("W3/sort")
from bounding_box import BoundingBox
from aicity_reader import read_annotations, read_detections, group_by_frame
from utils import draw_boxes
from bg_postprocess import temporal_filter, postprocess_fg, discard_overlapping_bboxes
from tracking import Tracking
from flow_reader import read_flow
from block_matching import estimate_flow

import motmetrics as mm
import Kalman
from sort import Sort
from flow_utils import plot_flow

def computeOpticalFlow(old, new, detection, option=1):

    # p0 = np.array([[x, y] for y in range(height) for x in range(width)], dtype=np.float32).reshape((-1, 1, 2))
    # p0 = []

    # Parameters for lucas kanade optical flow
    lk_params = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    
    if option == 1:
        ''' 
        OPTION 1: 
            Agafar tots els punts dintre la detecció i fer la mitja. 
            Aplicar aquesta mitja a les 4 coordenades de la detecció.'''
        # print("Option 1")
        height = int(detection.ybr - detection.ytl)
        width =  int(detection.xbr - detection.xtl)
        p0 = np.array([[x, y] for y in range(height) for x in range(width)], dtype=np.float32).reshape((-1, 1, 2))
        p1, st, err = cv2.calcOpticalFlowPyrLK(old, new, p0, None, **lk_params)
        flow = p1 - p0
        flow[st == 0] = 0
        # flow[:,:,1] = - flow[:,:,1]
        flow = np.reshape(flow, (width,height,2))
        flow = np.mean(flow, axis=(0,1))
        return flow

    elif option == 2:
        '''
        OPTION 2: 
            Aplicar el goodFeaturesToTrack a la detecció. 
            Fer la mitja del optical flow dels goodFeaturesToTrack. 
            Aplicar aquesta mitja a les 4 coordenades de la detecció. 
            És més ràpid ja que no es calculan tants OF vectors a tants pixels.'''

        # print("Option 2")
        # params for ShiTomasi corner detection
        feature_params = dict( maxCorners = 100,
                            qualityLevel = 0.3,
                            minDistance = 7,
                            blockSize = 7 )

        old_gray = cv2.cvtColor(old, cv2.COLOR_BGR2GRAY)
        good_points = cv2.goodFeaturesToTrack(old_gray[int(detection.xtl):int(detection.xbr), int(detection.ytl):int(detection.ybr)], mask = None, **feature_params)

        if good_points is not None:
            #Using good points when we have them
            p0 = good_points
        else:
            #Computing the OF for all the pixels inside detection if we do not have good points.
            height = int(detection.ybr - detection.ytl)
            width =  int(detection.xbr - detection.xtl)
            p0 = np.array([[x, y] for y in range(height) for x in range(width)], dtype=np.float32).reshape((-1, 1, 2))

        p1, st, err = cv2.calcOpticalFlowPyrLK(old, new, p0, None, **lk_params)
        flow = p1 - p0
        flow[st == 0] = 0
        flow[:,:,1] = - flow[:,:,1]
        # flow = np.reshape(flow, (int(detection.width),int(detection.height),2))
        flow = np.mean(flow, axis=(0,1))
        return flow
    
    elif option == 3:
        ''' 
        OPTION 3: 
            Agafar tots els punts dintre la detecció i fer la mediana. 
            Aplicar aquesta mediana a les 4 coordenades de la detecció.'''
        # print("Option 1")
        height = int(detection.ybr - detection.ytl)
        width =  int(detection.xbr - detection.xtl)
        p0 = np.array([[x, y] for y in range(height) for x in range(width)], dtype=np.float32).reshape((-1, 1, 2))
        p1, st, err = cv2.calcOpticalFlowPyrLK(old, new, p0, None, **lk_params)
        flow = p1 - p0
        flow[st == 0] = 0
        # flow[:,:,1] = - flow[:,:,1]
        flow = np.reshape(flow, (width,height,2))
        plot_flow(old, flow)
        flow = np.median(flow, axis=(0,1))
        return flow
    
    elif option == 4:
        '''
        OPTION 4: 
            Aplicar el goodFeaturesToTrack a la detecció. 
            Fer la mitja del optical flow dels goodFeaturesToTrack. 
            Aplicar aquesta mitja a les 4 coordenades de la detecció. 
            És més ràpid ja que no es calculan tants OF vectors a tants pixels.'''

        # print("Option 2")
        # params for ShiTomasi corner detection
        feature_params = dict( maxCorners = 100,
                            qualityLevel = 0.3,
                            minDistance = 7,
                            blockSize = 7 )

        old_gray = cv2.cvtColor(old, cv2.COLOR_BGR2GRAY)
        good_points = cv2.goodFeaturesToTrack(old_gray[int(detection.xtl):int(detection.xbr), int(detection.ytl):int(detection.ybr)], mask = None, **feature_params)

        if good_points is not None:
            #Using good points when we have them
            p0 = good_points
        else:
            #Computing the OF for all the pixels inside detection if we do not have good points.
            height = int(detection.ybr - detection.ytl)
            width =  int(detection.xbr - detection.xtl)
            p0 = np.array([[x, y] for y in range(height) for x in range(width)], dtype=np.float32).reshape((-1, 1, 2))

        p1, st, err = cv2.calcOpticalFlowPyrLK(old, new, p0, None, **lk_params)
        flow = p1 - p0
        flow[st == 0] = 0
        flow[:,:,1] = - flow[:,:,1]
        # flow = np.reshape(flow, (int(detection.width),int(detection.height),2))
        flow = np.median(flow, axis=(0,1))
        return flow

    elif option == 5:
        '''
        OPTION 5: Using Block Matching
        '''
        flow = estimate_flow('forward', 16, 32, 'ncc', old[int(detection.ytl):int(detection.ybr), int(detection.xtl):int(detection.xbr)], new)
        flow = np.median(flow, axis=(0,1))
        return flow

    else:
        TypeError("Bad Option number on computeOpticalFlow_LK")


def eval_tracking_MaximumOverlap(vidcap, test_len, params, opticalFlow=None):

    print("Evaluating Tracking")  
    gt = read_annotations(params["gt_path"], grouped=True, use_parked=True)
    det = read_detections(params["det_path"],grouped=True, confidenceThr=0.4)
    frame_id = int(vidcap.get(cv2.CAP_PROP_POS_FRAMES))
    first_frame_id = frame_id
    print(frame_id)

    detections = []
    annotations = {}
    list_positions = {}

    center_seen_last5frames ={}
    id_seen_last5frames = {}

    tracking = Tracking()
    det_bboxes_old = -1

    old_frame = None

    # Create an accumulator that will be updated during each frame
    accumulator = mm.MOTAccumulator(auto_id=True)

    for t in tqdm(range((train_len + test_len) - first_frame_id)):

        _ ,frame = vidcap.read()
        # cv2.imshow('Frame', frame)
        # keyboard = cv2.waitKey(30)

        if params['use_optical_flow'] and old_frame is not None:
            for d in det_bboxes_old:
                # print(d)
                flow = None
                # print("Computing optical flow")
                flow = computeOpticalFlow(old_frame, frame, d, option=params['optical_flow_option'])
                d.flow = flow

        det_bboxes = det[frame_id]
        det_bboxes = tracking.set_frame_ids(det_bboxes, det_bboxes_old)
        detections += det_bboxes

        id_seen = []
        gt_bboxes = []
        if frame_id in gt:
            gt_bboxes = gt[frame_id]
        annotations[frame_id] = gt_bboxes

        objs = [bbox.center for bbox in gt_bboxes]
        hyps = [bbox.center for bbox in det_bboxes]
  
        for object_bb in det_bboxes:
            if object_bb.id in list(list_positions.keys()):
                if t < 5:
                    id_seen_last5frames[object_bb.id] = object_bb.id
                    center_seen_last5frames[object_bb.id] = object_bb.center
                list_positions[object_bb.id].append([int(x) for x in object_bb.center])
            else:
                if (t < 5):
                    id_seen_last5frames[object_bb.id] = object_bb.id
                    center_seen_last5frames[object_bb.id] = object_bb.center

                id_seen.append(object_bb)
                list_positions[object_bb.id] = [[int(x) for x in object_bb.center]]


        # To detect pared cars
        for bbox in id_seen:
            for idx in list(id_seen_last5frames.keys()):
                if idx != bbox.id:
                    center = [center_seen_last5frames[idx]]
                    mse = (np.square(np.subtract(np.array(center),np.array([int(x) for x in bbox.center])))).mean()
                    if mse < 300:
                        setattr(bbox, 'id', idx)        

        accumulator.update(
            [bbox.id for bbox in gt_bboxes],             # Ground truth objects in this frame
            [bbox.id for bbox in det_bboxes],            # Detector hypotheses in this frame
            mm.distances.norm2squared_matrix(objs, hyps) # Distances from object 1 to hypotheses 1, 2, 3 and Distances from object 2 to hypotheses 1, 2, 3
        )

        if params['show_boxes']:
            drawed_frame = frame
            # frame = draw_boxes(image=drawed_frame, boxes=gt_bboxes, color='w', linewidth=3, boxIds=False, tracker= list_positions)
            drawed_frame = draw_boxes(image=drawed_frame, boxes=det_bboxes, color='r', linewidth=3, det=False, boxIds=True, tracker = list_positions)
            # if not det_bboxes_old==-1:
            #     drawed_frame = draw_boxes_old(image=drawed_frame, boxes=det_bboxes_old, color='r', linewidth=3, det=False, boxIds=True, tracker = list_positions)
            #     drawed_frame = draw_boxes(image=drawed_frame, boxes=det_bboxes_old, tracker = list_positions, color='r', linewidth=3, det=False, boxIds=True,old=True)
            # drawed_frame = draw_boxes(image=drawed_frame, boxes=det_bboxes, color='g', linewidth=3, boxIds=False)
            cv2.rectangle(drawed_frame, (10, 2), (120,20), (255,255,255), -1)
            cv2.putText(drawed_frame, str(vidcap.get(cv2.CAP_PROP_POS_FRAMES)), (15, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5 , (0,0,0))
            cv2.imshow('Frame', drawed_frame)
            keyboard = cv2.waitKey(30)
        
        if params['save_results'] and frame_id >= (355+535) and frame_id < (410+535) : # if frame_id >= 535 and frame_id < 550
            cv2.imwrite(params['results_path'] + f"tracking_{str(frame_id)}_IoU.jpg", drawed_frame.astype(int))

        frame_id += 1
        old_frame = frame
        det_bboxes_old = det_bboxes

    
    mh = mm.metrics.create()
    summary = mh.compute(accumulator, metrics=['precision','recall','idp','idr','idf1'], name='acc')
    print(summary)

if __name__ == "__main__":

    params = {
        'video_path': "./data/vdo.avi",
        'gt_path': "./data/AICity_data/train/S03/c010/ai_challenge_s03_c010-full_annotation.xml",
        'det_path': "./data/AICity_data/train/S03/c010/det/ian_detections.txt",
        # 'det_path': "./data/AICity_data/train/S03/c010/det/det_mask_rcnn.txt", #MASK RCNN
        'roi_path': "./data/AICity_data/train/S03/c010/roi.jpg",
        'show_boxes': True,
        'sota_method': "MOG2",
        'save_results': True,
        'results_path': "./W4/output_noOF_oldbb/",
        'use_optical_flow': True,
        'optical_flow_option': 3,
    }



    vidcap = cv2.VideoCapture(params['video_path'])
    frame_count = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    print("Total frames: ", frame_count)

    train_len = int(0.25*frame_count)
    test_len = frame_count-train_len

    vidcap.set(cv2.CAP_PROP_POS_FRAMES, train_len)

    print("Train frames: ", train_len)
    print("Test frames: ", test_len)

    pprint(params)

    eval_tracking_MaximumOverlap(vidcap, test_len, params)