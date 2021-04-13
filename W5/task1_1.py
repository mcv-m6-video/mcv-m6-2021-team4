import cv2
import numpy as np
from tqdm import tqdm
import sys
from pprint import pprint

sys.path.append("W1")
sys.path.append("W2")
sys.path.append("W3")
sys.path.append("W4")
sys.path.append("W3/sort")
from aicity_reader import read_annotations, read_detections, group_by_frame
from utils import draw_boxes, draw_boxes_old
from tracking import Tracking
from block_matching import estimate_flow

import motmetrics as mm
from flow_utils import plot_flow

from copy import deepcopy

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
        height = int(detection.ybr) - int(detection.ytl)
        width =  int(detection.xbr) - int(detection.xtl)
        p0 = np.array([[x, y] for y in range(int(detection.ytl), int(detection.ybr)) for x in
                       range(int(detection.xtl), int(detection.xbr))],
                      dtype=np.float32).reshape((-1, 1, 2))

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
        good_points = cv2.goodFeaturesToTrack(old_gray[int(detection.ytl):int(detection.ybr), int(detection.xtl):int(detection.xbr)],
                                              mask = None, **feature_params)

        if good_points is not None:
            #Using good points when we have them
            p0 = good_points
        else:
            #Computing the OF for all the pixels inside detection if we do not have good points.
            height = int(detection.ybr) - int(detection.ytl)
            width = int(detection.xbr) - int(detection.xtl)
            p0 = np.array([[x, y] for y in range(int(detection.ytl), int(detection.ybr)) for x in
                           range(int(detection.xtl), int(detection.xbr))],
                          dtype=np.float32).reshape((-1, 1, 2))

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
        height = int(detection.ybr) - int(detection.ytl)
        width = int(detection.xbr) - int(detection.xtl)
        p0 = np.array([[x, y] for y in range(int(detection.ytl), int(detection.ybr)) for x in range(int(detection.xtl), int(detection.xbr))],
                      dtype=np.float32).reshape((-1, 1, 2))
        p1, st, err = cv2.calcOpticalFlowPyrLK(old, new, p0, None, **lk_params)
        flow = p1 - p0
        flow[st == 0] = 0
        flow[:,:,1] = - flow[:,:,1]
        flow = np.reshape(flow, (height,width,2))
        flow = np.median(flow, axis=(0,1))
        # print('MEDIAN: ', flow)
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
        good_points = cv2.goodFeaturesToTrack(old_gray[int(detection.ytl):int(detection.ybr), int(detection.xtl):int(detection.xbr)],
                                              mask = None, **feature_params)

        if good_points is not None:
            #Using good points when we have them
            p0 = good_points
        else:
            #Computing the OF for all the pixels inside detection if we do not have good points.
            height = int(detection.ybr) - int(detection.ytl)
            width = int(detection.xbr) - int(detection.xtl)
            p0 = np.array([[x, y] for y in range(int(detection.ytl), int(detection.ybr)) for x in
                           range(int(detection.xtl), int(detection.xbr))],
                          dtype=np.float32).reshape((-1, 1, 2))

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


def eval_tracking_MaximumOverlap(vidcap, test_len, params, video, opticalFlow=None):

    print("Evaluating Tracking")  
    gt = read_detections(video["gt_path"], grouped=True)
    det = read_detections(video["det_path"], grouped=True)
    # det = read_detections(video["det_path"],grouped=True, confidenceThr=0.4)
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

    for t in tqdm(range((test_len) - first_frame_id)):

        _ ,frame = vidcap.read()
        # cv2.imshow('Frame', frame)
        # keyboard = cv2.waitKey(30)

        flow_aux = np.zeros(shape=(frame.shape[0], frame.shape[1], 2))

        if params['use_optical_flow'] and old_frame is not None:
            for d in det_bboxes_old:
                # print(d)
                flow = None
                # print("Computing optical flow")
                flow = computeOpticalFlow(old_frame, frame, d, option=params['optical_flow_option'])
                d.flow = [flow[0], -flow[1]]

                flow_aux[int(d.ytl):int(d.ybr), int(d.xtl):int(d.xbr), :] = flow

            # plot_flow(old_frame[:, :, [2, 1, 0]], flow_aux, step=16,
            #           fname='/home/oscar/workspace/master/modules/m6/project/mcv-m6-2021-team4/W4/OF_BB/'+f"tracking_{str(frame_id)}_IoU.png")

        try:
            det_bboxes = det[frame_id]
        except:    
            det_bboxes = []

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
            drawed_frame_aux = draw_boxes(image=frame, boxes=det_bboxes, color='r', linewidth=3, det=False, boxIds=True, tracker = list_positions)
            drawed_frame_aux = draw_boxes(image=drawed_frame_aux, boxes=gt_bboxes, color='g', linewidth=3, boxIds=False, tracker= list_positions)
            drawed_frame = deepcopy(drawed_frame_aux)

            # if not det_bboxes_old==-1:
            #     drawed_frame = draw_boxes_old(image=drawed_frame, boxes=det_bboxes_old, color='r', linewidth=3, det=False, boxIds=True, tracker = list_positions)
            cv2.rectangle(drawed_frame, (10, 2), (120,20), (255,255,255), -1)
            cv2.putText(drawed_frame, str(vidcap.get(cv2.CAP_PROP_POS_FRAMES)), (15, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5 , (0,0,0))
            cv2.imshow('Frame', drawed_frame)
            cv2.waitKey(30)
            cv2.imwrite(params['results_path'] + f"tracking_{str(frame_id)}_IoU.jpg", drawed_frame.astype(int))

            drawed_frame2 = deepcopy(drawed_frame_aux)
            # if not det_bboxes_old == -1:
            #     drawed_frame2 = draw_boxes_old(image=drawed_frame2, boxes=det_bboxes_old, color='r', linewidth=3,
            #                                   det=False, boxIds=True, tracker=list_positions, shifted=True)
            cv2.rectangle(drawed_frame2, (10, 2), (120, 20), (255, 255, 255), -1)
            cv2.putText(drawed_frame2, str(vidcap.get(cv2.CAP_PROP_POS_FRAMES)), (15, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 0, 0))
            cv2.imshow('Frame', drawed_frame2)
            cv2.waitKey(30)
            cv2.imwrite('./W4/OF_shifted_BB/' + f"tracking_{str(frame_id)}_IoU.jpg", drawed_frame2.astype(int))
        
        if params['save_results'] and frame_id >= (355+535) and frame_id < (410+535) : # if frame_id >= 535 and frame_id < 550
            drawed_frame = frame
            drawed_frame = draw_boxes(image=drawed_frame, boxes=det_bboxes, color='r', linewidth=3, det=False,
                                      boxIds=True, tracker=list_positions)
            if not det_bboxes_old == -1:
                drawed_frame = draw_boxes_old(image=drawed_frame, boxes=det_bboxes_old, color='r', linewidth=3,
                                              det=False, boxIds=True, tracker=list_positions)
            cv2.rectangle(drawed_frame, (10, 2), (120, 20), (255, 255, 255), -1)
            cv2.putText(drawed_frame, str(vidcap.get(cv2.CAP_PROP_POS_FRAMES)), (15, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 0, 0))

            cv2.imwrite(params['results_path'] + f"tracking_{str(frame_id)}_IoU.jpg", drawed_frame.astype(int))

        frame_id += 1
        old_frame = frame
        det_bboxes_old = det_bboxes

    
    mh = mm.metrics.create()
    summary = mh.compute(accumulator, metrics=['precision','recall','idp','idr','idf1'], name='acc')
    print(summary)

if __name__ == "__main__":

    params = {
        'show_boxes': True,
        'sota_method': "MOG2",
        'save_results': False,
        'results_path': "./W4/OF_NOTshifted_BB/",
        'use_optical_flow': False,
        'optical_flow_option': 3,
    }

    videos_s03 = [
        {
            "video_path": "./aic19-track1-mtmc-train/train/S03/c010/vdo.avi",
            "roi_path": "./aic19-track1-mtmc-train/train/S03/c010/roi.jpg",
            "det_path": "./aic19-track1-mtmc-train/train/S03/c010/det/det_yolo3.txt",
            # "det_path": "./aic19-track1-mtmc-train/train/S03/c010/gt/gt.txt",
            "gt_path": "./aic19-track1-mtmc-train/train/S03/c010/gt/gt.txt",
            "s_num": "S03",
            "c_num": "c010",
        },
        {
            "video_path": "./aic19-track1-mtmc-train/train/S03/c011/vdo.avi",
            "roi_path": "./aic19-track1-mtmc-train/train/S03/c011/roi.jpg",
            "det_path": "./aic19-track1-mtmc-train/train/S03/c011/det/det_yolo3.txt",
            # "det_path": "./aic19-track1-mtmc-train/train/S03/c011/gt/gt.txt",
            "gt_path": "./aic19-track1-mtmc-train/train/S03/c011/gt/gt.txt",
            "s_num": "S03",
            "c_num": "c011",
        },
        {
            "video_path": "./aic19-track1-mtmc-train/train/S03/c012/vdo.avi",
            "roi_path": "./aic19-track1-mtmc-train/train/S03/c012/roi.jpg",
            "det_path": "./aic19-track1-mtmc-train/train/S03/c012/det/det_yolo3.txt",
            # "det_path": "./aic19-track1-mtmc-train/train/S03/c012/gt/gt.txt",
            "gt_path": "./aic19-track1-mtmc-train/train/S03/c012/gt/gt.txt",
            "s_num": "S03",
            "c_num": "c012",
        },
        {
            "video_path": "./aic19-track1-mtmc-train/train/S03/c013/vdo.avi",
            "roi_path": "./aic19-track1-mtmc-train/train/S03/c013/roi.jpg",
            "det_path": "./aic19-track1-mtmc-train/train/S03/c013/det/det_yolo3.txt",
            # "det_path": "./aic19-track1-mtmc-train/train/S03/c013/gt/gt.txt",
            "gt_path": "./aic19-track1-mtmc-train/train/S03/c013/gt/gt.txt",
            "s_num": "S03",
            "c_num": "c013",
        },
        {
            "video_path": "./aic19-track1-mtmc-train/train/S03/c014/vdo.avi",
            "roi_path": "./aic19-track1-mtmc-train/train/S03/c014/roi.jpg",
            "det_path": "./aic19-track1-mtmc-train/train/S03/c014/det/det_yolo3.txt",
            # "det_path": "./aic19-track1-mtmc-train/train/S03/c014/gt/gt.txt",
            "gt_path": "./aic19-track1-mtmc-train/train/S03/c014/gt/gt.txt",
            "s_num": "S03",
            "c_num": "c014",
        },
        {
            "video_path": "./aic19-track1-mtmc-train/train/S03/c015/vdo.avi",
            "roi_path": "./aic19-track1-mtmc-train/train/S03/c015/roi.jpg",
            "det_path": "./aic19-track1-mtmc-train/train/S03/c015/det/det_yolo3.txt",
            # "det_path": "./aic19-track1-mtmc-train/train/S03/c015/gt/gt.txt",
            "gt_path": "./aic19-track1-mtmc-train/train/S03/c015/gt/gt.txt",
            "s_num": "S03",
            "c_num": "c015",
        }
    ]

    for video in videos_s03:

        vidcap = cv2.VideoCapture(video['video_path'])
        frame_count = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
        print("Total frames: ", frame_count)


        print("--------------- VIDEO ---------------")
        print(video["s_num"], video["c_num"])
        print("--------------- PARAMS ---------------")
        pprint(params)
        print("--------------------------------------")


        eval_tracking_MaximumOverlap(vidcap, frame_count, params, video)