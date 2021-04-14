import os, sys, cv2, argparse
import numpy as np
from tqdm import tqdm

sys.path.append('./W1')
from bounding_box import BoundingBox
from aicity_reader import read_annotations, read_detections, group_by_frame
from utils import draw_boxes

sys.path.append("./W3")
from tracking import Tracking
import motmetrics as mm

sys.path.append("./W3/sort")
from sort import Sort


det_filename = {
    'faster': 'detections.txt',
    'mog': 'detections.txt',
    'mask': 'det_mask_rcnn.txt',
    'ssd': 'det_ssd512.txt',
    'yolo': 'det_yolo3.txt'
}


def track_kalman(det_bboxes, tracker):
    bboxes = []
    for d in det_bboxes:
        bboxes.append([d.xtl, d.ytl, d.xbr, d.ybr, d.confidence])
    tracks = tracker.update(np.array(bboxes))

    tracked_bboxes = []
    for t in tracks:
        tracked_bboxes.append(BoundingBox(
            id=int(t[4]) - 1,
            label='car',
            frame=int(det_bboxes[0].frame),
            xtl=float(t[0]),
            ytl=float(t[1]),
            xbr=float(t[2]),
            ybr=float(t[3]),
            occluded=False,
            parked=False
        ))
    return tracked_bboxes, tracker


def track_overlap(det_bboxes, det_bboxes_old, tracker):
    return tracker.set_frame_ids(det_bboxes, det_bboxes_old)


def filter_bboxes_size(det_bboxes):
    filtered_bboxes = []
    for b in det_bboxes:
        if b.width >= 70 and b.height >= 56:
            filtered_bboxes.append(b)

    return filtered_bboxes


def filter_bboxes_parked(list_positions, list_positions_bboxes, var_thr):
    det_bboxes_filtered = []
    for index, centers_bboxes_same_id in list_positions.items():
        id_var = np.mean(np.std(centers_bboxes_same_id, axis=0))
        if id_var > var_thr:
            for b in list_positions_bboxes[index]:
                det_bboxes_filtered.append(b)

    return det_bboxes_filtered


def filter_detections_parked(params, mse_thr=300, var_thr=50):
    print("[INFO] Filtering detections to remove parked cars")

    vidcap = cv2.VideoCapture(params['video_path'])
    num_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))

    gt = read_detections(params["gt_path"], grouped=True)
    det = read_detections(params["det_path"], grouped=True, confidenceThr=0.5)

    detections = []
    list_positions = {}
    list_positions_bboxes = {}

    center_seen_last5frames = {}
    id_seen_last5frames = {}

    if params['track_method'] == 'overlap':
        tracker = Tracking()
    elif params['track_method'] == 'kalman':
        tracker = Sort()

    det_bboxes_old = -1

    esc_pressed = False

    for frame_id in tqdm(range(num_frames)):
        _, frame = vidcap.read()

        gt_bboxes = []
        if frame_id in gt:
            gt_bboxes = gt[frame_id]

        det_bboxes = []
        if frame_id in det:
            if params['track_method'] == 'overlap':
                det_bboxes = track_overlap(det[frame_id], det_bboxes_old, tracker)
            elif params['track_method'] == 'kalman':
                det_bboxes, tracker = track_kalman(det[frame_id], tracker)

        detections += det_bboxes

        id_seen = []

        for object_bb in det_bboxes:
            if object_bb.id in list(list_positions.keys()):
                if frame_id < 5:
                    id_seen_last5frames[object_bb.id] = object_bb.id
                    center_seen_last5frames[object_bb.id] = object_bb.center

                list_positions[object_bb.id].append([int(x) for x in object_bb.center])
                list_positions_bboxes[object_bb.id].append(object_bb)
            else:
                if frame_id < 5:
                    id_seen_last5frames[object_bb.id] = object_bb.id
                    center_seen_last5frames[object_bb.id] = object_bb.center

                id_seen.append(object_bb)
                list_positions[object_bb.id] = [[int(x) for x in object_bb.center]]
                list_positions_bboxes[object_bb.id] = [object_bb]

        # To detect pared cars
        for bbox in id_seen:
            for idx in list(id_seen_last5frames.keys()):
                if idx != bbox.id:
                    center = [center_seen_last5frames[idx]]
                    mse = (np.square(np.subtract(np.array(center), np.array([int(x) for x in bbox.center])))).mean()
                    if mse < mse_thr:
                        setattr(bbox, 'id', idx)

        if params['show_boxes'] and not esc_pressed:
            frame = draw_boxes(image=frame, boxes=gt_bboxes, color='g', linewidth=3, boxIds=False,
                               tracker=list_positions)
            frame = draw_boxes(image=frame, boxes=det_bboxes, color='r', linewidth=3, det=False, boxIds=True,
                               tracker=list_positions)
            cv2.putText(frame, str(vidcap.get(cv2.CAP_PROP_POS_FRAMES)), (15, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 0, 0))
            cv2.imshow('Frame', frame)
            pressed_key = cv2.waitKey(30)
            if pressed_key == 113:  # press q to quit
                esc_pressed = True
                cv2.destroyAllWindows()
            elif pressed_key == 27:
                sys.exit()

        det_bboxes_old = det_bboxes

    det_bboxes_filtered = filter_bboxes_parked(list_positions, list_positions_bboxes, var_thr)

    return group_by_frame(det_bboxes_filtered)


def eval_tracking(params, det):
    print("Evaluating Tracking")

    vidcap = cv2.VideoCapture(params['video_path'])
    num_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))

    gt = read_detections(params["gt_path"], grouped=True)

    # Create an accumulator that will be updated during each frame
    accumulator = mm.MOTAccumulator(auto_id=True)

    esc_pressed = False

    for frame_id in tqdm(range(num_frames)):
        _, frame = vidcap.read()
        gt_bboxes = []
        if frame_id in gt:
            gt_bboxes = gt[frame_id]

        det_bboxes = []
        if frame_id in det:
            det_bboxes = filter_bboxes_size(det[frame_id])

        objs = [bbox.center for bbox in gt_bboxes]
        hyps = [bbox.center for bbox in det_bboxes]

        if params['show_boxes'] and not esc_pressed:
            frame = draw_boxes(image=frame, boxes=gt_bboxes, color='g', linewidth=3, boxIds=False)
            frame = draw_boxes(image=frame, boxes=det_bboxes, color='r', linewidth=3, det=False, boxIds=True)
            cv2.putText(frame, str(vidcap.get(cv2.CAP_PROP_POS_FRAMES)), (15, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 0, 0))
            cv2.imshow('Frame', frame)
            pressed_key = cv2.waitKey(30)
            if pressed_key == 113:  # press q to quit
                esc_pressed = True
                cv2.destroyAllWindows()
            elif pressed_key == 27:
                sys.exit()

        accumulator.update(
            [bbox.id for bbox in gt_bboxes],  # Ground truth objects in this frame
            [bbox.id for bbox in det_bboxes],  # Detector hypotheses in this frame
            mm.distances.norm2squared_matrix(objs, hyps)
            # Distances from object 1 to hypotheses 1, 2, 3 and Distances from object 2 to hypotheses 1, 2, 3
        )

    mh = mm.metrics.create()
    summary = mh.compute(accumulator, metrics=['precision', 'recall', 'idp', 'idr', 'idf1'], name='acc')
    print(summary)


def parse_args(args=sys.argv[1:]):
    parser = argparse.ArgumentParser()

    parser.add_argument('--track_method', type=str, default='overlap',
                        choices=['overlap', 'kalman'],
                        help='method used to track cars')

    parser.add_argument('--det_method', type=str, default='faster',
                        choices=['faster', 'mog', 'mask', 'ssd', 'yolo'],
                        help='load detections obtained with this method')

    parser.add_argument('--det_dir', type=str, default="./W5/task1_1/detections",
                        help='path from where to load detections')

    parser.add_argument('--data_path', type=str, default="./data/AICity_data/train",
                        help='path to sequences of AICity')

    parser.add_argument('--seq', type=str, default='S03/c010',
                        help='sequence/camera from AICity dataset')

    parser.add_argument('--show_boxes', action='store_true',
                        help='show bounding boxes')

    # parser.add_argument('--save_results', action='store_true',
    #                     help='save detections')

    return parser.parse_args(args)


def args_to_params(args):
    return {
        'track_method': args.track_method,
        'det_path': os.path.join(args.det_dir, args.det_method, args.seq, det_filename[args.det_method]),
        'video_path': os.path.join(args.data_path, args.seq, 'vdo.avi'),
        'gt_path': os.path.join(args.data_path, args.seq, 'gt/gt.txt'),
        'show_boxes': args.show_boxes,
    }


if __name__ == "__main__":
    args = parse_args()
    params = args_to_params(args)

    det_filtered = filter_detections_parked(params)

    eval_tracking(params, det_filtered)