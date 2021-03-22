import sys, os, cv2
import torch
assert torch.__version__.startswith("1.7")   # need to manually install torch 1.8 if Colab changes its default version
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

import argparse
from tqdm import tqdm

sys.path.append('../W1')
from aicity_reader import read_annotations, read_detections, group_by_frame
from voc_evaluation import voc_eval
from utils import draw_boxes

import task1_1_faster, task1_1_retina

model_detect = {
    'faster' : task1_1_faster.detect,
    'retina' : task1_1_retina.detect
}

detection_paths = {
    'faster' : 'results/task1_1_faster/faster_rcnn_R_50_FPN_3x/detections.txt',
    'retina' : 'results/task1_1_retina/retinanet_R_50_FPN_3x/detections.txt',
    'yolo' : 'results/yolo/detections.txt'
}


def get_test_subset(annotations, num_frames=2141, test_perc=0.75):
    initial_test_frame = int((1-test_perc) * num_frames)
    test_annotations = []
    for annot in annotations:
        if annot.frame >= initial_test_frame:
            test_annotations.append(annot)

    return test_annotations


def parse_args(args=sys.argv[1:]):
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str, default='faster',
                        help='pre-trained model to run inference on AICity Challenge dataset')

    parser.add_argument('--video_path', type=str, default='/mnt/gpid08/users/ian.riera/AICity_data/train/S03/c010/vdo.avi',
                        help='path to AICity Challenge video')

    parser.add_argument('--gt_path', type=str, default='/mnt/gpid08/users/ian.riera/AICity_data/train/S03/c010/ai_challenge_s03_c010-full_annotation.xml',
                        help='path to AICity Challenge annotations')

    # parser.add_argument('--det_path', type=str, default='results/task1_1_faster/faster_rcnn_R_50_FPN_3x/detections.txt',
    #                     help='path to AICity Challenge detections')

    parser.add_argument('--detect', action='store_true',
                        help='detect cars using the specified model')

    parser.add_argument('--show_gt_det', action='store_true',
                        help='show groundtruth and detections for each frame')

    parser.add_argument('--evaluate', action='store_true',
                        help='evaluate the detections')

    return parser.parse_args(args)

if __name__ == '__main__':
    args = parse_args()

    det_path = detection_paths[args.model]

    if args.detect:
        det_path = model_detect[args.model](args.video_path)

    if args.show_gt_det:
        gt = read_annotations(args.gt_path, grouped=True, use_parked=True)
        det = read_detections(det_path, grouped=True)

        vidcap = cv2.VideoCapture(args.video_path)
        # vidcap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)  # to start from frame #frame_id
        num_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))

        for frame_id in range(num_frames):
            _, frame = vidcap.read()

            frame = draw_boxes(frame, gt[frame_id], color='g')
            frame = draw_boxes(frame, det[frame_id], color='b', det=True)

            cv2.imshow('frame', frame)
            if cv2.waitKey() == 113:  # press q to quit
                break

        cv2.destroyAllWindows()

    if args.evaluate:
        gt = read_annotations(args.gt_path, grouped=False, use_parked=True)
        det = read_detections(det_path)

        test_perc = 0.75
        test_gt = get_test_subset(gt, num_frames=2141, test_perc=test_perc)
        test_det = get_test_subset(det, num_frames=2141, test_perc=test_perc)

        iou_thr = 0.7
        rec, prec, ap = voc_eval(test_det, group_by_frame(test_gt), iou_thr, use_confidence=True)
        print('AP' + str(iou_thr) + ': ', ap)
