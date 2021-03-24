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

import imageio
import argparse
from tqdm import tqdm

sys.path.append('../W1')
from aicity_reader import read_annotations, read_detections, group_by_frame
from voc_evaluation import voc_eval
from utils import draw_boxes, save_gif

import task1_1_faster, task1_1_retina

model_detect = {
    'faster' : task1_1_faster.detect,
    'retina' : task1_1_retina.detect
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

    parser.add_argument('--min_conf', type=float, default=0.5,
                        help='minimum confidence score to accept a detection')

    parser.add_argument('--min_iou', type=float, default=0.5,
                        help='minimum intersection over union to consider a detection a true positive')

    parser.add_argument('--detect', action='store_true',
                        help='detect cars using the specified model')

    parser.add_argument('--visualize', action='store_true',
                        help='show groundtruth and detections for each frame')

    parser.add_argument('--create_gif', action='store_true',
                        help='create gif using groundtruth and detections for each frame')

    parser.add_argument('--evaluate', action='store_true',
                        help='evaluate the detections')

    return parser.parse_args(args)



def filter_by_conf(detections, conf_thr=0.5):
    filtered_detections = []
    for det in detections:
        if det.confidence >= conf_thr:
            filtered_detections.append(det)

    return filtered_detections



if __name__ == '__main__':
    args = parse_args()

    det_path = 'results/task1_1/' + args.model + '/detections.txt'

    if args.detect:
        det_path = model_detect[args.model](args.video_path)

    if args.visualize:
        gt = read_annotations(args.gt_path, grouped=True, use_parked=True)
        det = read_detections(det_path, grouped=False)

        det = group_by_frame(filter_by_conf(det, conf_thr=args.min_conf))

        vidcap = cv2.VideoCapture(args.video_path)
        # vidcap.set(cv2.CAP_PROP_POS_FRAMES, initial_frame)  # to start from frame #frame_id
        num_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))

        for frame_id in range(num_frames):
            _, frame = vidcap.read()

            if frame_id >= 1755 and frame_id <= 1835:
                frame = draw_boxes(frame, gt[frame_id], color='g')
                frame = draw_boxes(frame, det[frame_id], color='b', det=True)

                cv2.imshow('frame', frame)
                if cv2.waitKey() == 113:  # press q to quit
                    break

        cv2.destroyAllWindows()

    if args.create_gif:
        gt = read_annotations(args.gt_path, grouped=True, use_parked=True)
        det = read_detections(det_path, grouped=False)

        det = group_by_frame(filter_by_conf(det, conf_thr=args.min_conf))

        vidcap = cv2.VideoCapture(args.video_path)
        initial_frame = 1755
        final_frame = 1835
        vidcap.set(cv2.CAP_PROP_POS_FRAMES, initial_frame)  # to start from frame #frame_id

        video_output_path = det_path.split('.txt')[0] + '/'
        os.makedirs(video_output_path, exist_ok=True)

        for frame_id in tqdm(range(initial_frame, final_frame), desc='Saving frames to create a gif'):
            _, frame = vidcap.read()

            frame = draw_boxes(frame, gt[frame_id], color='g')
            frame = draw_boxes(frame, det[frame_id], color='b', det=True)

            cv2.imwrite(video_output_path + '{}.png'.format(frame_id), frame)

        save_gif(video_output_path, video_output_path.split('detections')[0] + 'detections.gif')

    if args.evaluate:
        gt = read_annotations(args.gt_path, grouped=False, use_parked=True)
        det = read_detections(det_path, grouped=False)

        det = filter_by_conf(det, conf_thr=args.min_conf)

        test_perc = 0.75
        test_gt = get_test_subset(gt, num_frames=2141, test_perc=test_perc)
        test_det = get_test_subset(det, num_frames=2141, test_perc=test_perc)

        rec, prec, ap = voc_eval(test_det, group_by_frame(test_gt), args.min_iou, use_confidence=True)
        print('AP' + str(args.min_iou) + ': ', ap)
